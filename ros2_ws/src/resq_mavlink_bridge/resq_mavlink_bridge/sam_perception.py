"""Open-vocabulary hazard perception for SAHARA (SAM 3.1 + Muse Spark vision).

Segments the drone's RGB camera by text ("smoke", "downed power line",
"landslide", ...) with Meta's SAM 3.1, and in parallel has Muse Spark read
the scene for hazards SAM misses (sahara_ai/perception.py), and publishes what it finds as hazard
sightings on /detections/hazards, the same topic and format as the thermal
detector, so the mission bridge projects them to the ground, merges them into
hazard zones and routes rescue teams around them.

Runs off the control loop. Two triggers:
  - an optional slow sweep while a mission is flying (RESQ_SAM_SWEEP_S, off by default)
  - on demand: a JSON request on /ai/sam/request {"id": .., "concepts": [..]}
    (the AI commander's inspect/find tools); the answer, with every box, goes
    to /ai/sam/result.
An annotated JPEG of the last analysed frame is published for the dashboard.
"""
import json
import os
import re
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import String


def _import_sahara_ai():
    """sahara_ai lives at the repo root; ros_env.sh puts it on PYTHONPATH."""
    try:
        import sahara_ai  # noqa: F401
    except ImportError:
        root = Path(os.environ.get("SAHARA_ROOT", Path.home() / "Desktop" / "SIH-Drone-Simulation"))
        sys.path.insert(0, str(root))
    from sahara_ai import MetaClient
    from sahara_ai.perception import CONCEPTS, scan
    return MetaClient, CONCEPTS, scan


MetaClient, CONCEPTS, scan = _import_sahara_ai()

# Periodic sweep is OFF by default (SAM 3.1 calls cost credits): SAM runs on
# demand from the commander. RESQ_SAM_SWEEP_S=20 turns a sweep on for demos.
SWEEP_S = float(os.environ.get("RESQ_SAM_SWEEP_S", "0"))
FLYING_STAGES = {"SEARCHING", "VERIFYING", "MARKING", "HOVERING", "RETURNING", "INSPECTING"}
MIN_ALT_M = 3.0
STATE_RE = re.compile(r"Altitude:\s*(-?[\d.]+)\s*m,\s*Mission:\s*(\w+)")
COLOURS = {"smoke": (139, 116, 100), "power_line": (8, 179, 234), "landslide": (14, 64, 146)}


class SamPerception(Node):
    def __init__(self):
        super().__init__("sam_perception")
        self.client = MetaClient()
        self.frame = None          # (bgr ndarray, wall time)
        self.altitude, self.stage = 0.0, "IDLE"
        self.busy = threading.Lock()
        self.last_sweep = 0.0
        self.hazards_pub = self.create_publisher(String, "/detections/hazards", 10)
        self.result_pub = self.create_publisher(String, "/ai/sam/result", 10)
        self.debug_pub = self.create_publisher(CompressedImage, "/ai/sam/debug_image/compressed", 2)
        self.create_subscription(Image, "/camera/image", self.on_image, qos_profile_sensor_data)
        self.create_subscription(String, "/resq/drone/state", self.on_state, 10)
        self.create_subscription(String, "/ai/sam/request", self.on_request, 10)
        self.create_timer(1.0, self.tick)
        mode = "MOCK (no MODEL_API_KEY)" if self.client.mock else self.client.cfg.sam_model
        sweep = f"every {SWEEP_S:.0f} s" if SWEEP_S > 0 else "off (on demand only)"
        self.get_logger().info(f"SAM perception online: {mode}, sweep {sweep}, concepts {list(CONCEPTS)}")

    # ------------------------------------------------------------ inputs
    def on_image(self, msg):
        try:
            buf = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)[:, : msg.width * 3]
            img = buf.reshape(msg.height, msg.width, 3)
            if msg.encoding.lower() == "rgb8":
                img = img[:, :, ::-1]
            self.frame = (img.copy(), time.time())
        except ValueError:
            pass

    def on_state(self, msg):
        m = STATE_RE.search(msg.data)
        if m:
            self.altitude, self.stage = float(m.group(1)), m.group(2)

    def on_request(self, msg):
        try:
            req = json.loads(msg.data)
        except ValueError:
            return
        concepts = {c: CONCEPTS.get(c, c.replace(" ", "_")) for c in req.get("concepts") or CONCEPTS}
        threading.Thread(target=self.run, args=(concepts, req.get("id")), daemon=True).start()

    # ------------------------------------------------------------- sweep
    def tick(self):
        if SWEEP_S <= 0 or self.stage not in FLYING_STAGES or self.altitude < MIN_ALT_M:
            return
        if time.time() - self.last_sweep >= SWEEP_S and not self.busy.locked():
            self.last_sweep = time.time()
            threading.Thread(target=self.run, args=(CONCEPTS, None), daemon=True).start()

    def run(self, concepts, request_id):
        if self.frame is None or not self.busy.acquire(blocking=request_id is not None):
            return
        try:
            img, captured = self.frame
            alt = self.altitude
            ok, jpeg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])
            if not ok:
                return
            t0 = time.time()
            hazards, report = scan(self.client, jpeg.tobytes(), concepts,
                                   size=(img.shape[1], img.shape[0]), alt=max(alt, 1.0))
            took = time.time() - t0
            # only hazard types the map knows go to the bridge; everything is reported back
            known = [h for h in hazards if h["type"] in COLOURS or h["type"] in ("fire", "flood", "structure")]
            if known and alt >= MIN_ALT_M:
                self.hazards_pub.publish(String(data=json.dumps({"stamp": captured, "alt": alt, "hazards": known})))
            self.publish_debug(img, hazards)   # first: the dashboard fetches it when the result lands
            self.result_pub.publish(String(data=json.dumps({
                "id": request_id, "stamp": captured, "alt": alt, "latency_s": round(took, 2),
                "hazards": hazards, "report": report, "mock": bool(self.client.mock)})))
            found = ", ".join(f"{h['type']} {h['conf']:.2f}" for h in hazards) or "nothing"
            errors = [f"{c}: {r['error']}" for c, r in report.items() if r.get("error")]
            self.get_logger().info(f"SAM 3.1 ({took:.1f} s, {len(concepts)} concepts): {found}"
                                   + (f" | errors: {'; '.join(errors)}" if errors else ""))
        finally:
            self.busy.release()

    def publish_debug(self, img, hazards):
        out = img.copy()
        for h in hazards:
            x1, y1, x2, y2 = h["box"]
            col = COLOURS.get(h["type"], (0, 200, 255))
            cv2.rectangle(out, (x1, y1), (x2, y2), col, 2)
            cv2.putText(out, f"{h['type']} {h['conf']:.2f}", (x1, max(y1 - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
        ok, jpeg = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if ok:
            msg = CompressedImage()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.format = "jpeg"
            msg.data = jpeg.tobytes()
            self.debug_pub.publish(msg)


def main():
    rclpy.init()
    node = SamPerception()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
