"""AI Mission Commander endpoints for the dashboard (sahara_ai on Meta's models).

  POST /api/ai/chat        {"text": ...}  -> {"turn": id}; the agent runs in the background
  GET  /api/ai/events      ?turn=id&from=N -> text/event-stream of that turn's events
                           (keepalives every 5 s; reconnect with from= to resume)
  POST /api/ai/confirm     {"action": "start_mission"}  operator confirmation
  POST /api/ai/cancel      stop the running turn (Cancel button)
  POST /api/ai/transcribe  raw WAV body (16 kHz mono PCM) -> {"text": ...}
  GET  /api/ai/status      mock/live, calls and estimated spend today
  POST /api/ai/reset       forget the conversation
  GET  /api/ai/sitrep      AI narrative for the SITREP (cached per mission state)
  GET  /api/ai/sam_frame   last frame SAM 3.1 analysed, with its boxes (JPEG)

Every newly confirmed casualty is triaged once by Muse Spark (vision) from
the hover frames; the note goes back to the bridge (annotate_victim) and so
appears on the map, the rescue cards, the SITREP and in the agent's context.

The commander runs in this process and talks to the drone through the
existing rclpy node (RosBridge): JSON commands on /resq/ai/command, SAM 3.1
requests on /ai/sam/request, results back on their result topics.
"""
import json
import os
import sys
import threading
import time
import uuid

from flask import Response, request, stream_with_context
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from sahara_ai import MetaClient  # noqa: E402
from sahara_ai.commander import Commander, summarize  # noqa: E402
from sahara_ai.triage import sitrep_narrative, triage  # noqa: E402

IDLE_STAGES = {"IDLE", "COMPLETE", "ABORTED", "PILOT", None}


class Waiters:
    """Thread-safe {id: result} rendezvous for JSON results on a topic."""

    def __init__(self):
        self.cond = threading.Condition()
        self.results = {}

    def on_msg(self, msg):
        try:
            data = json.loads(msg.data)
        except ValueError:
            return
        if data.get("id") is None:
            return
        with self.cond:
            self.results.setdefault(data["id"], []).append(data)
            self.cond.notify_all()

    def wait(self, rid, accept, timeout, cancel=None):
        deadline = time.time() + timeout
        with self.cond:
            while True:
                for r in self.results.get(rid, []):
                    if accept(r):
                        return r
                if cancel is not None and cancel.is_set():
                    return {"status": "interrupted", "reason": "a newer message from the operator"}
                left = deadline - time.time()
                if left <= 0:
                    return None
                self.cond.wait(min(left, 0.5))


class DroneBackend:
    """sahara_ai.commander's Backend, on top of the dashboard's RosBridge."""

    def __init__(self, ros, mission_file_fn, start_fn):
        self.cancel = threading.Event()
        self.ros, self.mission_file_fn, self.start_fn = ros, mission_file_fn, start_fn
        self.cmd_pub = ros.create_publisher(String, "/resq/ai/command", 10)
        self.sam_pub = ros.create_publisher(String, "/ai/sam/request", 10)
        self.cmd_results, self.sam_results = Waiters(), Waiters()
        ros.create_subscription(String, "/resq/ai/command_result", self.cmd_results.on_msg, 10)
        ros.create_subscription(String, "/ai/sam/result", self.sam_results.on_msg, 10)
        self.sam_jpeg = None
        ros.create_subscription(CompressedImage, "/ai/sam/debug_image/compressed",
                                lambda m: setattr(self, "sam_jpeg", bytes(m.data)), qos_profile_sensor_data)

    def landmarks(self):
        try:
            with open(self.mission_file_fn()) as f:
                m = json.load(f)
        except (OSError, ValueError):
            return []
        out = [{"label": z["label"], "east_m": z["east_m"], "north_m": z["north_m"]}
               for z in m.get("known_hazards", []) + m.get("prior_zones", []) if "label" in z]
        s = m.get("staging")
        if s:
            out.append({"label": s.get("label", "Staging"), "east_m": s["east_m"], "north_m": s["north_m"]})
        # duplicate names (e.g. several "Collapsed house"): number them
        seen = {}
        for lm in out:
            k = lm["label"]
            seen[k] = seen.get(k, 0) + 1
            if seen[k] > 1:
                lm["label"] = f"{k} {seen[k]}"
        return out

    def map(self):
        raw = self.ros.get_map()
        try:
            return json.loads(raw) if raw else None
        except ValueError:
            return None

    def mission_running(self):
        return (self.map() or {}).get("stage") not in IDLE_STAGES

    def command(self, cmd, wait_for=("ok", "accepted", "rejected"), timeout=10):
        if self.map() is None:
            return {"status": "error", "reason": "the mission bridge is not running (launch the simulation first)"}
        cid = cmd.get("id") or uuid.uuid4().hex[:8]
        self.cmd_pub.publish(String(data=json.dumps({**cmd, "id": cid})))
        res = self.cmd_results.wait(cid, lambda r: r.get("status") in wait_for or r.get("status") == "rejected",
                                    timeout, self.cancel)
        return res or {"status": "error", "reason": f"no answer from the drone within {timeout:.0f} s (is the sim running?)"}

    def sam(self, concepts, timeout=45):
        rid = uuid.uuid4().hex[:8]
        self.sam_pub.publish(String(data=json.dumps({"id": rid, "concepts": concepts})))
        res = self.sam_results.wait(rid, lambda r: True, timeout, self.cancel)
        return res or {"status": "error", "reason": "SAM perception node did not answer (is it running?)"}

    def land(self):
        ok, msg = self.ros.call_mission_land()
        return {"status": "ok" if ok else "error", "message": msg}

    def start(self):
        return self.start_fn()


def register(app, ros, mission_file_fn, start_fn):
    client = MetaClient()
    backend = DroneBackend(ros, mission_file_fn, start_fn)
    state = {"commander": Commander(client, backend)}
    busy = threading.Lock()

    turns = {}
    cond = threading.Condition()

    def run_turn(tid, text):
        t = turns[tid]
        try:
            for ev in state["commander"].run(text):
                with cond:
                    t["events"].append(ev)
                    cond.notify_all()
        except Exception as e:  # noqa: BLE001 - always finish the turn cleanly
            with cond:
                t["events"].append({"type": "error", "text": str(e)[:300]})
        finally:
            with cond:
                t["done"] = True
                cond.notify_all()
            busy.release()

    @app.post("/api/ai/chat")
    def ai_chat():
        text = ((request.get_json(silent=True) or {}).get("text") or "").strip()
        if not text:
            return {"ok": False, "message": "empty message"}, 400
        if not busy.acquire(blocking=False):
            # A newer message wins (e.g. "return home" during a long inspect):
            # interrupt the running turn and take over once it has stopped.
            backend.cancel.set()
            if not busy.acquire(timeout=20):
                return {"ok": False, "message": "the commander is still finishing the last action; try again"}, 409
        backend.cancel.clear()
        tid = uuid.uuid4().hex[:10]
        turns[tid] = {"events": [], "done": False, "t": time.time()}
        for old in [k for k, v in turns.items() if time.time() - v["t"] > 3600]:
            turns.pop(old, None)
        # Runs independently of any HTTP connection: a dropped stream (browser,
        # proxy, idle timeout during a long inspect) no longer stops the agent.
        threading.Thread(target=run_turn, args=(tid, text), daemon=True).start()
        return {"ok": True, "turn": tid}

    @app.get("/api/ai/events")
    def ai_events():
        tid = request.args.get("turn", "")
        start = int(request.args.get("from", 0) or 0)
        if tid not in turns:
            return {"ok": False, "message": "unknown turn"}, 404

        def gen():
            i = start
            while True:
                with cond:
                    t = turns[tid]
                    if i >= len(t["events"]) and not t["done"]:
                        cond.wait(5.0)
                    new, done = t["events"][i:], t["done"]
                for ev in new:
                    yield f"data: {json.dumps({**ev, 'i': i})}\n\n"
                    i += 1
                if done and i >= len(turns[tid]["events"]):
                    yield 'data: {"type": "done"}\n\n'
                    return
                if not new:
                    yield ": keepalive\n\n"

        return Response(stream_with_context(gen()), mimetype="text/event-stream",
                        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.post("/api/ai/confirm")
    def ai_confirm():
        action = (request.get_json(silent=True) or {}).get("action")
        if action != "start_mission":
            return {"ok": False, "message": f"unknown action {action!r}"}, 400
        # The running turn's start_mission is waiting for this; it sees the
        # drone take off on the map and carries on with the request.
        ok, msg = backend.start()
        return {"ok": ok, "message": msg}

    @app.post("/api/ai/cancel")
    def ai_cancel():
        """Operator pressed Cancel (or Stop): end the running turn."""
        backend.cancel.set()
        return {"ok": True}

    @app.post("/api/ai/transcribe")
    def ai_transcribe():
        audio = request.get_data()
        if not audio:
            return {"ok": False, "message": "no audio"}, 400
        if request.headers.get("Content-Type", "").split(";")[0] != "audio/wav":
            return {"ok": False, "message": "send 16-bit PCM WAV (the panel converts the recording)"}, 400
        try:
            return {"ok": True, "text": client.transcribe(audio)}
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "message": str(e)[:300]}, 502

    @app.get("/api/ai/status")
    def ai_status():
        spend = 0.0
        try:
            midnight = time.mktime(time.localtime()[:3] + (0, 0, 0, 0, 0, -1))
            with open(client.cfg.usage_log) as f:
                for line in f:
                    r = json.loads(line)
                    if not r.get("mock") and r.get("t", 0) >= midnight:
                        spend += r.get("est_cost_usd", 0)
        except (OSError, ValueError):
            pass
        return {"mock": bool(client.mock), "model": client.cfg.agent_model, "sam_model": client.cfg.sam_model,
                "calls_today": client.calls_today(), "call_cap": client.cfg.max_calls_per_day,
                "est_spend_today_usd": round(spend, 4)}

    @app.get("/api/ai/sam_frame")
    def ai_sam_frame():
        if backend.sam_jpeg is None:
            return {"ok": False, "message": "no SAM frame yet"}, 404
        return Response(backend.sam_jpeg, mimetype="image/jpeg", headers={"Cache-Control": "no-store"})

    sitrep_cache = {}

    @app.get("/api/ai/sitrep")
    def ai_sitrep():
        situation = summarize(backend.map())
        situation.pop("recent_events", None)
        key = json.dumps({k: situation.get(k) for k in ("casualties", "hazards", "stage")}, sort_keys=True)
        if key not in sitrep_cache:
            try:
                sitrep_cache[key] = sitrep_narrative(client, situation)
            except Exception as e:  # noqa: BLE001
                return {"ok": False, "message": str(e)[:300]}, 502
        return {"ok": True, "text": sitrep_cache[key], "model": client.cfg.agent_model, "mock": bool(client.mock)}

    @app.post("/api/ai/reset")
    def ai_reset():
        state["commander"] = Commander(client, backend)
        return {"ok": True}

    threading.Thread(target=_triage_watch, args=(client, backend, ros), daemon=True).start()
    return backend


def _triage_watch(client, backend, ros):
    """Triage each newly confirmed casualty once, from the frames at its hover."""
    done, last_t, failures, retry_at, frames = set(), 0.0, {}, {}, {}
    while True:
        time.sleep(1.0)
        m = backend.map()
        if not m:
            continue
        t = m.get("t") or 0.0
        if t < last_t:  # mission clock went back: a new mission, IDs restart at V1
            done.clear(); failures.clear(); retry_at.clear(); frames.clear()
        last_t = t
        for v in m.get("victims", []):
            vid = v.get("id")
            if vid in done or v.get("ai_triage"):
                done.add(vid)
                continue
            if time.time() < retry_at.get(vid, 0):
                continue
            if vid not in frames:
                # frames from the moment we first see the casualty (the drone is
                # still hovering over them); reused on retries
                frames[vid] = (ros.get_frame(), ros.get_thermal_frame())
            rgb, thermal = frames[vid]
            if rgb is None:
                done.add(vid)
                continue
            context = f"casualty {vid}; risks from the map: {', '.join(v.get('risks') or []) or 'none known'}; " \
                      f"{'seen moving' if v.get('moving') else 'not seen moving'}"
            try:
                note = triage(client, rgb, thermal, (m.get("drone") or {}).get("alt", 8.0), context)
            except Exception as e:  # noqa: BLE001 - triage is a bonus; never break the dashboard
                failures[vid] = failures.get(vid, 0) + 1
                if failures[vid] >= 4:
                    done.add(vid)
                    ros.get_logger().warn(f"AI triage for {vid} failed {failures[vid]} times, giving up: {e}")
                else:
                    retry_at[vid] = time.time() + 5.0 * failures[vid]
                    ros.get_logger().warn(f"AI triage for {vid} failed (attempt {failures[vid]}), retrying: {e}")
                continue
            done.add(vid)
            backend.command({"cmd": "annotate_victim", "victim": vid, "ai_triage": note})
