"""Open-vocabulary hazard perception with Meta's models (ROS-free logic).

Two models look at the same frame in parallel:
  - SAM 3.1 segments each text concept ("smoke", "downed power line", ...)
    and returns exact boxes and masks when it recognises the concept;
  - Muse Spark (vision) reads the whole scene and names the hazards it sees
    with boxes, which covers what SAM misses (in testing on simulator
    renders SAM found smoke but not cables lying on a street; Muse Spark
    found the downed power line).
Where both report a type, SAM's box is kept and Muse Spark is added as
corroboration. Results become the same hazard sightings the thermal detector
publishes on /detections/hazards, so the bridge projects, merges and routes
around them unchanged.
"""
import json
import math
from concurrent.futures import ThreadPoolExecutor

from .client import image_part

# concept phrase -> hazard type. One concept per SAM request (API rule).
CONCEPTS = {
    "smoke": "smoke",
    "downed power line": "power_line",
    "landslide": "landslide",
}
MIN_CONF = 0.55
# Ignore boxes covering nearly the whole frame: at 8 m a "smoke" box that big
# is the haze over everything, not a locatable plume.
MAX_AREA_FRAC = 0.85


def records_to_hazards(records, kind, min_conf=MIN_CONF):
    """SAM records for one concept -> hazard sightings for /detections/hazards."""
    out = []
    for r in records:
        w, h = r.size
        if not w or not h or r.conf < min_conf or r.area_frac > MAX_AREA_FRAC:
            continue
        x1, y1, x2, y2 = r.box
        cx, cy = r.center
        out.append({
            "type": kind,
            "cx": round(cx / w - 0.5, 4), "cy": round(cy / h - 0.5, 4),
            "r_px": round(0.5 * math.hypot(x2 - x1, y2 - y1), 1),
            "conf": round(r.conf, 3),
            "sensors": ["sam3.1"],
            "box": [x1, y1, x2, y2], "size": [w, h],
        })
    return out


# People are not in this list: they are the YOLO/thermal detector's job, and
# asking a vision model to box people from a drone tripped the API's content
# filter in testing.
VISION_TYPES = ["smoke", "fire", "flood", "power_line", "landslide", "collapsed_structure"]
VISION_SCHEMA = {"type": "json_schema", "json_schema": {"name": "hazards", "strict": True, "schema": {
    "type": "object", "additionalProperties": False, "required": ["hazards"],
    "properties": {"hazards": {"type": "array", "items": {
        "type": "object", "additionalProperties": False, "required": ["type", "box", "confidence", "evidence"],
        "properties": {
            "type": {"type": "string", "enum": VISION_TYPES},
            "box": {"type": "array", "items": {"type": "integer"},
                    "description": "x1, y1, x2, y2 in 0-1000 of the image width / height"},
            "confidence": {"type": "number"},
            "evidence": {"type": "string"}}}}}}}}
VISION_PROMPTS = (
    "Aerial site-survey photo taken straight down from about {alt} m. List the environmental safety hazards "
    "clearly visible: smoke, fire, standing flood water, downed power lines (cables lying on the ground or "
    "sagging low, fallen utility poles), landslide or mudflow, collapsed building. Box each one in 0-1000 "
    "coordinates. Only what is visible; an empty list if none.",
    # fallback wording if the first is filtered
    "Top-down survey image of an outdoor area. Identify visible infrastructure and terrain hazards (smoke, "
    "fire, water pooling, fallen utility poles or cables on the ground, mud or slope failure, damaged "
    "building) and give a 0-1000 bounding box for each. Return an empty list if there are none.",
)
# vision type -> SAHARA hazard type (structures and people are handled elsewhere)
VISION_TO_HAZARD = {"smoke": "smoke", "fire": "fire", "flood": "flood", "power_line": "power_line", "landslide": "landslide"}


def vision_hazards(client, jpeg, size, alt=8.0, min_conf=MIN_CONF):
    """Muse Spark vision pass -> hazard sightings (same format as SAM's)."""
    w, h = size
    for i, prompt in enumerate(VISION_PROMPTS):
        try:
            out = client.chat([{"role": "user", "content": [
                {"type": "text", "text": prompt.format(alt=round(alt or 8.0))}, image_part(jpeg)]}],
                model=client.cfg.vision_model, response_format=VISION_SCHEMA, temperature=0)
            break
        except Exception as e:  # noqa: BLE001 - retry once with other wording if filtered
            if "content_policy" not in str(e) or i == len(VISION_PROMPTS) - 1:
                raise
    found = json.loads(out["choices"][0]["message"]["content"]).get("hazards", [])
    hazards = []
    for f in found:
        kind = VISION_TO_HAZARD.get(f.get("type"))
        box = f.get("box") or []
        if kind is None or len(box) != 4 or float(f.get("confidence", 0)) < min_conf:
            continue
        x1, y1, x2, y2 = [int(round(v / 1000.0 * (w if i % 2 == 0 else h))) for i, v in enumerate(box)]
        x1, x2 = sorted((max(0, x1), min(w - 1, x2)))
        y1, y2 = sorted((max(0, y1), min(h - 1, y2)))
        hazards.append({
            "type": kind, "cx": round((x1 + x2) / 2 / w - 0.5, 4), "cy": round((y1 + y2) / 2 / h - 0.5, 4),
            "r_px": round(0.5 * math.hypot(x2 - x1, y2 - y1), 1), "conf": round(float(f["confidence"]), 3),
            "sensors": ["muse-spark"], "box": [x1, y1, x2, y2], "size": [w, h], "evidence": f.get("evidence", ""),
        })
    return hazards


def merge(sam_hazards, vision):
    """SAM boxes win per type; Muse Spark corroborates them or fills the gaps."""
    sam_types = {h["type"] for h in sam_hazards}
    merged = [dict(h) for h in sam_hazards]
    for h in merged:
        if any(v["type"] == h["type"] for v in vision):
            h["sensors"] = ["sam3.1", "muse-spark"]
    return merged + [v for v in vision if v["type"] not in sam_types]


def scan(client, jpeg, concepts=CONCEPTS, workers=4, vision=True, size=(640, 480), alt=8.0):
    """Run every concept on one JPEG concurrently.

    Returns (hazards, per-concept results {concept: {"n":.., "best":.., "error":..}}).
    A failed concept is reported, never raised: perception must not stop the node.
    """
    def one(item):
        concept, kind = item
        try:
            recs, _ = client.segment(concept, jpeg)
            return concept, records_to_hazards(recs, kind), {"n": len(recs), "best": max((r.conf for r in recs), default=None)}
        except Exception as e:  # noqa: BLE001 - network/API errors are data here
            return concept, [], {"n": 0, "best": None, "error": str(e)[:200]}

    hazards, report = [], {}
    with ThreadPoolExecutor(max_workers=workers + 1) as ex:
        vfut = ex.submit(vision_hazards, client, jpeg, size, alt) if vision else None
        for concept, hz, info in ex.map(one, concepts.items()):
            hazards += hz
            report[concept] = info
        if vfut is not None:
            try:
                vis = vfut.result()
                report["muse-spark vision"] = {"n": len(vis), "best": max((v["conf"] for v in vis), default=None)}
            except Exception as e:  # noqa: BLE001
                vis = []
                report["muse-spark vision"] = {"n": 0, "best": None, "error": str(e)[:200]}
            hazards = merge(hazards, vis)
    return hazards, report
