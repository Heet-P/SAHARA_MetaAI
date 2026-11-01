"""SAHARA AI Mission Commander: Muse Spark drives the drone through tools.

A responder says what they know ("reports of two people trapped at the
collapsed police station, check it"); the agent turns that into actions on
the running mission: weight the search toward the report, fly there and look
with SAM 3.1, mark hazards it is told about, report back in plain language.

Safety model: the agent only has the tools below. It cannot arm or take off
by itself: start_mission asks the operator to press Confirm. Everything it
commands still goes through the mission bridge's own guards (obstacle guard,
geofenced search area, pilot override, crash detection).

ROS-free: a Backend object (implemented by the dashboard server) does the
actual I/O, so the agent loop is unit-tested with a fake drone.
"""
import json
import re
import time

MAX_STEPS = 8            # model turns per user message
MAX_TOOL_CALLS = 8       # tool executions per user message
HISTORY_TURNS = 8        # past user/assistant messages kept in context
START_WAIT_S = 300       # operator Confirm + arm + takeoff (the sim can run below real time)
INSPECT_WAIT_S = 420     # fly there, possibly verify a casualty on the way

SYSTEM = """You are SAHARA's AI Mission Commander: the co-pilot of an autonomous \
search-and-rescue drone flying over a disaster zone. You talk to rescue \
responders. You act through tools, and only through tools.

Coordinates are metres east (E) / north (N) of the drone's launch point. \
{area}
Named places you can use:
{landmarks}

How to work:
- Turn what the responder says into actions. Reports of people somewhere -> \
prioritize_area there (the planner searches it sooner). "Check / look at X" -> \
inspect X (the drone flies there, hovers, and SAM 3.1 segments the camera image \
for the things you ask about). A hazard they tell you about -> report_hazard.
- If the mission is not running, call start_mission FIRST, before any other \
action: it waits until the responder presses Confirm and the drone is airborne \
and searching, then you continue (prioritize_area, inspect) in the same reply.
- Use get_situation when you need the current state; do not guess it.
- Report only what tools returned. Give SAM detections with their confidence. \
Never claim a person was found unless get_situation lists a confirmed casualty.
- Be brief: 1-3 short sentences, plain words, numbers with units. Use the \
casualty and hazard IDs (V1, H2) the system uses.
"""

TOOLS = [
    {"type": "function", "function": {
        "name": "get_situation",
        "description": "Current mission state: stage, drone position/altitude/battery, probability of success, "
                       "confirmed casualties (with priority and risks) and hazard zones.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "start_mission",
        "description": "Start the autonomous search mission (arm, take off, search). Asks the human operator "
                       "to press Confirm and waits until the drone is airborne and searching (status 'searching'), "
                       "or returns 'not_confirmed'.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "inspect",
        "description": "Fly to a place, hover, and look for the given things with SAM 3.1 open-vocabulary "
                       "segmentation. Use a named place or coordinates. Takes up to ~1 minute.",
        "parameters": {"type": "object", "properties": {
            "place": {"type": "string", "description": "A named place from the list, e.g. 'Collapsed police station'"},
            "east_m": {"type": "number"}, "north_m": {"type": "number"},
            "concepts": {"type": "array", "items": {"type": "string"},
                         "description": "Short noun phrases to look for, e.g. ['person', 'smoke', 'downed power line']"}},
            "required": ["concepts"]}}},
    {"type": "function", "function": {
        "name": "prioritize_area",
        "description": "New information that people may be at a place: shift the search probability there so the "
                       "planner searches it sooner.",
        "parameters": {"type": "object", "properties": {
            "place": {"type": "string"}, "east_m": {"type": "number"}, "north_m": {"type": "number"},
            "radius_m": {"type": "number", "description": "Area radius, default 6"},
            "strength": {"type": "string", "enum": ["low", "medium", "high"]}}}}},
    {"type": "function", "function": {
        "name": "report_hazard",
        "description": "Add a hazard the responder reports (not seen by the drone) to the map; rescue routes avoid it.",
        "parameters": {"type": "object", "properties": {
            "type": {"type": "string", "enum": ["fire", "flood", "structure", "smoke", "power_line", "landslide"]},
            "place": {"type": "string"}, "east_m": {"type": "number"}, "north_m": {"type": "number"},
            "radius_m": {"type": "number"}, "label": {"type": "string"}},
            "required": ["type"]}}},
    {"type": "function", "function": {
        "name": "find_now",
        "description": "Look at the drone's current camera view right now with SAM 3.1 for the given things.",
        "parameters": {"type": "object", "properties": {
            "concepts": {"type": "array", "items": {"type": "string"}}}, "required": ["concepts"]}}},
    {"type": "function", "function": {
        "name": "return_home",
        "description": "End the search and fly back to the launch point to land.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "land_now",
        "description": "Emergency: land immediately where the drone is.",
        "parameters": {"type": "object", "properties": {}}}},
]
STRENGTH = {"low": 3.0, "medium": 8.0, "high": 14.0}
SAM_HAZARD_TYPES = {"smoke": "smoke", "fire": "fire", "downed power line": "power_line", "power line": "power_line",
                    "landslide": "landslide", "mud": "landslide", "flood water": "flood", "water": "flood"}


def concepts_arg(v, default=("person",)):
    """Concept list from a tool argument. Models sometimes send a list as a
    string ("['person', 'smoke']" or "person, smoke"); accept all of those."""
    if isinstance(v, str):
        v = [x.strip(" '\"") for x in v.strip().strip("[]").split(",")]
    out = [str(x).strip() for x in (v or []) if str(x).strip()]
    return out[:4] or list(default)


def _words(s):
    return set(re.findall(r"[a-z0-9]+", s.lower()))


def resolve_place(args, landmarks):
    """(east, north, label) from explicit coordinates or a named place, else None."""
    if args.get("east_m") is not None and args.get("north_m") is not None:
        return float(args["east_m"]), float(args["north_m"]), args.get("place") or args.get("label") or ""
    name = (args.get("place") or args.get("label") or "").strip()
    if not name:
        return None
    want = _words(name) - {"the", "a", "of", "at", "near"}
    best, score = None, 0.0
    for lm in landmarks:
        have = _words(lm["label"])
        s = len(want & have) / max(1, len(want))
        if name.lower() in lm["label"].lower():
            s += 1.0
        if s > score:
            best, score = lm, s
    if best is None or score < 0.5:
        return None
    return float(best["east_m"]), float(best["north_m"]), best["label"]


def summarize(m):
    """Compact situation from the bridge's /resq/map JSON (keeps tokens low)."""
    if not m:
        return {"mission": "not running", "note": "no mission data yet (is the simulation running?)"}
    d = m.get("drone", {})
    return {
        "stage": m.get("stage"), "mission_time_s": m.get("t"), "sensor": m.get("sensor"),
        "drone": {"E": d.get("e"), "N": d.get("n"), "alt_m": d.get("alt"), "battery_pct": d.get("battery"),
                  "speed_ms": d.get("speed"), "armed": d.get("armed")},
        "probability_of_success": m.get("pos"),
        "casualties": [{"id": v.get("id"), "priority": v.get("priority"), "E": round(v.get("e", 0), 1),
                        "N": round(v.get("n", 0), 1), "moving": v.get("moving"), "risks": v.get("risks", []),
                        "action": v.get("action"), "ai_triage": v.get("ai_triage")}
                       for v in m.get("victims", [])],
        "hazards": [{"id": z.get("id"), "type": z.get("type"), "label": z.get("label"),
                     "E": round(z.get("e", 0), 1), "N": round(z.get("n", 0), 1),
                     "radius_m": round(z.get("radius", 0), 1), "source": z.get("source"),
                     "sensors": z.get("sensors")} for z in m.get("hazards", [])],
        "recent_events": [e.get("text") for e in m.get("events", [])[-5:]],
    }


class Commander:
    def __init__(self, client, backend):
        self.client, self.backend = client, backend
        self.history = []

    def system_prompt(self):
        lms = self.backend.landmarks()
        prob = (self.backend.map() or {}).get("prob") or {}
        inside = lambda lm: True
        area = "The search area is not loaded yet (no mission running)."
        if prob.get("w"):
            e0, n0, w, h = prob["e0"], prob["n0"], prob["w"], prob["h"]
            area = (f"The search area is E{e0:+.0f} to E{e0 + w:+.0f}, N{n0:+.0f} to N{n0 + h:+.0f}: the planner and "
                    "prioritize_area only work inside it; inspect can fly to places outside it.")
            inside = lambda lm: e0 <= lm["east_m"] <= e0 + w and n0 <= lm["north_m"] <= n0 + h
        table = "\n".join(f"- {lm['label']}: E{lm['east_m']:+.0f} N{lm['north_m']:+.0f}"
                          + ("" if inside(lm) else " (outside the search area)") for lm in lms) or "- (none loaded)"
        return SYSTEM.format(landmarks=table, area=area)

    def run(self, text):
        """Handle one responder message. Yields UI events:
        {"type": "status"|"tool_call"|"tool_progress"|"tool_result"|"confirm"|"reply"|"error", ...}"""
        messages = [{"role": "system", "content": self.system_prompt()}] + self.history[-HISTORY_TURNS:] \
            + [{"role": "user", "content": text}]
        calls_left = MAX_TOOL_CALLS
        reply = None
        cancel = getattr(self.backend, "cancel", None)
        stopped = lambda: cancel is not None and cancel.is_set()
        for _ in range(MAX_STEPS):
            if stopped():
                reply = "Stopped."
                break
            yield {"type": "status", "text": "thinking"}
            try:
                out = self.client.chat(messages, tools=TOOLS)
            except Exception as e:  # noqa: BLE001 - surface API errors to the operator
                yield {"type": "error", "text": f"AI unavailable: {e}"}
                return
            msg = out["choices"][0]["message"]
            calls = msg.get("tool_calls") or []
            messages.append({k: v for k, v in msg.items() if k in ("role", "content", "tool_calls")})
            if not calls:
                reply = (msg.get("content") or "").strip()
                break
            for call in calls:
                if stopped():
                    messages.append({"role": "tool", "tool_call_id": call.get("id"),
                                     "content": json.dumps({"status": "skipped", "reason": "interrupted"})})
                    continue
                name = call["function"]["name"]
                try:
                    args = json.loads(call["function"].get("arguments") or "{}")
                except ValueError:
                    args = {}
                yield {"type": "tool_call", "name": name, "args": args}
                if calls_left <= 0:
                    result = {"status": "skipped", "reason": "tool budget for this message used up"}
                else:
                    calls_left -= 1
                    result = yield from self.execute(name, args)
                yield {"type": "tool_result", "name": name, "result": result}
                messages.append({"role": "tool", "tool_call_id": call.get("id"), "content": json.dumps(result)})
        if reply is None:
            reply = "I ran out of steps for this request; ask me to continue if needed."
        self.history += [{"role": "user", "content": text}, {"role": "assistant", "content": reply}]
        yield {"type": "reply", "text": reply}

    # ------------------------------------------------------------ tools
    def execute(self, name, args):
        b = self.backend
        if name == "get_situation":
            return summarize(b.map())
        if name == "start_mission":
            if b.mission_running():
                return {"status": "already_running", "stage": (b.map() or {}).get("stage")}
            yield {"type": "confirm", "action": "start_mission",
                   "text": "Start the autonomous search? The drone will arm and take off."}
            yield {"type": "tool_progress", "text": "waiting for your confirmation"}
            cancel = getattr(b, "cancel", None)
            deadline, shown = time.time() + START_WAIT_S, None
            while time.time() < deadline and not (cancel is not None and cancel.is_set()):
                stage = (b.map() or {}).get("stage")
                if stage in ("SEARCHING", "INSPECTING", "VERIFYING", "MARKING"):
                    return {"status": "searching", "stage": stage}
                if b.mission_running() and stage != shown:
                    shown = stage
                    yield {"type": "tool_progress", "text": f"{str(stage).lower()}…"}
                if cancel is not None:
                    cancel.wait(1.0)
                else:
                    time.sleep(1.0)
            return {"status": "not_confirmed",
                    "note": "The operator did not confirm (or cancelled). Tell them; do not retry."}
        if name in ("inspect", "prioritize_area", "report_hazard"):
            place = resolve_place(args, b.landmarks())
            if place is None:
                return {"status": "error", "reason": "unknown place; give a name from the list or east_m/north_m"}
            e, n, label = place
        if name == "prioritize_area":
            factor = STRENGTH.get(args.get("strength", "medium"), 8.0)
            return b.command({"cmd": "prioritize", "east_m": e, "north_m": n, "label": label,
                              "radius_m": float(args.get("radius_m") or 6.0), "factor": factor})
        if name == "report_hazard":
            return b.command({"cmd": "avoid", "type": args["type"], "east_m": e, "north_m": n,
                              "radius_m": float(args.get("radius_m") or 5.0), "label": args.get("label") or label})
        if name == "inspect":
            concepts = concepts_arg(args.get("concepts"))
            yield {"type": "tool_progress", "text": f"flying to {label or f'E{e:+.0f} N{n:+.0f}'}"}
            res = b.command({"cmd": "inspect", "east_m": e, "north_m": n, "label": label, "hover_s": 8},
                            wait_for=("arrived", "rejected", "interrupted", "timeout", "superseded", "cancelled"),
                            timeout=INSPECT_WAIT_S)
            if res.get("status") != "arrived":
                return res
            yield {"type": "tool_progress", "text": f"hovering at {res.get('altitude')} m, looking with SAM 3.1"}
            sam = b.sam(concepts)
            return {"status": "inspected", "place": label, "altitude_m": res.get("altitude"),
                    "sam": _sam_summary(sam)}
        if name == "find_now":
            concepts = concepts_arg(args.get("concepts"))
            return {"status": "looked", "sam": _sam_summary(b.sam(concepts))}
        if name == "return_home":
            return b.command({"cmd": "return_home"})
        if name == "land_now":
            return b.land()
        return {"status": "error", "reason": f"unknown tool {name}"}


def _sam_summary(res):
    if not res or res.get("status") == "error":
        return {"error": (res or {}).get("reason", "no result")}
    found, by = {}, {}
    for h in res.get("hazards", []):
        found.setdefault(h["type"], []).append(round(h["conf"], 2))
        by.setdefault(h["type"], set()).update(h.get("sensors") or [])
    return {"found": {k: {"count": len(v), "best_conf": max(v), "by": sorted(by[k])} for k, v in found.items()},
            "nothing_found_for": [c for c, r in (res.get("report") or {}).items() if not r.get("n")],
            "errors": {c: r["error"] for c, r in (res.get("report") or {}).items() if r.get("error")},
            "latency_s": res.get("latency_s")}
