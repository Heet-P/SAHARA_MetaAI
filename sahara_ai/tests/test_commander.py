from sahara_ai import Config, MetaClient
from sahara_ai.commander import Commander, concepts_arg, resolve_place, summarize

LANDMARKS = [
    {"label": "Collapsed police station", "east_m": -30, "north_m": 15},
    {"label": "Collapsed fire station", "east_m": 9, "north_m": 53},
    {"label": "Ambulance / team staging", "east_m": -7, "north_m": -3},
]


class FakeDrone:
    def __init__(self, running=True):
        self.running, self.sent = running, []

    def landmarks(self):
        return LANDMARKS

    def map(self):
        return {"stage": "SEARCHING", "t": 42.0, "drone": {"e": 1, "n": 2, "alt": 8.0, "battery": 97},
                "pos": 0.31, "victims": [], "hazards": [], "events": []}

    def mission_running(self):
        return self.running

    def command(self, cmd, wait_for=None, timeout=10):
        self.sent.append(cmd)
        return {"status": "arrived", "altitude": 8.1} if cmd["cmd"] == "inspect" else {"status": "ok"}

    def sam(self, concepts):
        return {"hazards": [{"type": "smoke", "conf": 0.84}], "report": {c: {"n": int(c == "smoke")} for c in concepts},
                "latency_s": 3.1}

    def land(self):
        return {"status": "ok"}


def commander(tmp_path, drone):
    return Commander(MetaClient(Config(api_key="", usage_log=tmp_path / "u.jsonl")), drone)


def test_resolve_place_by_name_and_coords():
    assert resolve_place({"place": "the police station"}, LANDMARKS)[:2] == (-30.0, 15.0)
    assert resolve_place({"east_m": 3, "north_m": 4}, LANDMARKS)[:2] == (3.0, 4.0)
    assert resolve_place({"place": "moon base"}, LANDMARKS) is None


def test_inspect_and_prioritize_flow(tmp_path):
    drone = FakeDrone()
    events = list(commander(tmp_path, drone).run(
        "Inspect the collapsed police station, reports of two people trapped"))
    kinds = [e["type"] for e in events]
    assert "tool_call" in kinds and kinds[-1] == "reply"
    cmds = [c["cmd"] for c in drone.sent]
    assert cmds == ["inspect", "prioritize"]
    assert drone.sent[0]["east_m"] == -30.0
    sam = next(e for e in events if e["type"] == "tool_result" and e["name"] == "inspect")["result"]["sam"]
    assert sam["found"]["smoke"]["best_conf"] == 0.84


def test_start_requires_operator_confirmation(tmp_path, monkeypatch):
    import sahara_ai.commander as cm
    monkeypatch.setattr(cm, "START_WAIT_S", 0)  # nobody presses Confirm
    drone = FakeDrone(running=False)
    events = list(commander(tmp_path, drone).run("start the search"))
    assert any(e["type"] == "confirm" and e["action"] == "start_mission" for e in events)
    res = next(e for e in events if e["type"] == "tool_result")["result"]
    assert res["status"] == "not_confirmed" and drone.sent == []


def test_start_waits_until_searching(tmp_path, monkeypatch):
    import sahara_ai.commander as cm
    monkeypatch.setattr(cm, "START_WAIT_S", 5)
    drone = FakeDrone(running=False)  # FakeDrone.map() already reports SEARCHING: confirmed and airborne
    events = list(commander(tmp_path, drone).run("start the search"))
    res = next(e for e in events if e["type"] == "tool_result")["result"]
    assert res == {"status": "searching", "stage": "SEARCHING"}


def test_summarize_handles_empty_map():
    assert summarize(None)["mission"] == "not running"


def test_concepts_arg_accepts_stringified_lists():
    # seen from the live model: a list sent as a string
    assert concepts_arg("['person', 'smoke']") == ["person", "smoke"]
    assert concepts_arg("person, downed power line") == ["person", "downed power line"]
    assert concepts_arg(["smoke"]) == ["smoke"]
    assert concepts_arg(None) == ["person"]


def test_newer_message_interrupts_the_turn(tmp_path):
    import threading
    drone = FakeDrone()
    drone.cancel = threading.Event()
    run = commander(tmp_path, drone).run("Inspect the collapsed police station, reports of two people trapped")
    first = next(run)  # the turn has started
    drone.cancel.set()  # operator sent "return home"
    events = [first] + list(run)
    assert events[-1] == {"type": "reply", "text": "Stopped."}
    assert drone.sent == []
