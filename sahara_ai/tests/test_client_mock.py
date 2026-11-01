import json

from sahara_ai import Config, MetaClient

TOOLS = [{"type": "function", "function": {"name": n, "parameters": {"type": "object", "properties": {}}}}
         for n in ("inspect", "prioritize_area", "start_mission", "land_now", "get_situation")]


def client(tmp_path):
    return MetaClient(Config(api_key="", usage_log=tmp_path / "usage.jsonl"))


def test_mock_mode_without_key(tmp_path):
    assert client(tmp_path).mock is not None


def test_mock_tool_calls_and_usage_log(tmp_path):
    c = client(tmp_path)
    out = c.chat([{"role": "user", "content": "Inspect the collapsed police station, reports of two trapped"}], tools=TOOLS)
    calls = out["choices"][0]["message"]["tool_calls"]
    names = [x["function"]["name"] for x in calls]
    assert names[:2] == ["inspect", "prioritize_area"]
    assert json.loads(calls[0]["function"]["arguments"])["place"] == "Collapsed police station"
    log = [json.loads(l) for l in (tmp_path / "usage.jsonl").read_text().splitlines()]
    assert log[-1]["kind"] == "chat" and log[-1]["mock"] is True


def test_mock_json_schema(tmp_path):
    fmt = {"type": "json_schema", "json_schema": {"name": "t", "schema": {"type": "object", "properties": {
        "pose": {"type": "string", "enum": ["lying", "sitting"]}, "confidence": {"type": "number"},
        "visible_injury_signs": {"type": "array"}}}}}
    out = client(tmp_path).chat([{"role": "user", "content": "x"}], response_format=fmt)
    data = json.loads(out["choices"][0]["message"]["content"])
    assert data == {"pose": "lying", "confidence": 0.5, "visible_injury_signs": []}


def test_mock_segment(tmp_path):
    recs, raw = client(tmp_path).segment("smoke", b"\xff\xd8fake")
    assert len(recs) == 1 and recs[0].conf == 0.88 and raw
    assert client(tmp_path).segment("unicorn", b"x")[0] == []


def test_budget_guard_blocks_real_calls(tmp_path):
    import time as _t
    from sahara_ai.client import BudgetExceeded
    log = tmp_path / "u.jsonl"
    log.write_text("".join(json.dumps({"t": _t.time(), "mock": False}) + "\n" for _ in range(3)))
    c = MetaClient(Config(api_key="real-looking-key", usage_log=log, max_calls_per_day=3))
    assert c.mock is None and c.calls_today() == 3
    try:
        c.chat([{"role": "user", "content": "hi"}])
        assert False, "should have been blocked before any network call"
    except BudgetExceeded:
        pass
