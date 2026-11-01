from sahara_ai import Config, MetaClient
from sahara_ai.perception import merge, records_to_hazards, scan
from sahara_ai.sam import parse_records


def test_records_to_hazards_offsets_and_filters():
    text = ("<0f>0<|box;x1=320;y1=240;x2=419;y2=339;w=640;h=480;c=0.9|>"
            "1<|box;x1=0;y1=0;x2=10;y2=10;w=640;h=480;c=0.2|>"          # low confidence
            "2<|box;x1=0;y1=0;x2=639;y2=479;w=640;h=480;c=0.99|>")      # whole-frame haze
    (hz,) = records_to_hazards(parse_records(text), "smoke")
    assert hz["type"] == "smoke" and hz["sensors"] == ["sam3.1"]
    assert abs(hz["cx"] - (369.5 / 640 - 0.5)) < 1e-3
    assert abs(hz["cy"] - (289.5 / 480 - 0.5)) < 1e-3
    assert hz["r_px"] > 0


def test_scan_mock_reports_every_concept(tmp_path):
    c = MetaClient(Config(api_key="", usage_log=tmp_path / "u.jsonl"))
    hazards, report = scan(c, b"jpeg", {"smoke": "smoke", "unicorn": "landslide"}, vision=False)
    assert [h["type"] for h in hazards] == ["smoke"]
    assert set(report) == {"smoke", "unicorn"} and report["unicorn"]["n"] == 0


def test_merge_prefers_sam_and_fills_gaps():
    sam = [{"type": "smoke", "conf": 0.8, "sensors": ["sam3.1"]}]
    vis = [{"type": "smoke", "conf": 0.7, "sensors": ["muse-spark"]}, {"type": "power_line", "conf": 0.9, "sensors": ["muse-spark"]}]
    out = merge(sam, vis)
    assert [(h["type"], h["sensors"]) for h in out] == [("smoke", ["sam3.1", "muse-spark"]), ("power_line", ["muse-spark"])]
