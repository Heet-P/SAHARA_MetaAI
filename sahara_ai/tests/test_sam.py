from sahara_ai.sam import parse_records

DOC_EXAMPLE = "<0f>0<|box;x1=444;y1=625;x2=679;y2=850;w=1800;h=1200;c=0.96|><|mask;x=0;y=0;c=0.96;data=226,236,~abc|>"


def test_doc_example():
    (r,) = parse_records(DOC_EXAMPLE)
    assert (r.frame, r.obj) == (0, 0)
    assert r.box == (444, 625, 679, 850)
    assert r.size == (1800, 1200)
    assert r.conf == 0.96
    assert r.mask["data"] == "226,236,~abc"
    assert r.center == (561.5, 737.5)


def test_multiple_objects_and_frames():
    text = ("<0f>0<|box;x1=0;y1=0;x2=9;y2=9;w=100;h=100;c=0.5|>"
            "1<|box;x1=10;y1=10;x2=19;y2=19;w=100;h=100;c=0.7|><|mask;x=10;y=10;c=0.7;data=!xyz|>"
            "<3f>0<|box;x1=1;y1=2;x2=3;y2=4;w=100;h=100;c=0.9|>")
    rs = parse_records(text)
    assert [(r.frame, r.obj) for r in rs] == [(0, 0), (0, 1), (3, 0)]
    assert rs[0].mask == {}
    assert rs[1].mask["data"] == "!xyz"
    assert abs(rs[0].area_frac - 0.01) < 1e-9


def test_chunked_stream_joins():
    deltas = ["<0f>0<|box;x1=1;y1=", "2;x2=3;y2=4;w=10;h=10;c=0", ".8|>"]
    (r,) = parse_records("".join(deltas))
    assert r.box == (1, 2, 3, 4) and r.conf == 0.8


def test_empty_and_garbage():
    assert parse_records("") == []
    assert parse_records("no objects here") == []
