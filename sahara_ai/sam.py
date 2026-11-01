"""Parse SAM 3.1 segmentation records from the Meta Model API.

The API streams text like

    <0f>0<|box;x1=444;y1=625;x2=679;y2=850;w=1800;h=1200;c=0.96|><|mask;x=0;y=0;c=0.96;data=226,236,~...|>

meaning: frame 0, object 0, an inclusive box in a w x h image with confidence
c, then a mask whose pixel data starts at (x, y). A frame tag <Nf> applies to
the objects that follow it until the next frame tag.

The mask payload ("~" lossless, "!" one-bit) is kept raw here. Its encoding is
decoded in decode_mask once confirmed against live API output; everything
SAHARA needs for mapping (box, confidence, ids) is available without it.
"""
import re
from dataclasses import dataclass, field

_FRAME = re.compile(r"<(\d+)f>")
# object id, a box, then an optional mask; each field ends at the first "|>".
_OBJECT = re.compile(r"(\d+)<\|box;(.*?)\|>(?:<\|mask;(.*?)\|>)?", re.S)


@dataclass
class SamRecord:
    frame: int
    obj: int
    box: tuple            # (x1, y1, x2, y2), inclusive pixels
    size: tuple           # (w, h) of the image the box refers to
    conf: float
    mask: dict = field(default_factory=dict)   # {"x":..,"y":..,"c":..,"data": raw string} or {}

    @property
    def center(self):
        x1, y1, x2, y2 = self.box
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @property
    def area_frac(self):
        x1, y1, x2, y2 = self.box
        w, h = self.size
        return max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1) / float(w * h) if w and h else 0.0


def _kv(body):
    out = {}
    # "data" may itself contain commas and semicolons-free payload; it is last.
    head, sep, data = body.partition(";data=")
    for part in head.split(";"):
        if "=" in part:
            k, v = part.split("=", 1)
            out[k.strip()] = v.strip()
    if sep:
        out["data"] = data
    return out


def _num(v, cast=float, default=0):
    try:
        return cast(v)
    except (TypeError, ValueError):
        return default


def parse_records(text):
    """All complete records in a (possibly concatenated) SAM output string."""
    records = []
    # Split on frame tags, keeping the frame number for the chunk after it.
    pos, frame = 0, 0
    chunks = []
    for m in _FRAME.finditer(text):
        chunks.append((frame, text[pos:m.start()]))
        frame, pos = int(m.group(1)), m.end()
    chunks.append((frame, text[pos:]))
    for frame, chunk in chunks:
        for om in _OBJECT.finditer(chunk):
            box, mk = _kv(om.group(2)), _kv(om.group(3)) if om.group(3) is not None else None
            mask = {"x": _num(mk.get("x"), int), "y": _num(mk.get("y"), int),
                    "c": _num(mk.get("c")), "data": mk.get("data", "")} if mk else {}
            records.append(SamRecord(
                frame=frame, obj=int(om.group(1)),
                box=(_num(box.get("x1"), int), _num(box.get("y1"), int), _num(box.get("x2"), int), _num(box.get("y2"), int)),
                size=(_num(box.get("w"), int), _num(box.get("h"), int)),
                conf=_num(box.get("c"), float, 1.0),
                mask=mask,
            ))
    return records


def decode_mask(record):
    """Decode a record's mask into a list of rows of 0/1, or None if unknown.

    Placeholder until the payload format is confirmed on live output; callers
    fall back to the box.
    """
    return None
