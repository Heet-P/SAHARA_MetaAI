"""First contact with the Meta Model API: run once the key arrives.

  export MODEL_API_KEY=...
  python3 -m sahara_ai.smoke_test [--audio command.webm]

Checks, with latency for each:
  1. Muse Spark text reply
  2. tool calling (does the model emit a well-formed call?)
  3. image input as a base64 data URL + JSON-schema output (triage shape)
  4. SAM 3.1 on two simulator frames (person, smoke); raw outputs are saved
     to tests/fixtures/sam_live_<concept>.txt to pin down the mask encoding
  5. speech to text (only with --audio)
Runs in mock mode without a key, which only proves the script itself works.
"""
import argparse
import json
import time
from pathlib import Path

from . import Config, MetaClient
from .client import ApiError, image_part

FIX = Path(__file__).resolve().parent / "tests" / "fixtures"


def step(name, fn):
    t0 = time.time()
    try:
        detail = fn()
        print(f"  OK    {name:<34} {time.time() - t0:6.2f}s  {detail}")
        return True
    except (ApiError, KeyError, IndexError, ValueError, TypeError) as e:
        print(f"  FAIL  {name:<34} {time.time() - t0:6.2f}s  {e}")
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", type=Path)
    args = ap.parse_args()
    c = MetaClient(Config())
    print(f"Meta Model API smoke test  base={c.cfg.base_url}  mock={bool(c.mock)}")
    msg = lambda r: r["choices"][0]["message"]

    def text():
        r = c.chat([{"role": "user", "content": "Reply with exactly: SAHARA online"}], max_tokens=20)
        return repr(msg(r)["content"])

    def tools():
        t = [{"type": "function", "function": {
            "name": "inspect", "description": "Fly the drone to a named place and look for the given things.",
            "parameters": {"type": "object", "properties": {
                "target": {"type": "string"}, "concepts": {"type": "array", "items": {"type": "string"}}},
                "required": ["target"]}}}]
        r = c.chat([{"role": "user", "content": "Inspect the collapsed police station for people and smoke."}], tools=t)
        calls = msg(r).get("tool_calls") or []
        assert calls, "no tool call returned"
        return f"{calls[0]['function']['name']}({calls[0]['function']['arguments']})"

    def vision():
        fmt = {"type": "json_schema", "json_schema": {"name": "scene", "strict": True, "schema": {
            "type": "object", "additionalProperties": False,
            "required": ["person_visible", "pose", "notes"],
            "properties": {"person_visible": {"type": "boolean"},
                           "pose": {"type": "string", "enum": ["lying", "sitting", "standing", "unclear", "none"]},
                           "notes": {"type": "string"}}}}}
        img = (FIX / "feed_person.jpg").read_bytes()
        r = c.chat([{"role": "user", "content": [
            {"type": "text", "text": "Aerial drone view of a disaster site. Is a person visible? Describe briefly."},
            image_part(img)]}], response_format=fmt)
        return json.loads(msg(r)["content"])

    def sam(concept, fixture):
        def run():
            recs, raw = c.segment(concept, (FIX / fixture).read_bytes())
            if not c.mock:
                (FIX / f"sam_live_{concept.replace(' ', '_')}.txt").write_text(raw)
            best = max((r.conf for r in recs), default=None)
            return f"{len(recs)} object(s), best conf {best}, raw {len(raw)} chars"
        return run

    ok = [step("muse-spark text", text), step("muse-spark tool calling", tools),
          step("muse-spark image + json schema", vision)]
    for concept, fixture in (("person", "feed_person.jpg"), ("smoke", "world_smoke.jpg")):
        ok.append(step(f"sam-3.1 '{concept}'", sam(concept, fixture)))
    if args.audio:
        ok.append(step("voice transcription", lambda: repr(c.transcribe(args.audio.read_bytes(), args.audio.name))))
    est = 0.0
    try:
        with open(c.cfg.usage_log) as f:
            est = sum(json.loads(l).get("est_cost_usd", 0) for l in f.readlines()[-len(ok):])
    except OSError:
        pass
    print(f"{sum(ok)}/{len(ok)} passed · Muse Spark est. cost this run ${est:.4f} (SAM not priced) "
          f"· real calls today {c.calls_today()}/{c.cfg.max_calls_per_day} · log: {c.cfg.usage_log}")


if __name__ == "__main__":
    main()
