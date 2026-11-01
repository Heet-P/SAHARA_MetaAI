# sahara_ai — SAHARA's Meta AI layer

Built for the Meta Global AI Developer Hackathon (Oct 26 – Nov 4, 2026). Everything before git tag `pre-hackathon` existed beforehand.

| Module | Meta model | Role |
|---|---|---|
| `client.py` | Meta Model API (`https://api.meta.ai/v1`) | Chat with tools, images and JSON schema; SAM 3.1 streaming; speech to text; usage log |
| `sam.py` | `sam-3.1` | Parses SAM's `<Nf>id<\|box…\|><\|mask…\|>` records |
| `perception.py` | `sam-3.1` + `muse-spark-1.3` (vision) | Open-vocabulary hazards (smoke, downed power line, landslide…) from the drone camera; SAM masks where it fires, Muse Spark covers what SAM misses |
| `commander.py` | `muse-spark-1.3` | Mission Commander agent driving the ROS 2 / Gazebo drone through tool calls |
| `triage.py` | `muse-spark-1.3` (vision) | Aerial triage note at each confirmed casualty; SITREP narrative |

Measured on the simulator (Nov 1):
- Muse Spark: tool-call turns in 3–5 s; vision with a JSON schema in 7–19 s; about $0.002–0.012 per call.
- SAM 3.1: about 3 s per concept. It found the smoke plume (0.84) but nothing on cables or rocks in flat-shaded renders.
- Muse Spark vision found the downed power line on the same frame (0.92 / 0.88).

The package uses the standard library only, so the ROS 2 nodes can import it without touching the simulation's pinned Python environment.

## Setup

```bash
# either export it for this shell...
export MODEL_API_KEY=...
# ...or store it once in a private file outside the repo (read automatically)
mkdir -p ~/.config/sahara_ai && (umask 077; read -rsp "Meta API key: " k; echo "$k" > ~/.config/sahara_ai/key)
# without a key, everything runs in mock mode
python3 -m sahara_ai.smoke_test     # first contact: latency, tool calls, vision, SAM
```

Optional settings:
- `SAHARA_AI_AGENT_MODEL` (default `muse-spark-1.3`)
- `SAHARA_AI_SAM_MODEL` (default `sam-3.1`)
- `SAHARA_AI_PRICE_IN` / `SAHARA_AI_PRICE_OUT`: USD per million tokens. With these set, the usage log records cost.
- `SAHARA_AI_MOCK=1`: force mock mode.

The usage log is written to `logs/ai_usage.jsonl`.

## Tests

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q sahara_ai/tests
```

The plugin variable works around a user-installed `anyio` pytest plugin that is incompatible with the system pytest 6.2.
