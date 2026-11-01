"""Muse Spark visual triage and SITREP narrative.

triage(): when a casualty is confirmed, the drone is hovering over them. The
AI feed frame and the false-colour thermal frame from that hover go to Muse
Spark with a strict JSON schema: what is visibly there (pose, signs of
injury, responsiveness cues, nearby dangers, access) to brief the team before
they arrive. It is an aerial first look, not a diagnosis, and is labelled so.

sitrep_narrative(): a short written situation summary from the mission map,
citing only the IDs and numbers in it.
"""
import json

from .client import image_part

TRIAGE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {"name": "aerial_triage", "strict": True, "schema": {
        "type": "object", "additionalProperties": False,
        "required": ["pose", "visible_injury_signs", "responsiveness_cues", "nearby_dangers",
                     "access_notes", "suggested_kit", "confidence", "caveats"],
        "properties": {
            "pose": {"type": "string", "enum": ["lying face up", "lying face down", "lying on side",
                                                "lying, orientation unclear", "sitting",
                                                "crawling", "standing", "partly covered", "unclear"]},
            "visible_injury_signs": {"type": "array", "items": {"type": "string"},
                                     "description": "Only what is visible, e.g. 'possible bleeding on left leg'. Empty if none."},
            "responsiveness_cues": {"type": "string", "description": "e.g. 'no movement during hover', 'arm raised'"},
            "nearby_dangers": {"type": "array", "items": {"type": "string"}},
            "access_notes": {"type": "string", "description": "Ground access for a team: debris, open ground, obstacles"},
            "suggested_kit": {"type": "array", "items": {"type": "string"}},
            "confidence": {"type": "number", "description": "0-1, how clearly the images show the person"},
            "caveats": {"type": "string"},
        }}},
}

TRIAGE_PROMPT = """You are briefing a search-and-rescue team before they reach a casualty.
Image 1: the drone's colour camera, looking straight down from about {alt} m (green boxes are the \
detector's). Image 2: the thermal camera, false colour (bright = warm).
Known context: {context}
Describe only what is visible. Do not diagnose or guess medical conditions; say "not visible" \
when you cannot tell. This is a first look from the air for planning, to be verified on site."""

SITREP_PROMPT = """Write the narrative section of a search-and-rescue situation report for the \
incident commander, from this mission data (JSON). 4-6 sentences, plain and factual:
what was searched and how much (probability of success), each casualty by ID with priority, risks \
and the recommended action (include AI triage notes where present, labelled as aerial assessments), \
the hazard zones by ID, and what remains to be done. Use only facts in the data; no speculation.

{data}"""


def triage(client, rgb_jpeg, thermal_jpeg=None, alt=8.0, context=""):
    content = [{"type": "text", "text": TRIAGE_PROMPT.format(alt=round(alt or 8.0, 1), context=context or "none")},
               image_part(rgb_jpeg)]
    if thermal_jpeg:
        content.append(image_part(thermal_jpeg))
    out = client.chat([{"role": "user", "content": content}], model=client.cfg.vision_model,
                      response_format=TRIAGE_SCHEMA, temperature=0.1)
    note = json.loads(out["choices"][0]["message"]["content"])
    note["model"] = client.cfg.vision_model
    note["label"] = "AI aerial assessment - verify on site"
    return note


def sitrep_narrative(client, situation):
    # No max_tokens: Muse Spark reasons before it writes, and a tight cap
    # (700 in testing) was spent on reasoning, leaving the text empty.
    out = client.chat([{"role": "user", "content": SITREP_PROMPT.format(data=json.dumps(situation))}],
                      temperature=0.2)
    text = (out["choices"][0]["message"].get("content") or "").strip()
    if not text:
        raise RuntimeError("the model returned no text; try again")
    return text
