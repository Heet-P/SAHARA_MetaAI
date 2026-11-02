"""Edit the Meta hackathon video (SAHARA x Meta AI) from the OBS recordings.

  python3 demo/meta_edit.py stills      # first frame of every shot, zoomed -> demo/out/meta/stills/
  python3 demo/meta_edit.py build       # -> demo/out/sahara-meta.mp4

Shots are cut, retimed and zoomed from raw-videos/Meta_HACKATHON (plus the real
flight footage for the opening); cards, model badges and captions are rendered
by Chrome as PNGs and composited; the voice-over (VO_1..7) is laid over a
ducked music bed made with promo/sound.py. The recordings' audio is dropped.
Speed-ups are labelled on screen; chat moments with the AI play in real time.
"""
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import imageio_ffmpeg
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "raw-videos"
MH = RAW / "Meta_HACKATHON"
OUT = ROOT / "demo" / "out" / "meta"
FONTS = ROOT / "command_center" / "frontend" / "fonts"
FF = imageio_ffmpeg.get_ffmpeg_exe()
FPS = 30
W, H = 1920, 1080

SRC = {
    "ay": RAW / "Ayakashi.mp4",
    "wa": RAW / "WhatsApp Video 2026-09-29 at 16.12.17.mp4",
    "vc": MH / "Voice_Command_CMD_CENTER.mp4",
    "pl": MH / "Electric_Power_LIne_Down.mp4",
    "rh": MH / "RETURN_HOME_CMND.mp4",
    "sm": MH / "SITREP_MAP_CORRECTED.mp4",
}
SRC_W = {"ay": 3840, "wa": 1024}  # everything else is 1920 wide

FULL = (0, 0, 1920)  # every shot plays full frame; important parts get a highlight box instead of a zoom

# Highlight boxes (x, y, w, h) in 1920x1080 source pixels.
AI_STEPS = (1500, 240, 385, 330)     # AI panel: message, tool steps, Confirm card
AI_RESULT = (1505, 605, 375, 385)    # AI panel: inspect result, analysed frame, reply
MAP_HAZARD = (1053, 253, 420, 680)   # side map: hazard zone + re-planned route
TRIAGE_CARD = (1490, 77, 400, 556)   # Map view: rescue card with the AI triage table
SUMMARY_WRITING = (500, 355, 920, 110)
SUMMARY = (509, 387, 900, 337)       # SITREP: the AI situation summary

# VO files and where each scene starts (seconds in the final video).
VO = ["VO_1", "VO_2", "VO_3", "VO_4", "VO_5", "vo_6", "VO_7"]

# Scenes: (vo index, [shots]); shot = (src, a, b, speed, zoom).
# Output duration of a shot is (b - a) / speed.
CH = 2.4  # chapter card (transition) length
SCENES = [
    ("hook", 0, [("ay", 19.0, 22.5, 1, FULL), ("ay", 40.0, 43.0, 1, FULL), ("wa", 28.8, 31.3, 1, FULL)]),
    ("title", None, 3.6),
    ("ch_voice", None, CH),
    ("voice", 1, [("vc", 36.0, 46.0, 1.25, FULL), ("vc", 46.0, 52.5, 1, FULL), ("vc", 52.5, 85.0, 3.5, FULL)]),
    ("ch_hazard", None, CH),
    ("hazard", 2, [("pl", 7.0, 17.0, 2, FULL), ("pl", 17.0, 55.0, 6, FULL), ("pl", 55.0, 80.0, 4, FULL),
                   ("pl", 82.0, 94.0, 1, FULL)]),
    ("ch_triage", None, CH),
    ("triage", 3, [("sm", 28.0, 38.0, 2, FULL), ("sm", 134.0, 148.5, 1, FULL)]),
    ("ch_control", None, CH),
    ("control", 4, [("rh", 139.0, 147.0, 1, FULL), ("rh", 147.0, 200.0, 8, FULL)]),
    ("ch_sitrep", None, CH),
    ("sitrep", 5, [("sm", 84.5, 90.0, 2, FULL), ("sm", 90.0, 103.0, 4, FULL), ("sm", 103.0, 113.0, 1, FULL)]),
    ("arch", 6, 9.5),
    ("end", None, 4.0),
]
# Chapter cards between the sections: number, title, subtitle, models at work.
CHAPTERS = {
    "ch_voice": ("01", "Speak a mission", "A responder talks; the AI plans and asks before it flies",
                 [("Muse Voice Transcribe", "#60A5FA"), ("Muse Spark 1.3", "#A78BFA")]),
    "ch_hazard": ("02", "Find hazards it was never trained on", "Open-vocabulary perception on the live camera",
                  [("SAM 3.1", "#2DD4BF"), ("Muse Spark vision", "#A78BFA")]),
    "ch_triage": ("03", "Triage from the air", "A first look at each casualty before the team arrives",
                  [("Muse Spark vision", "#A78BFA")]),
    "ch_control": ("04", "The operator stays in charge", "Any new command interrupts the agent",
                   [("Muse Spark 1.3", "#A78BFA")]),
    "ch_sitrep": ("05", "The situation report writes itself", "From the drone's own mission map",
                  [("Muse Spark 1.3", "#A78BFA")]),
}
VO_LEAD = 0.5  # VO starts this long after its scene

# What is said in each VO, split into caption lines (model names spelled out).
CAPTIONS = [
    ["After a disaster, the first hour decides who survives.", "SAHARA is my search-and-rescue drone,",
     "and I gave it a brain built on Meta's AI models."],
    ["A responder just speaks.", "Muse Voice Transcribe turns speech into text,", "and Muse Spark plans the mission.",
     "It reads the live situation,", "raises the search priority where people were reported,", "and asks to launch.",
     "Arming always needs a human click.", "The AI proposes, and the operator decides.",
     "So it is a human-in-the-loop system."],
    ["For hazards our detector was never trained on,", "the drone uses open-vocabulary perception.",
     "SAM 3.1 segments each hazard we name,", "like smoke or a downed power line,",
     "while Muse Spark reads the whole scene in parallel.", "Every finding goes onto the live hazard map,",
     "and the rescue routes are re-planned around it automatically."],
    ["When a survivor is confirmed,", "Muse Spark looks at the colour and thermal frames from the hover",
     "and writes a triage note:", "pose, visible injuries, nearby dangers,",
     "how a team can reach them, and what kit to bring.", "It's clearly marked as an aerial assessment,",
     "and it has to be verified on site."],
    ["The operator stays in charge.", "A new command interrupts the agent instantly,",
     "and flight safety (obstacle avoidance, geofence, pilot override)", "never depends on the language model."],
    ["At the end, Muse Spark writes the situation report,", "the SITREP, for the incident commander,",
     "from the drone's own map:", "casualties by priority, hazards, and what's left to search,",
     "citing only real data."],
    ["Three Meta models, one drone, running on real flight software.",
     "SAHARA: from a voice command to a rescue plan.", "Thank you."],
]


def timeline():
    """Flatten SCENES into shots with global times."""
    t, shots, scenes = 0.0, [], {}
    for name, vo, body in SCENES:
        t0 = t
        if isinstance(body, list):
            for src, a, b, speed, zoom in body:
                d = (b - a) / speed
                shots.append({"kind": "clip", "src": src, "a": a, "b": b, "speed": speed, "zoom": zoom,
                              "t0": t, "t1": t + d, "scene": name})
                t += d
        else:
            shots.append({"kind": "card", "card": name, "t0": t, "t1": t + body, "scene": name})
            t += body
        scenes[name] = {"t0": t0, "t1": t, "vo": vo}
    return shots, scenes, t


def shot_time(shots, src, at):
    """Global time at which source time `at` of a clip from `src` plays."""
    for s in shots:
        if s["kind"] == "clip" and s["src"] == src and s["a"] <= at <= s["b"]:
            return s["t0"] + (at - s["a"]) / s["speed"]
    raise ValueError(f"{src}@{at} is not in any shot")


def overlays(shots, scenes):
    """(png name, html, t0, t1) for every badge and caption."""
    st = lambda src, at: shot_time(shots, src, at)
    sc = lambda n: scenes[n]
    out = []

    def lower(name, num, title, sub):
        s = sc(name)
        out.append((f"lower_{name}", lower_third(num, title, sub), s["t0"] + 0.3, s["t1"] - 0.2))

    def badge(key, model, what, color, t0, t1, slot=0):
        out.append((f"badge_{key}", model_badge(model, what, color, slot), t0, t1))

    def speed(key, x, t0, t1):
        out.append((f"speed_{key}", speed_tag(x), t0, t1))

    s = sc("hook")
    out.append(("lower_hook", lower_third("", "Our drone, flying for real", "S500 quadcopter · test flights and a payload drop"),
                s["t0"] + 0.4, s["t1"] - 0.2))

    def spot(key, rect, t0, t1):
        out.append((f"spot_{key}", spotlight(rect), t0, t1))

    spot("voice", AI_STEPS, st("vc", 37.0), st("vc", 52.5))
    badge("voice1", "Muse Voice Transcribe", "speech → text", "#60A5FA", st("vc", 38.2), st("vc", 46.0))
    badge("voice2", "Muse Spark 1.3", "agent · tool calling", "#A78BFA", st("vc", 45.4), sc("voice")["t1"] - 0.2)
    out.append(("note_confirm", note("A human presses Confirm before the drone arms"), st("vc", 48.4), st("vc", 52.5)))
    speed("voice", 3.5, st("vc", 52.5), sc("voice")["t1"] - 0.1)

    spot("haz1", AI_STEPS, st("pl", 8.0), st("pl", 17.0))
    badge("haz1", "Muse Spark 1.3", "logs the reported hazard · flies there", "#A78BFA", st("pl", 9.0), st("pl", 17.0))
    speed("haz1", 6, st("pl", 17.0), st("pl", 55.0))
    speed("haz2", 4, st("pl", 55.0), st("pl", 80.0))
    badge("haz2", "SAM 3.1", "open-vocabulary segmentation", "#2DD4BF", st("pl", 60.0), st("pl", 88.5))
    badge("haz3", "Muse Spark 1.3", "vision · reads the whole scene", "#A78BFA", st("pl", 60.0), st("pl", 88.5), 1)
    spot("haz2", AI_RESULT, st("pl", 83.5), st("pl", 89.5))
    spot("haz3", MAP_HAZARD, st("pl", 89.5), sc("hazard")["t1"])
    # The recording predates the fix that names which model saw the hazard; cover
    # that line with the accurate statement.
    out.append(("cover_sam", cover_line(), st("pl", 84.0), sc("hazard")["t1"]))

    speed("tri", 2, sc("triage")["t0"], st("sm", 38.0))
    spot("tri", TRIAGE_CARD, st("sm", 134.5), sc("triage")["t1"])
    badge("tri", "Muse Spark 1.3", "vision · aerial triage note", "#A78BFA", st("sm", 134.0), sc("triage")["t1"] - 0.2)
    out.append(("note_triage", note("Aerial assessment, to be verified on site"), st("sm", 136.0), sc("triage")["t1"] - 0.2))

    spot("ctl", AI_STEPS, st("rh", 139.5), st("rh", 147.0))
    badge("ctl", "Muse Spark 1.3", "agent · return_home", "#A78BFA", st("rh", 140.0), st("rh", 147.0))
    speed("ctl", 8, st("rh", 147.0), sc("control")["t1"] - 0.1)
    out.append(("note_ctl", note("Obstacle guard, geofence and pilot override never depend on the AI"),
                st("rh", 147.5), sc("control")["t1"] - 0.2))

    spot("sit1", SUMMARY_WRITING, sc("sitrep")["t0"] + 0.3, st("sm", 103.0))
    spot("sit2", SUMMARY, st("sm", 103.0), sc("sitrep")["t1"])
    for name, sc_ in scenes.items():
        if sc_["vo"] is None:
            continue
        for k, (text, a, b) in enumerate(caption_times(sc_["vo"], sc_["t0"] + VO_LEAD)):
            out.append((f"cap_{sc_['vo']}_{k}", caption(text), a, b))
    speed("sit", 4, st("sm", 90.0), st("sm", 103.0))
    badge("sit", "Muse Spark 1.3", "SITREP narrative", "#A78BFA", st("sm", 103.0), sc("sitrep")["t1"] - 0.2)
    return out


# ------------------------------------------------------------------ HTML

CSS = f"""
@font-face {{ font-family: Inter; src: url("file://{FONTS}/inter-latin-wght-normal.woff2"); font-weight: 100 900; }}
@font-face {{ font-family: JB; src: url("file://{FONTS}/jetbrains-mono-latin-wght-normal.woff2"); font-weight: 100 800; }}
* {{ box-sizing: border-box; margin: 0; }}
html, body {{ width: {W}px; height: {H}px; background: transparent; font-family: Inter, sans-serif; color: #EDEDEE;
  -webkit-font-smoothing: antialiased; }}
.glass {{ background: rgba(12, 12, 14, 0.86); border: 1px solid rgba(255,255,255,0.12); border-radius: 18px;
  box-shadow: 0 20px 50px -18px rgba(0,0,0,0.8); }}
"""
SPARK = '<svg viewBox="0 0 24 24" width="{s}" height="{s}"><path fill="{c}" d="M12 2c.6 4.8 2.2 7.4 8 10-5.8 2.6-7.4 5.2-8 10-.6-4.8-2.2-7.4-8-10 5.8-2.6 7.4-5.2 8-10z"/></svg>'


def page(body, bg="transparent"):
    return f"<!doctype html><html><head><meta charset=utf-8><style>{CSS} body{{background:{bg}}}</style></head><body>{body}</body></html>"


def lower_third(num, title, sub):
    n = f'<span style="font:700 22px JB;color:#F0701E;margin-right:18px">{num}</span>' if num else ""
    return page(f"""<div class=glass style="position:absolute;left:56px;top:56px;padding:20px 28px 20px 24px;
      border-left:4px solid #F0701E;max-width:1100px">
      <div style="display:flex;align-items:baseline">{n}<span style="font:650 36px Inter;letter-spacing:-.01em">{title}</span></div>
      <div style="font:500 21px Inter;color:#B4B4BA;margin-top:8px">{sub}</div></div>""")


def model_badge(model, what, color, slot):
    bottom = 196 + slot * 104  # left side, above the captions (the AI panel is on the right)
    return page(f"""<div class=glass style="position:absolute;left:56px;bottom:{bottom}px;display:flex;align-items:center;gap:16px;
      padding:16px 24px 16px 18px;border-color:{color}66">
      <div style="width:50px;height:50px;border-radius:14px;display:grid;place-items:center;background:{color}22">{SPARK.format(s=26, c=color)}</div>
      <div><div style="font:700 26px Inter;letter-spacing:-.01em">{model}</div>
      <div style="font:500 18px JB;color:{color};margin-top:3px">{what}</div></div></div>""")


def spotlight(rect, pad=10):
    x, y, w, h = rect
    x, y, w, h = x - pad, y - pad, w + 2 * pad, h + 2 * pad
    return page(f"""<div style="position:absolute;left:{x}px;top:{y}px;width:{w}px;height:{h}px;border-radius:16px;
      border:3px solid #F0701E;box-shadow:0 0 0 9999px rgba(0,0,0,.42),0 0 28px 4px rgba(240,112,30,.55)"></div>""")


def chapter_layers(name):
    num, title, sub, models = CHAPTERS[name]
    chips = "".join(chip(m, c) for m, c in models)
    bg = page(topo_svg(), "#0B0B0C")
    fg = page(f"""<div style="position:absolute;left:180px;top:0;bottom:0;display:flex;flex-direction:column;justify-content:center">
      <div style="font:700 40px JB;color:#F0701E;letter-spacing:.08em">{num}</div>
      <div style="font:700 84px Inter;letter-spacing:-.02em;margin-top:14px">{title}</div>
      <div style="font:500 32px Inter;color:#B4B4BA;margin-top:18px">{sub}</div>
      <div style="display:flex;gap:16px;margin-top:44px">{chips}</div></div>""")
    return bg, fg


def caption(text):
    return page(f"""<div style="position:absolute;left:50%;transform:translateX(-50%);bottom:52px;max-width:1500px;
      padding:12px 26px;border-radius:14px;background:rgba(8,8,10,.78);font:600 38px Inter;line-height:1.25;
      text-align:center;color:#fff;text-shadow:0 1px 2px rgba(0,0,0,.6)">{text}</div>""")


def vo_samples(i, sr):
    raw = subprocess.run([FF, "-loglevel", "error", "-i", str(MH / f"{VO[i]}.mp3.mpeg"), "-ac", "1", "-ar", str(sr),
                          "-f", "f64le", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float64)


def caption_times(i, t0, sr=16000):
    """Spread VO i's caption lines over its speech, cutting at the pauses nearest
    to where each line would end if speech were evenly paced."""
    v = vo_samples(i, sr)
    hop = int(0.02 * sr)
    e = np.sqrt(np.convolve(v ** 2, np.ones(hop) / hop, mode="same")[::hop])
    speech = e > max(0.02, 0.12 * np.percentile(e, 95))
    idx = np.flatnonzero(speech)
    s0, s1 = idx[0] * 0.02, idx[-1] * 0.02
    gaps, run = [], 0  # centres of silent runs >= 0.12 s
    for j, sp in enumerate(speech):
        if not sp:
            run += 1
        elif run:
            if run >= 6:
                gaps.append((j - run / 2) * 0.02)
            run = 0
    lines = CAPTIONS[i]
    chars = np.cumsum([len(x) for x in lines])
    cuts = [s0]
    for c in chars[:-1]:
        ideal = s0 + (s1 - s0) * c / chars[-1]
        near = [g for g in gaps if abs(g - ideal) < 0.8 and g > cuts[-1] + 0.4]
        cuts.append(min(near, key=lambda g: abs(g - ideal)) if near else ideal)
    cuts.append(s1 + 0.35)
    return [(text, t0 + cuts[k], t0 + cuts[k + 1]) for k, text in enumerate(lines)]


def speed_tag(x):
    return page(f"""<div class=glass style="position:absolute;right:56px;top:56px;padding:10px 18px;border-radius:12px;
      font:700 22px JB;letter-spacing:.04em">▶▶ {x:g}× SPEED</div>""")


def note(text):
    return page(f"""<div class=glass style="position:absolute;left:50%;transform:translateX(-50%);bottom:170px;padding:14px 26px;
      border-radius:999px;font:600 24px Inter;border-color:#F0701E88;white-space:nowrap">{text}</div>""")


def cover_line():
    # sits over the result line of the inspect row
    return page(f"""<div style="position:absolute;left:{COVER[0]}px;top:{COVER[1]}px;width:{COVER[2]}px;height:{COVER[3]}px;
      background:#16141F;display:flex;align-items:center;padding-left:6px;font:600 {COVER[4]}px JB;color:#86EFAC;white-space:nowrap">
      Found power line 0.88 · SAM 3.1 + Muse Spark vision</div>""")


COVER = (1538, 633, 340, 18, 11)  # x, y, w, h, font px: the mislabeled result line, full frame


def card(name):
    topo = topo_svg()
    if name == "title":
        chips = "".join(chip(m, c) for m, c in MODELS)
        return page(f"""{topo}<div style="position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center">
          <div style="display:flex;align-items:center;gap:28px">{LOGO.format(s=96)}
          <span style="font:700 128px Inter;letter-spacing:.28em;margin-right:-.28em">SAHARA</span></div>
          <div style="font:600 46px Inter;margin-top:26px;color:#C9C9CF">Search-and-rescue drone <span style="color:#F0701E">×</span> Meta AI</div>
          <div style="display:flex;gap:18px;margin-top:56px">{chips}</div></div>""", "#0B0B0C")
    if name == "arch":
        return page(f"""{topo}{ARCH}""", "#0B0B0C")
    if name == "end":
        return page(f"""{topo}<div style="position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center">
          <div style="display:flex;align-items:center;gap:24px">{LOGO.format(s=80)}
          <span style="font:700 104px Inter;letter-spacing:.28em;margin-right:-.28em">SAHARA</span></div>
          <div style="font:500 30px Inter;letter-spacing:.3em;color:#9D9DA3;margin-top:22px">SEARCH · DETECT · RESCUE</div>
          <div style="font:500 26px Inter;color:#C9C9CF;margin-top:60px">Built with Muse Spark 1.3 · SAM 3.1 · Muse Voice Transcribe</div></div>""",
                    "#0B0B0C")
    raise ValueError(name)


MODELS = [("Muse Spark 1.3", "#A78BFA"), ("SAM 3.1", "#2DD4BF"), ("Muse Voice Transcribe", "#60A5FA")]
LOGO = '<svg viewBox="0 0 32 32" width="{s}" height="{s}" fill="none" stroke="#EDEDEE" stroke-width="1.6" stroke-linejoin="round"><path d="M3 5 16 27 29 5 16 12z"/><path d="M16 12v15"/></svg>'


def chip(model, color):
    return (f'<div class=glass style="display:flex;align-items:center;gap:12px;padding:14px 22px;border-color:{color}77">'
            f'{SPARK.format(s=22, c=color)}<span style="font:650 26px Inter">{model}</span></div>')


def box(x, y, w, h, title, sub, color="#3F3F46", fill="rgba(24,24,27,.92)"):
    return (f'<div style="position:absolute;left:{x}px;top:{y}px;width:{w}px;height:{h}px;border:1.5px solid {color};'
            f'background:{fill};border-radius:16px;display:flex;flex-direction:column;justify-content:center;padding:0 22px">'
            f'<div style="font:650 26px Inter">{title}</div><div style="font:500 18px JB;color:#A1A1AA;margin-top:6px">{sub}</div></div>')


def arrow(x1, y1, x2, y2):
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#71717A" stroke-width="2.5" marker-end="url(#a)"/>'


def _arch():
    P, S, V = "#A78BFA", "#2DD4BF", "#60A5FA"
    o = '<div style="position:absolute;left:0;right:0;top:64px;text-align:center;font:700 48px Inter">Three Meta models, one drone</div>'
    rows = [
        # (y, [boxes]) boxes: (x, w, title, sub, color)
        (190, [(110, 300, "Responder", "voice or text", "#52525B"), (500, 380, "Muse Voice Transcribe", "speech → text", V),
               (970, 380, "Muse Spark 1.3", "agent · tool calling", P), (1440, 370, "Mission bridge", "inspect · prioritize · return", "#F0701E")]),
        (420, [(110, 300, "Drone camera", "RGB + thermal", "#52525B"), (500, 380, "SAM 3.1", "segments named hazards", S),
               (970, 380, "Muse Spark vision", "reads the whole scene", P), (1440, 370, "Hazard map", "routes re-planned", "#F0701E")]),
        (650, [(110, 300, "Casualty hover", "colour + thermal frames", "#52525B"), (500, 380, "Muse Spark vision", "aerial triage note", P),
               (970, 380, "Muse Spark 1.3", "SITREP narrative", P), (1440, 370, "Rescue team", "priorities · safe routes", "#F0701E")]),
    ]
    svg = ['<svg width="1920" height="1080" style="position:absolute;inset:0"><defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" '
           'markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0 10 5 0 10z" fill="#71717A"/></marker></defs>']
    for y, bs in rows:
        for (x, w, *_), (x2, *__) in zip(bs, bs[1:]):
            svg.append(arrow(x + w + 6, y + 60, x2 - 8, y + 60))
            o += ""
        for x, w, t, s, c in bs:
            o += box(x, y, w, 120, t, s, c)
    svg.append("</svg>")
    o += ('<div style="position:absolute;left:0;right:0;top:830px;text-align:center;font:500 26px Inter;color:#B4B4BA">'
          'Flight safety stays in the autopilot and the operator\'s hands: <span style="color:#EDEDEE">the AI proposes, a human confirms.</span></div>')
    return "".join(svg) + o


ARCH = _arch()


def topo_svg():
    import math
    paths = []
    for k in range(1, 15):
        r, pts = 36 + k * 24, []
        for i in range(61):
            a = -math.pi * 0.5 - 0.05 + i / 60 * math.pi * 0.6
            n = 1 + 0.10 * math.sin(3 * a + k * 0.35) + 0.06 * math.sin(7 * a - k * 0.6) + 0.03 * math.sin(13 * a + k)
            pts.append(f"{-40 + r * n * math.cos(a) * 1.15:.0f} {560 + r * n * math.sin(a):.0f}")
        paths.append(f'<path d="M{"L".join(pts)}" fill="none" stroke="#F0701E" stroke-opacity="{0.28 if k % 3 == 0 else 0.15}"/>')
    return f'<svg viewBox="0 0 360 560" style="position:absolute;left:0;bottom:0;width:540px;height:840px">{"".join(paths)}</svg>'


def render_pngs(items):
    """items: [(name, html, transparent)] -> OUT/png/name.png"""
    from playwright.sync_api import sync_playwright
    d = OUT / "png"
    d.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        b = p.chromium.launch(executable_path="/usr/bin/google-chrome", args=["--no-sandbox", "--allow-file-access-from-files"])
        pg = b.new_page(viewport={"width": W, "height": H})
        for name, html, transparent in items:
            f = d / f"{name}.html"
            f.write_text(html)
            pg.goto(f.as_uri())
            pg.evaluate("document.fonts.ready")
            pg.wait_for_timeout(80)
            pg.screenshot(path=str(d / f"{name}.png"), omit_background=transparent)
        b.close()
    return d


# ------------------------------------------------------------------ video

def zoom_filter(src, zoom):
    sw = SRC_W.get(src, 1920)
    k = sw / 1920
    x, y, w = zoom
    h = w * 9 / 16
    return f"crop={round(w * k)}:{round(h * k)}:{round(x * k)}:{round(y * k)},scale={W}:{H}:flags=lanczos"


def render_clip(s, f):
    d = s["t1"] - s["t0"]
    vf = f"setpts=(PTS-STARTPTS)/{s['speed']},{zoom_filter(s['src'], s['zoom'])},fps={FPS},format=yuv420p"
    subprocess.run([FF, "-loglevel", "error", "-y", "-ss", str(s["a"]), "-t", str(s["b"] - s["a"] + 0.5), "-i", str(SRC[s["src"]]),
                    "-an", "-vf", vf, "-t", f"{d:.4f}", "-c:v", "libx264", "-preset", "medium", "-crf", "16", "-r", str(FPS), str(f)],
                   check=True)


def render_card(png, d, f):
    fade = min(0.5, d / 4)
    subprocess.run([FF, "-loglevel", "error", "-y", "-loop", "1", "-t", f"{d:.4f}", "-i", str(png),
                    "-vf", f"fps={FPS},format=yuv420p,fade=in:st=0:d={fade},fade=out:st={d - fade}:d={fade}",
                    "-c:v", "libx264", "-preset", "medium", "-crf", "16", "-r", str(FPS), str(f)], check=True)


def render_chapter(bg, fg, d, f):
    """Transition card: slow push-in on the background, title slides up and fades in, all fades out."""
    n = round(d * FPS)
    flt = (f"[0:v]scale=2304:1296,zoompan=z='1+0.04*on/{n}':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d={n}:s={W}x{H}:fps={FPS}[b];"
           f"[1:v]format=rgba,fade=in:st=0.15:d=0.45:alpha=1[t];"
           f"[b][t]overlay=x=0:y='60*pow(max(0,1-t/0.6),3)':shortest=1,format=yuv420p,"
           f"fade=in:st=0:d=0.25,fade=out:st={d - 0.35:.3f}:d=0.35[v]")
    subprocess.run([FF, "-loglevel", "error", "-y", "-loop", "1", "-t", f"{d:.4f}", "-i", str(bg),
                    "-loop", "1", "-framerate", str(FPS), "-t", f"{d:.4f}", "-i", str(fg), "-filter_complex", flt, "-map", "[v]",
                    "-t", f"{d:.4f}", "-c:v", "libx264", "-preset", "medium", "-crf", "16", "-r", str(FPS), str(f)], check=True)


def soundtrack(scenes, dur):
    spec = importlib.util.spec_from_file_location("sound_demo", ROOT / "demo" / "sound_demo.py")
    sd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sd)
    S, SR = sd.S, sd.SR
    n = int((dur + 0.1) * SR)
    drums, music, send = np.zeros(n), np.zeros(n), np.zeros(n)
    side = np.ones(n)
    g0 = scenes["title"]["t0"]
    t = S.t_axis(g0 + 0.2)
    S.place(music, S.pad_chord((110.0, 164.8, 220.0), g0 + 0.2, 0.6) * np.clip(t / max(0.1, g0 - 0.4), 0, 1) ** 1.4, 0)
    sd.groove(music, send, drums, side, g0, scenes["end"]["t0"], [(scenes["arch"]["t0"], scenes["arch"]["t1"])])
    sd.resolve(music, send, scenes["end"]["t0"], scenes["end"]["t1"] - scenes["end"]["t0"])
    music *= side
    for name in CHAPTERS:
        S.place(send, S.whoosh(0.8) * 0.5, max(0.0, scenes[name]["t0"] - 0.375))
        S.place(drums, S.whoosh(0.8), max(0.0, scenes[name]["t0"] - 0.375))
    from scipy.signal import fftconvolve
    verb = fftconvolve(send, S.reverb_ir(), mode="full")[:n] * 0.22
    bed = drums * 0.8 + music + verb
    bed = bed / max(1e-6, np.percentile(np.abs(bed), 99.9))

    voice = np.zeros(n)
    for name, sc in scenes.items():
        if sc["vo"] is None:
            continue
        v = vo_samples(sc["vo"], SR)
        v = S.hp(v[None, :], 80)[0]                            # rumble off
        v = v / max(1e-6, np.percentile(np.abs(v), 99.5))
        v = np.tanh(v * 1.5) / np.tanh(1.5)                   # gentle compression: quiet words come up
        i = int((sc["t0"] + VO_LEAD) * SR)
        voice[i:i + len(v)] += v[: n - i]
    # duck the music under speech: smoothed speech envelope -> gain 1 .. 0.22
    env = np.abs(voice)
    k = int(0.25 * SR)
    env = np.convolve(env, np.ones(k) / k, mode="same")
    duck = 1 - 0.85 * np.clip(env / 0.06, 0, 1)
    duck = np.convolve(duck, np.ones(k) / k, mode="same")
    mix = voice + bed * 0.22 * duck
    tail = np.clip((dur - S.t_axis(dur + 0.1)) / 2.0, 0, 1)[:n]
    mix *= tail
    mix = np.tanh(mix / max(1e-6, np.percentile(np.abs(mix), 99.95)))  # soft limiter on the peaks
    mix *= 10 ** (-1.5 / 20) / np.max(np.abs(mix))
    f = OUT / "soundtrack.wav"
    S.write_wav(f, np.stack([mix, mix]))
    return f


def build():
    shots, scenes, dur = timeline()
    OUT.mkdir(parents=True, exist_ok=True)
    ovs = overlays(shots, scenes)
    layers = []
    for c in CHAPTERS:
        bg, fg = chapter_layers(c)
        layers += [(f"{c}_bg", bg, False), (f"{c}_fg", fg, True)]
    pngs = render_pngs([(f"card_{c}", card(c), False) for c in ("title", "arch", "end")] + layers
                       + [(name, html, True) for name, html, *_ in ovs])
    segd = OUT / "segs"
    segd.mkdir(exist_ok=True)
    files = []
    for i, s in enumerate(shots):
        f = segd / f"s{i:02d}.mp4"
        if s["kind"] == "clip":
            render_clip(s, f)
        elif s["card"] in CHAPTERS:
            render_chapter(pngs / f"{s['card']}_bg.png", pngs / f"{s['card']}_fg.png", s["t1"] - s["t0"], f)
        else:
            render_card(pngs / f"card_{s['card']}.png", s["t1"] - s["t0"], f)
        files.append(f)
    lst = segd / "list.txt"
    lst.write_text("".join(f"file '{f}'\n" for f in files))
    base = OUT / "base.mp4"
    subprocess.run([FF, "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", str(base)], check=True)

    audio = soundtrack(scenes, dur)
    args, chains, last = [FF, "-loglevel", "error", "-y", "-i", str(base), "-i", str(audio)], [], "[0:v]"
    # Each overlay is cropped to its visible pixels and its input starts at its
    # own time (-itsoffset), so ffmpeg holds a few small frames, not 60 full ones.
    from PIL import Image
    for k, (name, _html, t0, t1) in enumerate(ovs):
        im = Image.open(pngs / f"{name}.png")
        x0, y0, x1, y1 = im.getbbox()
        x0, y0 = x0 - x0 % 2, y0 - y0 % 2
        crop = pngs / f"{name}_crop.png"
        im.crop((x0, y0, x1 + (x1 - x0) % 2, y1 + (y1 - y0) % 2)).save(crop)
        d = t1 - t0
        fd = min(0.25, d / 3)
        args += ["-itsoffset", f"{t0:.3f}", "-loop", "1", "-framerate", str(FPS), "-t", f"{d:.3f}", "-i", str(crop)]
        j = k + 2
        chains.append(f"[{j}:v]format=rgba,fade=in:st={t0:.3f}:d={fd}:alpha=1,fade=out:st={t1 - fd:.3f}:d={fd}:alpha=1[o{k}]")
        nxt = f"[v{k}]"
        chains.append(f"{last}[o{k}]overlay={x0}:{y0}:eof_action=pass{nxt}")
        last = nxt
    out = ROOT / "demo" / "out" / "sahara-meta.mp4"
    args += ["-filter_complex", ";".join(chains), "-map", last, "-map", "1:a", "-c:v", "libx264", "-preset", "slow", "-crf", "18",
             "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", str(out)]
    subprocess.run(args, check=True)
    (OUT / "timeline.json").write_text(json.dumps({"duration": dur, "scenes": scenes}, indent=1))
    print(f"{out}  {dur:.1f} s")


def stills():
    shots, scenes, dur = timeline()
    d = OUT / "stills"
    d.mkdir(parents=True, exist_ok=True)
    for i, s in enumerate(shots):
        if s["kind"] != "clip":
            continue
        at = (s["a"] + s["b"]) / 2
        subprocess.run([FF, "-loglevel", "error", "-y", "-ss", str(at), "-i", str(SRC[s["src"]]), "-frames:v", "1",
                        "-vf", zoom_filter(s["src"], s["zoom"]), str(d / f"s{i:02d}_{s['scene']}.jpg")], check=True)
    print(json.dumps({k: (round(v["t0"], 1), round(v["t1"], 1)) for k, v in scenes.items()}), f"total {dur:.1f}s")


if __name__ == "__main__":
    {"build": build, "stills": stills}[sys.argv[1] if len(sys.argv) > 1 else "build"]()
