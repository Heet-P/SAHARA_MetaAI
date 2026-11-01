"""Minimal Meta Model API client (standard library only).

  chat()        Muse Spark chat completions: tools, images, JSON-schema output
  segment()     SAM 3.1 text-prompted segmentation of one image
  transcribe()  speech to text for voice commands

The API is OpenAI-compatible (https://api.meta.ai/v1). Every call is appended
to a JSONL usage log (model, latency, tokens, cost when prices are configured)
so the hackathon credits can be watched. Without an API key the same methods
are served by MockBackend, so the whole pipeline runs offline.
"""
import base64
import json
import time
import urllib.error
import urllib.request
import uuid

from .config import Config
from .sam import parse_records


class ApiError(RuntimeError):
    pass


def image_part(image, mime="image/jpeg"):
    """Chat-completions content part for an image (bytes or URL)."""
    url = image if isinstance(image, str) else f"data:{mime};base64,{base64.b64encode(image).decode()}"
    return {"type": "image_url", "image_url": {"url": url}}


class BudgetExceeded(ApiError):
    pass


class MetaClient:
    def __init__(self, config=None):
        self.cfg = config or Config()
        self.mock = MockBackend() if self.cfg.mock else None

    def calls_today(self):
        """Real (non-mock) API calls logged since local midnight."""
        midnight = time.mktime(time.localtime()[:3] + (0, 0, 0, 0, 0, -1))
        n = 0
        try:
            with open(self.cfg.usage_log) as f:
                for line in f:
                    try:
                        r = json.loads(line)
                    except ValueError:
                        continue
                    n += (not r.get("mock")) and r.get("t", 0) >= midnight
        except OSError:
            pass
        return n

    def _guard(self):
        if not self.mock and self.calls_today() >= self.cfg.max_calls_per_day:
            raise BudgetExceeded(f"daily API call cap reached ({self.cfg.max_calls_per_day}); "
                                 "raise SAHARA_AI_MAX_CALLS_PER_DAY to continue")

    # ------------------------------------------------------------- public
    def chat(self, messages, tools=None, response_format=None, model=None, temperature=0.2, max_tokens=None):
        """One chat completion. Returns the API's JSON (OpenAI chat shape)."""
        model = model or self.cfg.agent_model
        body = {"model": model, "messages": messages, "temperature": temperature}
        if tools:
            body["tools"] = tools
        if response_format:
            body["response_format"] = response_format
        if max_tokens:
            body["max_tokens"] = max_tokens
        self._guard()
        t0 = time.time()
        out = self.mock.chat(body) if self.mock else self._post_json("/chat/completions", body)
        self._log("chat", model, t0, out.get("usage"))
        return out

    def segment(self, concept, image, mime="image/jpeg"):
        """SAM 3.1 on one image. Returns (records, raw_text)."""
        model = self.cfg.sam_model
        url = image if isinstance(image, str) else f"data:{mime};base64,{base64.b64encode(image).decode()}"
        body = {
            "model": model,
            "input": [{"type": "message", "role": "user", "content": [
                {"type": "input_text", "text": concept},
                {"type": "input_image", "image_url": url},
            ]}],
            "stream": True,
            "metadata": {"include_confidence": "true", "mask_encoding": "one_bit"},
        }
        self._guard()
        t0 = time.time()
        text = self.mock.segment(concept) if self.mock else self._stream_text("/responses", body)
        self._log("segment", model, t0, None, extra={"concept": concept})
        return parse_records(text), text

    def transcribe(self, audio, filename="command.wav", mime="audio/wav"):
        """Speech to text (POST /asr/transcribe, WAV audio). Returns the transcript."""
        model = self.cfg.voice_model
        self._guard()
        t0 = time.time()
        if self.mock:
            text = self.mock.transcribe()
        else:
            settings = json.dumps({"model": model, "audioEncoding": "WAV", "mode": "PUSH_TO_TALK"})
            boundary = uuid.uuid4().hex
            body = (f'--{boundary}\r\nContent-Disposition: form-data; name="request"\r\n'
                    f'Content-Type: application/json\r\n\r\n{settings}\r\n'
                    f'--{boundary}\r\nContent-Disposition: form-data; name="audio"; filename="{filename}"\r\n'
                    f'Content-Type: {mime}\r\n\r\n').encode() + audio + f"\r\n--{boundary}--\r\n".encode()
            req = self._req("/asr/transcribe", body, f"multipart/form-data; boundary={boundary}")
            req.add_header("Accept", "text/plain")  # one turn per line
            with self._open(req) as r:
                text = " ".join(line.strip() for line in r.read().decode().splitlines() if line.strip())
        self._log("transcribe", model, t0, None)
        return text

    # ------------------------------------------------------------ transport
    def _req(self, path, data, ctype):
        req = urllib.request.Request(self.cfg.base_url + path, data=data, method="POST")
        req.add_header("Authorization", f"Bearer {self.cfg.api_key}")
        req.add_header("Content-Type", ctype)
        return req

    def _open(self, req):
        try:
            return urllib.request.urlopen(req, timeout=self.cfg.timeout_s)
        except urllib.error.HTTPError as e:
            raise ApiError(f"{e.code} {e.reason}: {e.read()[:500].decode(errors='replace')}") from None
        except urllib.error.URLError as e:
            raise ApiError(f"network error: {e.reason}") from None

    def _post_json(self, path, body):
        with self._open(self._req(path, json.dumps(body).encode(), "application/json")) as r:
            return json.loads(r.read())

    def _stream_text(self, path, body):
        """POST with stream=true; join response.output_text.delta events."""
        parts = []
        with self._open(self._req(path, json.dumps(body).encode(), "application/json")) as r:
            for raw in r:
                line = raw.decode(errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    ev = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                if ev.get("type") == "response.output_text.delta":
                    parts.append(ev.get("delta", ""))
                elif ev.get("type") == "error":
                    raise ApiError(str(ev))
        return "".join(parts)

    def _post_multipart(self, path, fields, files):
        boundary = uuid.uuid4().hex
        chunks = []
        for k, v in fields.items():
            chunks.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode())
        for k, (fname, mime, data) in files.items():
            chunks.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"; filename="{fname}"\r\n'
                          f'Content-Type: {mime}\r\n\r\n'.encode() + data + b"\r\n")
        chunks.append(f"--{boundary}--\r\n".encode())
        req = self._req(path, b"".join(chunks), f"multipart/form-data; boundary={boundary}")
        with self._open(req) as r:
            return json.loads(r.read())

    # ---------------------------------------------------------------- usage
    def _log(self, kind, model, t0, usage, extra=None):
        rec = {"t": round(t0, 3), "kind": kind, "model": model, "mock": bool(self.mock),
               "latency_s": round(time.time() - t0, 3)}
        if usage:
            pin, pout = usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)
            rec.update(tokens_in=pin, tokens_out=pout)
            if self.cfg.price_in is not None and self.cfg.price_out is not None:
                rec["est_cost_usd"] = round((pin * self.cfg.price_in + pout * self.cfg.price_out) / 1e6, 6)
        if extra:
            rec.update(extra)
        try:
            self.cfg.usage_log.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cfg.usage_log, "a") as f:
                f.write(json.dumps(rec) + "\n")
        except OSError:
            pass
        self.last = rec


class MockBackend:
    """Offline stand-in with the same shapes as the real API.

    Tool calls are chosen by simple keyword rules from the last user message,
    so commander tests and the dashboard can be exercised without a key.
    """
    SAM_FIXTURES = {
        "smoke": "<0f>0<|box;x1=380;y1=40;x2=600;y2=260;w=640;h=480;c=0.88|>",
        "fire": "<0f>0<|box;x1=470;y1=330;x2=560;y2=410;w=640;h=480;c=0.91|>",
        "person": "<0f>0<|box;x1=180;y1=200;x2=300;y2=330;w=640;h=480;c=0.93|>",
        "flood water": "<0f>0<|box;x1=0;y1=300;x2=260;y2=479;w=640;h=480;c=0.86|>",
    }

    def segment(self, concept):
        return self.SAM_FIXTURES.get(concept.lower().strip(), "")

    def transcribe(self):
        return "Inspect the collapsed police station, there are reports of two people trapped."

    def chat(self, body):
        msgs = body.get("messages", [])
        tools = {t["function"]["name"] for t in body.get("tools", []) if t.get("type") == "function"}
        fmt = body.get("response_format") or {}
        usage = {"prompt_tokens": 0, "completion_tokens": 0}
        if fmt.get("type") == "json_schema":
            return self._wrap({"role": "assistant", "content": json.dumps(self._schema_stub(fmt))}, usage)
        last = msgs[-1] if msgs else {}
        if last.get("role") == "tool" or not tools:
            done = [m for m in msgs if m.get("role") == "tool"]
            text = (f"Done. I ran {len(done)} action(s); the drone is carrying them out and the map will update."
                    if done else "Mock mode: no API key set, so I can only run scripted actions.")
            return self._wrap({"role": "assistant", "content": text}, usage)
        said = str(last.get("content", "")).lower()
        calls = []
        def call(name, **args):
            if name in tools:
                calls.append({"id": f"call_{len(calls)}", "type": "function",
                              "function": {"name": name, "arguments": json.dumps(args)}})
        if any(w in said for w in ("inspect", "check", "look at")):
            place = "Collapsed police station" if "police" in said else "the reported location"
            call("inspect", place=place, concepts=["person", "smoke", "fire"])
        if any(w in said for w in ("trapped", "reports", "people")):
            call("prioritize_area", place="Collapsed police station" if "police" in said else "the reported location",
                 strength="high")
        if any(w in said for w in ("start", "search", "begin")) and not calls:
            call("start_mission")
        if "land" in said:
            call("land_now")
        if "situation" in said or "status" in said or not calls:
            call("get_situation")
        return self._wrap({"role": "assistant", "content": None, "tool_calls": calls}, usage)

    @staticmethod
    def _schema_stub(fmt):
        props = (fmt.get("json_schema", {}).get("schema", {}) or {}).get("properties", {})
        out = {}
        for k, spec in props.items():
            t = spec.get("type")
            out[k] = ([] if t == "array" else 0.5 if t == "number" else False if t == "boolean"
                      else spec.get("enum", ["unknown"])[0] if "enum" in spec else "mock value")
        return out

    @staticmethod
    def _wrap(message, usage):
        return {"id": "mock", "object": "chat.completion", "model": "mock",
                "choices": [{"index": 0, "message": message,
                             "finish_reason": "tool_calls" if message.get("tool_calls") else "stop"}],
                "usage": usage}
