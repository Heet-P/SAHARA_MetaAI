// AI Mission Commander panel: chat with SAHARA's Muse Spark agent, which
// drives the drone through tools (see command_center/backend/ai_routes.py).
// Streams the agent's tool calls live, asks the operator to confirm a start,
// and takes voice commands (transcribed by Meta's speech model).
(() => {
  const $ = (id) => document.getElementById(id);
  const panel = $("ai-panel"), log = $("ai-log"), form = $("ai-form"), input = $("ai-text");
  if (!panel) return;

  const TOOL_LABEL = {
    get_situation: "Checking the situation",
    start_mission: "Starting the mission",
    inspect: "Inspecting",
    prioritize_area: "Prioritizing area",
    report_hazard: "Adding reported hazard",
    find_now: "Looking with SAM 3.1",
    return_home: "Returning home",
    land_now: "Landing now",
  };
  const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

  function toggle(open = !panel.classList.contains("open")) {
    panel.classList.toggle("open", open);
    panel.setAttribute("aria-hidden", String(!open));
    $("btn-ai").classList.toggle("on", open);
    document.body.classList.toggle("ai-open", open);
    if (open) { input.focus(); refreshStatus(); }
  }
  $("btn-ai").addEventListener("click", () => toggle());
  $("ai-close").addEventListener("click", () => toggle(false));
  document.addEventListener("keydown", (e) => {
    if (e.key.toLowerCase() === "a" && !/input|textarea|select/i.test(document.activeElement.tagName)) toggle();
  });
  $("ai-reset").addEventListener("click", async () => {
    await fetch("/api/ai/reset", { method: "POST" });
    log.innerHTML = '<li class="ai-hint">New conversation.</li>';
  });

  function add(cls, html) {
    const li = document.createElement("li");
    li.className = cls;
    li.innerHTML = html;
    log.appendChild(li);
    log.scrollTop = log.scrollHeight;
    return li;
  }

  function describeArgs(name, a) {
    if (!a) return "";
    const where = a.place || (a.east_m != null ? `E${Math.round(a.east_m)} N${Math.round(a.north_m)}` : "");
    const what = Array.isArray(a.concepts) ? a.concepts.join(", ") : a.concepts || a.type || "";
    return [where, what && `for ${what}`, a.strength && `(${a.strength})`].filter(Boolean).join(" ");
  }

  function describeResult(name, r) {
    if (!r) return "";
    if (r.status === "error" || r.status === "rejected") return `✗ ${r.reason || r.message || r.status}`;
    if (name === "inspect" || name === "find_now") {
      const sam = r.sam || {};
      // Name the model that actually saw it: SAM 3.1 masks, Muse Spark vision, or both.
      const MODEL = { "sam3.1": "SAM 3.1", "muse-spark": "Muse Spark" };
      const found = Object.entries(sam.found || {}).map(([k, v]) =>
        `${k.replace("_", " ")} ${v.best_conf} (${(v.by || ["sam3.1"]).map((m) => MODEL[m] || m).join(" + ")})`);
      if (sam.error) return `✗ perception: ${sam.error}`;
      return found.length ? `Found ${found.join(", ")}` : "SAM 3.1 + Muse Spark: nothing found";
    }
    if (name === "get_situation") {
      return `${r.stage || "no mission"} · ${(r.casualties || []).length} casualties · ${(r.hazards || []).length} hazards`;
    }
    if (name === "prioritize_area" && r.share_within_radius != null) return `${Math.round(r.share_within_radius * 100)}% of remaining probability now there`;
    if (name === "report_hazard" && r.hazard_id) return `added ${r.hazard_id}`;
    if (r.status === "needs_confirmation") return "waiting for your confirmation";
    return r.status || "done";
  }

  let toolRow = null;
  function onEvent(ev) {
    if (ev.type === "status") return;
    if (ev.type === "tool_call") {
      toolRow = add("ai-tool", `<span class="ai-dot"></span><b>${esc(TOOL_LABEL[ev.name] || ev.name)}</b> <span class="ai-args">${esc(describeArgs(ev.name, ev.args))}</span><div class="ai-prog"></div>`);
    } else if (ev.type === "tool_progress" && toolRow) {
      toolRow.querySelector(".ai-prog").textContent = ev.text;
    } else if (ev.type === "tool_result" && toolRow) {
      toolRow.classList.add("done");
      const r = ev.result || {};
      if (r.status === "error" || r.status === "rejected") toolRow.classList.add("bad");
      toolRow.querySelector(".ai-prog").textContent = describeResult(ev.name, r);
      if ((ev.name === "inspect" || ev.name === "find_now") && r.sam && !r.sam.error) {
        // the exact frame SAM 3.1 analysed, with its boxes
        const img = new Image();
        img.className = "ai-sam";
        img.alt = "SAM 3.1 view";
        let retried = false;
        img.onerror = () => {
          if (retried) { img.remove(); return; }
          retried = true;
          setTimeout(() => { img.src = `/api/ai/sam_frame?t=${Date.now()}`; }, 1500);
        };
        img.src = `/api/ai/sam_frame?t=${Date.now()}`;
        toolRow.appendChild(img);
      }
      toolRow = null;
    } else if (ev.type === "confirm") {
      const li = add("ai-confirm", `<div>${esc(ev.text)}</div><button class="ai-yes">Confirm start</button><button class="ai-no">Cancel</button>`);
      li.querySelector(".ai-yes").onclick = async () => {
        li.querySelectorAll("button").forEach((b) => (b.disabled = true));
        const r = await (await fetch("/api/ai/confirm", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ action: ev.action }) })).json();
        li.insertAdjacentHTML("beforeend", `<div class="ai-note">${r.ok ? "✓ Mission started" : "✗ " + esc(r.message)}</div>`);
      };
      li.querySelector(".ai-no").onclick = () => {
        li.querySelectorAll("button").forEach((b) => (b.disabled = true));
        fetch("/api/ai/cancel", { method: "POST" });
      };
    } else if (ev.type === "reply") {
      add("ai-msg ai-bot", esc(ev.text).replace(/\n/g, "<br>"));
    } else if (ev.type === "error") {
      add("ai-msg ai-err", esc(ev.text));
    }
  }

  // Read one events stream; returns {next, done}. Throws if the connection drops.
  async function follow(turn, from, thinking) {
    const res = await fetch(`/api/ai/events?turn=${turn}&from=${from}`);
    if (!res.ok) throw new Error(`events ${res.status}`);
    const reader = res.body.getReader(), dec = new TextDecoder();
    let buf = "", next = from;
    for (;;) {
      const { value, done } = await reader.read();
      if (done) return { next, done: false };
      buf += dec.decode(value, { stream: true });
      let i;
      while ((i = buf.indexOf("\n\n")) >= 0) {
        const chunk = buf.slice(0, i); buf = buf.slice(i + 2);
        const line = chunk.split("\n").find((l) => l.startsWith("data:"));
        if (!line) continue;  // keepalive
        let ev; try { ev = JSON.parse(line.slice(5)); } catch { continue; }
        if (ev.type === "done") return { next, done: true };
        if (typeof ev.i === "number") next = ev.i + 1;
        log.appendChild(thinking);  // keep the dots at the bottom while working
        onEvent(ev);
      }
    }
  }

  async function send(text) {
    text = text.trim();
    if (!text) return;
    add("ai-msg ai-me", esc(text));
    input.value = "";
    form.classList.add("busy");
    const thinking = add("ai-thinking", "<span></span><span></span><span></span>");
    try {
      const res = await fetch("/api/ai/chat", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ text }) });
      const j = await res.json().catch(() => ({}));
      if (!res.ok || !j.turn) { add("ai-msg ai-err", esc(j.message || `error ${res.status}`)); return; }
      // The agent keeps working server-side; if this stream drops, resume it.
      let from = 0, tries = 0;
      for (;;) {
        try {
          const r = await follow(j.turn, from, thinking);
          from = r.next;
          if (r.done) break;
        } catch (e) {
          if (++tries > 20) { add("ai-msg ai-err", esc(`lost the connection to the commander: ${e}`)); break; }
          await new Promise((ok) => setTimeout(ok, 1500));
        }
      }
    } catch (e) {
      add("ai-msg ai-err", esc(`connection lost: ${e}`));
    } finally {
      thinking.remove();
      form.classList.remove("busy");
      refreshStatus();
    }
  }
  form.addEventListener("submit", (e) => { e.preventDefault(); send(input.value); });

  // Voice: hold the mic button, release to transcribe and send.
  let rec = null, chunks = [];
  const mic = $("ai-mic");
  async function startRec() {
    if (rec || !navigator.mediaDevices) return;
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      rec = new MediaRecorder(stream);
      chunks = [];
      rec.ondataavailable = (e) => chunks.push(e.data);
      rec.onstop = async () => {
        stream.getTracks().forEach((t) => t.stop());
        const blob = new Blob(chunks, { type: rec.mimeType || "audio/webm" });
        rec = null; mic.classList.remove("rec");
        if (blob.size < 2000) return;
        input.placeholder = "Transcribing…";
        let r;
        try {
          const wav = await toWav(blob);
          r = await (await fetch("/api/ai/transcribe", { method: "POST", headers: { "Content-Type": "audio/wav" }, body: wav })).json();
        } catch (e) { r = { ok: false, message: `could not read the recording: ${e.message || e}` }; }
        input.placeholder = "Message the commander…";
        if (r.ok && r.text) send(r.text); else add("ai-msg ai-err", esc(r.message || "could not transcribe"));
      };
      rec.start();
      mic.classList.add("rec");
    } catch (e) { add("ai-msg ai-err", esc(`microphone: ${e.message}`)); }
  }
  // Meta's speech API takes WAV: decode the browser's recording, downmix to
  // 16 kHz mono and write 16-bit PCM.
  async function toWav(blob) {
    const ctx = new AudioContext();
    const decoded = await ctx.decodeAudioData(await blob.arrayBuffer());
    ctx.close();
    const rate = 16000;
    const off = new OfflineAudioContext(1, Math.ceil(decoded.duration * rate), rate);
    const src = off.createBufferSource();
    src.buffer = decoded; src.connect(off.destination); src.start();
    const pcm = (await off.startRendering()).getChannelData(0);
    const buf = new DataView(new ArrayBuffer(44 + pcm.length * 2));
    const str = (o, t) => [...t].forEach((c, i) => buf.setUint8(o + i, c.charCodeAt(0)));
    str(0, "RIFF"); buf.setUint32(4, 36 + pcm.length * 2, true); str(8, "WAVEfmt ");
    buf.setUint32(16, 16, true); buf.setUint16(20, 1, true); buf.setUint16(22, 1, true);
    buf.setUint32(24, rate, true); buf.setUint32(28, rate * 2, true); buf.setUint16(32, 2, true); buf.setUint16(34, 16, true);
    str(36, "data"); buf.setUint32(40, pcm.length * 2, true);
    pcm.forEach((v, i) => buf.setInt16(44 + i * 2, Math.max(-1, Math.min(1, v)) * 0x7fff, true));
    return new Blob([buf], { type: "audio/wav" });
  }
  const stopRec = () => rec && rec.state === "recording" && rec.stop();
  mic.addEventListener("mousedown", startRec);
  mic.addEventListener("touchstart", (e) => { e.preventDefault(); startRec(); });
  ["mouseup", "mouseleave", "touchend"].forEach((ev) => mic.addEventListener(ev, stopRec));

  async function refreshStatus() {
    try {
      const s = await (await fetch("/api/ai/status")).json();
      $("ai-mode").textContent = s.mock ? "MOCK" : "LIVE";
      $("ai-mode").className = `ai-mode ${s.mock ? "mock" : "live"}`;
      $("ai-sub").textContent = `${s.model} · ${s.sam_model}${s.mock ? " · mock mode (no API key)" : ""}`;
      $("ai-spend").textContent = `${s.calls_today}/${s.call_cap} calls · ~$${s.est_spend_today_usd.toFixed(2)}`;
    } catch { $("ai-mode").textContent = "OFF"; }
  }
  refreshStatus();
})();
