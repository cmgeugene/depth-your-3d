import {
  $, el, api, postJSON, toast, makeDropzone, uploadAndPrepare, jobCard, recentJobs, watchQueue, store,
} from "./common.js";

const MAX_UNDO = 15;
const MASK_RGB = "255, 70, 120"; // overlay tint for the area that will be cleaned

let status;
let media = null;
let opts;
let img = null; // source image
let result = null; // { img, name } shown in compare mode
let showing = "src";
const mask = document.createElement("canvas"); // alpha = area to clean
const tint = document.createElement("canvas");
const mctx = mask.getContext("2d", { willReadFrequently: true });
const tctx = tint.getContext("2d");
const view = $("#view");
const vctx = view.getContext("2d");
let tool = "brush";
let brush = 40;
let undo = [];
let cursor = null; // pointer position in image px
let last = null; // last stroke point while drawing
let coverage = 0; // fraction of the image inside the mask
let split = 0.5; // split view: original left of this fraction of the width, result right of it
let splitting = false; // dragging the split line
const SHOW_CYCLE = ["src", "out", "split"];

const jobsBox = $("#jobs");
const syncEmpty = () => ($("#jobsEmpty").hidden = jobsBox.children.length > 0);

// ---------- drawing ----------

let frame = 0;
function render() {
  if (frame) return;
  frame = requestAnimationFrame(() => {
    frame = 0;
    if (!img) return;
    vctx.globalAlpha = 1;
    vctx.drawImage(showing === "out" && result ? result.img : img, 0, 0);
    const scale = view.width / view.getBoundingClientRect().width; // image px per screen px
    if (showing === "split" && result) {
      const x = Math.round(split * view.width);
      vctx.save();
      vctx.beginPath();
      vctx.rect(x, 0, view.width - x, view.height);
      vctx.clip();
      vctx.drawImage(result.img, 0, 0);
      vctx.restore();
    }
    if ($("#showMask").checked) {
      tctx.globalCompositeOperation = "copy";
      tctx.drawImage(mask, 0, 0);
      tctx.globalCompositeOperation = "source-in";
      tctx.fillStyle = `rgb(${MASK_RGB})`;
      tctx.fillRect(0, 0, tint.width, tint.height);
      vctx.globalAlpha = 0.45;
      vctx.drawImage(tint, 0, 0);
      vctx.globalAlpha = 1;
    }
    if (showing === "split" && result) drawSplitLine(scale);
    else if (cursor) {
      vctx.lineWidth = 1.5 * scale; // keep the outline ~1.5 screen px
      vctx.strokeStyle = tool === "brush" ? "#fff" : "#ffd27a";
      vctx.beginPath();
      vctx.arc(cursor.x, cursor.y, brush / 2, 0, Math.PI * 2);
      vctx.stroke();
      vctx.strokeStyle = "rgba(0,0,0,.6)";
      vctx.beginPath();
      vctx.arc(cursor.x, cursor.y, brush / 2 + vctx.lineWidth, 0, Math.PI * 2);
      vctx.stroke();
    }
  });
}

function drawSplitLine(scale) {
  const x = split * view.width;
  vctx.fillStyle = "#fff";
  vctx.fillRect(x - scale, 0, 2 * scale, view.height);
  const r = 14 * scale, cy = view.height / 2;
  vctx.beginPath();
  vctx.arc(x, cy, r, 0, Math.PI * 2);
  vctx.fill();
  vctx.fillStyle = "#111";
  vctx.font = `${Math.round(15 * scale)}px sans-serif`;
  vctx.textAlign = "center";
  vctx.textBaseline = "middle";
  vctx.fillText("⇆", x, cy + scale);
  const tag = (text, left) => {
    vctx.font = `${Math.round(12 * scale)}px sans-serif`;
    const w = vctx.measureText(text).width + 16 * scale, h = 22 * scale, pad = 8 * scale;
    const tx = left ? pad : view.width - pad - w;
    vctx.fillStyle = "rgba(10,12,14,.75)";
    vctx.fillRect(tx, pad, w, h);
    vctx.fillStyle = "#fff";
    vctx.fillText(text, tx + w / 2, pad + h / 2);
  };
  tag("원본", true);
  tag("결과", false);
}

function toImage(e) {
  const r = view.getBoundingClientRect();
  return { x: ((e.clientX - r.left) * view.width) / r.width, y: ((e.clientY - r.top) * view.height) / r.height };
}

function stroke(a, b) {
  mctx.globalCompositeOperation = tool === "brush" ? "source-over" : "destination-out";
  mctx.strokeStyle = mctx.fillStyle = "#fff";
  mctx.lineWidth = brush;
  mctx.lineCap = mctx.lineJoin = "round";
  mctx.beginPath();
  if (a.x === b.x && a.y === b.y) {
    mctx.arc(a.x, a.y, brush / 2, 0, Math.PI * 2);
    mctx.fill();
  } else {
    mctx.moveTo(a.x, a.y);
    mctx.lineTo(b.x, b.y);
    mctx.stroke();
  }
  mctx.globalCompositeOperation = "source-over";
}

function snapshot() {
  undo.push(mctx.getImageData(0, 0, mask.width, mask.height));
  if (undo.length > MAX_UNDO) undo.shift();
  $("#undoBtn").disabled = false;
}

function maskChanged() {
  const a = mctx.getImageData(0, 0, mask.width, mask.height).data;
  let on = 0;
  for (let i = 3; i < a.length; i += 4) if (a[i] > 127) on++;
  coverage = on / (mask.width * mask.height);
  render();
  sync();
}

function invert() {
  snapshot();
  mctx.globalCompositeOperation = "xor"; // opaque fill XOR mask = 1 - alpha
  mctx.fillStyle = "#fff";
  mctx.fillRect(0, 0, mask.width, mask.height);
  mctx.globalCompositeOperation = "source-over";
  maskChanged();
}

// SAM 3.1 finds every instance of the prompt ("hair" -> each character's hair); add it to or cut it from the mask.
async function autoSelect(subtract) {
  const prompt = $("#autoPrompt").value.trim();
  if (!prompt || !media) return;
  const buttons = [$("#autoAdd"), $("#autoSub")];
  buttons.forEach((b) => (b.disabled = true));
  $("#autoNote").textContent = "찾는 중…";
  try {
    // The worker leaves after a while without requests (and after a failure), so ask whether it is up now.
    const loaded = await api("/api/status").then((s) => s.cleanup.sam.loaded).catch(() => true);
    if (!loaded) $("#autoNote").textContent = "찾는 중… (SAM 3.1을 올리는 중이라 1분쯤 걸립니다)";
    const res = await fetch(`/api/media/${media.id}/segment`, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ prompt }),
    });
    if (!res.ok) {
      let msg = `HTTP ${res.status}`;
      try { msg = (await res.json()).detail || msg; } catch {}
      throw new Error(msg);
    }
    const found = await createImageBitmap(await res.blob());
    const c = document.createElement("canvas");
    c.width = mask.width;
    c.height = mask.height;
    const cx = c.getContext("2d", { willReadFrequently: true });
    cx.drawImage(found, 0, 0, c.width, c.height);
    const px = cx.getImageData(0, 0, c.width, c.height);
    for (let i = 0; i < px.data.length; i += 4) { // white-on-black -> white with that as alpha
      px.data[i + 3] = px.data[i];
      px.data[i] = px.data[i + 1] = px.data[i + 2] = 255;
    }
    cx.putImageData(px, 0, 0);
    snapshot();
    mctx.globalCompositeOperation = subtract ? "destination-out" : "source-over";
    mctx.drawImage(c, 0, 0);
    mctx.globalCompositeOperation = "source-over";
    $("#showMask").checked = true;
    $("#autoNote").textContent = `"${prompt}" 영역을 ${subtract ? "뺐습니다" : "추가했습니다"}. 브러시·지우개로 다듬을 수 있습니다.`;
    maskChanged();
  } catch (e) {
    $("#autoNote").textContent = "";
    toast(e.message, 5000);
  } finally {
    buttons.forEach((b) => (b.disabled = !status.cleanup.sam.available));
  }
}

function clearMask() {
  snapshot();
  mctx.clearRect(0, 0, mask.width, mask.height);
  maskChanged();
}

function undoMask() {
  const prev = undo.pop();
  if (prev) mctx.putImageData(prev, 0, 0);
  $("#undoBtn").disabled = !undo.length;
  maskChanged();
}

function setTool(t) {
  tool = t;
  $("#toolBrush").classList.toggle("on", t === "brush");
  $("#toolEraser").classList.toggle("on", t === "eraser");
  render();
}

function setBrush(v) {
  brush = Math.max(4, Math.min(400, Math.round(v)));
  $("#brushSize").value = brush;
  $("#brushVal").textContent = `${brush}px`;
  render();
}

function setShowing(v) {
  showing = result ? v : "src";
  for (const b of $("#compareCtrl").querySelectorAll("button")) b.classList.toggle("on", b.dataset.v === showing);
  view.classList.toggle("splitting", showing === "split"); // split view: dragging moves the line, not the brush
  render();
}

function moveSplit(e) {
  split = Math.min(1, Math.max(0, toImage(e).x / view.width));
  render();
}

view.addEventListener("pointerdown", (e) => {
  if (e.button !== 0 || !img) return;
  view.setPointerCapture(e.pointerId);
  if (showing === "split" && result) {
    splitting = true;
    moveSplit(e);
    return;
  }
  snapshot();
  last = toImage(e);
  stroke(last, last);
  render();
});
view.addEventListener("pointermove", (e) => {
  if (splitting) return moveSplit(e);
  cursor = toImage(e);
  if (last) {
    stroke(last, cursor);
    last = cursor;
  }
  render();
});
const endStroke = () => {
  splitting = false;
  if (!last) return;
  last = null;
  maskChanged();
};
view.addEventListener("pointerup", endStroke);
view.addEventListener("pointercancel", endStroke);
view.addEventListener("pointerleave", () => { cursor = null; render(); });
view.addEventListener("wheel", (e) => {
  // Over the image the wheel sizes the brush; elsewhere it scrolls the page as usual.
  e.preventDefault();
  setBrush(brush * (e.deltaY < 0 ? 1.15 : 1 / 1.15));
}, { passive: false });

// ---------- help ----------

const HELP_SEEN = "ds-cleanup-help-seen";

function openHelp() {
  // Freeze the page behind the dialog; pad for the vanished scrollbar so the layout doesn't jump.
  document.body.style.paddingRight = `${window.innerWidth - document.documentElement.clientWidth}px`;
  document.body.classList.add("modal-open");
  $("#helpModal").hidden = false;
  $("#helpClose").focus({ preventScroll: true });
}

function closeHelp() {
  document.body.classList.remove("modal-open");
  document.body.style.paddingRight = "";
  $("#helpModal").hidden = true;
  store(HELP_SEEN, true);
  $("#helpBtn").focus({ preventScroll: true }); // give focus back without jumping to the top
}

function helpTab(name) {
  for (const b of document.querySelectorAll(".help-tabs button")) b.classList.toggle("on", b.dataset.tab === name);
  for (const p of document.querySelectorAll(".help-pane")) p.hidden = p.dataset.pane !== name;
}

$(".help-tabs").addEventListener("click", (e) => { const b = e.target.closest("button"); if (b) helpTab(b.dataset.tab); });
for (const box of document.querySelectorAll(".ba")) {
  box.querySelector(".ba-range").addEventListener("input", (e) => box.style.setProperty("--pos", `${e.target.value}%`));
}
$("#helpBtn").addEventListener("click", openHelp);
$("#helpClose").addEventListener("click", closeHelp);
$("#helpModal").addEventListener("click", (e) => { if (e.target === e.currentTarget) closeHelp(); });
// A file dropped on the dialog would otherwise make the browser open it in place of the page.
for (const type of ["dragover", "drop"]) $("#helpModal").addEventListener(type, (e) => e.preventDefault());

document.addEventListener("keydown", (e) => {
  if (!$("#helpModal").hidden) { // the dialog takes the keyboard while it is open
    if (e.key === "Escape") { e.preventDefault(); closeHelp(); }
    return;
  }
  if (e.target instanceof Element && e.target.closest("input, textarea, select")) return;
  if (e.key === "?") { e.preventDefault(); openHelp(); return; }
  if (!img) return;
  const k = e.key.toLowerCase();
  if ((e.ctrlKey || e.metaKey) && k === "z") { e.preventDefault(); undoMask(); return; }
  if (e.ctrlKey || e.metaKey || e.altKey) return;
  if (k === "b") setTool("brush");
  else if (k === "e") setTool("eraser");
  else if (k === "i") invert();
  else if (k === "m") { $("#showMask").checked = !$("#showMask").checked; render(); }
  else if (k === "c" && result) setShowing(SHOW_CYCLE[(SHOW_CYCLE.indexOf(showing) + 1) % SHOW_CYCLE.length]);
  else if (e.key === "[") setBrush(brush / 1.2);
  else if (e.key === "]") setBrush(brush * 1.2);
  else return;
  e.preventDefault();
});

// ---------- options ----------

function slider(id, key, [min, max], fmt) {
  const input = $(`#${id}`);
  input.min = min;
  input.max = max;
  input.addEventListener("input", () => { opts[key] = +input.value; sync(); });
  return () => {
    input.value = opts[key];
    $(`#${id}Val`).textContent = fmt(opts[key]);
  };
}

let syncSliders = [];

// Reflects opts in the controls, the summary and the warnings; remembers opts per browser.
function sync() {
  const st = status.cleanup;
  syncSliders.forEach((f) => f());
  for (const b of document.querySelectorAll('.seg-ctrl[data-key="count"] button')) {
    const on = +b.dataset.v === opts.count;
    b.classList.toggle("on", on);
    b.setAttribute("aria-pressed", on);
  }
  if ($("#prompt").value !== opts.prompt) $("#prompt").value = opts.prompt;
  store("ds-cleanup-opts", opts);

  $("#engineWarn").hidden = st.available;
  $("#engineWarn").textContent = st.available ? "" : `얼룩 정리를 쓸 수 없습니다: ${st.reason}`;
  if (!media) return;

  const empty = coverage === 0, full = coverage > 0.9999;
  $("#maskInfo").textContent = empty
    ? "마스크 없음 — 이미지 전체를 정리합니다. 정리할 곳만 칠하거나, 지킬 곳(얼굴 등)을 칠한 뒤 반전하세요."
    : full ? "마스크가 이미지 전체 — 전체를 정리합니다."
      : `칠한 영역(${(coverage * 100).toFixed(1)}%)만 정리합니다. 나머지 픽셀은 원본 그대로 둡니다.`;
  $("#clearBtn").disabled = empty;
  $("#feather").disabled = empty || full;

  const mp = (media.width * media.height) / 1e6;
  const tooBig = mp > st.max_mp;
  $("#outInfo").textContent = `출력 ${media.width}×${media.height} PNG${media.has_alpha ? " (투명도 유지)" : ""} · ${st.model} · `
    + `실제 ${opts.steps}스텝${st.offload ? " · VRAM 절약 모드(느림)" : ""}`;
  const warns = [];
  if (opts.denoise > 0.28) warns.push("0.28보다 높으면 선이나 얼굴이 바뀌기 쉽습니다.");
  if (tooBig) warns.push(`이미지가 서버 한도(${st.max_mp}MP)보다 큽니다.`);
  $("#outWarn").hidden = !warns.length;
  $("#outWarn").textContent = warns.join(" ");
  $("#startBtn").disabled = !st.available || tooBig;
}

// ---------- media ----------

async function showMedia(m) {
  media = m;
  img = await loadImage(`/api/media/${m.id}/image`);
  for (const c of [view, mask, tint]) {
    c.width = img.naturalWidth;
    c.height = img.naturalHeight;
  }
  result = null;
  undo = [];
  $("#undoBtn").disabled = true;
  $("#compareCtrl").hidden = true;
  setShowing("src");
  setBrush(Math.max(img.naturalWidth, img.naturalHeight) / 30);
  $("#uploadBox").hidden = true;
  $("#paintBox").hidden = false;
  $("#optionsCard").hidden = false;
  history.replaceState(null, "", `?m=${m.id}`);
  maskChanged();
}

function loadImage(src) {
  return new Promise((resolve, reject) => {
    const i = new Image();
    i.onload = () => resolve(i);
    i.onerror = () => reject(new Error("이미지를 불러오지 못했습니다."));
    i.src = src;
  });
}

function reset() {
  media = img = result = null;
  $("#paintBox").hidden = true;
  $("#optionsCard").hidden = true;
  $("#uploadBox").hidden = false;
  $("#uploadBox").replaceChildren(dropzone);
  history.replaceState(null, "", location.pathname);
}

async function onFile(file) {
  const box = $("#uploadBox");
  const holder = el("div", { class: "stack" });
  box.replaceChildren(holder);
  try {
    await showMedia(await uploadAndPrepare(file, holder, "이미지 확인 중…"));
  } catch (e) {
    holder.replaceChildren(el("div", { class: "job-err" }, e.message), el("button", { onclick: reset }, "다시 시도"));
  }
}

// ---------- jobs ----------

async function showResult(j, out) {
  if (!media || j.media_id !== media.id) {
    // A job from an earlier visit or another image: open its source first, so there is something to compare with.
    if (media && coverage > 0 && !confirm("다른 이미지의 결과입니다. 지금 칠한 마스크를 버리고 그 이미지를 열까요?")) return;
    let m = null;
    try {
      m = await api(`/api/media/${j.media_id}`);
    } catch {}
    if (!m || m.status !== "ready" || m.kind !== "image") {
      toast("원본 이미지가 서버에서 지워져 비교할 수 없습니다. 결과 파일만 내려받습니다.", 4000);
      window.open(`/api/jobs/${j.id}/files/${encodeURIComponent(out.name)}`, "_blank");
      return;
    }
    await showMedia(m);
  }
  result = { img: await loadImage(`/api/jobs/${j.id}/files/${encodeURIComponent(out.name)}`), name: out.name };
  $("#compareCtrl").hidden = false;
  $("#showMask").checked = false;
  setShowing("out");
  $("#paint").scrollIntoView({ block: "nearest", behavior: "smooth" });
  toast("결과를 보여줍니다. C 키로 원본·결과·스플릿을 바꿔 볼 수 있습니다.", 3000);
}

function jobActions(j) {
  return j.outputs.map((o) => {
    const src = `/api/jobs/${j.id}/files/${encodeURIComponent(o.name)}`;
    return el("button", { class: "thumb-btn", title: `${o.label} — 위 화면에서 원본과 비교`, onclick: () => showResult(j, o) },
      el("img", { src, alt: "", loading: "lazy" }), "비교");
  });
}

function addJob(id, prepend = true) {
  const card = jobCard(id, { onGone: syncEmpty, kind: "cleanup", actions: jobActions });
  prepend ? jobsBox.prepend(card) : jobsBox.append(card);
  syncEmpty();
}

async function start() {
  $("#startBtn").disabled = true;
  try {
    const partial = coverage > 0 && coverage < 0.9999;
    const body = {
      media_id: media.id, denoise: opts.denoise, steps: opts.steps, guidance: opts.guidance,
      feather: opts.feather, count: opts.count, prompt: opts.prompt,
      seed: $("#seed").value === "" ? null : Math.max(0, Math.floor(+$("#seed").value)),
      mask: partial ? mask.toDataURL("image/png") : null,
    };
    const job = await postJSON("/api/cleanup-jobs", body);
    recentJobs(job.id);
    addJob(job.id);
    toast("대기열에 추가했습니다.");
  } catch (e) {
    toast(e.message, 4000);
  } finally {
    sync();
  }
}

const dropzone = $("#dropzone");

async function init() {
  status = await api("/api/status");
  const st = status.cleanup;
  opts = { ...st.defaults, ...(store("ds-cleanup-opts") || {}) };
  for (const [k, [lo, hi]] of Object.entries(st.limits)) opts[k] = Math.min(hi, Math.max(lo, +opts[k] || st.defaults[k]));
  $("#ttl").textContent = status.ttl_hours;
  $("#uploadHint").textContent = `png · jpg · webp — 최대 ${st.max_mp}MP (2048×2048 정도)`;
  makeDropzone(dropzone, onFile, "image/png,image/jpeg,image/webp,.png,.jpg,.jpeg,.webp");

  syncSliders = [
    slider("denoise", "denoise", [0.05, 0.4], (v) => v.toFixed(2)),
    slider("feather", "feather", st.limits.feather, (v) => (v ? `${v}px` : "끔")),
    slider("steps", "steps", st.limits.steps, (v) => String(v)),
    slider("guidance", "guidance", [1, 6], (v) => v.toFixed(1)),
  ];
  $('.seg-ctrl[data-key="count"]').addEventListener("click", (e) => {
    const b = e.target.closest("button");
    if (b) { opts.count = +b.dataset.v; sync(); }
  });
  $("#prompt").addEventListener("input", (e) => { opts.prompt = e.target.value; store("ds-cleanup-opts", opts); });
  $("#promptReset").addEventListener("click", () => { opts.prompt = st.defaults.prompt; sync(); });
  $("#toolBrush").addEventListener("click", () => setTool("brush"));
  $("#toolEraser").addEventListener("click", () => setTool("eraser"));
  $("#brushSize").addEventListener("input", (e) => setBrush(+e.target.value));
  $("#invertBtn").addEventListener("click", invert);
  $("#undoBtn").addEventListener("click", undoMask);
  $("#clearBtn").addEventListener("click", clearMask);
  const sam = st.sam;
  $("#autoAdd").addEventListener("click", () => autoSelect(false));
  $("#autoSub").addEventListener("click", () => autoSelect(true));
  $("#autoPrompt").addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); autoSelect(false); } });
  $("#autoAdd").disabled = $("#autoSub").disabled = $("#autoPrompt").disabled = !sam.available;
  if (!sam.available) $("#autoNote").textContent = `자동 선택을 쓸 수 없습니다: ${sam.reason}`;
  $("#showMask").addEventListener("change", render);
  $("#compareCtrl").addEventListener("click", (e) => { const b = e.target.closest("button"); if (b) setShowing(b.dataset.v); });
  $("#otherBtn").addEventListener("click", reset);
  $("#startBtn").addEventListener("click", start);
  setTool("brush");
  sync();
  for (const id of recentJobs()) addJob(id, false);
  if (!store(HELP_SEEN)) openHelp(); // first visit
  watchQueue(status);

  const m = new URLSearchParams(location.search).get("m");
  if (m) api(`/api/media/${m}`).then((x) => x.status === "ready" && x.kind === "image" && showMedia(x)).catch(() => {});
}

init();
