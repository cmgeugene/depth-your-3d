// Frame timing editor: step live-action footage on 1s/2s/3s like 2D animation.
//
// The editable spec S is small and serializable (segments, overrides, cut edits);
// everything shown is derived from it in derive(): which frames are new drawings (keys)
// and, for every frame, which source frame is shown (the timing map sent to the server).

import {
  $, el, api, postJSON, toast, store, fmtTC, detectCuts, makeDropzone, uploadAndPrepare,
  waitMedia, depthOptionsForm, jobCard, recentJobs, audioToggle,
} from "./common.js";

const STEP_COLORS = { 1: "#5aa9ff", 2: "#4cc38a", 3: "#f0b44c", 4: "#b38cff" };
const stepColor = (s) => (s === 0 ? "#7b8494" : STEP_COLORS[s] || "#3cc6c6");
const stepLabel = (s) => (s === 0 ? "홀드" : `${s}s`);

// Timeline rows (CSS px); must match .tl-labels heights in editor.html.
const ROW_Y = { ruler: 0, cuts: 20, seg: 50, frames: 72 };
const ROW_H = { ruler: 20, cuts: 30, seg: 22, frames: 64 };
const TL_H = 136;

let status, media, N, fps, depthForm;
let S; // editable timing spec
let D; // derived timing
const view = { cur: 0, sel: null, playing: false, mode: "step", onion: false, loop: true, speed: 1, cellW: 16, follow: true, wipe: 0.5 };
const hist = { undo: [], redo: [] };
const previews = [];
const thumbs = [];

const viewCanvas = $("#viewCanvas");
const tlCanvas = $("#tlCanvas");
const tlScroll = $("#tlScroll");

// ---------- spec ----------

function defaultSpec() {
  return {
    version: 1,
    segments: [{ start: 0, step: 2 }],
    overrides: {},
    cutAdd: [],
    cutRemove: [],
    threshold: status.cut_threshold,
    minShot: status.cut_min_shot,
    resetAtCuts: true,
  };
}

function sanitizeSpec(raw) {
  const d = defaultSpec();
  const s = { ...d, ...(raw && typeof raw === "object" ? raw : {}) };
  const isInt = (v, lo, hi) => Number.isInteger(v) && v >= lo && v <= hi;
  const seen = new Set();
  s.segments = (Array.isArray(s.segments) ? s.segments : [])
    .filter((x) => x && isInt(x.start, 0, N - 1) && isInt(x.step, 0, 24))
    .sort((a, b) => a.start - b.start)
    .filter((x) => !seen.has(x.start) && seen.add(x.start))
    .map((x) => ({ start: x.start, step: x.step }));
  if (!s.segments.length || s.segments[0].start !== 0) s.segments.unshift({ start: 0, step: s.segments[0]?.step ?? 2 });
  const ov = {};
  for (const [k, v] of Object.entries(s.overrides || {})) if (isInt(+k, 0, N - 1) && (v === "key" || v === "hold")) ov[+k] = v;
  s.overrides = ov;
  s.cutAdd = (s.cutAdd || []).filter((c) => isInt(c, 1, N - 1));
  s.cutRemove = (s.cutRemove || []).filter((c) => isInt(c, 1, N - 1));
  s.threshold = Number.isFinite(+s.threshold) ? +s.threshold : d.threshold;
  s.minShot = isInt(+s.minShot, 1, 96) ? +s.minShot : d.minShot;
  s.resetAtCuts = s.resetAtCuts !== false;
  return s;
}

function derive() {
  const auto = detectCuts(media.scores, S.threshold, S.minShot);
  const set = new Set(auto);
  S.cutAdd.forEach((c) => set.add(c));
  S.cutRemove.forEach((c) => set.delete(c));
  const cuts = [...set].filter((c) => c > 0 && c < N).sort((a, b) => a - b);
  const cutSet = new Set(cuts);

  const map = new Int32Array(N), isKey = new Uint8Array(N), drawNo = new Int32Array(N);
  const shotOf = new Int32Array(N), segOf = new Int32Array(N);
  const segs = S.segments;
  let si = 0, anchor = 0, last = 0, draws = 0, shot = 0;
  for (let i = 0; i < N; i++) {
    while (si + 1 < segs.length && segs[si + 1].start <= i) si++;
    if (cutSet.has(i)) shot++;
    const seg = segs[si];
    const ov = S.overrides[i];
    let key;
    if (i === 0) { key = true; anchor = 0; }
    else if (ov === "hold") key = false;
    else if (ov === "key" || i === seg.start || (S.resetAtCuts && cutSet.has(i))) { key = true; anchor = i; }
    else if (seg.step === 0) key = false;
    else key = (i - anchor) % seg.step === 0;
    if (key) { last = i; draws++; }
    map[i] = last; isKey[i] = key ? 1 : 0; drawNo[i] = draws; shotOf[i] = shot; segOf[i] = si;
  }
  D = { auto: new Set(auto), cuts, cutSet, map, isKey, drawNo, shotOf, segOf, shots: [0, ...cuts, N], draws };
}

const segEnd = (i) => (S.segments[i + 1]?.start ?? N) - 1;
const shotRange = (f) => [D.shots[D.shotOf[f]], D.shots[D.shotOf[f] + 1] - 1];

function target() {
  if (view.sel) return { a: view.sel.a, b: view.sel.b, label: "선택 구간" };
  const [a, b] = shotRange(view.cur);
  return { a, b, label: `현재 샷 ${D.shotOf[view.cur] + 1}` };
}

// ---------- history ----------

let saveTimer;
function afterChange() {
  derive();
  clearTimeout(saveTimer);
  saveTimer = setTimeout(() => store(`ds-timing-${media.id}`, S), 300);
  renderInspector();
  schedule("viewer", "timeline");
}

function commit(fn) {
  hist.undo.push(JSON.stringify(S));
  if (hist.undo.length > 300) hist.undo.shift();
  hist.redo = [];
  fn();
  afterChange();
}

function undo() {
  if (!hist.undo.length) return;
  hist.redo.push(JSON.stringify(S));
  S = JSON.parse(hist.undo.pop());
  afterChange();
}

function redo() {
  if (!hist.redo.length) return;
  hist.undo.push(JSON.stringify(S));
  S = JSON.parse(hist.redo.pop());
  afterChange();
}

// ---------- edits ----------

function splitAt(f) {
  if (f <= 0 || f >= N) return;
  const segs = S.segments;
  let i = segs.findIndex((s) => s.start >= f);
  if (i === -1) i = segs.length;
  if (segs[i]?.start === f) return;
  segs.splice(i, 0, { start: f, step: segs[i - 1].step });
}

function setStep(step) {
  if (!Number.isInteger(step) || step < 0 || step > 24) return toast("스텝은 0~24 사이 정수입니다.");
  const { a, b, label } = target();
  commit(() => {
    splitAt(a);
    splitAt(b + 1);
    for (const s of S.segments) if (s.start >= a && s.start <= b) s.step = step;
  });
  toast(`${label} F${a}–${b} → ${stepLabel(step)}`);
}

function setOverride(f, v) {
  commit(() => {
    if (v) S.overrides[f] = v;
    else delete S.overrides[f];
  });
}

function cycleOverride(f) {
  if (f === 0) return;
  const cur = S.overrides[f];
  setOverride(f, cur ? null : D.isKey[f] ? "hold" : "key");
}

function clearOverrides() {
  const { a, b } = target();
  commit(() => {
    for (const k of Object.keys(S.overrides)) if (+k >= a && +k <= b) delete S.overrides[k];
  });
}

function toggleCut(f) {
  if (f <= 0 || f >= N) return;
  commit(() => {
    const add = new Set(S.cutAdd), rem = new Set(S.cutRemove);
    if (D.cutSet.has(f)) add.has(f) ? add.delete(f) : rem.add(f);
    else rem.has(f) ? rem.delete(f) : add.add(f);
    S.cutAdd = [...add];
    S.cutRemove = [...rem];
  });
}

// ---------- navigation & playback ----------

function setCur(f) {
  view.cur = Math.max(0, Math.min(N - 1, f));
  if (view.follow) ensureVisible(view.cur);
  renderCurInfo();
  schedule("viewer", "timeline");
}

function ensureVisible(f) {
  const w = tlScroll.clientWidth, cw = view.cellW, x = f * cw;
  if (x < tlScroll.scrollLeft || x + cw > tlScroll.scrollLeft + w) tlScroll.scrollLeft = Math.max(0, x - w * 0.3);
}

const prevKey = () => { for (let i = view.cur - 1; i >= 0; i--) if (D.isKey[i]) return setCur(i); };
const nextKey = () => { for (let i = view.cur + 1; i < N; i++) if (D.isKey[i]) return setCur(i); };
const prevCut = () => setCur([...D.shots].reverse().find((c) => c < view.cur) ?? 0);
const nextCut = () => setCur(D.shots.find((c) => c > view.cur && c < N) ?? N - 1);

let playT0 = 0, playF0 = 0;
function play() {
  if (view.playing) return stop();
  const [a, b] = playRange();
  if (view.cur >= b || view.cur < a) view.cur = a;
  view.playing = true;
  playT0 = performance.now();
  playF0 = view.cur;
  $("#tPlay").textContent = "⏸";
  requestAnimationFrame(tick);
}

function stop() {
  view.playing = false;
  $("#tPlay").textContent = "▶";
}

const playRange = () => (view.loop && view.sel ? [view.sel.a, view.sel.b] : [0, N - 1]);

function tick(now) {
  if (!view.playing) return;
  const [a, b] = playRange();
  let f = playF0 + Math.floor(((now - playT0) / 1000) * fps * view.speed);
  if (f > b) {
    if (!view.loop) { setCur(b); return stop(); }
    playF0 = a; playT0 = now; f = a;
  }
  if (f !== view.cur) setCur(f);
  requestAnimationFrame(tick);
}

// ---------- images ----------

function loadPreviews() {
  let next = 0, done = 0;
  const label = $("#loadStatus");
  const one = () => {
    if (next >= N) return;
    const i = next++;
    const img = new Image();
    img.onload = () => { previews[i] = img; finish(i); };
    img.onerror = () => finish(i);
    img.src = `/api/media/${media.id}/preview/${i}.jpg`;
  };
  const finish = (i) => {
    done++;
    label.textContent = done < N ? `미리보기 로딩 ${done}/${N}` : "";
    if (i === view.cur || i === D.map[view.cur]) schedule("viewer");
    one();
  };
  for (let k = 0; k < 8; k++) one();
}

function thumb(i) {
  if (!thumbs[i]) {
    const img = new Image();
    img.onload = () => schedule("timeline");
    img.src = `/api/media/${media.id}/thumb/${i}.jpg`;
    thumbs[i] = img;
  }
  const t = thumbs[i];
  return t.complete && t.naturalWidth ? t : previews[i] || null;
}

// ---------- drawing ----------

const dirty = new Set();
let rafPending = false;
function schedule(...kinds) {
  kinds.forEach((k) => dirty.add(k));
  if (rafPending) return;
  rafPending = true;
  requestAnimationFrame(() => {
    rafPending = false;
    if (dirty.has("viewer")) drawViewer();
    if (dirty.has("timeline")) drawTimeline();
    dirty.clear();
  });
}

function fitCanvas(canvas, wCss, hCss) {
  const dpr = window.devicePixelRatio || 1;
  const w = Math.round(wCss * dpr), h = Math.round(hCss * dpr);
  if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return ctx;
}

let imgRect = { x: 0, y: 0, w: 0, h: 0 };
function drawViewer() {
  const box = $("#viewer").getBoundingClientRect();
  const ctx = fitCanvas(viewCanvas, box.width, box.height);
  ctx.fillStyle = "#0b0c0e";
  ctx.fillRect(0, 0, box.width, box.height);
  const ar = media.width / media.height;
  let w = box.width - 16, h = w / ar;
  if (h > box.height - 16) { h = box.height - 16; w = h * ar; }
  const x = (box.width - w) / 2, y = (box.height - h) / 2;
  imgRect = { x, y, w, h };
  const draw = (idx, alpha = 1) => {
    const img = previews[idx] || thumb(idx);
    if (!img) return;
    ctx.globalAlpha = alpha;
    ctx.drawImage(img, x, y, w, h);
    ctx.globalAlpha = 1;
  };
  const f = view.cur, shown = D.map[f];
  if (view.mode === "orig") {
    draw(f);
  } else {
    draw(shown);
    if (view.onion && shown > 0) draw(D.map[shown - 1], 0.35);
    if (view.mode === "compare") {
      const wx = x + w * view.wipe;
      ctx.save();
      ctx.beginPath();
      ctx.rect(x, y, wx - x, h);
      ctx.clip();
      draw(f);
      ctx.restore();
      ctx.fillStyle = "#5aa9ff";
      ctx.fillRect(wx - 1, y, 2, h);
      ctx.font = "12px sans-serif";
      ctx.fillStyle = "rgba(10,12,14,.75)";
      ctx.fillRect(x + 8, y + h - 26, 38, 18);
      ctx.fillRect(x + w - 46, y + h - 26, 38, 18);
      ctx.fillStyle = "#e6e9ee";
      ctx.fillText("원본", x + 14, y + h - 13);
      ctx.fillText("스텝", x + w - 40, y + h - 13);
    }
  }
}

function drawCover(ctx, img, x, y, w, h) {
  const iar = img.naturalWidth / img.naturalHeight, car = w / h;
  let sw = img.naturalWidth, sh = img.naturalHeight, sx = 0, sy = 0;
  if (iar > car) { sw = sh * car; sx = (img.naturalWidth - sw) / 2; }
  else { sh = sw / car; sy = (img.naturalHeight - sh) / 2; }
  ctx.drawImage(img, sx, sy, sw, sh, x, y, w, h);
}

function drawTimeline() {
  const W = tlCanvas.parentElement.clientWidth;
  tlCanvas.style.height = `${TL_H}px`;
  const ctx = fitCanvas(tlCanvas, W, TL_H);
  const cw = view.cellW, sl = tlScroll.scrollLeft;
  const f0 = Math.max(0, Math.floor(sl / cw)), f1 = Math.min(N - 1, Math.floor((sl + W) / cw));
  const X = (f) => f * cw - sl;

  ctx.fillStyle = "#16191d"; ctx.fillRect(0, 0, W, TL_H);
  ctx.fillStyle = "#1b1f24"; ctx.fillRect(0, ROW_Y.cuts, W, ROW_H.cuts);
  ctx.fillStyle = "#121417"; ctx.fillRect(0, ROW_Y.frames, W, ROW_H.frames);
  ctx.strokeStyle = "#2c3139"; ctx.lineWidth = 1;
  for (const y of [ROW_Y.cuts, ROW_Y.seg, ROW_Y.frames]) { ctx.beginPath(); ctx.moveTo(0, y + 0.5); ctx.lineTo(W, y + 0.5); ctx.stroke(); }

  // ruler
  const every = [1, 2, 5, 10, 12, 24, 48, 96, 120, 240, 480, 960, 2400].find((s) => s * cw >= 46) || 4800;
  const sec = Math.round(fps);
  ctx.font = "10px Consolas, monospace";
  ctx.textBaseline = "top";
  for (let f = f0; f <= f1; f++) {
    const x = Math.round(X(f)) + 0.5;
    const major = f % every === 0;
    if (major || cw >= 5 || f % sec === 0) {
      ctx.strokeStyle = major ? "#8b939f" : f % sec === 0 ? "#5b6370" : "#353b44";
      ctx.beginPath(); ctx.moveTo(x, major ? 10 : 15); ctx.lineTo(x, 20); ctx.stroke();
    }
    if (major) { ctx.fillStyle = "#8b939f"; ctx.fillText(String(f), x + 3, 2); }
  }

  // scene scores + threshold
  const T = S.threshold, barMax = ROW_H.cuts - 6, base = ROW_Y.cuts + ROW_H.cuts - 3;
  for (let f = f0; f <= f1; f++) {
    const sc = media.scores[f];
    const h = Math.min(1, sc / (2 * T)) * barMax;
    if (h < 0.5) continue;
    ctx.fillStyle = sc >= T ? "#a8524f" : "#3d4654";
    ctx.fillRect(X(f) + (cw > 3 ? 1 : 0), base - h, Math.max(1, cw - (cw > 3 ? 2 : 0)), h);
  }
  ctx.strokeStyle = "rgba(239,100,97,.45)"; ctx.setLineDash([3, 3]);
  ctx.beginPath(); ctx.moveTo(0, base - barMax / 2 + 0.5); ctx.lineTo(W, base - barMax / 2 + 0.5); ctx.stroke();
  ctx.setLineDash([]);

  // segments
  ctx.textBaseline = "middle";
  ctx.font = "bold 11px sans-serif";
  S.segments.forEach((seg, i) => {
    const s = seg.start, e = segEnd(i);
    if (e < f0 || s > f1) return;
    const x1 = X(s), x2 = X(e + 1);
    ctx.fillStyle = stepColor(seg.step);
    ctx.globalAlpha = 0.8;
    ctx.fillRect(x1, ROW_Y.seg + 2, x2 - x1, ROW_H.seg - 4);
    ctx.globalAlpha = 1;
    ctx.fillStyle = "#16191d";
    ctx.fillRect(x1, ROW_Y.seg + 2, 2, ROW_H.seg - 4);
    if (x2 - x1 > 26) {
      ctx.fillStyle = "#0e1013";
      ctx.fillText(stepLabel(seg.step), Math.max(x1, 0) + 6, ROW_Y.seg + ROW_H.seg / 2);
    }
  });

  // frames
  const fy = ROW_Y.frames, labelH = 13, thumbY = fy + labelH, thumbH = ROW_H.frames - labelH - 2;
  ctx.font = "10px Consolas, monospace";
  for (let f = f0; f <= f1; f++) {
    const x = X(f), key = D.isKey[f], src = D.map[f];
    if (cw >= 12) {
      const img = thumb(src);
      if (img) drawCover(ctx, img, x + 0.5, thumbY, cw - 1, thumbH);
      else { ctx.fillStyle = "#2a2f37"; ctx.fillRect(x + 0.5, thumbY, cw - 1, thumbH); }
      if (!key) { ctx.fillStyle = "rgba(10,12,14,.62)"; ctx.fillRect(x + 0.5, thumbY, cw - 1, thumbH); }
    } else {
      ctx.fillStyle = key ? "#aeb7c3" : "#323842";
      ctx.fillRect(x, thumbY, Math.max(1, cw - (cw > 3 ? 1 : 0)), thumbH);
    }
    if (key) {
      ctx.fillStyle = stepColor(S.segments[D.segOf[f]].step);
      ctx.fillRect(x + 0.5, fy + 1, Math.max(1, cw - 1), 3);
      if (cw >= 20) { ctx.fillStyle = "#e6e9ee"; ctx.textBaseline = "top"; ctx.fillText(String(D.drawNo[f]), x + 3, fy + 3); }
    } else if (cw >= 6) {
      ctx.strokeStyle = "#9aa3af";
      ctx.beginPath(); ctx.moveTo(x, thumbY + thumbH / 2 + 0.5); ctx.lineTo(x + cw, thumbY + thumbH / 2 + 0.5); ctx.stroke();
    }
    const ov = S.overrides[f];
    if (ov && cw >= 8) {
      ctx.fillStyle = ov === "key" ? "#f0b44c" : "#3cc6c6";
      ctx.fillRect(x + 1, fy + ROW_H.frames - 14, 11, 12);
      ctx.fillStyle = "#0e1013"; ctx.textBaseline = "top";
      ctx.fillText(ov === "key" ? "K" : "H", x + 3.5, fy + ROW_H.frames - 13);
    } else if (ov) {
      ctx.fillStyle = ov === "key" ? "#f0b44c" : "#3cc6c6";
      ctx.fillRect(x, fy + ROW_H.frames - 4, Math.max(2, cw), 3);
    }
    if (cw >= 8) { ctx.fillStyle = "#0e1013"; ctx.fillRect(x + cw - 0.5, thumbY, 1, thumbH); }
  }

  // cuts (active), and auto cuts the user removed
  for (const c of S.cutRemove) {
    if (c < f0 || c > f1 + 1) continue;
    ctx.strokeStyle = "rgba(139,147,159,.6)"; ctx.setLineDash([2, 3]);
    ctx.beginPath(); ctx.moveTo(Math.round(X(c)) + 0.5, ROW_Y.cuts); ctx.lineTo(Math.round(X(c)) + 0.5, ROW_Y.seg); ctx.stroke();
    ctx.setLineDash([]);
  }
  for (const c of D.cuts) {
    if (c < f0 || c > f1 + 1) continue;
    const x = Math.round(X(c)) + 0.5;
    ctx.strokeStyle = "#ef6461"; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(x, ROW_Y.cuts); ctx.lineTo(x, TL_H); ctx.stroke();
    ctx.lineWidth = 1;
    ctx.fillStyle = D.auto.has(c) ? "#ef6461" : "#ffb3b1";
    ctx.beginPath(); ctx.moveTo(x - 5, ROW_Y.cuts + 1); ctx.lineTo(x + 5, ROW_Y.cuts + 1); ctx.lineTo(x, ROW_Y.cuts + 8); ctx.fill();
  }

  // selection
  if (view.sel) {
    const x1 = X(view.sel.a), x2 = X(view.sel.b + 1);
    ctx.fillStyle = "rgba(90,169,255,.16)";
    ctx.fillRect(x1, 0, x2 - x1, TL_H);
    ctx.strokeStyle = "#5aa9ff";
    ctx.strokeRect(x1 + 0.5, 0.5, x2 - x1 - 1, TL_H - 1);
  }

  // playhead
  const px = X(view.cur);
  ctx.strokeStyle = "#ffffff"; ctx.lineWidth = 1.5;
  ctx.strokeRect(px + 0.5, fy + 0.5, Math.max(2, cw - 1), ROW_H.frames - 1);
  ctx.lineWidth = 1;
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(px + cw / 2 - 0.5, 0, 1, TL_H);
  ctx.beginPath(); ctx.moveTo(px + cw / 2 - 5, 0); ctx.lineTo(px + cw / 2 + 5, 0); ctx.lineTo(px + cw / 2, 7); ctx.fill();
}

// ---------- inspector ----------

function renderCurInfo() {
  const f = view.cur, key = D.isKey[f], shown = D.map[f];
  const seg = S.segments[D.segOf[f]];
  const ov = S.overrides[f];
  const [sa, sb] = shotRange(f);
  $("#curInfo").replaceChildren(
    el("dt", {}, "프레임"), el("dd", { class: "mono" }, `${f} / ${N - 1}`),
    el("dt", {}, "타임코드"), el("dd", { class: "mono" }, fmtTC(f, fps)),
    el("dt", {}, "샷"), el("dd", {}, `${D.shotOf[f] + 1} / ${D.shots.length - 1}  (F${sa}–${sb})`),
    el("dt", {}, "상태"), el("dd", {}, key ? `새 그림 #${D.drawNo[f]}` : `홀드 — 그림 #${D.drawNo[f]} 유지`),
    el("dt", {}, "표시 원본"), el("dd", { class: "mono" }, `F${shown}${shown !== f ? `  (${f - shown}프레임 전)` : ""}`),
    el("dt", {}, "구간 스텝"), el("dd", {}, stepLabel(seg.step)),
    el("dt", {}, "오버라이드"), el("dd", {}, ov === "key" ? "강제 키" : ov === "hold" ? "강제 홀드" : "자동"),
  );
  for (const b of $("#ovCtrl").children) b.classList.toggle("on", (b.dataset.ov || undefined) === ov);
  $("#hud").replaceChildren(...[
    el("span", {}, `F ${f}`),
    el("span", {}, fmtTC(f, fps)),
    el("span", {}, `샷 ${D.shotOf[f] + 1}`),
    key ? el("span", { class: "key" }, `KEY #${D.drawNo[f]}`) : el("span", { class: "hold" }, `HOLD ← F${shown}`),
    ov ? el("span", { class: "ov" }, ov === "key" ? "강제 키" : "강제 홀드") : null,
    view.mode === "orig" ? el("span", {}, "원본 보기") : null,
  ].filter(Boolean));
  $("#tFrame").value = f;
  $("#tTC").textContent = fmtTC(f, fps);
  for (const item of $("#segList").children) item.classList.toggle("active", +item.dataset.i === D.segOf[f]);
  if (!view.sel) renderTarget();
}

function renderTarget() {
  const { a, b, label } = target();
  const steps = new Set();
  S.segments.forEach((s, i) => { if (s.start <= b && segEnd(i) >= a) steps.add(s.step); });
  $("#targetInfo").replaceChildren("대상: ", el("b", {}, label), ` F${a}–${b} (${b - a + 1}프레임)`,
    steps.size > 1 ? " · 여러 스텝 섞임" : "");
  for (const btn of $("#stepBtns").children) btn.classList.toggle("on", steps.size === 1 && steps.has(+btn.dataset.step));
}

function renderInspector() {
  renderCurInfo();
  renderTarget();
  $("#cutThreshold").value = S.threshold;
  $("#cutThresholdVal").textContent = S.threshold;
  $("#minShot").value = S.minShot;
  $("#resetAtCuts").checked = S.resetAtCuts;
  $("#undoBtn").disabled = !hist.undo.length;
  $("#redoBtn").disabled = !hist.redo.length;

  const cutItems = D.cuts.map((c) =>
    el("div", { class: "item" },
      el("i", { class: "swatch", style: `background:${D.auto.has(c) ? "#ef6461" : "#ffb3b1"}` }),
      el("span", { class: "range", onclick: () => setCur(c) }, `F${c}  ${fmtTC(c, fps)}`),
      el("span", { class: "note" }, D.auto.has(c) ? "자동" : "수동"),
      el("button", { title: "컷 삭제", onclick: () => toggleCut(c) }, "✕")));
  const removed = S.cutRemove.filter((c) => D.auto.has(c)).sort((a, b) => a - b).map((c) =>
    el("div", { class: "item" },
      el("i", { class: "swatch", style: "background:#3a414b" }),
      el("span", { class: "range muted", onclick: () => setCur(c) }, `F${c}  제외됨`),
      el("button", { onclick: () => toggleCut(c) }, "복원")));
  $("#cutList").replaceChildren(...(cutItems.length || removed.length ? [...cutItems, ...removed] : [el("div", { class: "note" }, "컷 없음 — 전체가 하나의 샷입니다.")]));

  const options = [1, 2, 3, 4, 5, 6, 8, 0];
  $("#segList").replaceChildren(...S.segments.map((seg, i) => {
    const e = segEnd(i);
    const sel = el("select", {
      title: "이 구간의 스텝",
      onchange: (ev) => commit(() => (S.segments[i].step = +ev.target.value)),
    }, [...new Set([...options, seg.step])].map((s) => el("option", { value: s, selected: s === seg.step }, stepLabel(s))));
    return el("div", { class: `item${D.segOf[view.cur] === i ? " active" : ""}`, "data-i": i },
      el("i", { class: "swatch", style: `background:${stepColor(seg.step)}` }),
      el("span", {
        class: "range",
        title: "클릭: 이 구간 선택",
        onclick: () => { view.sel = { a: seg.start, b: e }; setCur(seg.start); renderTarget(); schedule("timeline"); },
      }, `F${seg.start}–${e} (${e - seg.start + 1})`),
      sel,
      i > 0 ? el("button", { title: "경계를 지우고 앞 구간에 합치기", onclick: () => commit(() => S.segments.splice(i, 1)) }, "합치기") : null);
  }));

  const keyed = D.draws;
  $("#stats").replaceChildren(
    stat(N, "전체 프레임"),
    stat(keyed, "그림 수 (키)"),
    stat((N / keyed).toFixed(2), "평균 프레임/그림"),
    stat(D.shots.length - 1, "샷 수"),
  );
}

const stat = (v, k) => el("div", { class: "stat" }, el("div", { class: "v" }, String(v)), el("div", { class: "k" }, k));

// ---------- timeline input ----------

function frameAt(clientX) {
  const r = tlCanvas.getBoundingClientRect();
  return Math.max(0, Math.min(N - 1, Math.floor((clientX - r.left + tlScroll.scrollLeft) / view.cellW)));
}

function rowAt(clientY) {
  const y = clientY - tlCanvas.getBoundingClientRect().top;
  return y < ROW_Y.cuts ? "ruler" : y < ROW_Y.seg ? "cuts" : y < ROW_Y.frames ? "seg" : "frames";
}

let drag = null;
tlCanvas.addEventListener("pointerdown", (e) => {
  const f = frameAt(e.clientX), row = rowAt(e.clientY);
  if (row === "cuts") return toggleCut(f === 0 ? 1 : f);
  if (row === "seg") {
    const i = D.segOf[f];
    view.sel = { a: S.segments[i].start, b: segEnd(i) };
    setCur(f);
    renderTarget();
    return;
  }
  if (e.shiftKey) {
    const anchor = view.sel ? (f < view.sel.a ? view.sel.b : view.sel.a) : view.cur;
    view.sel = { a: Math.min(anchor, f), b: Math.max(anchor, f) };
    setCur(f);
    renderTarget();
    return;
  }
  drag = { anchor: f };
  view.sel = null;
  tlCanvas.setPointerCapture(e.pointerId);
  setCur(f);
});

tlCanvas.addEventListener("pointermove", (e) => {
  if (!drag) return;
  const r = tlCanvas.getBoundingClientRect();
  if (e.clientX > r.right - 24) tlScroll.scrollLeft += view.cellW;
  if (e.clientX < r.left + 24) tlScroll.scrollLeft -= view.cellW;
  const f = frameAt(e.clientX);
  view.sel = f === drag.anchor ? null : { a: Math.min(drag.anchor, f), b: Math.max(drag.anchor, f) };
  setCur(f);
});

tlCanvas.addEventListener("pointerup", () => {
  if (!drag) return;
  drag = null;
  renderTarget();
});

tlCanvas.addEventListener("dblclick", (e) => {
  if (rowAt(e.clientY) === "frames") cycleOverride(frameAt(e.clientX));
});

tlCanvas.addEventListener("wheel", (e) => {
  e.preventDefault();
  if (e.ctrlKey) {
    const r = tlCanvas.getBoundingClientRect();
    setZoom(view.cellW * (e.deltaY < 0 ? 1.2 : 1 / 1.2), e.clientX - r.left);
  } else {
    tlScroll.scrollLeft += e.deltaY || e.deltaX;
  }
}, { passive: false });

tlScroll.addEventListener("scroll", () => schedule("timeline"));

function setZoom(cw, anchorX = tlScroll.clientWidth / 2) {
  cw = Math.max(3, Math.min(96, Math.round(cw)));
  const frame = (anchorX + tlScroll.scrollLeft) / view.cellW;
  view.cellW = cw;
  $("#zoom").value = cw;
  $("#tlSpacer").style.width = `${N * cw + 40}px`;
  tlScroll.scrollLeft = frame * cw - anchorX;
  schedule("timeline");
}

// viewer: compare wipe drag, click to play
let wiping = false;
viewCanvas.addEventListener("pointerdown", (e) => {
  if (view.mode !== "compare") return play();
  wiping = true;
  viewCanvas.setPointerCapture(e.pointerId);
  moveWipe(e);
});
viewCanvas.addEventListener("pointermove", (e) => wiping && moveWipe(e));
viewCanvas.addEventListener("pointerup", () => (wiping = false));
function moveWipe(e) {
  const r = viewCanvas.getBoundingClientRect();
  view.wipe = Math.max(0, Math.min(1, (e.clientX - r.left - imgRect.x) / imgRect.w));
  schedule("viewer");
}

function setMode(mode) {
  view.mode = mode;
  for (const b of $("#modeCtrl").children) b.classList.toggle("on", b.dataset.mode === mode);
  renderCurInfo();
  schedule("viewer");
}

// ---------- files ----------

function download(name, text, type) {
  const a = el("a", { href: URL.createObjectURL(new Blob([text], { type })), download: name });
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 2000);
}

const stem = () => media.name.replace(/\.[^.]+$/, "");

function exportJSON() {
  download(`${stem()}_timing.json`, JSON.stringify({
    app: "depth-studio", version: 1, source: media.name, frames: N, fps: media.fps_str,
    spec: S, cuts: D.cuts, map: Array.from(D.map),
  }, null, 1), "application/json");
}

function exportCSV() {
  const rows = ["frame,timecode,shot,drawing,source_frame,type,step,override"];
  for (let f = 0; f < N; f++) {
    rows.push([f, fmtTC(f, fps), D.shotOf[f] + 1, D.drawNo[f], D.map[f], D.isKey[f] ? "key" : "hold",
      stepLabel(S.segments[D.segOf[f]].step), S.overrides[f] || ""].join(","));
  }
  download(`${stem()}_xsheet.csv`, "﻿" + rows.join("\n"), "text/csv");
}

async function importJSON(file) {
  try {
    const obj = JSON.parse(await file.text());
    if (obj.frames && obj.frames !== N) toast(`프레임 수가 다릅니다 (${obj.frames} → ${N}). 범위를 벗어난 값은 버립니다.`, 4000);
    commit(() => (S = sanitizeSpec(obj.spec || obj)));
    toast("타이밍을 불러왔습니다.");
  } catch (e) {
    toast(`불러오기 실패: ${e.message}`, 4000);
  }
}

// ---------- render job ----------

async function startRender() {
  const outputs = { stepped_mp4: $("#outStepped").checked, depth_mp4: $("#outDepth").checked, depth_png: $("#outPng").checked, audio: $("#outAudio").checked };
  if (!outputs.stepped_mp4 && !outputs.depth_mp4 && !outputs.depth_png) return toast("출력을 하나 이상 선택하세요.");
  $("#renderBtn").disabled = true;
  try {
    const job = await postJSON("/api/jobs", {
      media_id: media.id, outputs, depth: depthForm.get(), map: Array.from(D.map), cuts: D.cuts, timing: S,
    });
    recentJobs(job.id);
    $("#jobs").prepend(jobCard(job.id));
    toast("렌더 대기열에 추가했습니다.");
  } catch (e) {
    toast(e.message, 4000);
  } finally {
    $("#renderBtn").disabled = false;
  }
}

// ---------- keyboard ----------

document.addEventListener("keydown", (e) => {
  if (!D || !$("#helpModal").hidden && e.key !== "Escape" && e.key !== "?") return;
  const t = e.target;
  if (t.matches?.("input:not([type=checkbox]):not([type=range]), select, textarea")) return;
  const k = e.key, ctrl = e.ctrlKey || e.metaKey;
  const handled = () => e.preventDefault();

  if (ctrl && k.toLowerCase() === "z") return handled(), e.shiftKey ? redo() : undo();
  if (ctrl && k.toLowerCase() === "y") return handled(), redo();
  if (ctrl && k.toLowerCase() === "a") { handled(); view.sel = { a: 0, b: N - 1 }; renderTarget(); return schedule("timeline"); }
  if (ctrl) return;

  switch (k) {
    case " ": handled(); return play();
    case "ArrowLeft": handled(); return setCur(view.cur - (e.shiftKey ? 10 : 1));
    case "ArrowRight": handled(); return setCur(view.cur + (e.shiftKey ? 10 : 1));
    case "ArrowUp": handled(); return prevCut();
    case "ArrowDown": handled(); return nextCut();
    case "[": return prevKey();
    case "]": return nextKey();
    case "Home": handled(); return setCur(0);
    case "End": handled(); return setCur(N - 1);
    case "Escape":
      if (!$("#helpModal").hidden) return ($("#helpModal").hidden = true);
      view.sel = null; renderTarget(); return schedule("timeline");
    case "?": $("#helpModal").hidden = !$("#helpModal").hidden; return;
    case "+": case "=": return setZoom(view.cellW * 1.25);
    case "-": case "_": return setZoom(view.cellW / 1.25);
  }
  const lower = k.toLowerCase();
  if (/^[0-8]$/.test(k)) return setStep(+k);
  if (lower === "k") return view.cur > 0 && setOverride(view.cur, "key");
  if (lower === "h") return view.cur > 0 && setOverride(view.cur, "hold");
  if (lower === "a") return setOverride(view.cur, null);
  if (lower === "c") return toggleCut(view.cur);
  if (lower === "i" || lower === "o") {
    const s = view.sel || { a: view.cur, b: view.cur };
    view.sel = lower === "i" ? { a: view.cur, b: Math.max(view.cur, s.b) } : { a: Math.min(view.cur, s.a), b: view.cur };
    renderTarget(); return schedule("timeline");
  }
  if (lower === "s") { const [a, b] = shotRange(view.cur); view.sel = { a, b }; renderTarget(); return schedule("timeline"); }
  if (lower === "l") { view.loop = !view.loop; $("#tLoop").classList.toggle("on", view.loop); return; }
  if (lower === "v") return setMode({ step: "orig", orig: "compare", compare: "step" }[view.mode]);
  if (lower === "n") { $("#onion").checked = view.onion = !view.onion; return schedule("viewer"); }
});

// ---------- setup ----------

function bindUI() {
  $("#undoBtn").onclick = undo;
  $("#redoBtn").onclick = redo;
  $("#exportBtn").onclick = exportJSON;
  $("#csvBtn").onclick = exportCSV;
  $("#importBtn").onclick = () => $("#importFile").click();
  $("#importFile").onchange = (e) => { if (e.target.files[0]) importJSON(e.target.files[0]); e.target.value = ""; };
  $("#resetBtn").onclick = () => confirm("타이밍 편집을 모두 초기화할까요? (실행 취소로 되돌릴 수 있습니다)") && commit(() => (S = defaultSpec()));
  $("#helpBtn").onclick = () => ($("#helpModal").hidden = false);
  $("#helpClose").onclick = () => ($("#helpModal").hidden = true);

  for (const b of $("#modeCtrl").children) b.onclick = () => setMode(b.dataset.mode);
  $("#onion").onchange = (e) => { view.onion = e.target.checked; schedule("viewer"); };
  for (const b of $("#ovCtrl").children) b.onclick = () => view.cur > 0 && setOverride(view.cur, b.dataset.ov || null);
  for (const b of $("#stepBtns").children) b.onclick = () => setStep(+b.dataset.step);
  $("#customStepBtn").onclick = () => setStep(parseInt($("#customStep").value, 10));
  $("#selShotBtn").onclick = () => { const [a, b] = shotRange(view.cur); view.sel = { a, b }; renderTarget(); schedule("timeline"); };
  $("#selAllBtn").onclick = () => { view.sel = { a: 0, b: N - 1 }; renderTarget(); schedule("timeline"); };
  $("#selClearBtn").onclick = () => { view.sel = null; renderTarget(); schedule("timeline"); };
  $("#clearOvBtn").onclick = clearOverrides;
  $("#mergeBtn").onclick = () => commit(() => (S.segments = S.segments.filter((s, i, arr) => i === 0 || s.step !== arr[i - 1].step)));

  // Slider edits update live; one undo step per drag.
  let pre = null;
  const live = (fn) => (e) => { if (pre === null) pre = JSON.stringify(S); fn(e); afterChange(); };
  const settle = () => { if (pre !== null) { hist.undo.push(pre); hist.redo = []; pre = null; renderInspector(); } };
  $("#cutThreshold").addEventListener("input", live((e) => (S.threshold = +e.target.value)));
  $("#cutThreshold").addEventListener("change", settle);
  $("#minShot").addEventListener("change", (e) => {
    const v = Math.max(1, Math.min(96, parseInt(e.target.value, 10) || 1));
    commit(() => (S.minShot = v));
  });
  $("#resetAtCuts").onchange = (e) => commit(() => (S.resetAtCuts = e.target.checked));
  $("#resetCutsBtn").onclick = () => commit(() => { S.cutAdd = []; S.cutRemove = []; });

  $("#tFirst").onclick = () => setCur(0);
  $("#tLast").onclick = () => setCur(N - 1);
  $("#tPrev").onclick = () => setCur(view.cur - 1);
  $("#tNext").onclick = () => setCur(view.cur + 1);
  $("#tPrevKey").onclick = prevKey;
  $("#tNextKey").onclick = nextKey;
  $("#tPlay").onclick = play;
  $("#tLoop").onclick = () => { view.loop = !view.loop; $("#tLoop").classList.toggle("on", view.loop); };
  $("#tSpeed").onchange = (e) => { view.speed = +e.target.value; if (view.playing) { playT0 = performance.now(); playF0 = view.cur; } };
  $("#tFrame").max = N - 1;
  $("#tFrame").addEventListener("keydown", (e) => { if (e.key === "Enter") { setCur(parseInt(e.target.value, 10) || 0); e.target.blur(); } });

  $("#zoom").oninput = (e) => setZoom(+e.target.value);
  $("#follow").onchange = (e) => (view.follow = e.target.checked);
  $("#renderBtn").onclick = startRender;

  new ResizeObserver(() => schedule("viewer")).observe($("#viewer"));
  new ResizeObserver(() => schedule("timeline")).observe(tlCanvas.parentElement);
}

function showUpload(message) {
  $("#uploadView").hidden = false;
  const box = $("#editorUploadBox");
  if (message) box.prepend(el("p", { class: "job-err" }, message));
  makeDropzone($("#editorDropzone"), async (file) => {
    const holder = el("div", { class: "stack" });
    box.replaceChildren(holder);
    try {
      const m = await uploadAndPrepare(file, holder);
      location.search = `?m=${m.id}`;
    } catch (e) {
      holder.replaceChildren(el("div", { class: "job-err" }, e.message), el("a", { class: "btn", href: "/editor" }, "다시 시도"));
    }
  });
}

async function init() {
  status = await api("/api/status");
  const id = new URLSearchParams(location.search).get("m");
  if (!id) return showUpload();
  try {
    media = await waitMedia(id);
  } catch (e) {
    return showUpload(e.message);
  }
  if (!media.editor) return showUpload(`이 영상은 편집기를 쓸 수 없습니다 (${media.frames}프레임 — 최대 ${media.max_editor_frames}프레임).`);

  N = media.frames;
  fps = media.fps;
  const saved = store(`ds-timing-${id}`);
  S = sanitizeSpec(saved);
  derive();

  document.title = `${media.name} — Frame Timing Editor`;
  $("#fileName").textContent = media.name;
  $("#fileName").title = media.name;
  $("#fileMeta").textContent = `${media.width}×${media.height} · ${+media.fps.toFixed(3)}fps · ${N}f`;
  $("#editorView").hidden = false;
  depthForm = depthOptionsForm($("#depthOptions"), status);
  audioToggle($("#outAudio"), media);
  bindUI();
  setZoom(Math.max(12, Math.min(28, Math.floor(tlScroll.clientWidth / Math.min(N, 96)))), 0);
  renderInspector();
  schedule("viewer", "timeline");
  loadPreviews();
  if (saved) toast("이 영상의 이전 편집을 불러왔습니다.");
}

init().catch((e) => toast(`초기화 실패: ${e.message}`, 6000));
