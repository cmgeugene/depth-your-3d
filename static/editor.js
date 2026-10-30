// Frame timing editor: step live-action footage on 1s/1.5s/2s… like 2D animation.
//
// The editable spec S is small and serializable (segments, overrides, cut edits, auto settings);
// everything shown is derived from it in derive(): which frames are new drawings (keys)
// and, for every frame, which source frame is shown (the timing map sent to the server).
//
// Auto timing writes segments too: a base step, and a faster step where the server's motion
// analysis says the camera or the action is fast. A shot whose segments still match that
// result counts as "auto" and follows later changes to the auto settings or the cuts.

import {
  $, el, api, postJSON, toast, store, fmtTC, detectCuts, makeDropzone, uploadAndPrepare,
  waitMedia, jobCard, recentJobs, audioToggle,
} from "./common.js";

const STEP_COLORS = { 1: "#5aa9ff", 1.5: "#9ad65c", 2: "#4cc38a", 3: "#f0b44c", 4: "#b38cff" };
const stepColor = (s) => (s === 0 ? "#7b8494" : STEP_COLORS[s] || "#3cc6c6");
const stepLabel = (s) => (s === 0 ? "멈춤" : `${s}s`);
// 0 = hold, otherwise 1–24 in half-frame steps (1.5s alternates 2,1,2,1… frames per drawing).
const isStep = (v) => v === 0 || (typeof v === "number" && v >= 1 && v <= 24 && Number.isInteger(v * 2));
const STEP_HINT = {
  auto: "자동: 기본 스텝으로, 빠른 곳만 '빠를 때' 스텝으로",
  1: "1s — 매 프레임 새 그림",
  1.5: "1.5s — 그림 하나를 2,1,2,1프레임씩 번갈아",
  2: "2s — 2프레임마다 새 그림",
  3: "3s — 3프레임마다 새 그림",
  4: "4s — 4프레임마다 새 그림",
  0: "멈춤 — 첫 그림을 계속 보여 줌",
};

// Motion counted as fast at sensitivity 5, in % of frame width per second. Calibrated on
// concert footage: a slow push-in measures about 3 (camera) / 5 (action), a dance shot 11 / 24.
const CAMERA_FAST = 7;
const ACTION_FAST = 22;

// Timeline rows; the last one takes whatever height is left.
const ROWS = [
  { k: "ruler", h: 18, label: "프레임" },
  { k: "cuts", h: 22, label: "컷" },
  { k: "motion", h: 22, label: "움직임" },
  { k: "seg", h: 20, label: "스텝" },
  { k: "frames", h: 0, label: "그림" },
];
const ROW_Y = {}, ROW_H = {};
let TL_H = 0;

const LAYOUT_DEFAULT = { insp: 380, tl: 230 };

let status, media, N, fps;
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
    version: 2,
    segments: [{ start: 0, step: 1.5 }],
    overrides: {},
    cutAdd: [],
    cutRemove: [],
    threshold: status.cut_threshold,
    minShot: status.cut_min_shot,
    resetAtCuts: true,
    auto: { base: 1.5, fast: 1, sens: 5 },
  };
}

function sanitizeSpec(raw) {
  const d = defaultSpec();
  const s = { ...d, ...(raw && typeof raw === "object" ? raw : {}) };
  const isInt = (v, lo, hi) => Number.isInteger(v) && v >= lo && v <= hi;
  const seen = new Set();
  s.segments = (Array.isArray(s.segments) ? s.segments : [])
    .filter((x) => x && isInt(x.start, 0, N - 1) && isStep(x.step))
    .sort((a, b) => a.start - b.start)
    .filter((x) => !seen.has(x.start) && seen.add(x.start))
    .map((x) => ({ start: x.start, step: x.step }));
  if (!s.segments.length || s.segments[0].start !== 0) s.segments.unshift({ start: 0, step: s.segments[0]?.step ?? 1.5 });
  s.segments = mergeSegs(s.segments);
  const ov = {};
  for (const [k, v] of Object.entries(s.overrides || {})) if (isInt(+k, 0, N - 1) && (v === "key" || v === "hold")) ov[+k] = v;
  s.overrides = ov;
  s.cutAdd = (s.cutAdd || []).filter((c) => isInt(c, 1, N - 1));
  s.cutRemove = (s.cutRemove || []).filter((c) => isInt(c, 1, N - 1));
  s.threshold = Number.isFinite(+s.threshold) ? +s.threshold : d.threshold;
  s.minShot = isInt(+s.minShot, 1, 96) ? +s.minShot : d.minShot;
  s.resetAtCuts = s.resetAtCuts !== false;
  const a = s.auto && typeof s.auto === "object" ? s.auto : {};
  const okStep = (v) => isStep(v) && v > 0;
  s.auto = { base: okStep(a.base) ? a.base : 1.5, fast: okStep(a.fast) ? a.fast : 1, sens: isInt(a.sens, 1, 10) ? a.sens : 5 };
  s.version = 2;
  return s;
}

// A segment start always begins a new drawing, so neighbours with the same step are merged.
const mergeSegs = (segs) => segs.filter((s, i) => i === 0 || s.step !== segs[i - 1].step);
const sameSegs = (x, y) => x.length === y.length && x.every((s, i) => s.start === y[i].start && s.step === y[i].step);

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
    else {
      // New drawing whenever floor(d / step) ticks over; in half-frame units so 1.5 stays exact.
      const d = i - anchor, h = seg.step * 2;
      key = Math.floor((2 * d) / h) !== Math.floor((2 * (d - 1)) / h);
    }
    if (key) { last = i; draws++; }
    map[i] = last; isKey[i] = key ? 1 : 0; drawNo[i] = draws; shotOf[i] = shot; segOf[i] = si;
  }
  D = { auto: new Set(auto), cuts, cutSet, map, isKey, drawNo, shotOf, segOf, shots: [0, ...cuts, N], draws };
}

const segEnd = (i) => (S.segments[i + 1]?.start ?? N) - 1;
const shotRange = (f) => [D.shots[D.shotOf[f]], D.shots[D.shotOf[f] + 1] - 1];
const shots = () => D.shots.slice(0, -1).map((a, k) => [a, D.shots[k + 1] - 1]);

function target() {
  if (view.sel) return { a: view.sel.a, b: view.sel.b, label: "고른 구간" };
  const [a, b] = shotRange(view.cur);
  return { a, b, label: `샷 ${D.shotOf[view.cur] + 1}` };
}

// ---------- auto timing ----------

// Returns f => score, where a score of 1 or more means fast at the current sensitivity.
function scorer() {
  const m = media.motion;
  if (!m) return () => 0;
  const k = 2 ** ((5 - S.auto.sens) / 2.5); // sensitivity 10 → thresholds ×0.25, 1 → ×3
  const cam = (CAMERA_FAST * k) / fps, act = (ACTION_FAST * k) / fps;
  return (f) => Math.max(m.camera[f] / cam, m.action[f] / act);
}

// Runs of `value` shorter than minLen flip to the other value (only between other runs if interiorOnly).
function fillRuns(arr, value, minLen, interiorOnly) {
  for (let i = 0; i < arr.length;) {
    if (arr[i] !== value) { i++; continue; }
    let j = i;
    while (j < arr.length && arr[j] === value) j++;
    if (j - i < minLen && (!interiorOnly || (i > 0 && j < arr.length))) arr.fill(1 - value, i, j);
    i = j;
  }
}

function autoSegments(a, b) {
  const { base, fast } = S.auto;
  if (!media.motion || base === fast) return [{ start: a, step: base }];
  const n = b - a + 1, score = scorer();
  // A shot's first frame is the cut itself, so it borrows the next frame's motion.
  const raw = Float32Array.from({ length: n }, (_, i) => score(Math.min(b, a + Math.max(i, 1))));
  // Average over ±2 frames so one noisy frame doesn't flip the step, then drop short blips.
  const isFast = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    let sum = 0, cnt = 0;
    for (let j = Math.max(0, i - 2); j <= Math.min(n - 1, i + 2); j++) { sum += raw[j]; cnt++; }
    isFast[i] = sum / cnt >= 1 ? 1 : 0;
  }
  const minRun = Math.max(3, Math.round(fps / 4));
  fillRuns(isFast, 0, minRun, true);
  fillRuns(isFast, 1, minRun, false);
  const segs = [];
  for (let i = 0; i < n; i++) if (i === 0 || isFast[i] !== isFast[i - 1]) segs.push({ start: a + i, step: isFast[i] ? fast : base });
  return segs;
}

function segsIn(a, b) {
  return mergeSegs([{ start: a, step: S.segments[D.segOf[a]].step },
    ...S.segments.filter((s) => s.start > a && s.start <= b).map((s) => ({ ...s }))]);
}

function shotMode(a, b) {
  const segs = segsIn(a, b);
  if (sameSegs(segs, autoSegments(a, b))) return { kind: "auto", segs };
  if (segs.length === 1) return { kind: "fixed", step: segs[0].step, segs };
  return { kind: "custom", segs };
}

function replaceRange(a, b, segs) {
  splitAt(b + 1);
  S.segments = mergeSegs([...S.segments.filter((s) => s.start < a), ...segs, ...S.segments.filter((s) => s.start > b)]);
}

function applyAutoAll() {
  for (const [a, b] of shots()) replaceRange(a, b, autoSegments(a, b));
}

// Runs an edit that changes the auto settings or the shot boundaries, then re-applies auto
// timing to every shot that was auto before it. Shots set by hand stay as they are.
function keepAuto(fn) {
  const wasAuto = new Uint8Array(N);
  for (const [a, b] of shots()) if (shotMode(a, b).kind === "auto") wasAuto.fill(1, a, b + 1);
  fn();
  derive();
  for (const [a, b] of shots()) if (wasAuto.subarray(a, b + 1).every(Boolean)) replaceRange(a, b, autoSegments(a, b));
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
  if (!isStep(step)) return toast("스텝은 1~24 사이, 0.5 단위입니다 (0 = 멈춤).");
  const { a, b, label } = target();
  commit(() => replaceRange(a, b, [{ start: a, step }]));
  toast(`${label} F${a}–${b} → ${stepLabel(step)}`);
}

function setShot(a, b, step) {
  commit(() => replaceRange(a, b, step === "auto" ? autoSegments(a, b) : [{ start: a, step }]));
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
  commit(() => keepAuto(() => {
    const add = new Set(S.cutAdd), rem = new Set(S.cutRemove);
    if (D.cutSet.has(f)) add.has(f) ? add.delete(f) : rem.add(f);
    else rem.has(f) ? rem.delete(f) : add.add(f);
    S.cutAdd = [...add];
    S.cutRemove = [...rem];
  }));
}

function select(sel) {
  view.sel = sel;
  renderSelection();
  schedule("timeline");
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
    label.textContent = done < N ? `미리보기 불러오는 중 ${done}/${N}` : "";
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

function layoutRows(h) {
  if (h === TL_H) return;
  TL_H = h;
  let y = 0;
  for (const r of ROWS) {
    ROW_Y[r.k] = y;
    ROW_H[r.k] = r.h || Math.max(36, h - y);
    y += ROW_H[r.k];
  }
  $("#tlLabels").replaceChildren(...ROWS.map((r) => el("div", { style: `height:${ROW_H[r.k]}px` }, r.label)));
}

function drawTimeline() {
  const W = tlCanvas.parentElement.clientWidth;
  layoutRows(Math.max(120, tlCanvas.clientHeight));
  const ctx = fitCanvas(tlCanvas, W, TL_H);
  const cw = view.cellW, sl = tlScroll.scrollLeft;
  const f0 = Math.max(0, Math.floor(sl / cw)), f1 = Math.min(N - 1, Math.floor((sl + W) / cw));
  const X = (f) => f * cw - sl;

  ctx.fillStyle = "#16191d"; ctx.fillRect(0, 0, W, TL_H);
  ctx.fillStyle = "#1b1f24"; ctx.fillRect(0, ROW_Y.cuts, W, ROW_H.cuts);
  ctx.fillStyle = "#181b20"; ctx.fillRect(0, ROW_Y.motion, W, ROW_H.motion);
  ctx.fillStyle = "#121417"; ctx.fillRect(0, ROW_Y.frames, W, ROW_H.frames);
  ctx.strokeStyle = "#2c3139"; ctx.lineWidth = 1;
  for (const r of ROWS.slice(1)) { const y = ROW_Y[r.k]; ctx.beginPath(); ctx.moveTo(0, y + 0.5); ctx.lineTo(W, y + 0.5); ctx.stroke(); }

  // ruler
  const every = [1, 2, 5, 10, 12, 24, 48, 96, 120, 240, 480, 960, 2400].find((s) => s * cw >= 46) || 4800;
  const sec = Math.round(fps), rb = ROW_H.ruler;
  ctx.font = "10px Consolas, monospace";
  ctx.textBaseline = "top";
  for (let f = f0; f <= f1; f++) {
    const x = Math.round(X(f)) + 0.5;
    const major = f % every === 0;
    if (major || cw >= 5 || f % sec === 0) {
      ctx.strokeStyle = major ? "#8b939f" : f % sec === 0 ? "#5b6370" : "#353b44";
      ctx.beginPath(); ctx.moveTo(x, major ? rb - 10 : rb - 5); ctx.lineTo(x, rb); ctx.stroke();
    }
    if (major) { ctx.fillStyle = "#8b939f"; ctx.fillText(String(f), x + 3, 1); }
  }

  // bar graph helper: value 1 sits on the dashed line at half height
  const bars = (row, value, colorOf) => {
    const barMax = ROW_H[row] - 5, base = ROW_Y[row] + ROW_H[row] - 2;
    for (let f = f0; f <= f1; f++) {
      const v = value(f), h = Math.min(1, v / 2) * barMax;
      if (h < 0.5) continue;
      ctx.fillStyle = colorOf(v);
      ctx.fillRect(X(f) + (cw > 3 ? 1 : 0), base - h, Math.max(1, cw - (cw > 3 ? 2 : 0)), h);
    }
    ctx.setLineDash([3, 3]);
    ctx.beginPath(); ctx.moveTo(0, base - barMax / 2 + 0.5); ctx.lineTo(W, base - barMax / 2 + 0.5); ctx.stroke();
    ctx.setLineDash([]);
  };

  // scene scores vs cut threshold
  ctx.strokeStyle = "rgba(239,100,97,.45)";
  bars("cuts", (f) => media.scores[f] / S.threshold, (v) => (v >= 1 ? "#a8524f" : "#3d4654"));

  // motion vs the auto timing threshold
  if (media.motion) {
    const score = scorer(), fastColor = stepColor(S.auto.fast);
    ctx.strokeStyle = "rgba(154,214,92,.45)";
    bars("motion", (f) => (f === 0 || D.cutSet.has(f) ? 0 : score(f)), (v) => (v >= 1 ? fastColor : "#3d4654"));
  } else {
    ctx.fillStyle = "#5b6370"; ctx.font = "11px sans-serif"; ctx.textBaseline = "middle";
    ctx.fillText("움직임 정보 없음", 8, ROW_Y.motion + ROW_H.motion / 2);
  }

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
    if (x2 - x1 > 30) {
      ctx.fillStyle = "#0e1013";
      ctx.fillText(stepLabel(seg.step), Math.max(x1, 0) + 6, ROW_Y.seg + ROW_H.seg / 2);
    }
  });

  // frames
  const fy = ROW_Y.frames, fh = ROW_H.frames, labelH = 13, thumbY = fy + labelH, thumbH = fh - labelH - 2;
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
      ctx.fillRect(x + 1, fy + fh - 14, 11, 12);
      ctx.fillStyle = "#0e1013"; ctx.textBaseline = "top";
      ctx.fillText(ov === "key" ? "K" : "H", x + 3.5, fy + fh - 13);
    } else if (ov) {
      ctx.fillStyle = ov === "key" ? "#f0b44c" : "#3cc6c6";
      ctx.fillRect(x, fy + fh - 4, Math.max(2, cw), 3);
    }
    if (cw >= 8) { ctx.fillStyle = "#0e1013"; ctx.fillRect(x + cw - 0.5, thumbY, 1, thumbH); }
  }

  // cuts (active), and auto cuts the user removed
  for (const c of S.cutRemove) {
    if (c < f0 || c > f1 + 1) continue;
    ctx.strokeStyle = "rgba(139,147,159,.6)"; ctx.setLineDash([2, 3]);
    ctx.beginPath(); ctx.moveTo(Math.round(X(c)) + 0.5, ROW_Y.cuts); ctx.lineTo(Math.round(X(c)) + 0.5, ROW_Y.motion); ctx.stroke();
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
  ctx.strokeRect(px + 0.5, fy + 0.5, Math.max(2, cw - 1), fh - 1);
  ctx.lineWidth = 1;
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(px + cw / 2 - 0.5, 0, 1, TL_H);
  ctx.beginPath(); ctx.moveTo(px + cw / 2 - 5, 0); ctx.lineTo(px + cw / 2 + 5, 0); ctx.lineTo(px + cw / 2, 7); ctx.fill();
}

// ---------- inspector ----------

// Step chips; `current` is a value or a list of values to highlight.
function chips(values, current, onPick, { disabled = false } = {}) {
  const on = new Set([].concat(current));
  return values.map((v) => el("button", {
    class: `chip${on.has(v) ? " on" : ""}`, "data-step": v, title: STEP_HINT[v], disabled, onclick: () => onPick(v),
  }, v === "auto" ? "자동" : stepLabel(v)));
}

function renderCurInfo() {
  const f = view.cur, key = D.isKey[f], shown = D.map[f], ov = S.overrides[f];
  const step = stepLabel(S.segments[D.segOf[f]].step);
  $("#curFrameLabel").textContent = `F${f}`;
  $("#curState").textContent = key
    ? `새 그림 #${D.drawNo[f]} · ${step} 구간`
    : `앞 그림 #${D.drawNo[f]} (F${shown})을 유지 · ${step} 구간`;
  for (const b of $("#ovCtrl").children) b.classList.toggle("on", (b.dataset.ov || undefined) === ov);
  $("#hud").replaceChildren(...[
    el("span", {}, `F ${f}`),
    el("span", {}, fmtTC(f, fps)),
    el("span", {}, `샷 ${D.shotOf[f] + 1}/${D.shots.length - 1}`),
    key ? el("span", { class: "key" }, `새 그림 #${D.drawNo[f]}`) : el("span", { class: "hold" }, `유지 ← F${shown}`),
    ov ? el("span", { class: "ov" }, ov === "key" ? "직접: 새 그림" : "직접: 유지") : null,
    view.mode === "orig" ? el("span", {}, "원본 보기") : null,
  ].filter(Boolean));
  $("#tFrame").value = f;
  $("#tTC").textContent = fmtTC(f, fps);
  for (const card of $("#shotList").children) card.classList.toggle("active", +card.dataset.k === D.shotOf[f]);
}

function renderSelection() {
  const sel = view.sel;
  let cur = null;
  if (sel) {
    const steps = new Set();
    S.segments.forEach((s, i) => { if (s.start <= sel.b && segEnd(i) >= sel.a) steps.add(s.step); });
    if (steps.size === 1) cur = [...steps][0];
  }
  $("#selInfo").textContent = sel
    ? `F${sel.a}–${sel.b} (${sel.b - sel.a + 1}프레임)에 적용합니다.`
    : "타임라인을 드래그해서 구간을 고르면, 그 부분만 다른 스텝으로 바꿀 수 있어요.";
  $("#selSteps").replaceChildren(...chips([1, 1.5, 2, 3, 4, 0], cur, setStep, { disabled: !sel }));
  $("#customStepBtn").disabled = !sel;
  $("#selClearBtn").disabled = !sel;
}

function renderAuto() {
  const pick = (patch) => commit(() => keepAuto(() => Object.assign(S.auto, patch)));
  $("#autoBase").replaceChildren(...chips([1, 1.5, 2, 3], S.auto.base, (v) => pick({ base: v })));
  $("#autoFast").replaceChildren(...chips([1, 1.5, 2], S.auto.fast, (v) => pick({ fast: v })));
  $("#autoSens").value = S.auto.sens;
  $("#autoSens").disabled = !media.motion;
  $("#autoNote").hidden = !!media.motion;
  $("#autoNote").textContent = "이 영상은 움직임 분석 결과가 없어 모두 기본 스텝으로 둡니다. 영상을 다시 올리면 분석합니다.";
}

function renderShots() {
  const list = shots();
  $("#shotCount").textContent = `${list.length}개`;
  $("#shotList").replaceChildren(...list.map(([a, b], k) => {
    const mode = shotMode(a, b);
    let draws = 0;
    for (let f = a; f <= b; f++) draws += D.isKey[f];
    let desc, on;
    if (mode.kind === "auto") {
      const { base, fast } = S.auto;
      const fastRuns = base === fast ? 0 : mode.segs.filter((s) => s.step === fast).length;
      desc = !fastRuns ? `자동 · 전부 ${stepLabel(base)}`
        : mode.segs.length === 1 ? `자동 · 전부 ${stepLabel(fast)} (빠른 샷)`
        : `자동 · ${stepLabel(base)}, 빠른 곳 ${fastRuns}군데는 ${stepLabel(fast)}`;
      on = mode.segs.length === 1 ? ["auto", mode.segs[0].step] : "auto";
    } else if (mode.kind === "fixed") {
      desc = `${stepLabel(mode.step)} 고정`;
      on = mode.step;
    } else {
      desc = `직접 나눔 · ${[...new Set(mode.segs.map((s) => stepLabel(s.step)))].join(" + ")}`;
      on = null;
    }
    const go = () => { select(null); setCur(a); };
    return el("div", { class: `shot${D.shotOf[view.cur] === k ? " active" : ""}`, "data-k": k },
      el("img", { class: "shot-thumb", src: `/api/media/${media.id}/thumb/${a}.jpg`, alt: "", title: "이 샷으로 이동", onclick: go }),
      el("div", { class: "shot-body" },
        el("div", { class: "shot-title", title: "이 샷으로 이동", onclick: go },
          el("b", {}, `샷 ${k + 1}`), el("span", { class: "note mono" }, `${((b - a + 1) / fps).toFixed(1)}초 · F${a}–${b} · 그림 ${draws}장`)),
        el("div", { class: `shot-desc ${mode.kind}` }, desc),
        el("div", { class: "chips" }, chips(["auto", 1, 1.5, 2, 3, 0], on, (v) => setShot(a, b, v)))));
  }));
}

function renderCuts() {
  $("#cutStrength").value = 62 - S.threshold;
  $("#minShot").value = S.minShot;
  $("#resetAtCuts").checked = S.resetAtCuts;
  $("#cutSummary").textContent = `— 컷 ${D.cuts.length}개`;
  const cutItems = D.cuts.map((c) =>
    el("div", { class: "item" },
      el("i", { class: "swatch", style: `background:${D.auto.has(c) ? "#ef6461" : "#ffb3b1"}` }),
      el("span", { class: "range", onclick: () => setCur(c) }, `F${c}  ${fmtTC(c, fps)}`),
      el("span", { class: "note" }, D.auto.has(c) ? "자동" : "직접"),
      el("button", { title: "이 컷 빼기", onclick: () => toggleCut(c) }, "✕")));
  const removed = S.cutRemove.filter((c) => D.auto.has(c)).sort((a, b) => a - b).map((c) =>
    el("div", { class: "item" },
      el("i", { class: "swatch", style: "background:#3a414b" }),
      el("span", { class: "range muted", onclick: () => setCur(c) }, `F${c}  뺀 컷`),
      el("button", { onclick: () => toggleCut(c) }, "되살리기")));
  $("#cutList").replaceChildren(...(cutItems.length || removed.length ? [...cutItems, ...removed] : [el("div", { class: "note" }, "컷 없음 — 전체가 하나의 샷입니다.")]));
}

function renderInspector() {
  renderAuto();
  renderShots();
  renderCuts();
  renderCurInfo();
  renderSelection();
  $("#statsChip").textContent = `그림 ${D.draws}장 · 평균 ${(N / D.draws).toFixed(2)}프레임`;
  $("#statsChip").title = `${N}프레임 중 새 그림 ${D.draws}장`;
  $("#undoBtn").disabled = !hist.undo.length;
  $("#redoBtn").disabled = !hist.redo.length;
}

// ---------- timeline input ----------

function frameAt(clientX) {
  const r = tlCanvas.getBoundingClientRect();
  return Math.max(0, Math.min(N - 1, Math.floor((clientX - r.left + tlScroll.scrollLeft) / view.cellW)));
}

function rowAt(clientY) {
  const y = clientY - tlCanvas.getBoundingClientRect().top;
  return [...ROWS].reverse().find((r) => y >= ROW_Y[r.k])?.k ?? "ruler";
}

let drag = null;
tlCanvas.addEventListener("pointerdown", (e) => {
  const f = frameAt(e.clientX), row = rowAt(e.clientY);
  if (row === "cuts") return toggleCut(f === 0 ? 1 : f);
  if (row === "seg") {
    const i = D.segOf[f];
    view.sel = { a: S.segments[i].start, b: segEnd(i) };
    $("#detailSec").open = true;
    setCur(f);
    renderSelection();
    return;
  }
  if (e.shiftKey) {
    const anchor = view.sel ? (f < view.sel.a ? view.sel.b : view.sel.a) : view.cur;
    view.sel = { a: Math.min(anchor, f), b: Math.max(anchor, f) };
    setCur(f);
    renderSelection();
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
  if (view.sel) $("#detailSec").open = true;
  renderSelection();
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

// ---------- panel sizes ----------

function initSplitters() {
  const grid = $("#editorView");
  const lay = { ...LAYOUT_DEFAULT, ...(store("ds-editor-layout") || {}) };
  const apply = () => {
    lay.insp = Math.round(Math.max(300, Math.min(lay.insp, innerWidth * 0.6)));
    lay.tl = Math.round(Math.max(150, Math.min(lay.tl, innerHeight * 0.65)));
    grid.style.setProperty("--insp-w", `${lay.insp}px`);
    grid.style.setProperty("--tl-h", `${lay.tl}px`);
  };
  const bind = (handle, onMove) => {
    handle.addEventListener("pointerdown", (e) => {
      const start = { x: e.clientX, y: e.clientY, ...lay };
      handle.setPointerCapture(e.pointerId);
      document.body.classList.add("resizing");
      const move = (ev) => { onMove(ev.clientX - start.x, ev.clientY - start.y, start); apply(); };
      const up = () => {
        handle.removeEventListener("pointermove", move);
        handle.removeEventListener("pointerup", up);
        document.body.classList.remove("resizing");
        store("ds-editor-layout", lay);
      };
      handle.addEventListener("pointermove", move);
      handle.addEventListener("pointerup", up);
    });
    handle.addEventListener("dblclick", () => { Object.assign(lay, LAYOUT_DEFAULT); apply(); store("ds-editor-layout", lay); });
  };
  bind($("#splitV"), (dx, dy, s) => (lay.insp = s.insp - dx));
  bind($("#splitH"), (dx, dy, s) => (lay.tl = s.tl - dy));
  apply();
  addEventListener("resize", apply);
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
    app: "depth-studio", version: 2, source: media.name, frames: N, fps: media.fps_str,
    spec: S, cuts: D.cuts, map: Array.from(D.map),
  }, null, 1), "application/json");
}

function exportCSV() {
  const rows = ["frame,timecode,shot,drawing,source_frame,type,step,override"];
  for (let f = 0; f < N; f++) {
    rows.push([f, fmtTC(f, fps), D.shotOf[f] + 1, D.drawNo[f], D.map[f], D.isKey[f] ? "key" : "hold",
      S.segments[D.segOf[f]].step === 0 ? "hold" : `${S.segments[D.segOf[f]].step}s`, S.overrides[f] || ""].join(","));
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

function openRender() {
  $("#renderSummary").textContent = `${N}프레임 → 새 그림 ${D.draws}장. 원본 프레임을 이 타이밍대로 골라 붙인 영상을 만듭니다 (길이와 fps는 원본과 같음).`;
  $("#renderModal").hidden = false;
}

async function startRender() {
  const outputs = { stepped_mp4: true, depth_mp4: false, depth_png: false, audio: $("#outAudio").checked };
  $("#renderBtn").disabled = true;
  try {
    const job = await postJSON("/api/jobs", {
      media_id: media.id, outputs, map: Array.from(D.map), cuts: D.cuts, timing: S,
    });
    recentJobs(job.id);
    $("#jobs").prepend(jobCard(job.id));
    toast("렌더 대기열에 넣었습니다.");
  } catch (e) {
    toast(e.message, 4000);
  } finally {
    $("#renderBtn").disabled = false;
  }
}

// Top-bar button shows how many of this page's renders are still waiting or running.
function watchJobs() {
  setInterval(() => {
    const n = document.querySelectorAll("#jobs .badge.queued, #jobs .badge.running").length;
    $("#openRenderBtn").textContent = n ? `렌더 · ${n}건 진행 중` : "렌더";
  }, 1000);
}

// ---------- keyboard ----------

const closeModals = () => { $("#helpModal").hidden = true; $("#renderModal").hidden = true; $("#fileMenu").open = false; };

document.addEventListener("keydown", (e) => {
  if (!D) return;
  const t = e.target;
  if (t.matches?.("input:not([type=checkbox]):not([type=range]), select, textarea")) return;
  if (!$("#helpModal").hidden || !$("#renderModal").hidden) {
    if (e.key === "Escape" || (e.key === "?" && !$("#helpModal").hidden)) closeModals();
    return;
  }
  const k = e.key, ctrl = e.ctrlKey || e.metaKey;
  const handled = () => e.preventDefault();

  if (ctrl && k.toLowerCase() === "z") return handled(), e.shiftKey ? redo() : undo();
  if (ctrl && k.toLowerCase() === "y") return handled(), redo();
  if (ctrl && k.toLowerCase() === "a") { handled(); return select({ a: 0, b: N - 1 }); }
  if (ctrl) return;
  // Shift+digit = half step (Shift+1 → 1.5s). Matched by key position so keyboard layout doesn't matter.
  if (e.shiftKey && /^Digit[1-8]$/.test(e.code)) return setStep(+e.code.slice(5) + 0.5);

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
    case "Escape": closeModals(); return select(null);
    case "?": $("#helpModal").hidden = false; return;
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
    return select(lower === "i" ? { a: view.cur, b: Math.max(view.cur, s.b) } : { a: Math.min(view.cur, s.a), b: view.cur });
  }
  if (lower === "s") { const [a, b] = shotRange(view.cur); return select({ a, b }); }
  if (lower === "l") { view.loop = !view.loop; $("#tLoop").classList.toggle("on", view.loop); return; }
  if (lower === "v") return setMode({ step: "orig", orig: "compare", compare: "step" }[view.mode]);
  if (lower === "n") { $("#onion").checked = view.onion = !view.onion; return schedule("viewer"); }
});

// ---------- setup ----------

function bindUI() {
  const menu = $("#fileMenu");
  const fromMenu = (fn) => () => { menu.open = false; fn(); };
  document.addEventListener("pointerdown", (e) => { if (menu.open && !menu.contains(e.target)) menu.open = false; });
  $("#undoBtn").onclick = undo;
  $("#redoBtn").onclick = redo;
  $("#exportBtn").onclick = fromMenu(exportJSON);
  $("#csvBtn").onclick = fromMenu(exportCSV);
  $("#importBtn").onclick = fromMenu(() => $("#importFile").click());
  $("#importFile").onchange = (e) => { if (e.target.files[0]) importJSON(e.target.files[0]); e.target.value = ""; };
  $("#resetBtn").onclick = fromMenu(() => confirm("타이밍 편집을 모두 처음 상태(자동)로 되돌릴까요? (되돌리기로 취소할 수 있습니다)")
    && commit(() => { S = defaultSpec(); derive(); applyAutoAll(); }));
  $("#helpBtn").onclick = () => ($("#helpModal").hidden = false);
  $("#helpClose").onclick = () => ($("#helpModal").hidden = true);
  $("#openRenderBtn").onclick = openRender;
  $("#renderClose").onclick = () => ($("#renderModal").hidden = true);
  for (const id of ["#renderModal", "#helpModal"]) $(id).addEventListener("pointerdown", (e) => { if (e.target === $(id)) $(id).hidden = true; });
  $("#renderBtn").onclick = startRender;

  for (const b of $("#modeCtrl").children) b.onclick = () => setMode(b.dataset.mode);
  $("#onion").onchange = (e) => { view.onion = e.target.checked; schedule("viewer"); };
  for (const b of $("#ovCtrl").children) b.onclick = () => view.cur > 0 && setOverride(view.cur, b.dataset.ov || null);
  $("#customStepBtn").onclick = () => setStep(parseFloat($("#customStep").value));
  $("#selClearBtn").onclick = () => select(null);
  $("#clearOvBtn").onclick = clearOverrides;
  $("#autoAllBtn").onclick = () => commit(applyAutoAll);

  // Slider edits update live; one undo step per drag.
  let pre = null;
  const live = (fn) => (e) => { if (pre === null) pre = JSON.stringify(S); fn(e); afterChange(); };
  const settle = () => { if (pre !== null) { hist.undo.push(pre); hist.redo = []; pre = null; renderInspector(); } };
  $("#autoSens").addEventListener("input", live((e) => keepAuto(() => (S.auto.sens = +e.target.value))));
  $("#autoSens").addEventListener("change", settle);
  // The slider reads "more cuts" to the right, so it runs opposite to the score threshold.
  $("#cutStrength").addEventListener("input", live((e) => keepAuto(() => (S.threshold = 62 - +e.target.value))));
  $("#cutStrength").addEventListener("change", settle);
  $("#minShot").addEventListener("change", (e) => {
    const v = Math.max(1, Math.min(96, parseInt(e.target.value, 10) || 1));
    commit(() => keepAuto(() => (S.minShot = v)));
  });
  $("#resetAtCuts").onchange = (e) => commit(() => (S.resetAtCuts = e.target.checked));
  $("#resetCutsBtn").onclick = () => commit(() => keepAuto(() => { S.cutAdd = []; S.cutRemove = []; }));

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
  if (saved) {
    S = sanitizeSpec(saved);
    derive();
  } else {
    S = defaultSpec();
    derive();
    applyAutoAll();
    derive();
  }

  document.title = `${media.name} — Frame Timing Editor`;
  $("#fileName").textContent = media.name;
  $("#fileName").title = media.name;
  $("#fileMeta").textContent = `${media.width}×${media.height} · ${+media.fps.toFixed(3)}fps · ${N}f`;
  $("#editorView").hidden = false;
  initSplitters();
  audioToggle($("#outAudio"), media);
  bindUI();
  watchJobs();
  setZoom(Math.max(12, Math.min(28, Math.floor(tlScroll.clientWidth / Math.min(N, 96)))), 0);
  renderInspector();
  schedule("viewer", "timeline");
  loadPreviews();
  if (saved) toast("이 영상의 이전 편집을 불러왔습니다.");
}

init().catch((e) => toast(`초기화 실패: ${e.message}`, 6000));
