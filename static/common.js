// Shared helpers for the converter page and the timing editor.

export const $ = (sel, root = document) => root.querySelector(sel);

export function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v == null || v === false) continue;
    if (k === "class") node.className = v;
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else if (k === "html") node.innerHTML = v;
    else node.setAttribute(k, v === true ? "" : v);
  }
  for (const c of children.flat()) if (c != null) node.append(c);
  return node;
}

export async function api(path, opts = {}) {
  const res = await fetch(path, opts);
  if (!res.ok) {
    let msg = res.statusText;
    try {
      const body = await res.json();
      msg = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {}
    throw new Error(msg || `HTTP ${res.status}`);
  }
  return res.json();
}

export const postJSON = (path, body) =>
  api(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });

export const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

export function store(key, value) {
  try {
    if (value === undefined) return JSON.parse(localStorage.getItem(key));
    localStorage.setItem(key, JSON.stringify(value));
  } catch {
    return null;
  }
}

// "Keep source audio" checkbox: remembered per browser, disabled when the video has no audio.
export function audioToggle(input, media) {
  input.checked = media.has_audio && store("ds-keep-audio") !== false;
  input.disabled = !media.has_audio;
  input.closest("label").title = media.has_audio ? "원본 오디오 트랙을 mp4 결과물에 넣습니다." : "이 영상에는 오디오가 없습니다.";
  input.onchange = () => store("ds-keep-audio", input.checked);
}

export function toast(text, ms = 2200) {
  const t = el("div", { class: "toast" }, text);
  document.body.append(t);
  setTimeout(() => t.remove(), ms);
}

export function fmtBytes(n) {
  if (n < 1024) return `${n} B`;
  if (n < 1024 ** 2) return `${(n / 1024).toFixed(0)} KB`;
  if (n < 1024 ** 3) return `${(n / 1024 ** 2).toFixed(1)} MB`;
  return `${(n / 1024 ** 3).toFixed(2)} GB`;
}

export function fmtTC(frame, fps) {
  const r = Math.round(fps);
  const s = Math.floor(frame / r);
  const pad = (v) => String(v).padStart(2, "0");
  return `${pad(Math.floor(s / 3600))}:${pad(Math.floor(s / 60) % 60)}:${pad(s % 60)}:${pad(frame % r)}`;
}

export function fmtDuration(sec) {
  sec = Math.max(0, Math.round(sec));
  return sec >= 60 ? `${Math.floor(sec / 60)}분 ${sec % 60}초` : `${sec}초`;
}

// Same rule as server/video.py detect_cuts.
export function detectCuts(scores, threshold, minShot) {
  const cuts = [];
  let last = 0;
  for (let i = 1; i < scores.length; i++) {
    if (scores[i] >= threshold && i - last >= minShot) {
      cuts.push(i);
      last = i;
    }
  }
  return cuts;
}

export function uploadFile(file, onProgress) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    const form = new FormData();
    form.append("file", file);
    xhr.open("POST", "/api/media");
    xhr.upload.onprogress = (e) => e.lengthComputable && onProgress(e.loaded / e.total);
    xhr.onload = () => {
      let body = {};
      try { body = JSON.parse(xhr.responseText); } catch {}
      if (xhr.status >= 200 && xhr.status < 300) resolve(body);
      else reject(new Error(body.detail || `업로드 실패 (${xhr.status})`));
    };
    xhr.onerror = () => reject(new Error("네트워크 오류로 업로드에 실패했습니다."));
    xhr.send(form);
  });
}

export async function waitMedia(id) {
  for (;;) {
    const m = await api(`/api/media/${id}`);
    if (m.status !== "preparing") {
      if (m.status === "error") throw new Error(m.error || "영상 분석에 실패했습니다.");
      return m;
    }
    await sleep(700);
  }
}

// Drop zone + file picker. Calls onFile(File).
export function makeDropzone(zone, onFile) {
  const input = el("input", { type: "file", accept: "video/*,.mp4,.mov,.mkv,.avi,.webm", hidden: true });
  zone.append(input);
  zone.addEventListener("click", () => input.click());
  input.addEventListener("change", () => input.files[0] && onFile(input.files[0]));
  zone.addEventListener("dragover", (e) => { e.preventDefault(); zone.classList.add("drag"); });
  zone.addEventListener("dragleave", () => zone.classList.remove("drag"));
  zone.addEventListener("drop", (e) => {
    e.preventDefault();
    zone.classList.remove("drag");
    if (e.dataTransfer.files[0]) onFile(e.dataTransfer.files[0]);
  });
}

// Upload a file and wait for the server-side analysis, rendering progress into `box`.
export async function uploadAndPrepare(file, box) {
  box.replaceChildren(
    el("div", { class: "row" }, el("span", {}, file.name), el("span", { class: "muted" }, fmtBytes(file.size))),
    el("div", { class: "progress" }, el("div")),
    el("div", { class: "note" }, "업로드 중…"),
  );
  const bar = box.querySelector(".progress > div");
  const note = box.querySelector(".note");
  const media = await uploadFile(file, (p) => {
    bar.style.width = `${(p * 100).toFixed(1)}%`;
    note.textContent = `업로드 중… ${(p * 100).toFixed(0)}%`;
  });
  box.querySelector(".progress").classList.add("indeterminate");
  note.textContent = "영상 분석 중 (프레임 수, 컷 감지, 미리보기 생성)…";
  return waitMedia(media.id);
}

// ---------- depth options form ----------

const DEPTH_DEFAULTS = {
  model: null, input_size: 518, norm: "shot", adaptive_window: 1.0,
  clip_lo: 1.0, clip_hi: 1.0, temporal: 0, gamma: 1.0, invert: false,
};

export function depthOptionsForm(container, status) {
  const saved = { ...DEPTH_DEFAULTS, model: status.default_model, ...(store("ds-depth-opts") || {}) };
  if (!status.models.some((m) => m.key === saved.model)) saved.model = status.default_model;
  if (!status.input_sizes.includes(saved.input_size)) saved.input_size = 518;

  const model = el("select", {}, status.models.map((m) =>
    el("option", { value: m.key }, `${m.label} · ${m.license}`)));
  const size = el("select", {}, status.input_sizes.map((s) =>
    el("option", { value: s }, s === 518 ? "518 (기본)" : s >= 1000 ? `${s} (고해상도, 느림)` : String(s))));
  const norm = el("select", {},
    el("option", { value: "shot" }, "컷 단위 고정 (권장)"),
    el("option", { value: "adaptive" }, "컷 안에서 천천히 적응"),
    el("option", { value: "clip" }, "영상 전체 고정"),
    el("option", { value: "frame" }, "프레임별 (깜빡임 생김)"));
  const win = el("input", { type: "number", min: 0.1, max: 10, step: 0.1 });
  const range = (min, max, step) => el("input", { type: "range", min, max, step });
  const clipLo = range(0, 10, 0.5), clipHi = range(0, 10, 0.5), temporal = range(0, 3, 0.25), gamma = range(0.4, 2.5, 0.05);
  const invert = el("input", { type: "checkbox" });
  const licenseNote = el("div", { class: "note warn full" });
  const winRow = [el("label", {}, "적응 속도(초)"), el("div", { class: "inline" }, win, el("span", { class: "note" }, "클수록 밝기 변화가 느림"))];

  const withVal = (input, fmt) => {
    const out = el("span", { class: "val" });
    const sync = () => (out.textContent = fmt(+input.value));
    input.addEventListener("input", sync);
    return [el("div", { class: "inline" }, input, out), sync];
  };
  const [loRow, syncLo] = withVal(clipLo, (v) => `${v}%`);
  const [hiRow, syncHi] = withVal(clipHi, (v) => `${v}%`);
  const [tRow, syncT] = withVal(temporal, (v) => (v ? v.toFixed(2) : "끔"));
  const [gRow, syncG] = withVal(gamma, (v) => v.toFixed(2));

  container.replaceChildren(el("div", { class: "form" },
    el("label", {}, "모델"), model,
    licenseNote,
    el("label", {}, "입력 해상도"), size,
    el("label", {}, "밝기 정규화"), norm,
    ...winRow,
    el("details", { class: "adv full" },
      el("summary", {}, "고급 설정"),
      el("div", { class: "form" },
        el("label", {}, "먼 쪽 클리핑"), loRow,
        el("label", {}, "가까운 쪽 클리핑"), hiRow,
        el("label", {}, "시간 스무딩"), tRow,
        el("label", {}, "감마"), gRow,
        el("label", {}, "반전"), el("label", { class: "check" }, invert, "가까운 곳을 검정으로"),
      )),
  ));

  const set = (o) => {
    model.value = o.model; size.value = o.input_size; norm.value = o.norm; win.value = o.adaptive_window;
    clipLo.value = o.clip_lo; clipHi.value = o.clip_hi; temporal.value = o.temporal; gamma.value = o.gamma;
    invert.checked = o.invert;
    syncLo(); syncHi(); syncT(); syncG(); refresh();
  };
  const get = () => ({
    model: model.value, input_size: +size.value, norm: norm.value, adaptive_window: +win.value || 1,
    clip_lo: +clipLo.value, clip_hi: +clipHi.value, temporal: +temporal.value, gamma: +gamma.value,
    invert: invert.checked,
  });
  const refresh = () => {
    const m = status.models.find((x) => x.key === model.value);
    licenseNote.textContent = m && m.license.includes("NC") ? `${m.label}은(는) ${m.license} — 비상업 용도로만 사용 가능합니다.` : "";
    licenseNote.hidden = !licenseNote.textContent;
    winRow.forEach((n) => (n.hidden = norm.value !== "adaptive"));
    store("ds-depth-opts", get());
  };
  container.addEventListener("input", refresh);
  container.addEventListener("change", refresh);
  set(saved);
  return { get, set };
}

// ---------- job cards ----------

const STATUS_LABEL = { queued: "대기", running: "처리 중", done: "완료", error: "실패", cancelled: "취소" };

export function recentJobs(add) {
  const list = store("ds-recent-jobs") || [];
  if (add) {
    store("ds-recent-jobs", [add, ...list.filter((id) => id !== add)].slice(0, 20));
    return;
  }
  return list;
}

// Renders a self-updating card for a job. Returns the element.
export function jobCard(jobId, { onGone } = {}) {
  const card = el("div", { class: "job" });
  let stopped = false;

  const render = (j) => {
    const elapsed = j.started ? ((j.finished || Date.now() / 1000) - j.started) : 0;
    const head = el("div", { class: "job-head" },
      el("span", { class: `badge ${j.status}` }, STATUS_LABEL[j.status] || j.status),
      el("span", { class: "job-title", title: j.media_name }, j.media_name || j.id.slice(0, 8)),
      j.status === "queued" || j.status === "running"
        ? el("button", { class: "danger", onclick: async () => { await postJSON(`/api/jobs/${j.id}/cancel`, {}); poll(); } }, "취소")
        : null);
    const parts = [head];
    if (j.status === "queued") {
      parts.push(el("div", { class: "job-stage" }, j.lane === "cpu"
        ? "GPU가 필요 없는 작업이라 대기열을 거치지 않고 곧 시작합니다."
        : `대기열 ${j.position}번째 — 앞 작업이 끝나면 자동으로 시작합니다.`));
    }
    if (j.status === "running") {
      const bar = el("div", { class: "progress" }, el("div"));
      bar.firstChild.style.width = `${(j.progress * 100).toFixed(1)}%`;
      parts.push(bar, el("div", { class: "job-stage" }, `${j.stage} · ${(j.progress * 100).toFixed(0)}% · ${fmtDuration(elapsed)} 경과`));
    }
    if (j.status === "done") {
      parts.push(el("div", { class: "job-stage" }, `${fmtDuration(elapsed)} 걸림`));
      parts.push(el("div", { class: "job-files" }, j.outputs.map((o) =>
        el("a", { href: `/api/jobs/${j.id}/files/${encodeURIComponent(o.name)}`, download: o.name },
          el("span", {}, `⬇ ${o.label}`), el("span", { class: "muted mono" }, fmtBytes(o.size))))));
    }
    if (j.status === "error") parts.push(el("div", { class: "job-err" }, j.error));
    card.replaceChildren(...parts);
  };

  const poll = async () => {
    try {
      const j = await api(`/api/jobs/${jobId}`);
      render(j);
      return j.status === "queued" || j.status === "running";
    } catch (e) {
      stopped = true;
      card.remove();
      onGone && onGone(jobId);
      return false;
    }
  };

  (async () => {
    while (!stopped && (await poll())) {
      await sleep(1000);
      if (!card.isConnected) break;
    }
  })();
  return card;
}
