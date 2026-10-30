import {
  $, el, api, postJSON, toast, fmtDuration, makeDropzone, uploadAndPrepare,
  jobCard, recentJobs, audioToggle, watchQueue, store,
} from "./common.js";

const DEFAULTS = { mode: "anime", scale: 2, backend: "ai", denoise: "none", format: "mp4_h265", quality: "balanced", interpolation: false };

let status;
let media = null;
let opts = { ...DEFAULTS, ...(store("ds-upscale-opts") || {}) };

const jobsBox = $("#jobs");
const syncEmpty = () => ($("#jobsEmpty").hidden = jobsBox.children.length > 0);

async function toEditor(job, button) {
  button.disabled = true;
  try {
    const m = await postJSON(`/api/jobs/${job.id}/to-editor`, {});
    location.href = `/editor?m=${m.id}`;
  } catch (e) {
    toast(e.message, 4000);
    button.disabled = false;
  }
}

function jobActions(j) {
  const out = j.outputs[0];
  const editorBtn = el("button", { onclick: () => toEditor(j, editorBtn) }, "프레임 편집기로 보내기");
  return [
    el("a", { class: "btn primary", href: `/api/jobs/${j.id}/files/${encodeURIComponent(out.name)}`, download: out.name }, "⬇ 영상 다운로드"),
    editorBtn,
  ];
}

function addJob(id, prepend = true) {
  const card = jobCard(id, { onGone: syncEmpty, kind: "upscale", actions: jobActions });
  prepend ? jobsBox.prepend(card) : jobsBox.append(card);
  syncEmpty();
}

function modelName() {
  if (opts.backend === "lanczos") return "Lanczos (AI 없음)";
  if (opts.mode === "anime") return "realesr-animevideov3";
  return opts.scale === 2 ? "realesrgan-x4plus (4배 추론 후 2배 축소)" : "realesrgan-x4plus";
}

// Reflects opts in the controls, the output summary and the warnings; remembers opts per browser.
function sync() {
  const eng = status.upscale;
  if (!eng.realesrgan) opts.backend = "lanczos";
  if (!eng.rife) opts.interpolation = false;
  for (const group of document.querySelectorAll(".seg-ctrl[data-key]")) {
    for (const b of group.querySelectorAll("button")) {
      const on = String(opts[group.dataset.key]) === b.dataset.v;
      b.classList.toggle("on", on);
      b.setAttribute("aria-pressed", on);
    }
  }
  $('.seg-ctrl[data-key="backend"] button[data-v="ai"]').disabled = !eng.realesrgan;
  $("#rife").checked = opts.interpolation;
  $("#rife").disabled = !eng.rife;
  $("#format").value = opts.format;
  $("#quality").value = opts.quality;
  store("ds-upscale-opts", opts);

  const missing = [!eng.realesrgan && "Real-ESRGAN", !eng.rife && "RIFE"].filter(Boolean);
  $("#engineWarn").hidden = !missing.length;
  $("#engineWarn").textContent = missing.length
    ? `서버에 ${missing.join(", ")} 엔진이 설치되지 않아 해당 기능을 쓸 수 없습니다. 관리자에게 setup-upscale-engines.bat 실행을 요청하세요.`
    : "";
  if (!media) return;

  const w = media.width * opts.scale, h = media.height * opts.scale;
  const fps = media.fps * (opts.interpolation ? 2 : 1);
  $("#outInfo").textContent = `출력 ${w}×${h} · ${+fps.toFixed(3)}fps · 모델 ${modelName()}`;
  const tooBig = Math.max(w, h) > eng.max_side;
  const warns = [];
  if (opts.backend === "ai" && opts.mode === "real") warns.push("실사 모드는 2D 애니메이션에서 피부가 매끈해지고 선이 바뀔 수 있습니다.");
  if (tooBig) warns.push(`출력 긴 변이 서버 한도(${eng.max_side}px)를 넘습니다. 배율을 낮추세요.`);
  $("#outWarn").hidden = !warns.length;
  $("#outWarn").textContent = warns.join(" ");
  $("#startBtn").disabled = tooBig;
}

function showMedia(m) {
  media = m;
  const box = $("#mediaBox");
  box.hidden = false;
  const thumb = m.editor ? el("img", { src: `/api/media/${m.id}/preview/0.jpg`, alt: "" }) : el("div");
  box.replaceChildren(
    el("div", { class: "media-info" }, thumb,
      el("dl", { class: "kv" },
        el("dt", {}, "파일"), el("dd", {}, m.name),
        el("dt", {}, "해상도"), el("dd", {}, `${m.width}×${m.height}`),
        el("dt", {}, "프레임"), el("dd", {}, `${m.frames}프레임 · ${m.fps.toFixed(3).replace(/\.?0+$/, "")}fps · ${fmtDuration(m.duration)}`),
        el("dt", {}, "오디오"), el("dd", {}, m.has_audio ? "있음" : "없음"))),
    el("div", { class: "row" }, el("button", { onclick: reset }, "다른 영상")),
  );
  $("#uploadBox").hidden = true;
  $("#optionsCard").hidden = false;
  audioToggle($("#outAudio"), m);
  sync();
}

function reset() {
  media = null;
  $("#mediaBox").hidden = true;
  $("#optionsCard").hidden = true;
  $("#uploadBox").hidden = false;
  $("#uploadBox").replaceChildren(dropzone);
}

async function onFile(file) {
  const box = $("#uploadBox");
  const holder = el("div", { class: "stack" });
  box.replaceChildren(holder);
  try {
    showMedia(await uploadAndPrepare(file, holder));
  } catch (e) {
    holder.replaceChildren(el("div", { class: "job-err" }, e.message), el("button", { onclick: reset }, "다시 시도"));
  }
}

async function start() {
  $("#startBtn").disabled = true;
  try {
    const job = await postJSON("/api/upscale-jobs", { media_id: media.id, ...opts, audio: $("#outAudio").checked });
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
  $("#ttl").textContent = status.ttl_hours;
  $("#uploadHint").textContent = `mp4 · mov · mkv · webm — 최대 ${status.max_upload_mb >= 1024 ? `${(status.max_upload_mb / 1024).toFixed(0)}GB` : `${status.max_upload_mb}MB`}`;
  makeDropzone(dropzone, onFile);
  for (const group of document.querySelectorAll(".seg-ctrl[data-key]")) {
    group.addEventListener("click", (e) => {
      const b = e.target.closest("button");
      if (!b || b.disabled) return;
      const key = group.dataset.key;
      opts[key] = key === "scale" ? +b.dataset.v : b.dataset.v;
      sync();
    });
  }
  $("#rife").addEventListener("change", (e) => { opts.interpolation = e.target.checked; sync(); });
  $("#format").addEventListener("change", (e) => { opts.format = e.target.value; sync(); });
  $("#quality").addEventListener("change", (e) => { opts.quality = e.target.value; sync(); });
  $("#startBtn").addEventListener("click", start);
  sync();
  for (const id of recentJobs()) addJob(id, false);
  watchQueue(status);

  const m = new URLSearchParams(location.search).get("m");
  if (m) api(`/api/media/${m}`).then((x) => x.status === "ready" && showMedia(x)).catch(() => {});
}

init();
