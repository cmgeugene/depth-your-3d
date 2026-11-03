import {
  $, el, api, postJSON, toast, fmtDuration, makeDropzone, uploadAndPrepare,
  jobCard, recentJobs, audioToggle, watchQueue, store,
} from "./common.js";

const DEFAULTS = {
  mode: "anime", scale: 2, backend: "ai", denoise: "none", format: "mp4_h265", quality: "balanced",
  interpolation: false, image_format: "png", image_quality: 92,
};
const VIDEO_EXT = /\.(mp4|mov|mkv|avi|webm|m4v|ts|mts)$/i;
const IMAGE_EXT = /\.(png|jpe?g|webp|bmp|tiff?)$/i;
const MAX_IMAGES = 200;

let status;
let media = null; // the video, or the first image of a batch
let images = []; // image batch (empty for a video)
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
  const href = `/api/jobs/${j.id}/files/${encodeURIComponent(out.name)}`;
  if (!VIDEO_EXT.test(out.name)) {
    return [el("a", { class: "btn primary", href, download: out.name }, out.name.endsWith(".zip") ? "⬇ 이미지 zip 다운로드" : "⬇ 이미지 다운로드")];
  }
  const editorBtn = el("button", { onclick: () => toEditor(j, editorBtn) }, "프레임 편집기로 보내기");
  return [el("a", { class: "btn primary", href, download: out.name }, "⬇ 영상 다운로드"), editorBtn];
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
  $("#imgQuality").value = String(opts.image_quality);
  $("#imgQuality").disabled = opts.image_format === "png";
  store("ds-upscale-opts", opts);

  const missing = [!eng.realesrgan && "Real-ESRGAN", !eng.rife && !images.length && "RIFE"].filter(Boolean);
  $("#engineWarn").hidden = !missing.length;
  $("#engineWarn").textContent = missing.length
    ? `서버에 ${missing.join(", ")} 엔진이 설치되지 않아 해당 기능을 쓸 수 없습니다. 관리자에게 setup-upscale-engines.bat 실행을 요청하세요.`
    : "";
  if (!media) return;

  const warns = [];
  let tooBig;
  if (images.length) {
    const big = images.filter((m) => Math.max(m.width, m.height) * opts.scale > eng.max_side);
    tooBig = big.length > 0;
    const w = media.width * opts.scale, h = media.height * opts.scale;
    const fmt = { png: "PNG", jpg: "JPG", webp: "WebP" }[opts.image_format];
    $("#outInfo").textContent = `출력 ${w}×${h}${images.length > 1 ? ` 외 ${images.length - 1}장 (zip으로 받음)` : ""} · ${fmt} · 모델 ${modelName()}`;
    if (tooBig) warns.push(`${big.length}장의 출력 긴 변이 서버 한도(${eng.max_side}px)를 넘습니다. 배율을 낮추세요.`);
    if (opts.image_format === "jpg" && images.some((m) => m.has_alpha)) warns.push("JPG는 투명도를 지원하지 않아 투명한 부분이 흰색으로 채워집니다.");
    if (opts.denoise !== "none" && images.some((m) => m.has_alpha)) warns.push("투명도가 있는 이미지는 노이즈 제거를 건너뜁니다.");
  } else {
    const w = media.width * opts.scale, h = media.height * opts.scale;
    const fps = media.fps * (opts.interpolation ? 2 : 1);
    $("#outInfo").textContent = `출력 ${w}×${h} · ${+fps.toFixed(3)}fps · 모델 ${modelName()}`;
    tooBig = Math.max(w, h) > eng.max_side;
    if (tooBig) warns.push(`출력 긴 변이 서버 한도(${eng.max_side}px)를 넘습니다. 배율을 낮추세요.`);
  }
  if (opts.backend === "ai" && opts.mode === "real") warns.unshift("실사 모드는 2D 애니메이션에서 피부가 매끈해지고 선이 바뀔 수 있습니다.");
  $("#outWarn").hidden = !warns.length;
  $("#outWarn").textContent = warns.join(" ");
  $("#startBtn").disabled = tooBig;
}

function setKind(kind) {
  document.body.classList.toggle("upscale-image", kind === "image");
  document.body.classList.toggle("upscale-video", kind === "video");
}

function showMedia(m) {
  media = m;
  images = [];
  setKind("video");
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
    el("div", { class: "row" }, el("button", { onclick: reset }, "다른 파일")),
  );
  $("#uploadBox").hidden = true;
  $("#optionsCard").hidden = false;
  audioToggle($("#outAudio"), m);
  sync();
}

function showImages(list) {
  images = list;
  media = list[0];
  setKind("image");
  const box = $("#mediaBox");
  box.hidden = false;
  const sizes = [...new Set(list.map((m) => `${m.width}×${m.height}`))];
  const shown = list.slice(0, 11);
  box.replaceChildren(
    el("div", { class: "thumb-strip" },
      shown.map((m) => el("img", { src: `/api/media/${m.id}/image`, alt: "", title: m.name, loading: "lazy" })),
      list.length > shown.length ? el("span", {}, `+${list.length - shown.length}`) : null),
    el("dl", { class: "kv" },
      el("dt", {}, "이미지"), el("dd", {}, list.length > 1 ? `${list.length}장` : list[0].name),
      el("dt", {}, "해상도"), el("dd", {}, sizes.length > 3 ? `${sizes.slice(0, 3).join(", ")} 외` : sizes.join(", ")),
      el("dt", {}, "투명도"), el("dd", {}, list.some((m) => m.has_alpha) ? "있는 이미지 포함" : "없음")),
    el("div", { class: "row" }, el("button", { onclick: reset }, "다른 파일")),
  );
  $("#uploadBox").hidden = true;
  $("#optionsCard").hidden = false;
  sync();
}

function reset() {
  media = null;
  images = [];
  document.body.classList.remove("upscale-image", "upscale-video");
  $("#mediaBox").hidden = true;
  $("#optionsCard").hidden = true;
  $("#uploadBox").hidden = false;
  $("#uploadBox").replaceChildren(dropzone);
}

async function onFiles(files) {
  const box = $("#uploadBox");
  const holder = el("div", { class: "stack" });
  const fail = (msg) => holder.replaceChildren(el("div", { class: "job-err" }, msg), el("button", { onclick: reset }, "다시 시도"));
  box.replaceChildren(holder);
  const videos = files.filter((f) => VIDEO_EXT.test(f.name) || f.type.startsWith("video/"));
  const pics = files.filter((f) => IMAGE_EXT.test(f.name));
  if (videos.length && (files.length > 1)) return fail("영상은 한 번에 1개만 올릴 수 있습니다. 이미지는 여러 장을 함께 올릴 수 있습니다.");
  if (!videos.length && pics.length !== files.length) return fail("지원하지 않는 파일이 섞여 있습니다 (png · jpg · webp · bmp · tif만 가능).");
  if (pics.length > MAX_IMAGES) return fail(`이미지는 한 번에 ${MAX_IMAGES}장까지 올릴 수 있습니다.`);
  try {
    if (videos.length) return showMedia(await uploadAndPrepare(videos[0], holder));
    const done = [];
    for (const [i, f] of pics.entries()) {
      const note = pics.length > 1 ? `이미지 ${i + 1}/${pics.length} 확인 중…` : "이미지 확인 중…";
      done.push(await uploadAndPrepare(f, holder, note));
    }
    showImages(done);
  } catch (e) {
    fail(e.message);
  }
}

async function start() {
  $("#startBtn").disabled = true;
  try {
    const body = images.length
      ? { media_id: images[0].id, media_ids: images.map((m) => m.id), scale: opts.scale, mode: opts.mode,
          backend: opts.backend, denoise: opts.denoise, image_format: opts.image_format, image_quality: opts.image_quality }
      : { media_id: media.id, ...opts, audio: $("#outAudio").checked };
    const job = await postJSON("/api/upscale-jobs", body);
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
  const max = status.max_upload_mb >= 1024 ? `${(status.max_upload_mb / 1024).toFixed(0)}GB` : `${status.max_upload_mb}MB`;
  $("#uploadHint").textContent = `영상 mp4 · mov · mkv · webm (최대 ${max}) / 이미지 png · jpg · webp · bmp · tif (최대 ${MAX_IMAGES}장)`;
  makeDropzone(dropzone, onFiles, "video/*,.mp4,.mov,.mkv,.avi,.webm,image/*,.png,.jpg,.jpeg,.webp,.bmp,.tif,.tiff", true);
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
  $("#imgQuality").addEventListener("change", (e) => { opts.image_quality = +e.target.value; sync(); });
  $("#startBtn").addEventListener("click", start);
  sync();
  for (const id of recentJobs()) addJob(id, false);
  watchQueue(status);

  const m = new URLSearchParams(location.search).get("m");
  if (m) api(`/api/media/${m}`).then((x) => x.status === "ready" && x.kind !== "image" && showMedia(x)).catch(() => {});
}

init();
