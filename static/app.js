import {
  $, el, api, postJSON, toast, fmtDuration, detectCuts, makeDropzone, uploadAndPrepare,
  depthOptionsForm, jobCard, recentJobs, audioToggle, watchQueue,
} from "./common.js";

let status;
let media = null;
let depthForm;

const jobsBox = $("#jobs");
const syncEmpty = () => ($("#jobsEmpty").hidden = jobsBox.children.length > 0);

function addJob(id, prepend = true) {
  const card = jobCard(id, { onGone: syncEmpty, kind: "depth" });
  prepend ? jobsBox.prepend(card) : jobsBox.append(card);
  syncEmpty();
}

function cuts() {
  return detectCuts(media.scores, +$("#cutThreshold").value, status.cut_min_shot);
}

function syncCuts() {
  $("#cutThresholdVal").textContent = $("#cutThreshold").value;
  $("#cutInfo").textContent = `샷 ${cuts().length + 1}개 감지 — 샷마다 밝기 범위를 따로 맞춥니다. 컷이 아닌데 끊기면 값을 올리세요.`;
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
    el("div", { class: "row" },
      m.editor
        ? el("a", { class: "btn", href: `/editor?m=${m.id}` }, "이 영상으로 프레임 편집기 열기 ", el("span", { class: "beta" }, "BETA"))
        : el("span", { class: "note" }, `편집기는 ${m.max_editor_frames}프레임 이하 영상만 지원합니다.`),
      el("button", { onclick: reset }, "다른 영상")),
  );
  $("#uploadBox").hidden = true;
  $("#optionsCard").hidden = false;
  $("#cutThreshold").value = status.cut_threshold;
  syncCuts();
  audioToggle($("#outAudio"), m);
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
  const outputs = { depth_mp4: $("#outMp4").checked, depth_png: $("#outPng").checked, stepped_mp4: false, audio: $("#outAudio").checked };
  if (!outputs.depth_mp4 && !outputs.depth_png) return toast("출력을 하나 이상 선택하세요.");
  $("#startBtn").disabled = true;
  try {
    const job = await postJSON("/api/jobs", { media_id: media.id, outputs, depth: depthForm.get(), cuts: cuts() });
    recentJobs(job.id);
    addJob(job.id);
    toast("대기열에 추가했습니다.");
  } catch (e) {
    toast(e.message, 4000);
  } finally {
    $("#startBtn").disabled = false;
  }
}

const dropzone = $("#dropzone");

async function init() {
  status = await api("/api/status");
  $("#ttl").textContent = status.ttl_hours;
  $("#uploadHint").textContent = `mp4 · mov · mkv · webm — 최대 ${status.max_upload_mb >= 1024 ? `${(status.max_upload_mb / 1024).toFixed(0)}GB` : `${status.max_upload_mb}MB`}`;
  makeDropzone(dropzone, onFile);
  depthForm = depthOptionsForm($("#depthOptions"), status);
  $("#cutThreshold").addEventListener("input", syncCuts);
  $("#startBtn").addEventListener("click", start);
  for (const id of recentJobs()) addJob(id, false);
  watchQueue(status);

  const m = new URLSearchParams(location.search).get("m");
  if (m) api(`/api/media/${m}`).then((x) => x.status === "ready" && showMedia(x)).catch(() => {});
}

init();
