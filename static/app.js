import {
  $, el, api, postJSON, toast, fmtDuration, detectCuts, makeDropzone, uploadAndPrepare,
  depthOptionsForm, jobCard, recentJobs, audioToggle, watchQueue, store,
} from "./common.js";

// kind: depth | clay | depth_clay. Clay settings apply to the two clay kinds.
const DEFAULTS = { kind: "depth", mode: "head", smooth: 2, bg_mode: "overlay", max_people: 2, hands: "full" };
const HANDS_NOTE = {
  full: "손을 따로 확대해 한 번 더 추정함. 물건을 쥔 손가락까지 맞음.",
  fast: "몸만 추정해 약 3배 빠름. 손은 반쯤 벌린 모양이 되어 쥔 모양이 약해짐.",
};
const KIND_NOTE = {
  depth: "실사 영상을 흑백 depth로 바꿈. 공간 · 실루엣 · 카메라가 넘어감.",
  clay: "인물을 회색 클레이 마네킹으로 바꿈 (SAM 3D Body). 몸 동작 · 고개 각도 · 카메라만 넘기고 배우 얼굴형은 넘기지 않음. 배경은 비어 있음.",
  depth_clay: "depth 위에 클레이 마네킹을 얹음. 방 배치 · 소품까지 넘어가고 얼굴형은 넘어가지 않음. Seedance v2v에서 지금까지 가장 결과가 좋은 조합. depth mp4도 함께 받음.",
};

let status;
let media = null;
let depthForm;
let opts = { ...DEFAULTS, ...(store("ds-motion-opts") || {}) };

const jobsBox = $("#jobs");
const syncEmpty = () => ($("#jobsEmpty").hidden = jobsBox.children.length > 0);

function addJob(id, prepend = true) {
  const card = jobCard(id, { onGone: syncEmpty, kind: ["depth", "clay"] });
  prepend ? jobsBox.prepend(card) : jobsBox.append(card);
  syncEmpty();
}

function cuts() {
  return detectCuts(media.scores, +$("#cutThreshold").value, status.cut_min_shot);
}

function syncCuts() {
  if (!media) return;
  $("#cutThresholdVal").textContent = $("#cutThreshold").value;
  const n = cuts().length + 1;
  const what = opts.kind === "depth" ? "샷마다 밝기 범위를 따로 맞춥니다"
    : opts.kind === "clay" ? "샷마다 인물 추적과 초점거리를 새로 잡습니다"
      : "샷마다 depth 밝기 범위, 인물 추적, 초점거리를 새로 잡습니다";
  $("#cutInfo").textContent = `샷 ${n}개 감지 — ${what}. 컷이 아닌데 끊기면(빠른 푸시인 등) 값을 올리세요.`;
}

// Shows the controls for the chosen result kind; remembers the clay settings per browser.
function sync() {
  const clayOk = status.clay && status.clay.available;
  if (!clayOk && opts.kind !== "depth") opts.kind = "depth";
  for (const group of document.querySelectorAll(".seg-ctrl[data-key]")) {
    for (const b of group.querySelectorAll("button")) {
      const on = String(opts[group.dataset.key]) === b.dataset.v;
      b.classList.toggle("on", on);
      b.setAttribute("aria-pressed", on);
      if (group.dataset.key === "kind" && b.dataset.v !== "depth") b.disabled = !clayOk;
    }
  }
  $("#clayWarn").hidden = clayOk;
  $("#clayWarn").textContent = clayOk ? "" : `이 서버에서는 클레이를 쓸 수 없습니다: ${status.clay ? status.clay.reason : ""}`;
  const k = opts.kind;
  for (const e of document.querySelectorAll(".k-depth")) e.hidden = k === "clay";
  for (const e of document.querySelectorAll(".k-clay")) e.hidden = k === "depth";
  for (const e of document.querySelectorAll(".k-both")) e.hidden = k !== "depth_clay";
  for (const e of document.querySelectorAll(".k-only-depth")) e.hidden = k !== "depth";
  $("#maxPeople").value = String(opts.max_people);
  $("#kindNote").textContent = KIND_NOTE[k];
  $("#handsNote").textContent = HANDS_NOTE[opts.hands] || "";
  const tooLong = media && k !== "depth" && media.frames > status.clay.max_frames;
  $("#startBtn").disabled = !!tooLong;
  $("#startNote").textContent = tooLong ? `클레이는 ${status.clay.max_frames}프레임까지 가능합니다.`
    : k !== "depth" && !status.clay.loaded ? "첫 클레이 작업은 모델을 불러오느라 1~3분 더 걸림." : "";
  $("#startBtn").textContent = k === "depth" ? "변환 시작" : "클레이 만들기";
  store("ds-motion-opts", opts);
  syncCuts();
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
  let path, body;
  if (opts.kind === "depth") {
    const outputs = { depth_mp4: $("#outMp4").checked, depth_png: $("#outPng").checked, stepped_mp4: false, audio: $("#outAudio").checked };
    if (!outputs.depth_mp4 && !outputs.depth_png) return toast("출력을 하나 이상 선택하세요.");
    path = "/api/jobs";
    body = { media_id: media.id, outputs, depth: depthForm.get(), cuts: cuts() };
  } else {
    path = "/api/clay-jobs";
    body = {
      media_id: media.id, mode: opts.mode, smooth: +opts.smooth, max_people: +opts.max_people, hands: opts.hands,
      bg: opts.kind === "depth_clay" ? "auto" : "none", bg_mode: opts.bg_mode, depth: depthForm.get(), cuts: cuts(),
    };
  }
  $("#startBtn").disabled = true;
  try {
    const job = await postJSON(path, body);
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
  depthForm = depthOptionsForm($("#depthOptions"), status);
  $("#cutThreshold").addEventListener("input", syncCuts);
  for (const group of document.querySelectorAll(".seg-ctrl[data-key]")) {
    group.addEventListener("click", (e) => {
      const b = e.target.closest("button");
      if (!b || b.disabled) return;
      const key = group.dataset.key;
      opts[key] = key === "smooth" ? +b.dataset.v : b.dataset.v;
      sync();
    });
  }
  $("#maxPeople").addEventListener("change", (e) => { opts.max_people = +e.target.value; sync(); });
  $("#startBtn").addEventListener("click", start);
  sync();
  for (const id of recentJobs()) addJob(id, false);
  watchQueue(status);

  const m = new URLSearchParams(location.search).get("m");
  if (m) api(`/api/media/${m}`).then((x) => x.status === "ready" && showMedia(x)).catch(() => {});
}

init();
