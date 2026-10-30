import {
  $, el, api, postJSON, toast, fmtDuration, detectCuts, makeDropzone, uploadAndPrepare,
  depthOptionsForm, jobCard, recentJobs, audioToggle,
} from "./common.js";

let status;
let media = null;
let depthForm;

const jobsBox = $("#jobs");
const syncEmpty = () => ($("#jobsEmpty").hidden = jobsBox.children.length > 0);

function addJob(id, prepend = true) {
  const card = jobCard(id, { onGone: syncEmpty });
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

function queueRow(j, now, mine) {
  const isMine = mine.has(j.tag);
  const work = [
    j.resolution,
    j.drawings !== j.frames ? `${j.frames}프레임 (그림 ${j.drawings})` : `${j.frames}프레임`,
    j.model,
    j.outputs.join(" + "),
  ].filter(Boolean).join(" · ");
  const cpu = j.lane === "cpu";
  let state;
  if (j.status === "running") {
    state = `${cpu ? "CPU · " : ""}${j.stage} · ${(j.progress * 100).toFixed(0)}% · ${fmtDuration(now - j.started)} 경과`;
  } else if (j.status === "queued") {
    state = cpu ? "CPU 작업 · 곧 시작 (GPU 대기열과 별도)" : `${fmtDuration(now - j.created)}째 대기 중`;
  } else {
    const label = { done: "완료", error: "실패", cancelled: "취소" }[j.status];
    const took = j.started ? ` · ${fmtDuration(j.finished - j.started)} 걸림` : "";
    state = `${label}${took} · ${fmtDuration(now - j.finished)} 전`;
  }
  const parts = [
    el("div", { class: "qpos" }, j.status === "running" ? "▶" : j.status === "queued" ? (cpu ? "·" : String(j.position)) : "✓"),
    el("div", { class: "qname" }, el("span", { title: j.name }, j.name || "(만료된 영상)"), isMine ? el("span", { class: "mine-tag" }, "내 작업") : null),
    el("div", { class: "qmeta" }, state),
  ];
  if (j.status === "running") {
    const bar = el("div", { class: "progress" }, el("div"));
    bar.firstChild.style.width = `${(j.progress * 100).toFixed(1)}%`;
    parts.push(bar);
  }
  parts.push(el("div", { class: "qmeta" }, work));
  const cls = ["qrow", j.status === "running" ? "running" : "", j.status === "queued" ? "" : j.status === "running" ? "" : "done-row", isMine ? "mine" : ""];
  return el("div", { class: cls.filter(Boolean).join(" ") }, parts);
}

async function refreshQueue() {
  let q;
  try {
    q = await api("/api/queue");
  } catch {
    $("#serverStatus").textContent = "서버 연결 끊김";
    return;
  }
  const busy = q.running.length + q.queued.length;
  $("#serverStatus").textContent = `${status.gpu} · ${busy ? `처리 중 ${q.running.length} · 대기 ${q.queued.length}` : "대기 없음"}`;
  $("#queueSummary").textContent = busy ? `— 처리 중 ${q.running.length}건, 대기 ${q.queued.length}건` : "— 비어 있음";

  const mine = new Set(recentJobs().map((id) => id.slice(0, 8)));
  const list = [];
  list.push(el("h4", {}, "처리 중 · 대기"));
  if (busy) {
    for (const j of [...q.running, ...q.queued]) list.push(queueRow(j, q.now, mine));
  } else {
    list.push(el("div", { class: "note" }, "지금 돌고 있는 작업이 없습니다. 변환을 시작하면 바로 처리됩니다."));
  }
  if (q.recent.length) {
    list.push(el("h4", {}, "최근 끝난 작업"));
    for (const j of q.recent) list.push(queueRow(j, q.now, mine));
  }
  $("#queueList").replaceChildren(...list);
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
  refreshQueue();
  setInterval(refreshQueue, 2000);

  const m = new URLSearchParams(location.search).get("m");
  if (m) api(`/api/media/${m}`).then((x) => x.status === "ready" && showMedia(x)).catch(() => {});
}

init();
