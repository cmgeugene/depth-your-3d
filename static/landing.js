import { $, api, watchQueue } from "./common.js";

async function init() {
  const status = await api("/api/status");
  $("#ttl").textContent = status.ttl_hours;
  watchQueue(status);
}

init();
