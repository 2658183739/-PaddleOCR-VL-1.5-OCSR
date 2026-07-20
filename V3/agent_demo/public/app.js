"use strict";

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];
const state = { image: "", filename: "", sampleId: "", diagnostics: {}, result: null };

const sampleSvg = `<svg xmlns="http://www.w3.org/2000/svg" width="960" height="700" viewBox="0 0 960 700">
  <rect width="960" height="700" fill="#fbfcfd"/>
  <g stroke="#17212d" stroke-width="9" stroke-linecap="round" stroke-linejoin="round" fill="none">
    <path d="M245 250 L370 174 L500 245 L495 392 L365 466 L240 392 Z"/>
    <path d="M500 245 L630 202 L718 305 L640 416 L495 392"/>
    <path d="M267 270 L370 208 M470 266 L466 370 M270 374 L365 430" stroke-width="5"/>
    <path d="M630 202 L646 111 M640 416 L700 498 M365 174 L322 92 M240 392 L148 441"/>
    <path d="M718 305 L808 305 M500 245 L548 145" stroke-width="7"/>
  </g>
  <g fill="#087f72" font-family="Arial, sans-serif" font-weight="700" font-size="42">
    <text x="350" y="190">N</text><text x="470" y="257">C</text><text x="475" y="410">N</text>
    <text x="610" y="220">N</text><text x="695" y="320">N</text><text x="620" y="432">C</text>
  </g>
  <g fill="#b42318" font-family="Arial, sans-serif" font-weight="700" font-size="38">
    <text x="300" y="80">O</text><text x="120" y="470">O</text>
  </g>
  <g fill="#17212d" font-family="Arial, sans-serif" font-size="27">
    <text x="290" y="64">CH₃</text><text x="685" y="535">CH₃</text><text x="812" y="315">CH₃</text>
  </g>
  <text x="480" y="625" text-anchor="middle" fill="#667185" font-family="Arial, sans-serif" font-size="22">CAFFEINE · GUIDED DEMO SAMPLE</text>
</svg>`;

function setRuntime(modelAvailable) {
  $("#runtimeDot").classList.add("online");
  $("#runtimeText").textContent = modelAvailable ? "V3 GPU 模型就绪" : "引导演示模式";
}

async function api(url, options = {}) {
  const response = await fetch(url, { headers: { "Content-Type": "application/json" }, ...options });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || data.notice || "请求失败");
  return data;
}

function switchView(name) {
  $$(".tab").forEach((tab) => tab.classList.toggle("active", tab.dataset.view === name));
  $$(".view").forEach((view) => view.classList.remove("active"));
  $(`#${name}View`).classList.add("active");
  if (name === "history") loadHistory();
}

async function inspectImage(dataUrl) {
  const image = new Image();
  await new Promise((resolve, reject) => { image.onload = resolve; image.onerror = reject; image.src = dataUrl; });
  const canvas = $("#analysisCanvas");
  canvas.width = 128;
  canvas.height = 128;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.clearRect(0, 0, 128, 128);
  ctx.fillStyle = "#fff";
  ctx.fillRect(0, 0, 128, 128);
  ctx.drawImage(image, 0, 0, 128, 128);
  const pixels = ctx.getImageData(0, 0, 128, 128).data;
  let sum = 0;
  const values = [];
  for (let i = 0; i < pixels.length; i += 4) {
    const lum = .2126 * pixels[i] + .7152 * pixels[i + 1] + .0722 * pixels[i + 2];
    values.push(lum); sum += lum;
  }
  const brightness = sum / values.length;
  const contrast = Math.sqrt(values.reduce((acc, value) => acc + (value - brightness) ** 2, 0) / values.length);
  return { width: image.naturalWidth, height: image.naturalHeight, brightness: Math.round(brightness), contrast: Math.round(contrast) };
}

async function loadDataUrl(dataUrl, filename, sampleId = "") {
  state.image = dataUrl;
  state.filename = filename;
  state.sampleId = sampleId;
  state.diagnostics = await inspectImage(dataUrl);
  $("#preview").src = dataUrl;
  $("#dropzone").classList.add("has-image");
  $("#diagSize").textContent = `${state.diagnostics.width} × ${state.diagnostics.height}`;
  $("#diagBrightness").textContent = `${state.diagnostics.brightness} / 255`;
  $("#diagContrast").textContent = `${state.diagnostics.contrast} σ`;
  const review = Math.min(state.diagnostics.width, state.diagnostics.height) < 256 || state.diagnostics.contrast < 18;
  $("#diagGate").textContent = review ? "需复核" : "通过";
  $("#diagGate").className = `gate ${review ? "review" : "pass"}`;
  $("#runBtn").disabled = false;
  resetResult(false);
}

async function loadFile(file) {
  if (!file || !/^image\/(png|jpeg|webp)$/.test(file.type) || file.size > 15 * 1024 * 1024) {
    alert("请选择 15 MB 以内的 PNG、JPEG 或 WebP 图片。"); return;
  }
  const reader = new FileReader();
  reader.onload = () => loadDataUrl(reader.result, file.name);
  reader.readAsDataURL(file);
}

function loadSample() {
  const encoded = `data:image/svg+xml;base64,${btoa(unescape(encodeURIComponent(sampleSvg)))}`;
  const image = new Image();
  image.onload = () => {
    const canvas = document.createElement("canvas");
    canvas.width = 960;
    canvas.height = 700;
    canvas.getContext("2d").drawImage(image, 0, 0);
    loadDataUrl(canvas.toDataURL("image/png"), "caffeine_demo.png", "caffeine-v1");
  };
  image.src = encoded;
}

function renderTrace(items) {
  const nodes = $$("#trace li");
  items.forEach((item, index) => {
    const node = nodes[index];
    node.className = item.status;
    node.querySelector("small").textContent = item.detail;
    node.querySelector("em").textContent = item.duration_ms ? `${item.duration_ms} ms` : item.status;
  });
}

function renderCandidates(items) {
  $("#candidateRows").innerHTML = items.length ? items.map((item) => `<tr>
    <td><strong>#${item.rank}</strong></td><td>${escapeHtml(item.prediction)}</td>
    <td><span class="${item.valid ? "valid-chip" : "invalid-chip"}">${item.valid ? "VALID" : "REJECT"}</span></td>
    <td>${item.votes ?? "—"}</td><td>${typeof item.score === "number" ? item.score.toFixed(2) : "—"}</td><td>${item.penalty ?? "—"}</td>
  </tr>`).join("") : `<tr class="placeholder-row"><td colspan="6">当前运行没有候选结果</td></tr>`;
}

function renderResult(result) {
  state.result = result;
  $("#emptyResult").hidden = true;
  $("#resultContent").hidden = false;
  $("#modeBadge").textContent = result.mode === "gpu_model" ? "GPU MODEL" : result.mode === "guided_demo" ? "GUIDED DEMO" : "MODEL REQUIRED";
  $("#validityBadge").textContent = result.valid ? "RDKit 有效" : "无有效结果";
  $("#validityBadge").className = `validity ${result.valid ? "" : "invalid"}`;
  $("#elapsed").textContent = `${result.elapsed_ms} ms`;
  $("#smilesOutput").textContent = result.canonical_prediction || result.prediction || "—";
  const confidence = result.confidence === null ? 0 : Math.round(result.confidence * 100);
  $("#confidenceBar").style.width = `${confidence}%`;
  $("#confidenceText").textContent = result.confidence === null ? "—" : `${confidence}%`;
  $("#selectionReason").textContent = result.selection_reason || "模型尚未加载，未执行候选选择。";
  $("#resultNotice").textContent = result.notice;
  renderTrace(result.trace || []);
  renderCandidates(result.candidates || []);
  loadHistoryCount();
}

async function runAgent() {
  if (!state.image) return;
  const button = $("#runBtn");
  button.disabled = true;
  button.querySelector("span").textContent = "Agent 正在执行…";
  try {
    const result = await api("/api/agent/run", { method: "POST", body: JSON.stringify({
      image: state.image.replace("image/svg+xml", "image/png"),
      filename: state.filename,
      sample_id: state.sampleId,
      client_diagnostics: state.diagnostics,
      settings: { beams: Number($("#beams").value), returns: Number($("#returns").value), max_tokens: Number($("#maxTokens").value), tta: $("#tta").checked },
    }) });
    renderResult(result);
  } catch (error) { alert(error.message); }
  finally { button.disabled = false; button.querySelector("span").textContent = "运行 OCSR Agent"; }
}

function resetResult(clearImage = false) {
  state.result = null;
  $("#emptyResult").hidden = false;
  $("#resultContent").hidden = true;
  $("#modeBadge").textContent = "未运行";
  renderCandidates([]);
  if (clearImage) {
    state.image = ""; state.filename = ""; state.sampleId = ""; state.diagnostics = {};
    $("#preview").src = ""; $("#dropzone").classList.remove("has-image"); $("#runBtn").disabled = true;
    $("#diagSize").textContent = $("#diagBrightness").textContent = $("#diagContrast").textContent = "—";
    $("#diagGate").textContent = "待输入"; $("#diagGate").className = "gate";
  }
  $$("#trace li").forEach((node) => { node.className = ""; node.querySelector("em").textContent = "等待"; });
}

async function loadHistoryCount() {
  try { const data = await api("/api/history"); $("#historyCount").textContent = data.items.length; }
  catch { /* service status already communicates availability */ }
}

async function loadHistory() {
  const data = await api("/api/history");
  $("#historyList").innerHTML = data.items.length ? data.items.map((item) => `<div class="history-item">
    <div><strong>${new Date(item.time).toLocaleString("zh-CN")}</strong><br><small>${escapeHtml(item.filename)}</small></div>
    <code>${escapeHtml(item.prediction || "无结果")}</code>
    <span>${item.mode.replaceAll("_", " ")}</span><span>${item.elapsed_ms} ms</span>
  </div>`).join("") : `<div class="history-empty">还没有运行记录</div>`;
  $("#historyCount").textContent = data.items.length;
}

function escapeHtml(value) { return String(value).replace(/[&<>"']/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;" })[char]); }

function exportJson() {
  if (!state.result) return;
  const blob = new Blob([JSON.stringify(state.result, null, 2)], { type: "application/json" });
  const link = document.createElement("a"); link.href = URL.createObjectURL(blob); link.download = `${state.result.id}.json`; link.click(); URL.revokeObjectURL(link.href);
}

$$('.tab').forEach((tab) => tab.addEventListener('click', () => switchView(tab.dataset.view)));
$("#sampleBtn").addEventListener("click", loadSample);
$("#fileInput").addEventListener("change", (event) => loadFile(event.target.files[0]));
$("#beams").addEventListener("input", () => { $("#beamOut").textContent = $("#beams").value; });
$("#runBtn").addEventListener("click", runAgent);
$("#clearBtn").addEventListener("click", () => resetResult(true));
$("#downloadBtn").addEventListener("click", exportJson);
$("#copyBtn").addEventListener("click", () => navigator.clipboard.writeText($("#smilesOutput").textContent));
$("#clearHistory").addEventListener("click", async () => { await api("/api/history", { method: "DELETE" }); loadHistory(); });
$("#dropzone").addEventListener("dragover", (event) => { event.preventDefault(); $("#dropzone").classList.add("drag"); });
$("#dropzone").addEventListener("dragleave", () => $("#dropzone").classList.remove("drag"));
$("#dropzone").addEventListener("drop", (event) => { event.preventDefault(); $("#dropzone").classList.remove("drag"); loadFile(event.dataTransfer.files[0]); });
document.addEventListener("keydown", (event) => { if (event.ctrlKey && event.key === "Enter") runAgent(); });

api("/api/health").then((data) => { setRuntime(data.model_available); loadHistoryCount(); }).catch(() => { $("#runtimeText").textContent = "后端未连接"; });
