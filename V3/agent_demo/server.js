"use strict";

const http = require("node:http");
const fs = require("node:fs");
const path = require("node:path");
const os = require("node:os");
const crypto = require("node:crypto");
const { spawn } = require("node:child_process");

const HOST = process.env.HOST || "127.0.0.1";
const PORT = Number(process.env.PORT || 8787);
const PUBLIC_DIR = path.join(__dirname, "public");
const V3_ROOT = path.resolve(__dirname, "..");
const MODEL_DIR = path.resolve(process.env.V3_MODEL_DIR || path.join(V3_ROOT, "models", "final_best_export"));
const PYTHON_BIN = process.env.PYTHON_BIN || "python";
const INFERENCE_SCRIPT = path.join(V3_ROOT, "scripts", "infer_ocsr_transformers.py");
const MAX_BODY_BYTES = 15 * 1024 * 1024;
const history = [];

const MIME = {
  ".html": "text/html; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".svg": "image/svg+xml",
  ".png": "image/png",
};

function modelAvailable() {
  return fs.existsSync(MODEL_DIR) && fs.existsSync(path.join(MODEL_DIR, "config.json"));
}

function sha12(value) {
  return crypto.createHash("sha256").update(value).digest("hex").slice(0, 12);
}

function parseDataUrl(value) {
  const match = /^data:(image\/(?:png|jpeg|webp));base64,([A-Za-z0-9+/=]+)$/.exec(String(value || ""));
  if (!match) throw new Error("仅支持 PNG、JPEG 或 WebP 的 base64 图片");
  const buffer = Buffer.from(match[2], "base64");
  if (!buffer.length || buffer.length > MAX_BODY_BYTES) throw new Error("图片为空或超过 15 MB");
  return { mime: match[1], buffer };
}

function validateSmilesLight(input) {
  const text = String(input || "").replace(/\s+/g, "");
  const balanced = (open, close) => [...text].filter((x) => x === open).length === [...text].filter((x) => x === close).length;
  const allowed = /^[A-Za-z0-9@+\-\[\]()=#%\\/.]+$/;
  const issues = [];
  if (!text) issues.push("empty");
  if (text.includes(".")) issues.push("multi_fragment");
  if (!allowed.test(text)) issues.push("unsupported_character");
  if (!balanced("(", ")") || !balanced("[", "]")) issues.push("unbalanced_delimiter");
  return {
    input: text,
    valid: issues.length === 0,
    validator: "lexical_fallback",
    issues,
    note: "轻量检查不替代 RDKit 化学解析；真实推理候选由 V3 脚本调用 RDKit 校验。",
  };
}

function clamp(value, min, max, fallback) {
  const number = Number(value);
  return Number.isFinite(number) ? Math.min(max, Math.max(min, number)) : fallback;
}

function normalizeSettings(settings = {}) {
  return {
    beams: Math.round(clamp(settings.beams, 1, 4, 4)),
    returns: Math.round(clamp(settings.returns, 1, 4, 4)),
    max_tokens: Math.round(clamp(settings.max_tokens, 64, 512, 256)),
    tta: settings.tta === true,
  };
}

function imageDiagnostics(payload, parsed) {
  const client = payload.client_diagnostics || {};
  const width = Math.round(clamp(client.width, 0, 12000, 0));
  const height = Math.round(clamp(client.height, 0, 12000, 0));
  const brightness = clamp(client.brightness, 0, 255, null);
  const contrast = clamp(client.contrast, 0, 128, null);
  const warnings = [];
  if (width && height && Math.min(width, height) < 256) warnings.push("短边低于 256 px");
  if (brightness !== null && (brightness < 45 || brightness > 235)) warnings.push("曝光可能影响键线识别");
  if (contrast !== null && contrast < 18) warnings.push("图像对比度偏低");
  return {
    format: parsed.mime.replace("image/", "").toUpperCase(),
    bytes: parsed.buffer.length,
    width,
    height,
    brightness,
    contrast,
    quality_gate: warnings.length ? "review" : "pass",
    warnings,
  };
}

function demoResult(payload, parsed) {
  const settings = normalizeSettings(payload.settings);
  const diagnostics = imageDiagnostics(payload, parsed);
  const started = Date.now();
  const isCaffeine = payload.sample_id === "caffeine-v1";
  const trace = [
    ["输入契约", "核对格式、大小与隐私策略", "pass", 8],
    ["视觉质检", diagnostics.warnings.length ? diagnostics.warnings.join("；") : "分辨率、亮度与对比度通过", diagnostics.quality_gate, 19],
    ["候选生成", isCaffeine ? `${settings.beams} beam × ${settings.tta ? "4 TTA" : "原图"}` : "等待 GPU 模型", isCaffeine ? "pass" : "blocked", isCaffeine ? 74 : 2],
    ["化学校验", isCaffeine ? "RDKit 规则口径；演示候选为预置已知结构" : "未生成候选", isCaffeine ? "pass" : "blocked", isCaffeine ? 14 : 0],
    ["一致性排序", isCaffeine ? "有效性 > 投票数 > 生成分数 > 结构惩罚" : "未执行", isCaffeine ? "pass" : "blocked", isCaffeine ? 11 : 0],
    ["证据汇总", isCaffeine ? "输出候选、选择理由与运行边界" : "返回模型配置提示", "pass", 5],
  ].map(([title, detail, status, duration_ms], index) => ({ index: index + 1, title, detail, status, duration_ms }));

  if (!isCaffeine) {
    return {
      id: `run_${Date.now().toString(36)}`,
      mode: "model_unavailable",
      status: "needs_model",
      notice: "当前未加载 GPU 模型。请设置 V3_MODEL_DIR 与 PYTHON_BIN 后重启；系统不会为任意上传图伪造结果。",
      prediction: "",
      canonical_prediction: "",
      valid: false,
      confidence: null,
      candidates: [],
      diagnostics,
      settings,
      trace,
      elapsed_ms: Date.now() - started + trace.reduce((sum, item) => sum + item.duration_ms, 0),
    };
  }

  const candidates = [
    { rank: 1, prediction: "Cn1c(=O)n(C)c2ncn(C)c2c1=O", canonical: "Cn1c(=O)n(C)c2ncn(C)c2c1=O", valid: true, votes: 3, score: -0.41, penalty: 0 },
    { rank: 2, prediction: "CN1C(=O)N(C)c2ncn(C)c2C1=O", canonical: "Cn1c(=O)n(C)c2ncn(C)c2c1=O", valid: true, votes: 1, score: -0.57, penalty: 0 },
    { rank: 3, prediction: "Cn1c(=O)n(C)c2nc[nH]c2c1=O", canonical: "Cn1c(=O)[nH]c2ncn(C)c2c1=O", valid: true, votes: 1, score: -1.03, penalty: 0 },
    { rank: 4, prediction: "Cn1c(=O)n(C)c2ncn(C)c2c1O", canonical: "", valid: false, votes: 0, score: -1.48, penalty: 4 },
  ];
  return {
    id: `run_${Date.now().toString(36)}`,
    mode: "guided_demo",
    status: "completed",
    notice: "内置咖啡因样例用于展示完整 Agent 工作流；结果来自已知标签，不计作在线模型推理证据。",
    prediction: candidates[0].prediction,
    canonical_prediction: candidates[0].canonical,
    valid: true,
    confidence: 0.88,
    selection_reason: "3/4 有效候选 canonical 一致；按有效投票与生成分数选择。",
    candidates,
    diagnostics,
    settings,
    trace,
    elapsed_ms: Date.now() - started + trace.reduce((sum, item) => sum + item.duration_ms, 0),
  };
}

function runInference(payload, parsed) {
  return new Promise((resolve, reject) => {
    const suffix = parsed.mime === "image/png" ? ".png" : parsed.mime === "image/webp" ? ".webp" : ".jpg";
    const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), "ocsr-agent-"));
    const imagePath = path.join(tempDir, `input${suffix}`);
    fs.writeFileSync(imagePath, parsed.buffer);
    const settings = normalizeSettings(payload.settings);
    const args = [
      INFERENCE_SCRIPT,
      "--model-dir", MODEL_DIR,
      "--image", imagePath,
      "--num-beams", String(settings.beams),
      "--num-return-sequences", String(settings.returns),
      "--max-new-tokens", String(settings.max_tokens),
      "--tta-preset", settings.tta ? "light" : "none",
      "--save-candidates",
      "--device", process.env.V3_DEVICE || "auto",
    ];
    const started = Date.now();
    const child = spawn(PYTHON_BIN, args, { windowsHide: true });
    let stdout = "";
    let stderr = "";
    child.stdout.on("data", (chunk) => { stdout += chunk; });
    child.stderr.on("data", (chunk) => { stderr += chunk; });
    const timeout = setTimeout(() => child.kill(), Number(process.env.V3_INFERENCE_TIMEOUT_MS || 180000));
    child.on("error", reject);
    child.on("close", (code) => {
      clearTimeout(timeout);
      fs.rmSync(tempDir, { recursive: true, force: true });
      if (code !== 0) return reject(new Error(stderr.trim() || `推理进程退出码 ${code}`));
      try {
        const raw = JSON.parse(stdout.slice(stdout.indexOf("{")));
        const diagnostics = imageDiagnostics(payload, parsed);
        const candidates = (raw.candidates || []).map((item, index) => ({
          rank: index + 1,
          prediction: item.prediction || "",
          canonical: item.canonical_prediction || "",
          valid: Boolean(item.canonical_prediction),
          votes: item.vote_count || 0,
          score: item.generation_score,
          penalty: item.smiles_structure_penalty,
        }));
        resolve({
          id: `run_${Date.now().toString(36)}`,
          mode: "gpu_model",
          status: "completed",
          notice: "结果由本地 V3 模型生成，并按 RDKit 有效性、投票和生成分数排序。",
          prediction: raw.prediction || "",
          canonical_prediction: raw.canonical_prediction || "",
          valid: Boolean(raw.canonical_prediction),
          confidence: null,
          selection_reason: raw.selection_reason || "V3 candidate rerank",
          candidates,
          diagnostics,
          settings,
          trace: [
            { index: 1, title: "输入契约", detail: "图片格式与大小通过", status: "pass", duration_ms: 4 },
            { index: 2, title: "视觉质检", detail: diagnostics.warnings.join("；") || "质量门通过", status: diagnostics.quality_gate, duration_ms: 9 },
            { index: 3, title: "候选生成", detail: `${settings.beams} beam / ${settings.returns} return`, status: "pass", duration_ms: Date.now() - started },
            { index: 4, title: "化学校验", detail: "RDKit canonicalization", status: "pass", duration_ms: 0 },
            { index: 5, title: "一致性排序", detail: raw.selection_reason || "valid vote score rerank", status: "pass", duration_ms: 0 },
            { index: 6, title: "证据汇总", detail: "候选与边界已记录", status: "pass", duration_ms: 1 },
          ],
          elapsed_ms: Date.now() - started,
        });
      } catch (error) {
        reject(new Error(`无法解析推理输出：${error.message}`));
      }
    });
  });
}

function json(res, status, value) {
  const body = JSON.stringify(value);
  res.writeHead(status, { "Content-Type": "application/json; charset=utf-8", "Content-Length": Buffer.byteLength(body), "Cache-Control": "no-store" });
  res.end(body);
}

function readJson(req) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    let size = 0;
    req.on("data", (chunk) => {
      size += chunk.length;
      if (size > MAX_BODY_BYTES) {
        reject(new Error("请求超过 15 MB"));
        req.destroy();
      } else chunks.push(chunk);
    });
    req.on("end", () => {
      try { resolve(JSON.parse(Buffer.concat(chunks).toString("utf8") || "{}")); }
      catch { reject(new Error("JSON 请求格式无效")); }
    });
    req.on("error", reject);
  });
}

function serveStatic(req, res) {
  const requested = decodeURIComponent(new URL(req.url, "http://localhost").pathname);
  const relative = requested === "/" ? "index.html" : requested.replace(/^\/+/, "");
  const filePath = path.resolve(PUBLIC_DIR, relative);
  if (!filePath.startsWith(PUBLIC_DIR) || !fs.existsSync(filePath) || fs.statSync(filePath).isDirectory()) {
    res.writeHead(404); res.end("Not found"); return;
  }
  res.writeHead(200, { "Content-Type": MIME[path.extname(filePath)] || "application/octet-stream", "Cache-Control": "no-cache" });
  fs.createReadStream(filePath).pipe(res);
}

async function handler(req, res) {
  const url = new URL(req.url, "http://localhost");
  try {
    if (req.method === "GET" && url.pathname === "/api/health") {
      return json(res, 200, { status: "ok", service: "ocsr-agent", version: "3.0.0", model_available: modelAvailable(), history_count: history.length });
    }
    if (req.method === "GET" && url.pathname === "/api/model") {
      return json(res, 200, {
        model: "PaddleOCR-VL-1.5 OCSR V3 final",
        model_available: modelAvailable(),
        model_dir: modelAvailable() ? MODEL_DIR : null,
        checkpoint: "checkpoint-1400",
        decoder: "beam4 / return4",
        development_exact: 0.4207,
        locked_wild_exact: 0.2292,
        hf_revision: "e496110ec222c1a70ebca287990c07dae47a2daa",
      });
    }
    if (req.method === "GET" && url.pathname === "/api/history") return json(res, 200, { items: history });
    if (req.method === "DELETE" && url.pathname === "/api/history") { history.splice(0); return json(res, 200, { cleared: true }); }
    if (req.method === "POST" && url.pathname === "/api/validate") {
      const body = await readJson(req);
      return json(res, 200, validateSmilesLight(body.smiles));
    }
    if (req.method === "POST" && url.pathname === "/api/agent/run") {
      const body = await readJson(req);
      const parsed = parseDataUrl(body.image);
      let result;
      if (modelAvailable()) {
        try { result = await runInference(body, parsed); }
        catch (error) {
          result = demoResult({ ...body, sample_id: "" }, parsed);
          result.status = "error";
          result.notice = `真实模型推理失败：${error.message}`;
        }
      } else result = demoResult(body, parsed);
      const record = {
        id: result.id,
        time: new Date().toISOString(),
        filename: String(body.filename || "upload").slice(0, 120),
        fingerprint: sha12(parsed.buffer),
        mode: result.mode,
        status: result.status,
        prediction: result.canonical_prediction || result.prediction || "",
        valid: result.valid,
        elapsed_ms: result.elapsed_ms,
      };
      history.unshift(record);
      history.splice(20);
      return json(res, result.status === "error" ? 500 : 200, result);
    }
    if (url.pathname.startsWith("/api/")) return json(res, 404, { error: "API not found" });
    return serveStatic(req, res);
  } catch (error) {
    return json(res, 400, { error: error.message || "请求失败" });
  }
}

function createServer() { return http.createServer(handler); }

if (require.main === module) {
  createServer().listen(PORT, HOST, () => {
    console.log(`OCSR Agent running at http://${HOST}:${PORT}`);
    console.log(`Model mode: ${modelAvailable() ? "GPU model" : "guided demo"}`);
  });
}

module.exports = { createServer, parseDataUrl, validateSmilesLight, normalizeSettings, demoResult };
