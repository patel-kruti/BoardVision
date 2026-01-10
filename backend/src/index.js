import express from "express";
import cors from "cors";
import multer from "multer";
import path from "path";
import fs from "fs";
import { fileURLToPath } from "url";
import axios from "axios";
import { createJob, getJob, updateJob } from "./jobs.js";
import { buildPptxFromSemantic } from "./ppt.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const PORT = process.env.PORT ? Number(process.env.PORT) : 3001;
const MCP_URL = process.env.MCP_URL || "http://localhost:8080";

const uploadsDir = path.join(__dirname, "..", "uploads");
const outputsDir = path.join(__dirname, "..", "outputs");
fs.mkdirSync(uploadsDir, { recursive: true });
fs.mkdirSync(outputsDir, { recursive: true });

function safeBasename(p) {
  return path.basename(String(p || ""));
}

function resolveMaybeRelativePptPath(pptFile) {
  if (!pptFile) return null;
  const s = String(pptFile);

  // If MCP returns a relative path like "workdir\\slide_x.pptx", resolve from repo root
  if (!path.isAbsolute(s)) {
    const repoRoot = path.resolve(__dirname, "..", "..");
    return path.resolve(repoRoot, s);
  }
  return s;
}

const upload = multer({ dest: uploadsDir });

const app = express();
app.use(cors());
app.use(express.json({ limit: "10mb" }));
app.use("/downloads", express.static(outputsDir));

app.get("/health", (req, res) => res.json({ ok: true, service: "backend", port: PORT }));

// POST /process-image
// Accepts multipart/form-data: image=<file>
app.post("/process-image", upload.single("image"), async (req, res) => {
  const job = createJob();
  const imagePath = req.file?.path;

  if (!imagePath) {
    updateJob(job.id, { status: "error", error: "Missing image file" });
    return res.status(400).json({ error: "Missing image file" });
  }

  updateJob(job.id, { status: "processing", imagePath });
  res.json({ id: job.id });

  // Async processing: Step 1) detect_visuals (returns visual_json)
  (async () => {
    try {
      const mcpResp = await axios.post(`${MCP_URL}/tools/detect_visuals`, {
        file_id: imagePath
      });

      const visualJson = mcpResp.data?.visual_json || { elements: [] };

      updateJob(job.id, {
        status: "done",
        result: { visualJson }
      });
    } catch (e) {
      updateJob(job.id, {
        status: "error",
        error: e?.response?.data?.error || e?.message || String(e)
      });
    }
  })();
});

// GET /status/:id
app.get("/status/:id", (req, res) => {
  const job = getJob(req.params.id);
  if (!job) return res.status(404).json({ error: "Not found" });
  res.json({
    id: job.id,
    status: job.status,
    error: job.error,
    result: job.result,
    updatedAt: job.updatedAt
  });
});

// POST /generate-ppt
// body: { id: string, visualJson?: object }
// Behavior:
// - Step 2) reason_semantics(visual_json) via MCP
// - Step 3) render PPT locally (pptxgenjs) using semanticJson
app.post("/generate-ppt", async (req, res) => {
  const { id, visualJson } = req.body || {};
  if (!id) return res.status(400).json({ error: "Missing id" });

  const job = getJob(id);
  if (!job) return res.status(404).json({ error: "Unknown job id" });

  try {
    const vjson =
      (visualJson && typeof visualJson === "object" ? visualJson : null) ||
      job.result?.visualJson ||
      null;

    if (!vjson) return res.status(400).json({ error: "Missing visualJson (process image first)" });

    // Step 2: reason_semantics(visual_json) -> semantic_json
    const semResp = await axios.post(`${MCP_URL}/tools/reason_semantics`, {
      visual_json: vjson
    });
    const semanticJson = semResp.data?.semantic_json || {
      diagram_type: "flowchart",
      nodes: [],
      edges: [],
      summary: ""
    };

    // Step 3: render_ppt(semantic_json) -> ppt_file
    // If the MCP server is a proxy to local_mcp_new.py, this will produce the real PPT in its workdir.
    let renderedPptPath = null;
    try {
      const renderResp = await axios.post(`${MCP_URL}/tools/render_ppt`, {
        semantic_json: semanticJson
      });
      renderedPptPath = resolveMaybeRelativePptPath(renderResp.data?.ppt_file);
    } catch {
      // If MCP doesn't implement /tools/render_ppt, fall back to local JS PPT renderer.
    }

    const filename = `boardvision_${id}_${Date.now()}.pptx`;
    const outPath = path.join(outputsDir, filename);

    if (renderedPptPath && fs.existsSync(renderedPptPath)) {
      fs.copyFileSync(renderedPptPath, outPath);
    } else {
      await buildPptxFromSemantic(semanticJson, outPath);
    }

    // Return a URL that works from the Vite dev server via proxy (/api -> backend).
    // This also works when opened from a phone on the same WiFi.
    const downloadUrl = `/api/downloads/${encodeURIComponent(filename)}`;
    res.json({ ok: true, filename, downloadUrl, semanticJson });
  } catch (e) {
    res.status(500).json({ error: e?.message || String(e) });
  }
});

app.listen(PORT, "0.0.0.0", () => {
  // Bind to 0.0.0.0 so phone can reach backend too (optional, but helpful).
  // Frontend's only hard requirement is Vite host:0.0.0.0, but this avoids surprises.
  // eslint-disable-next-line no-console
  console.log(`[backend] listening on http://0.0.0.0:${PORT} (MCP: ${MCP_URL})`);
});


