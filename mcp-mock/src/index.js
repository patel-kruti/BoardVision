import express from "express";
import axios from "axios";

const app = express();
app.use(express.json({ limit: "10mb" }));

const PORT = process.env.PORT ? Number(process.env.PORT) : 8080;
const REAL_MCP_URL = process.env.REAL_MCP_URL || "";

app.get("/health", (req, res) => res.json({ ok: true, service: "mcp-mock", port: PORT }));

async function maybeProxy(req, res, path) {
  if (!REAL_MCP_URL) return false;
  try {
    const upstream = await axios.post(`${REAL_MCP_URL}${path}`, req.body, {
      timeout: 10 * 60 * 1000
    });
    res.status(upstream.status).json(upstream.data);
    return true;
  } catch (e) {
    res.status(502).json({
      status: "error",
      detail: "Failed to reach REAL_MCP_URL",
      REAL_MCP_URL,
      error: e?.message || String(e)
    });
    return true;
  }
}

function makeVisualJson() {
  return {
    elements: [
      { id: "e1", type: "shape", shape: "terminator", text: "Start", bbox: [120, 40, 180, 60] },
      { id: "e2", type: "shape", shape: "process", text: "Process", bbox: [120, 120, 180, 60] },
      { id: "e3", type: "shape", shape: "decision", text: "Decision", bbox: [120, 220, 180, 120] },
      { id: "e4", type: "shape", shape: "data", text: "Data", bbox: [120, 380, 200, 70] },
      { id: "e5", type: "shape", shape: "document", text: "Document", bbox: [120, 470, 200, 70] },
      { id: "e6", type: "shape", shape: "terminator", text: "End", bbox: [120, 560, 180, 60] }
    ]
  };
}

function makeSemanticFromVisual(visual_json) {
  const elements = Array.isArray(visual_json?.elements) ? visual_json.elements : [];
  const nodes = elements
    .filter((e) => e.type === "shape" && e.text)
    .map((e, idx) => {
      const shape = e.shape || "unknown";
      const role =
        shape === "terminator"
          ? idx === 0
            ? "start"
            : idx === elements.length - 1
              ? "end"
              : "process"
          : shape;
      return { id: `n${idx + 1}`, role, text: e.text };
    });

  // Simple edge inference: sequential with a decision fork if a decision exists.
  const edges = [];
  for (let i = 0; i < Math.max(0, nodes.length - 1); i++) {
    edges.push({ from: nodes[i].id, to: nodes[i + 1].id });
  }
  const decisionIdx = nodes.findIndex((n) => n.role === "decision");
  if (decisionIdx !== -1 && nodes.length >= decisionIdx + 3) {
    // Make a fork and converge (mock behavior)
    edges.splice(decisionIdx, 2,
      { from: nodes[decisionIdx].id, to: nodes[decisionIdx + 1].id, label: "Yes" },
      { from: nodes[decisionIdx].id, to: nodes[decisionIdx + 2].id, label: "No" }
    );
  }

  return {
    diagram_type: "flowchart",
    nodes,
    edges,
    summary: "Mock semantic inference from detected visuals."
  };
}

// 1) Detect visuals (like local_mcp_new.py /tools/detect_visuals)
app.post("/tools/detect_visuals", async (req, res) => {
  if (await maybeProxy(req, res, "/tools/detect_visuals")) return;
  const { file_id } = req.body || {};
  if (!file_id) return res.status(400).json({ status: "error", detail: "file_id required" });
  await new Promise((r) => setTimeout(r, 450));
  res.json({ status: "ok", visual_json: makeVisualJson() });
});

// 2) Reason semantics (like local_mcp_new.py /tools/reason_semantics)
app.post("/tools/reason_semantics", async (req, res) => {
  if (await maybeProxy(req, res, "/tools/reason_semantics")) return;
  const { visual_json } = req.body || {};
  if (!visual_json) return res.status(400).json({ status: "error", detail: "visual_json required" });
  await new Promise((r) => setTimeout(r, 450));
  res.json({ status: "ok", semantic_json: makeSemanticFromVisual(visual_json) });
});

// 3) Render PPT (like local_mcp_new.py /tools/render_ppt)
app.post("/tools/render_ppt", async (req, res) => {
  if (await maybeProxy(req, res, "/tools/render_ppt")) return;
  // Mock: we don't generate a real pptx here; backend will fall back to JS PPT rendering.
  res.json({ status: "ok", ppt_file: null });
});

// 3) Orchestrator (like local_mcp_new.py /tools/whiteboard_to_ppt)
app.post("/tools/whiteboard_to_ppt", async (req, res) => {
  if (await maybeProxy(req, res, "/tools/whiteboard_to_ppt")) return;
  const { file_id } = req.body || {};
  if (!file_id) return res.status(400).json({ error: "file_id required" });

  // Simulate compute time (first run usually slower)
  await new Promise((r) => setTimeout(r, 900));

  const visual_json = makeVisualJson();
  const semantic_json = makeSemanticFromVisual(visual_json);

  res.json({
    status: "ok",
    file_id,
    ppt_file: "mock://not-generated-here.pptx",
    visual_json,
    semantic_json
  });
});

app.listen(PORT, "0.0.0.0", () => {
  // eslint-disable-next-line no-console
  console.log(`[mcp-mock] listening on http://0.0.0.0:${PORT}`);
});


