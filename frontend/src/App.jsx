import React, { useMemo, useRef, useState } from "react";
import Mascot from "./components/Mascot.jsx";
import { generatePpt, getStatus, processImage } from "./lib/api.js";

function Badge({ children, tone = "slate" }) {
  const tones = {
    slate: "bg-slate-100 text-slate-700 border-slate-200",
    green: "bg-emerald-50 text-emerald-700 border-emerald-200",
    amber: "bg-amber-50 text-amber-700 border-amber-200",
    red: "bg-rose-50 text-rose-700 border-rose-200",
    blue: "bg-sky-50 text-sky-700 border-sky-200"
  };
  return (
    <span
      className={`inline-flex items-center rounded-full border px-2.5 py-1 text-xs font-medium ${tones[tone] || tones.slate}`}
    >
      {children}
    </span>
  );
}

function Card({ title, right, children }) {
  return (
    <div className="rounded-2xl border border-slate-200 bg-white shadow-sm">
      <div className="flex items-center justify-between px-5 py-4 border-b border-slate-100">
        <div className="text-sm font-semibold text-slate-900">{title}</div>
        <div>{right}</div>
      </div>
      <div className="p-5">{children}</div>
    </div>
  );
}

export default function App() {
  const inputRef = useRef(null);

  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [jobId, setJobId] = useState("");

  const [status, setStatus] = useState("idle"); // idle | uploading | processing | ready | generating | done | error
  const [message, setMessage] = useState("");

  const [result, setResult] = useState(null); // { visualJson, semanticJson? }
  const [visualText, setVisualText] = useState("");
  const [semanticPreview, setSemanticPreview] = useState(null);
  const [pptUrl, setPptUrl] = useState("");

  const visualJson = useMemo(() => {
    try {
      return visualText ? JSON.parse(visualText) : null;
    } catch {
      return null;
    }
  }, [visualText]);

  function onPickFile(f) {
    if (!f) return;
    setFile(f);
    const url = URL.createObjectURL(f);
    setPreviewUrl(url);
    setResult(null);
    setVisualText("");
    setSemanticPreview(null);
    setPptUrl("");
    setJobId("");
    setStatus("idle");
    setMessage("");
  }

  async function startProcess() {
    if (!file) return;
    setStatus("uploading");
    setMessage("Uploading image…");
    try {
      const { id } = await processImage(file);
      setJobId(id);
      setStatus("processing");
      setMessage("Processing… (this may take ~20–60s on first run)");
      await pollStatus(id);
    } catch (e) {
      setStatus("error");
      setMessage(String(e?.message || e));
    }
  }

  async function pollStatus(id) {
    const started = Date.now();
    const timeoutMs = 3 * 60 * 1000;

    // eslint-disable-next-line no-constant-condition
    while (true) {
      if (Date.now() - started > timeoutMs) {
        setStatus("error");
        setMessage("Timed out waiting for processing. Try again.");
        return;
      }
      const s = await getStatus(id);
      if (s.status === "done") {
        setResult(s.result);
        setVisualText(JSON.stringify(s.result.visualJson, null, 2));
        setStatus("ready");
        setMessage("Ready. Review/edit the visual JSON, then Generate PPT.");
        return;
      }
      if (s.status === "error") {
        setStatus("error");
        setMessage(s.error || "Processing failed");
        return;
      }
      await new Promise((r) => setTimeout(r, 1200));
    }
  }

  async function onGenerate() {
    if (!jobId) return;
    if (!visualJson) {
      setStatus("error");
      setMessage("Visual JSON is invalid. Fix JSON before generating PPT.");
      return;
    }
    setStatus("generating");
    setMessage("Running reason_semantics → generating PPT…");
    try {
      const resp = await generatePpt({ id: jobId, visualJson });
      setPptUrl(resp.downloadUrl);
      setSemanticPreview(resp.semanticJson || null);
      setStatus("done");
      setMessage("PPT generated.");
    } catch (e) {
      setStatus("error");
      setMessage(String(e?.message || e));
    }
  }

  const statusTone =
    status === "done"
      ? "green"
      : status === "error"
        ? "red"
        : status === "processing" || status === "uploading" || status === "generating"
          ? "amber"
          : "slate";

  return (
    <div className="min-h-screen">
      {/* Header */}
      <header className="sticky top-0 z-10 border-b border-slate-200 bg-white/80 backdrop-blur">
        <div className="mx-auto max-w-6xl px-4 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="h-10 w-10 rounded-2xl bg-gradient-to-br from-sky-400 to-emerald-400" />
            <div>
              <div className="text-lg font-semibold leading-tight">BoardVision</div>
              <div className="text-xs text-slate-600">
                Flowchart photo → editable structure → PowerPoint
              </div>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <Badge tone={statusTone}>{status.toUpperCase()}</Badge>
            <a
              className="text-xs text-slate-600 hover:text-slate-900 underline underline-offset-4"
              href="http://127.0.0.1:8000/docs"
              target="_blank"
              rel="noreferrer"
              title="(Optional) Python API docs, if running"
            >
              API docs
            </a>
          </div>
        </div>
      </header>

      {/* Body */}
      <main className="mx-auto max-w-6xl px-4 py-8">
        <div className="grid gap-6 lg:grid-cols-12">
          {/* Left column */}
          <div className="lg:col-span-5 space-y-6">
            <Card
              title="1) Upload a whiteboard / flowchart image"
              right={
                <button
                  className="text-xs rounded-lg border border-slate-200 px-3 py-1.5 hover:bg-slate-50"
                  onClick={() => inputRef.current?.click()}
                >
                  Choose file
                </button>
              }
            >
              <input
                ref={inputRef}
                type="file"
                accept="image/*"
                className="hidden"
                onChange={(e) => onPickFile(e.target.files?.[0])}
              />

              <div
                className="rounded-2xl border-2 border-dashed border-slate-200 bg-slate-50 p-6 text-center"
                onDragOver={(e) => e.preventDefault()}
                onDrop={(e) => {
                  e.preventDefault();
                  const f = e.dataTransfer.files?.[0];
                  onPickFile(f);
                }}
              >
                <div className="text-sm font-medium text-slate-900">
                  Drag & drop an image here
                </div>
                <div className="text-xs text-slate-600 mt-1">
                  PNG/JPG recommended. Keep it high-contrast.
                </div>
              </div>

              {previewUrl ? (
                <div className="mt-5">
                  <div className="text-xs font-semibold text-slate-700 mb-2">Preview</div>
                  <div className="overflow-hidden rounded-2xl border border-slate-200 bg-white">
                    <img src={previewUrl} alt="preview" className="w-full h-auto" />
                  </div>
                  <div className="mt-3 flex gap-2">
                    <button
                      className="flex-1 rounded-xl bg-slate-900 text-white text-sm font-semibold py-2.5 hover:bg-slate-800 disabled:opacity-50"
                      disabled={!file || status === "uploading" || status === "processing"}
                      onClick={startProcess}
                    >
                      Process image
                    </button>
                    <button
                      className="rounded-xl border border-slate-200 text-sm font-semibold px-4 py-2.5 hover:bg-slate-50"
                      onClick={() => onPickFile(null)}
                    >
                      Reset
                    </button>
                  </div>
                </div>
              ) : (
                <div className="mt-5 text-xs text-slate-600">
                  Tip: Use your phone camera and keep the diagram centered.
                </div>
              )}
            </Card>

            <Card title="Status">
              <div className="flex items-start justify-between gap-3">
                <div>
                  <div className="text-sm font-medium text-slate-900">Pipeline</div>
                  <div className="text-xs text-slate-600 mt-1">
                    Upload → MCP mock → editable JSON → PPTX
                  </div>
                </div>
                {jobId ? <Badge tone="blue">JOB: {jobId.slice(0, 8)}</Badge> : null}
              </div>
              <div className="mt-4 rounded-xl border border-slate-200 bg-slate-50 p-3 text-sm">
                {message || "Waiting for an upload."}
              </div>
              {pptUrl ? (
                <div className="mt-4">
                  <a
                    className="inline-flex items-center justify-center rounded-xl bg-emerald-600 text-white text-sm font-semibold px-4 py-2.5 hover:bg-emerald-500"
                    href={pptUrl}
                    target="_blank"
                    rel="noreferrer"
                  >
                    Download PPTX
                  </a>
                </div>
              ) : null}
            </Card>
          </div>

          {/* Right column */}
          <div className="lg:col-span-7 space-y-6">
            <Card
              title="2) Preview snippets"
              right={result ? <Badge tone="green">Detected</Badge> : <Badge>Waiting</Badge>}
            >
              {result ? (
                <div className="grid gap-4 md:grid-cols-2">
                  <div className="rounded-xl border border-slate-200 bg-white p-4">
                    <div className="text-xs font-semibold text-slate-700">Visual elements</div>
                    <div className="mt-2 space-y-2">
                      {result.visualJson?.elements?.slice(0, 6)?.map((e) => (
                        <div
                          key={e.id}
                          className="flex items-center justify-between rounded-lg bg-slate-50 px-3 py-2"
                        >
                          <div className="text-sm font-medium">{e.text || e.id}</div>
                          <Badge>{e.type}</Badge>
                        </div>
                      ))}
                    </div>
                  </div>
                  <div className="rounded-xl border border-slate-200 bg-white p-4">
                    <div className="text-xs font-semibold text-slate-700">Semantic (after Generate)</div>
                    <div className="mt-2 text-xs text-slate-600">
                      {semanticPreview
                        ? `Nodes: ${semanticPreview.nodes?.length || 0}, Edges: ${semanticPreview.edges?.length || 0}`
                        : "Not generated yet (Generate PPT calls reason_semantics)."}
                    </div>
                  </div>
                </div>
              ) : (
                <div className="text-sm text-slate-600">
                  After processing, you’ll see a quick preview here.
                </div>
              )}
            </Card>

            <Card
              title="3) Edit detected visuals (JSON)"
              right={
                visualText ? (
                  visualJson ? (
                    <Badge tone="green">Valid JSON</Badge>
                  ) : (
                    <Badge tone="red">Invalid JSON</Badge>
                  )
                ) : (
                  <Badge>Waiting</Badge>
                )
              }
            >
              <textarea
                value={visualText}
                onChange={(e) => setVisualText(e.target.value)}
                placeholder='{\n  "diagram_type": "flowchart",\n  "nodes": [...],\n  "edges": [...]\n}'
                className="w-full h-72 resize-none rounded-xl border border-slate-200 bg-white p-4 font-mono text-xs leading-relaxed focus:outline-none focus:ring-2 focus:ring-sky-300"
              />
              <div className="mt-3 flex gap-2">
                <button
                  className="flex-1 rounded-xl bg-sky-600 text-white text-sm font-semibold py-2.5 hover:bg-sky-500 disabled:opacity-50"
                  disabled={!jobId || !visualJson || status === "generating"}
                  onClick={onGenerate}
                >
                  Generate PPT
                </button>
                <button
                  className="rounded-xl border border-slate-200 text-sm font-semibold px-4 py-2.5 hover:bg-slate-50 disabled:opacity-50"
                  disabled={!result?.visualJson}
                  onClick={() => setVisualText(JSON.stringify(result.visualJson, null, 2))}
                >
                  Reset JSON
                </button>
              </div>
            </Card>
          </div>
        </div>
      </main>

      <Mascot />
    </div>
  );
}


