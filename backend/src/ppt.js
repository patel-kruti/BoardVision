import PptxGenJS from "pptxgenjs";

function safeText(v) {
  if (v == null) return "";
  return String(v);
}

export async function buildPptxFromSemantic(semanticJson, outPath) {
  const pptx = new PptxGenJS();
  pptx.layout = "LAYOUT_WIDE";

  const slide = pptx.addSlide();
  slide.background = { color: "F8FAFC" };

  slide.addText("BoardVision", {
    x: 0.5,
    y: 0.35,
    w: 12.3,
    h: 0.6,
    fontFace: "Calibri",
    fontSize: 28,
    bold: true,
    color: "0F172A"
  });

  slide.addText("Generated from image → semantic JSON", {
    x: 0.52,
    y: 1.0,
    w: 12.0,
    h: 0.35,
    fontFace: "Calibri",
    fontSize: 14,
    color: "475569"
  });

  const nodes = Array.isArray(semanticJson?.nodes) ? semanticJson.nodes : [];
  const edges = Array.isArray(semanticJson?.edges) ? semanticJson.edges : [];

  // Left column: nodes
  slide.addShape(pptx.ShapeType.roundRect, {
    x: 0.6,
    y: 1.55,
    w: 6.1,
    h: 5.4,
    fill: { color: "FFFFFF" },
    line: { color: "E2E8F0", width: 1 }
  });
  slide.addText("Nodes", {
    x: 0.85,
    y: 1.75,
    w: 5.6,
    h: 0.4,
    fontSize: 16,
    bold: true,
    color: "0F172A"
  });
  const nodeLines = nodes.slice(0, 18).map((n) => `• ${safeText(n.id)}  [${safeText(n.role)}]  ${safeText(n.text)}`);
  slide.addText(nodeLines.join("\n") || "No nodes", {
    x: 0.85,
    y: 2.25,
    w: 5.7,
    h: 4.6,
    fontSize: 12,
    color: "0F172A"
  });

  // Right column: edges
  slide.addShape(pptx.ShapeType.roundRect, {
    x: 6.95,
    y: 1.55,
    w: 6.1,
    h: 5.4,
    fill: { color: "FFFFFF" },
    line: { color: "E2E8F0", width: 1 }
  });
  slide.addText("Edges", {
    x: 7.2,
    y: 1.75,
    w: 5.6,
    h: 0.4,
    fontSize: 16,
    bold: true,
    color: "0F172A"
  });
  const edgeLines = edges.slice(0, 22).map((e) => {
    const label = e.label ? ` (${safeText(e.label)})` : "";
    return `• ${safeText(e.from)} → ${safeText(e.to)}${label}`;
  });
  slide.addText(edgeLines.join("\n") || "No edges", {
    x: 7.2,
    y: 2.25,
    w: 5.7,
    h: 4.6,
    fontSize: 12,
    color: "0F172A"
  });

  await pptx.writeFile({ fileName: outPath });
}


