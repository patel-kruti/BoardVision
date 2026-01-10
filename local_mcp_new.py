from fastapi import FastAPI
from pathlib import Path
from PIL import Image
import uuid
import json
import torch
import re
import os

from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from pptx import Presentation
from pptx.util import Inches
from pptx.enum.shapes import MSO_AUTO_SHAPE_TYPE, MSO_CONNECTOR_TYPE

from pptx.enum.text import PP_ALIGN
from pptx.util import Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR_TYPE


from layout_engine import LayoutEngine


# =========================================================
# CONFIG
# =========================================================

app = FastAPI()

WORKDIR = Path("workdir")
WORKDIR.mkdir(exist_ok=True)

SLIDE_WIDTH_IN = 10
SLIDE_HEIGHT_IN = 7.5
NODE_WIDTH_IN = 2.5
NODE_HEIGHT_IN = 1.2
HORIZONTAL_SPACING_IN = 3.0
VERTICAL_CENTER_IN = 3.5

# =========================================================
# LOAD QWEN2-VL (STABLE MODE)
# =========================================================

QWEN_MODEL = "Qwen/Qwen2-VL-2B-Instruct"

os.environ["ACCELERATE_DISABLE_RICH"] = "1"

print("Loading Qwen2-VL...")

qwen_processor = AutoProcessor.from_pretrained(
    QWEN_MODEL, trust_remote_code=True
)

qwen_model = Qwen2VLForConditionalGeneration.from_pretrained(
    QWEN_MODEL,
    trust_remote_code=True,
    dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True
)

# Determine device for tensors (will use GPU if available, otherwise CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
qwen_model.eval()

print("Qwen2-VL loaded on", device)

# =========================================================
# JSON EXTRACTION (HARDENED)
# =========================================================

def extract_json_from_text(text: str):
    if not text:
        return None

    # Strip markdown fences if present
    text = re.sub(r"```json|```", "", text).strip()

    # Direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Fallback: try to extract and repair incomplete JSON
    match = re.search(r"\{.*", text, re.DOTALL)
    if match:
        json_str = match.group(0)
        
        # Count braces to find complete JSON
        brace_count = 0
        end_idx = 0
        for i, char in enumerate(json_str):
            if char == "{":
                brace_count += 1
            elif char == "}":
                brace_count -= 1
                if brace_count == 0:
                    end_idx = i + 1
                    break
        
        if end_idx > 0:
            json_str = json_str[:end_idx]
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass
        else:
            # Incomplete JSON - aggressively repair it
            # 1. Remove trailing incomplete content (after last complete field)
            # Find the last complete field ending (}, ], or a complete value)
            # Remove everything after the last comma that doesn't complete
            
            # Strategy: find last successfully closed element and truncate after it
            # Walk backwards to find last }, ], or " followed by comma
            last_good_pos = -1
            for i in range(len(json_str) - 1, -1, -1):
                if json_str[i] in '}]"' and i + 1 < len(json_str):
                    # Check if followed by optional whitespace then comma or closing bracket/brace
                    rest = json_str[i+1:].lstrip()
                    if rest and rest[0] in ',}]':
                        last_good_pos = i + 1
                        # Scan past whitespace and comma
                        j = i + 1
                        while j < len(json_str) and json_str[j] in ' \t\n\r':
                            j += 1
                        if j < len(json_str) and json_str[j] == ',':
                            j += 1
                        while j < len(json_str) and json_str[j] in ' \t\n\r':
                            j += 1
                        last_good_pos = j if j < len(json_str) else i + 1
                        break
            
            if last_good_pos > 0:
                json_str = json_str[:last_good_pos]
            
            # 2. Close open arrays
            bracket_count = json_str.count("[") - json_str.count("]")
            if bracket_count > 0:
                json_str += "]" * bracket_count
            
            # 3. Close open braces
            brace_count = json_str.count("{") - json_str.count("}")
            if brace_count > 0:
                json_str += "}" * brace_count
            
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass
    
    # Last resort: return None to signal failure
    return None

def validate_semantics(semantic_json):
    nodes = semantic_json.get("nodes", [])
    edges = semantic_json.get("edges", [])

    node_ids = {n["id"] for n in nodes}

    ALLOWED_ROLES = {"start", "process", "decision", "end", "connector", "data", "document", "unknown"}

    # Fix invalid roles
    for n in nodes:
        if n["role"] not in ALLOWED_ROLES:
            n["role"] = "unknown"

    # Remove invalid edges
    valid_edges = []
    for e in edges:
        if e["from"] in node_ids and e["to"] in node_ids:
            valid_edges.append(e)

    semantic_json["edges"] = valid_edges
    return semantic_json

def bootstrap_nodes_from_visuals(visual_json):
    nodes = []
    for idx, e in enumerate(visual_json["elements"]):
        if e["type"] == "shape" and e.get("text"):
            SHAPE_TO_ROLE = {
                "terminator": "start",    # refined later
                "process": "process",
                "decision": "decision",
                "data": "data",
                "document": "document",
                "connector": "connector"
            }

            role = SHAPE_TO_ROLE.get(e.get("shape"), "unknown")

            nodes.append({
                "id": f"n{idx+1}",
                "role": role,
                "text": e["text"]
            })
    return nodes



# =========================================================
# LAYER 1 — VISUAL DETECTION (GEOMETRY ONLY)
# =========================================================

@app.post("/tools/detect_visuals")
async def detect_visuals(data: dict):
    file_id = data.get("file_id")
    if not file_id:
        return {"status": "error", "detail": "file_id required"}

    image = Image.open(file_id).convert("RGB")

    prompt = """You are a VISUAL ELEMENT DETECTOR for hand-drawn diagrams and whiteboards.

Your task is to detect ALL visible elements EXACTLY as they appear.
Do NOT infer meaning or flow logic.

============================
GENERAL RULES
============================

- Be COMPLETE, not selective.
- If unsure, STILL include the element.
- NEVER merge elements.
- Use "unknown" if classification is unclear.

============================
ELEMENT TYPES
============================

Detect and output THREE element types ONLY:

1. shape
   - Represents a drawn container (box, oval, diamond, circle, etc.)
   - If a shape contains text, put that text in the SAME element.

2. text
   - Text NOT enclosed inside a shape
   - Examples: titles, annotations, legends

3. arrow
   - Represents a directional connector between shapes
   - Use this ONLY if direction is visually implied

============================
SHAPE CLASSIFICATION
============================

============================
FLOWCHART SHAPE CLASSIFICATION (MANDATORY)
============================

If the shape visually matches a STANDARD FLOWCHART SYMBOL,
you MUST classify it as one of the following:

- terminator        (start / end – rounded capsule)
- process           (rectangle)
- decision          (diamond)
- data              (parallelogram)
- document          (rectangle with curved bottom)
- connector         (small circle with letter or number)

If unsure, use "unknown".

IMPORTANT:
- Do NOT collapse different symbols into "oval".
- Prefer semantic flowchart symbols over raw geometry.


============================
OUTPUT FIELDS (MANDATORY)
============================

For EACH visible element, output:

- id: unique string (e1, e2, e3, ...)
- type: "shape" | "text" | "arrow"
- text: detected text (empty string if none)
- shape: shape type (ONLY for type="shape", else "unknown")
- bbox: [x, y, w, h]

============================
BOUNDING BOX RULES
============================

- bbox values must be integers
- Use SMALL approximate values (0–1000 range)
- Coarse estimation is acceptable

============================
CRITICAL CONSTRAINTS
============================

- If text appears INSIDE a shape, it MUST be included in the SAME shape element.
- DO NOT output separate text elements for text inside shapes.
- Titles or decorative text OUTSIDE shapes must be type="text".
- EVERY arrow must be type="arrow".
- Multiple elements are EXPECTED.

============================
OUTPUT FORMAT (STRICT)
============================

Return ONLY valid JSON.
NO markdown.
NO comments.
NO explanations.

{
  "elements": [
    {
      "id": "e1",
      "type": "shape",
      "text": "Start",
      "shape": "oval",
      "bbox": [120, 80, 140, 60]
    }
  ]
}

============================
FINAL REQUIREMENT
============================

End your response ONLY after the final closing brace '}'.
Do NOT stop early.

"""

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        },
    ]

    text = qwen_processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = qwen_processor(
        text=[text],
        images=[image],
        return_tensors="pt"
    )
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    with torch.no_grad():
        output = qwen_model.generate(
            **inputs,
            max_new_tokens=800,
            eos_token_id=qwen_processor.tokenizer.eos_token_id,
        )

    generated_ids = output[:, inputs["input_ids"].size(1):]
    raw = qwen_processor.batch_decode(
        generated_ids, skip_special_tokens=True
    )[0]
    # raw = raw.strip()
    # raw = raw[raw.find("{") : raw.rfind("}") + 1]

    visual_json = extract_json_from_text(raw)

    if visual_json is None:
        return {
            "status": "error",
            "detail": "Failed to extract valid JSON from model output",
            "raw_output": raw[:800],
        }

    if "elements" not in visual_json:
        visual_json["elements"] = []

    return {"status": "ok", "visual_json": visual_json}

# =========================================================
# LAYER 2 — SEMANTIC REASONING (NO GEOMETRY)
# =========================================================

@app.post("/tools/reason_semantics")
async def reason_semantics(data: dict):
    visual_json = data.get("visual_json")
    if not visual_json:
        return {"status": "error", "detail": "visual_json required"}

    seed_nodes = bootstrap_nodes_from_visuals(visual_json)
    
#     prompt = """
# You are a FLOWCHART STRUCTURE ANALYZER.

# You are given VISUAL ELEMENTS extracted from a whiteboard image.
# Each element represents either a shape, text, or arrow.

# Your task is to convert these visual elements into a STRUCTURED FLOWCHART GRAPH.

# ============================
# IMPORTANT — READ CAREFULLY
# ============================

# This diagram IS a FLOWCHART.
# Do NOT summarize, simplify, or invent structure.
# Preserve ALL logical steps exactly as shown.

# You MUST follow the rules below EXACTLY.
# Violating any rule makes the output invalid.

# ============================
# INPUT: VISUAL ELEMENTS
# ============================

# """ + visual_json_str + """

# ============================
# MANDATORY TWO-PHASE CONSTRUCTION
# ============================

# You MUST work in TWO PHASES internally:

# PHASE 1 — NODE ENUMERATION (MANDATORY)
# - Identify ALL visual SHAPES that contain text.
# - Create EXACTLY ONE semantic node for EACH such shape.
# - List ALL nodes completely BEFORE considering edges.
# - If unsure about a role, still create the node with role = "unknown".

# PHASE 2 — EDGE CONSTRUCTION
# - Create edges ONLY AFTER all nodes exist.
# - EVERY edge MUST reference an EXISTING node id.
# - You are NOT ALLOWED to reference an id that is not in the nodes list.

# If fewer than 2 nodes are detected visually,
# you MUST still output all detected nodes.
# Do NOT collapse nodes.

# ============================
# NODE EXTRACTION RULES
# ============================

# 1. One shape → one node (NO exceptions).
# 2. Text inside the shape is the node text.
# 3. Role assignment:
#    - Oval → start or end
#    - Rectangle → process
#    - Diamond → decision
#    - Small labeled circles (A, B, etc.) → connector
#    - Unknown shape → unknown role
# 4. Node IDs:
#    - n1, n2, n3, ...
#    - Sequential
#    - Unique
# 5. Connector nodes:
#    - Same label = same connector node
#    - Multiple arrows may connect through it

# ============================
# EDGE EXTRACTION RULES
# ============================

# 1. Create an edge for EVERY arrow or connector.
# 2. Direction MUST follow arrow direction.
# 3. Arrow text (Yes / No / True / False) → edge.label.
# 4. Decision nodes MUST have at least 2 outgoing edges.
# 5. Merges MUST converge to the same node.

# ============================
# STRUCTURAL RULES
# ============================

# - Multiple start nodes allowed.
# - Multiple end nodes allowed.
# - Cycles (loops) allowed.
# - Parallel paths allowed.
# - A flowchart DOES NOT have only one node unless ONLY ONE shape exists visually.

# ============================
# OUTPUT FORMAT (STRICT)
# ============================

# Return ONLY valid JSON.
# NO markdown.
# NO comments.
# NO explanations.
# NO trailing text.

# {
#   "diagram_type": "flowchart",
#   "nodes": [
#     { "id": "n1", "role": "start", "text": "..." }
#   ],
#   "edges": [
#     { "from": "n1", "to": "n2", "label": "Yes" }
#   ],
#   "summary": "One concise sentence describing the flowchart."
# }

# ============================
# FINAL VALIDATION (MANDATORY)
# ============================

# Before responding:
# - Count visual shapes with text.
# - Ensure the SAME number of nodes exist.
# - Ensure NO edge references missing node ids.
# - Ensure decision nodes have multiple edges.
# - If any rule is violated, FIX IT before returning.

# """
    prompt = """
You are a FLOWCHART EDGE INFERENCE ENGINE.

STRICT OUTPUT RULES (NON-NEGOTIABLE):
- You MUST output VALID JSON ONLY.
- NO explanations.
- NO prose.
- NO markdown.
- NO comments.
- NO text before or after JSON.

If you violate this, the output will be discarded.

====================================
PREDEFINED NODES (DO NOT MODIFY)
====================================
""" + json.dumps(seed_nodes, indent=2) + """

====================================
VISUAL ELEMENTS (FOR REFERENCE)
====================================
""" + json.dumps(visual_json, indent=2) + """

====================================
YOUR TASK
====================================

- Create EDGES between EXISTING node IDs only.
- Use arrow direction/labels if present in visual elements.
- If no arrow text is visible, omit the "label" field.
- Decision nodes may have multiple outgoing edges (Yes/No).
- You are NOT allowed to create new nodes.

====================================
OUTPUT FORMAT (EXACT)
====================================

{
  "edges": [
    { "from": "n1", "to": "n2" },
    { "from": "n2", "to": "n3", "label": "Yes" }
  ],
  "summary": "One concise sentence."
}
"""

    print("=== SEMANTIC PROMPT ===")
    print(prompt)


    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": prompt}],
        },
    ]

    text = qwen_processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = qwen_processor(text=[text], return_tensors="pt")
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    with torch.no_grad():
        output = qwen_model.generate(
            **inputs,
            max_new_tokens=600,
            eos_token_id=qwen_processor.tokenizer.eos_token_id,
        )

    generated_ids = output[:, inputs["input_ids"].size(1):]
    raw = qwen_processor.batch_decode(
        generated_ids, skip_special_tokens=True
    )[0]
    # raw = raw.strip()
    # raw = raw[raw.find("{") : raw.rfind("}") + 1]

    edges_json = extract_json_from_text(raw)
    
    # Check for extraction failure BEFORE validation
    if edges_json is None:
        # If model failed to return JSON, create default edges (sequential flow)
        print("Failed to extract edges JSON, creating sequential edges...")
        edges_json = {"edges": []}
        for i in range(len(seed_nodes) - 1):
            edges_json["edges"].append({
                "from": seed_nodes[i]["id"],
                "to": seed_nodes[i+1]["id"]
            })
    
    # Construct complete semantic JSON
    semantic_json = {
        "diagram_type": "flowchart",
        "nodes": seed_nodes,
        "edges": edges_json.get("edges", []),
        "summary": edges_json.get("summary", "")
    }
    
    from layout_engine import validate_and_score_semantics

    try:
        semantic_json, confidence = validate_and_score_semantics(
            semantic_json,
            visual_json
        )
        print("Semantic confidence:", confidence)
    except ValueError as e:
        return {
            "status": "error",
            "detail": str(e),
            "raw_semantic": semantic_json,
        }

    semantic_json.setdefault("nodes", [])
    semantic_json.setdefault("edges", [])
    semantic_json.setdefault("diagram_type", "unknown")
    semantic_json.setdefault("summary", "")

    return {"status": "ok", "semantic_json": semantic_json}

# =========================================================
# LAYER 3 — DETERMINISTIC PPT RENDERING
# =========================================================

def map_role_to_shape(role: str):
    return {
        "start": MSO_AUTO_SHAPE_TYPE.FLOWCHART_TERMINATOR,
        "end": MSO_AUTO_SHAPE_TYPE.FLOWCHART_TERMINATOR,
        "process": MSO_AUTO_SHAPE_TYPE.FLOWCHART_PROCESS,
        "decision": MSO_AUTO_SHAPE_TYPE.FLOWCHART_DECISION,
        "data": MSO_AUTO_SHAPE_TYPE.FLOWCHART_DATA,
        "document": MSO_AUTO_SHAPE_TYPE.FLOWCHART_DOCUMENT,
        "connector": MSO_AUTO_SHAPE_TYPE.FLOWCHART_CONNECTOR,
    }.get(role, MSO_AUTO_SHAPE_TYPE.RECTANGLE)

@app.post("/tools/render_ppt")
async def render_ppt(data: dict):
    semantic = data.get("semantic_json")
    if not semantic:
        return {"status": "error", "detail": "semantic_json required"}

    prs = Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[6])

    # nodes = semantic.get("nodes", [])
    # edges = semantic.get("edges", [])
    engine = LayoutEngine()
    layout = engine.layout(semantic)

    layout_nodes = layout["nodes"]
    layout_edges = layout["edges"]

    ppt_nodes = {}

    for node in layout_nodes:
        shape = slide.shapes.add_shape(
            map_role_to_shape(node["role"]),
            Inches(node["x"]),
            Inches(node["y"]),
            Inches(node["width"]),
            Inches(node["height"])
        )

        # ---- Style shape ----
        shape.fill.solid()
        shape.fill.fore_color.rgb = RGBColor(235, 242, 255)
        shape.line.width = Pt(1.5)

        # ---- Text formatting ----
        tf = shape.text_frame
        tf.clear()
        p = tf.paragraphs[0]
        p.text = node["text"]
        p.font.size = Pt(18)
        p.font.bold = True
        p.alignment = PP_ALIGN.CENTER

        ppt_nodes[node["id"]] = shape


    for edge in layout_edges:
        src = ppt_nodes.get(edge["from"])
        dst = ppt_nodes.get(edge["to"])
        if not src or not dst:
            continue

        connector = slide.shapes.add_connector(
            MSO_CONNECTOR_TYPE.STRAIGHT,
            int(src.left + src.width),
            int(src.top + src.height / 2),
            dst.left,
            int(dst.top + dst.height / 2),
        )

        connector.line.width = Pt(1.5)
        connector.line.end_arrowhead = True



    # ppt_nodes = {}
    # x = 1.0

    # for node in nodes:
    #     shape = slide.shapes.add_shape(
    #         map_role_to_shape(node["role"]),
    #         Inches(x),
    #         Inches(VERTICAL_CENTER_IN),
    #         Inches(NODE_WIDTH_IN),
    #         Inches(NODE_HEIGHT_IN),
    #     )
    #     shape.text = node["text"]
    #     ppt_nodes[node["id"]] = shape
    #     x += HORIZONTAL_SPACING_IN

    # for edge in edges:
    #     src = ppt_nodes.get(edge["from"])
    #     dst = ppt_nodes.get(edge["to"])
    #     if not src or not dst:
    #         continue

    #     connector = slide.shapes.add_connector(
    #         MSO_CONNECTOR_TYPE.STRAIGHT,
    #         src.left + src.width,
    #         src.top + src.height // 2,
    #         dst.left - src.left,
    #         0,
    #     )
    #     connector.line.end_arrowhead = True

    # if semantic.get("summary"):
    #     tb = slide.shapes.add_textbox(
    #         Inches(0.5),
    #         Inches(6.3),
    #         Inches(9),
    #         Inches(0.6),
    #     )
    #     tb.text = semantic["summary"]

    if semantic.get("summary"):
        tb = slide.shapes.add_textbox(
            Inches(6.0),
            Inches(6.5),
            Inches(3.5),
            Inches(0.6),
        )
        tf = tb.text_frame
        tf.text = semantic["summary"]
        tf.paragraphs[0].font.size = Pt(11)
        tf.paragraphs[0].font.italic = True

    outpath = WORKDIR / f"slide_{uuid.uuid4().hex}.pptx"
    prs.save(outpath)

    return {"status": "ok", "ppt_file": str(outpath)}

# =========================================================
# ORCHESTRATOR
# =========================================================

@app.post("/tools/whiteboard_to_ppt")
async def whiteboard_to_ppt(data: dict):
    visuals = await detect_visuals(data)
    if visuals["status"] != "ok":
        return visuals

    semantics = await reason_semantics({"visual_json": visuals["visual_json"]})
    if semantics["status"] != "ok":
        return semantics

    ppt = await render_ppt({"semantic_json": semantics["semantic_json"]})
    if ppt["status"] != "ok":
        return ppt

    return {
        "status": "ok",
        "ppt_file": ppt["ppt_file"],
        "semantic_json": semantics["semantic_json"],
    }
