# layout_engine.py

from typing import Dict, List
from collections import defaultdict

# =========================================================
# Slide constants (in inches)
# =========================================================

SLIDE_WIDTH_IN = 10.0
SLIDE_HEIGHT_IN = 7.5

LEFT_MARGIN_IN = 0.5
TOP_MARGIN_IN = 0.5

COL_SPACING_IN = 2.2
ROW_SPACING_IN = 1.4

# =========================================================
# Role-based sizes (presentation-first)
# =========================================================

ROLE_SIZES = {
    "start": (1.5, 0.7),
    "end": (1.5, 0.7),
    "process": (1.8, 0.9),
    "decision": (1.8, 1.2),
    "data": (1.8, 0.9),
    "document": (1.8, 0.9),
    "connector": (0.4, 0.4),   # small jump connector (A, B, etc.)
    "unknown": (1.6, 0.8),
}

# =========================================================
# Layout Engine
# =========================================================

class LayoutEngine:
    """
    General-purpose deterministic flowchart layout engine.

    Handles:
    - Linear flows
    - Branching (decision nodes)
    - Merging
    - Parallel paths
    - Connector / jump nodes (A → A)
    - Cycles (safe handling)

    This engine is SEMANTIC-first, not geometry-first.
    """

    # -----------------------------------------------------
    # Public API
    # -----------------------------------------------------

    def layout(self, semantic_json: Dict) -> Dict:
        nodes = semantic_json.get("nodes", [])
        edges = semantic_json.get("edges", [])

        id_to_node = {n["id"]: n for n in nodes}
        outgoing, incoming = self._build_graph(edges)

        start_nodes = self._find_start_nodes(nodes, incoming)
        paths = self._enumerate_paths(start_nodes, outgoing)

        # Detect flow direction from visual bboxes
        flow_direction = self._detect_flow_direction(nodes)
        
        node_columns = self._compute_columns(paths, id_to_node)
        node_rows = self._compute_rows(paths)

        layout_nodes = self._assign_positions(
            id_to_node, node_columns, node_rows, flow_direction
        )

        return {
            "nodes": layout_nodes,
            "edges": edges,
            "summary": semantic_json.get("summary", ""),
            "flow_direction": flow_direction,
        }

    # -----------------------------------------------------
    # Flow direction detection
    # -----------------------------------------------------

    def _detect_flow_direction(self, nodes):
        """
        Detect if flowchart flows horizontally or vertically
        based on visual bbox positions.
        Returns: "horizontal" or "vertical"
        """
        if len(nodes) < 2:
            return "vertical"  # default
        
        # Calculate spread in X vs Y from visual bboxes
        bboxes = [n.get("_bbox") for n in nodes if n.get("_bbox")]
        if not bboxes:
            return "vertical"
        
        x_coords = [bbox[0] + bbox[2]/2 for bbox in bboxes]  # center X
        y_coords = [bbox[1] + bbox[3]/2 for bbox in bboxes]  # center Y
        
        x_spread = max(x_coords) - min(x_coords)
        y_spread = max(y_coords) - min(y_coords)
        
        # If X spread is significantly larger, it's horizontal
        if x_spread > y_spread * 1.3:
            return "horizontal"
        else:
            return "vertical"

    # -----------------------------------------------------
    # Graph utilities
    # -----------------------------------------------------

    def _build_graph(self, edges):
        outgoing = defaultdict(list)
        incoming = defaultdict(list)

        for e in edges:
            outgoing[e["from"]].append(e["to"])
            incoming[e["to"]].append(e["from"])

        return outgoing, incoming

    def _find_start_nodes(self, nodes, incoming):
        return [n["id"] for n in nodes if n["id"] not in incoming]

    # -----------------------------------------------------
    # Path enumeration (cycle-safe)
    # -----------------------------------------------------

    def _enumerate_paths(self, starts, outgoing):
        """
        Enumerate all root-to-leaf execution paths.
        Cycles are broken safely.
        """
        paths = []

        def dfs(node, path, visited):
            if node in visited:
                return  # break cycles safely
            visited.add(node)

            path.append(node)

            if node not in outgoing or not outgoing[node]:
                paths.append(list(path))
            else:
                for nxt in outgoing[node]:
                    dfs(nxt, path, visited.copy())

            path.pop()

        for s in starts:
            dfs(s, [], set())

        return paths

    # -----------------------------------------------------
    # Column assignment (X-axis / flow)
    # -----------------------------------------------------

    def _compute_columns(self, paths, id_to_node):
        """
        Assigns a flow column per node.
        Connector nodes do NOT advance the column.
        """
        columns = {}

        for path in paths:
            col = 0
            for nid in path:
                role = id_to_node.get(nid, {}).get("role", "unknown")
                # Set/relax to earliest column we see for this node
                columns[nid] = min(columns.get(nid, col), col) if nid in columns else col
                # Connectors do not advance the flow column
                if role != "connector":
                    col += 1

        # Ensure every node has a column
        for nid in id_to_node.keys():
            columns.setdefault(nid, 0)

        return columns

    # -----------------------------------------------------
    # Row assignment (Y-axis / branching)
    # -----------------------------------------------------

    def _compute_rows(self, paths):
        """
        Rows are assigned by path index.
        Merge nodes are centered by averaging rows.
        """
        row_accumulator = defaultdict(list)

        for row_idx, path in enumerate(paths):
            for nid in path:
                row_accumulator[nid].append(row_idx)

        return {
            nid: sum(rows) / len(rows)
            for nid, rows in row_accumulator.items()
        }

    # -----------------------------------------------------
    # Absolute position assignment
    # -----------------------------------------------------

    def _assign_positions(self, id_to_node, cols, rows, flow_direction="vertical"):
        layout_nodes = []
        
        # Adjust spacing based on flow direction and node count
        num_nodes = len(id_to_node)
        max_col = max(cols.values()) if cols else 0
        max_row = max(rows.values()) if rows else 0
        
        # Calculate required space based on actual shape sizes
        base_width, base_height = 1.8, 1.2  # max shape size
        
        if flow_direction == "horizontal":
            # For horizontal flow: X is flow, Y is branching
            required_width = (max_col + 1) * base_width
            required_height = (max_row + 1) * base_height
            
            # Scale spacing to fit slide with padding
            col_spacing = (SLIDE_WIDTH_IN - 2 * LEFT_MARGIN_IN) / max(max_col + 1, 1)
            row_spacing = (SLIDE_HEIGHT_IN - 2 * TOP_MARGIN_IN) / max(max_row + 1, 1)
            
            # Ensure minimum spacing between nodes
            col_spacing = max(col_spacing, base_width * 1.2)
            row_spacing = max(row_spacing, base_height * 1.1)
        else:
            # For vertical flow: Y is flow, X is branching
            required_width = (max_col + 1) * base_width
            required_height = (max_row + 1) * base_height
            
            col_spacing = (SLIDE_WIDTH_IN - 2 * LEFT_MARGIN_IN) / max(max_col + 1, 1)
            row_spacing = (SLIDE_HEIGHT_IN - 2 * TOP_MARGIN_IN) / max(max_row + 1, 1)
            
            col_spacing = max(col_spacing, base_width * 1.1)
            row_spacing = max(row_spacing, base_height * 1.2)

        for nid, node in id_to_node.items():
            col = cols.get(nid, 0)
            row = rows.get(nid, 0)

            width, height = ROLE_SIZES.get(
                node["role"], ROLE_SIZES["unknown"]
            )
            
            # Dynamically scale down for dense flowcharts
            if num_nodes > 8:
                scale_factor = min(0.7, 8.0 / num_nodes)
                width *= scale_factor
                height *= scale_factor
            elif num_nodes > 5:
                width *= 0.85
                height *= 0.85

            if flow_direction == "horizontal":
                # Horizontal: col drives X, row drives Y
                x = LEFT_MARGIN_IN + col * col_spacing
                y = TOP_MARGIN_IN + row * row_spacing
            else:
                # Vertical: col drives X, row drives Y
                x = LEFT_MARGIN_IN + col * col_spacing
                y = TOP_MARGIN_IN + row * row_spacing

            # Keep within slide bounds (strict clipping)
            x = max(0.2, min(x, SLIDE_WIDTH_IN - width - 0.2))
            y = max(0.2, min(y, SLIDE_HEIGHT_IN - height - 0.2))

            layout_nodes.append({
                "id": nid,
                "text": node["text"],
                "role": node["role"],
                "x": x,
                "y": y,
                "width": width,
                "height": height,
            })

        return layout_nodes


# semantic_guard.py

from typing import Dict, List, Tuple

# =========================================================
# Configuration
# =========================================================

ALLOWED_ROLES = {
    "start",
    "process",
    "decision",
    "end",
    "data",
    "document",
    "connector",
    "unknown",
}

MIN_NODE_SHAPE_RATIO = 0.6   # at least 60% of shapes must become nodes
MIN_NODE_COUNT = 2           # single-node flowcharts are almost always wrong


# =========================================================
# Public API
# =========================================================

def validate_and_score_semantics(
    semantic_json: Dict,
    visual_json: Dict,
) -> Tuple[Dict, Dict]:
    """
    Validates semantic output and returns:
    - cleaned semantic_json
    - confidence report (for logging / debugging / UI)

    Raises ValueError if semantics are unusable.
    """

    confidence = {
        "status": "ok",
        "issues": [],
        "metrics": {},
    }

    nodes = semantic_json.get("nodes", [])
    edges = semantic_json.get("edges", [])

    # -----------------------------
    # 1. Basic sanity checks
    # -----------------------------

    if not nodes:
        raise ValueError("Semantic extraction failed: no nodes detected.")

    if len(nodes) < MIN_NODE_COUNT:
        confidence["issues"].append(
            f"Only {len(nodes)} node(s) detected — likely under-extraction."
        )

    # -----------------------------
    # 2. Role validation
    # -----------------------------

    for n in nodes:
        if n.get("role") not in ALLOWED_ROLES:
            n["role"] = "unknown"
            confidence["issues"].append(
                f"Invalid role corrected to 'unknown' for node {n.get('id')}"
            )

    # -----------------------------
    # 3. Shape → node coverage check (CRITICAL)
    # -----------------------------

    shape_elements = [
        e for e in visual_json.get("elements", [])
        if e.get("type") == "shape"
    ]

    node_count = len(nodes)
    shape_count = len(shape_elements)

    confidence["metrics"]["shape_count"] = shape_count
    confidence["metrics"]["node_count"] = node_count

    if shape_count > 0:
        ratio = node_count / shape_count
        confidence["metrics"]["node_shape_ratio"] = round(ratio, 2)

        if ratio < MIN_NODE_SHAPE_RATIO:
            confidence["issues"].append(
                f"Only {node_count}/{shape_count} shapes mapped to nodes "
                f"({ratio:.2f}). Semantic under-extraction likely."
            )

    # -----------------------------
    # 4. Edge integrity check
    # -----------------------------

    node_ids = {n["id"] for n in nodes}
    valid_edges = []

    for e in edges:
        if e.get("from") in node_ids and e.get("to") in node_ids:
            valid_edges.append(e)
        else:
            confidence["issues"].append(
                f"Dropped invalid edge: {e}"
            )

    semantic_json["edges"] = valid_edges

    # -----------------------------
    # 5. Structural expectations
    # -----------------------------

    roles_present = {n["role"] for n in nodes}

    if "decision" in roles_present and len(valid_edges) < 2:
        confidence["issues"].append(
            "Decision node detected but insufficient branching edges."
        )

    # -----------------------------
    # 6. Final decision
    # -----------------------------

    if confidence["issues"]:
        confidence["status"] = "warning"

    # Hard fail conditions (DO NOT render PPT)
    # if (
    #     node_count < MIN_NODE_COUNT
    #     or shape_count > 0 and node_count / max(shape_count, 1) < 0.3
    # ):
    #     raise ValueError(
    #         "Semantic extraction confidence too low. "
    #         "Refusing to generate PPT.\n"
    #         f"Issues: {confidence['issues']}"
    #     )

    return semantic_json, confidence
