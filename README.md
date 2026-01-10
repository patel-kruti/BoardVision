## BoardVision (React + Tailwind + Node + Mock MCP)

### Requirement: open from a phone on the same WiFi

The Vite dev server is configured to bind to:

- **host**: `0.0.0.0`
- **port**: `5173`

So from your phone (same WiFi), open:

- `http://<laptop-local-ip>:5173`

### Install & run

From repo root:

```bash
npm install
npm run dev
```

You should see:

- Frontend (Vite): `http://localhost:5173` and a **Network** URL like `http://192.168.x.x:5173`
- Backend (Express): `http://localhost:3001`
- Mock MCP server: `http://localhost:8080`

### Find laptop IP

- **Windows**: `ipconfig` → look for **IPv4 Address**
- **macOS/Linux**: `ifconfig` (or `ip a`)

### What happens in the UI

- **Process image**: uploads the image to Express → Express calls MCP `POST /tools/detect_visuals` → UI shows **visualJson**
- **Generate PPT**: UI sends visualJson to Express → Express calls MCP `POST /tools/reason_semantics` → Express generates a `.pptx` and returns a download link

### Express endpoints

- `POST /process-image`
- `GET  /status/:id`
- `POST /generate-ppt`

### Mock MCP (port 8080)

- `POST /tools/detect_visuals`
- `POST /tools/reason_semantics`
- `POST /tools/whiteboard_to_ppt` (orchestrator-style mock)

#### Use the REAL Python MCP (`local_mcp_new.py`) instead of mock output

By default the mock server returns hardcoded data (that’s why you see `makeVisualJson()`).

If you want the web app to call your real Python endpoints:

1) Start your Python server:

```bash
python start_server.py
```

2) Start the web stack with `REAL_MCP_URL` set (so the mock server proxies to Python):

- Windows CMD:

```bat
set REAL_MCP_URL=http://127.0.0.1:8000
npm run dev
```

Now, clicking **Process image** calls:

- `POST http://127.0.0.1:8000/tools/detect_visuals`

and clicking **Generate PPT** calls:

- `POST http://127.0.0.1:8000/tools/reason_semantics`

### Docker (optional)

```bash
docker compose up --build
```

Then open from phone:

- `http://<laptop-local-ip>:5173`
