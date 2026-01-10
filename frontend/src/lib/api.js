const API_BASE = "/api";

export async function processImage(file) {
  const form = new FormData();
  form.append("image", file);

  const res = await fetch(`${API_BASE}/process-image`, {
    method: "POST",
    body: form
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function getStatus(id) {
  const res = await fetch(`${API_BASE}/status/${encodeURIComponent(id)}`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function generatePpt({ id, visualJson }) {
  const res = await fetch(`${API_BASE}/generate-ppt`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ id, visualJson })
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}


