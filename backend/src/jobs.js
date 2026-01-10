import { nanoid } from "nanoid";

const jobs = new Map();

export function createJob() {
  const id = nanoid(10);
  jobs.set(id, {
    id,
    status: "queued", // queued | processing | done | error
    createdAt: Date.now(),
    updatedAt: Date.now(),
    result: null,
    error: null,
    imagePath: null
  });
  return jobs.get(id);
}

export function getJob(id) {
  return jobs.get(id) || null;
}

export function updateJob(id, patch) {
  const j = jobs.get(id);
  if (!j) return null;
  const next = { ...j, ...patch, updatedAt: Date.now() };
  jobs.set(id, next);
  return next;
}


