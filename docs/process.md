# How it works

CubeXpress is built to handle **large**, **rate-limited**, and **oversized**
requests against Google Earth Engine without you having to manage any of that by
hand. Four ideas do the heavy lifting.

---

## 1. Batched discovery ⚙️

When you pass a **list** of rts to `discover_images`, CubeXpress does not loop
over them one by one. It groups them into batches and resolves each batch
server-side, so many locations are mapped to image footprints in a single round
trip. This is dramatically faster than querying point-by-point.

---

## 2. Adaptive concurrency (AIMD) 🔁

Large runs push against GEE's rate limits. CubeXpress watches the responses:

- while requests succeed, it **grows** the worker pool (additive increase);
- on a rate-limit signal, it **halves** it (multiplicative decrease).

This AIMD loop finds the safe throughput on its own — fast when GEE is happy,
backing off the moment it pushes back. Worker count is about **network I/O**, not
CPU, so values like 8 workers over small batches work well.

---

## 3. Crash-safe checkpoints 💾

Pass `checkpoint="run.jsonl"` and every resolved location is appended to a small
journal. If the run is interrupted, calling it again with the same checkpoint
**resumes exactly where it stopped** — only the missing locations are
rediscovered. A signature guards against resuming with a different request list.

---

## 4. Reactive retiling on download 🔄

GEE rejects a request that is too large. Instead of guessing a tile size up
front, CubeXpress **probes once**: it tries the whole download, and only if GEE
rejects it on size does it learn the bytes-per-pixel from the error, split the
request into tiles that fit, download them in parallel, and merge them back into
a single GeoTIFF. The same learned cost is reused across same-shaped requests, so
a homogeneous table probes only once.

For polygons, `express_clip` grids the bounding box, keeps only the tiles that
**touch** the polygon (the rest become nodata, with no download cost), merges,
and masks the result to the polygon's shape.

---

Together these let you go from a handful of points to a 250k-tile run with the
same few function calls, while CubeXpress keeps the throughput safe and the
output correct.