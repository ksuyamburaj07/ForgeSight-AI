# ForgeSight AI · Inspection Studio

React + TypeScript + Vite interface for the genuine FastAPI CPU inference service.

[Open the live Inspection Studio](https://forgesight-ai-6d5yatbeia-el.a.run.app/).
The frozen preprocessing, model, banks, thresholds, and open scientific gate are preserved.

## Run locally

Install the Python requirements and privately provision the original
authorized model snapshot and four verified banks using the
[CPU runtime guide](../docs/engineering/RELEASE_CPU_RUNTIME.md).
These model binaries and bank files are not supplied by this repository.

From the repository root, with the Python environment activated,
start one CPU service worker:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m uvicorn forgesight.api.app:app --host 127.0.0.1 --port 8000 --workers 1
```

In a second terminal, from the repository root (Node.js 24 or newer):

```bash
cd frontend
npm ci
npm run dev
```

Open http://127.0.0.1:5173/. The Vite development server proxies `/api` to the local FastAPI process on port 8000. Both processes are needed. An unavailable service gives a reconnect action; no fallback results are generated. Model loading remains lazy on the first valid inspection. The original verified model cache and frozen artifact directory must remain available.

For a local production-build preview:

```bash
npm run build
npm run preview
```

Open http://127.0.0.1:4173/. Preview also proxies `/api` to port 8000. The `dist` directory alone requires equivalent same-origin API routing if hosted elsewhere. The live Cloud Run application uses a separate private deployment context containing its authorized runtime artifacts and routing; that artifact-bearing context is not distributed in this source repository.

## Inspect

1. Select the matching PCB1–PCB4 reference bank, then upload one PNG/JPEG/WebP.
2. Select **Run inspection**. Original bytes are SHA-256 checked against the response before evidence is shown.
3. View the orientation-corrected source, returned distance heatmap, and returned overlay. Zoom controls affect all three views. Drag any image to pan. Select a point in any image to inspect its raw patch distance.
4. Alternatively enter zero-based row/column indices. With an image focused, arrow keys pan; `+`/`-` zoom; `0` fits; Enter/Space selects the visible centre. Range controls support keyboard input.
5. **Blend** crossfades the original and the service overlay. 100% means the existing service overlay, whose opacity already depends on evidence strength. It does not recalculate the visualization.
6. **Evidence explorer** displays the live record and downloads the complete response JSON, including raw distances and the eligibility mask. Only the current response is held in browser memory; a reload clears it.

**Use synthetic input** creates a deliberately artificial board drawing in the browser for integration checking. It only loads an input; it does not preload any score or result. Its rasterization is browser-dependent and is not the byte-identical Python M3.15C fixture. Do not compare its score to the Python checkpoint as numerical reproduction evidence. The byte-identical Pillow fixture is available through the script below.

## Checks

```bash
# From the repository root:
cd frontend
npm test
npm run build
cd ..
source .venv/bin/activate
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q tests/test_service_api.py
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q --ignore=tests/test_m312b3_evidence_review.py
# With the local API running; synthetic input only:
PYTHONDONTWRITEBYTECODE=1 python scripts/m315_real_http_smoke.py --category all
```

The frontend's Node tests cover raw-coordinate mapping through letterboxing and zoom, padding eligibility, API response/provenance validation, multipart uploads, service failures, and SHA-256 calculation. Transport fixtures in these tests are invented solely for tests and are never loaded into the UI. They are not scientific results.

Service tests use HTTPX AsyncClient + ASGITransport, following [FastAPI's async testing guidance](https://fastapi.tiangolo.com/advanced/async-tests/). There are no lifespan handlers in the current source API; ASGITransport does not trigger startup/shutdown if those are introduced later. Six historical evidence integrity tests require original ignored research files and are excluded from source CI; see the [test policy](../docs/engineering/CLEAN_CHECKOUT_TEST_POLICY.md).

## Scientific interpretation

- The CPU cross-platform score gate remains open. Three of four tracked M3.14B scores met the original absolute tolerance; the UI reproduces the tracked note, not a fresh benchmark.
- Scores are cosine-distance statistics, not probabilities. Review flags are research indicators. Threshold non-exceedance does not establish manufacturing acceptance.
- The 48×48 map is letterboxed patch evidence. Patches are eligible by their centres; padding zeros are not low-distance measurements. Views map clicks through the exact response geometry.
- Heatmaps are interpolated for display, not pixel-level defect masks. Color scaling is per image; hues cannot be quantitatively compared across inputs. The existing scientific visualization palette is retained.
- Background/boundary sensitivity remains a limitation. No foreground filtering, threshold tuning, automatic manufacturing decision, sealed-image browsing, or new accuracy claim is introduced.

Research views use the tracked `docs/engineering/M314B_CPU_REPRODUCTION_NOTE.md`, `M314C2_FOUR_PCB_VISUAL_AUDIT.md`, frozen inference code, and explicitly attributed public research assets. The separate historical Case Atlas shows four reused retrospective cases; the dataset gallery shows 16 normal-training examples. Neither is a fresh independent evaluation or a substitute for the original reference banks. Bank fingerprints are not supplied by the inspection API and are not invented in the UI.

## Verification and limits

At main commit `d68672e893fbe2a3934962c11f199d074079905b`, GitHub Actions passed **68 frontend tests**, the TypeScript/Vite production build, and **225 self-contained Python tests** on fresh runners. Six additional historical evidence tests passed separately in the original local research environment. These counts supersede earlier 67/227 checkpoint counts.

The hosted application returned real CPU inspection records, patch evidence and visualizations for synthetic inputs across PCB1–PCB4. Public HTTPS checks passed and the project owner confirmed browser testing. This establishes a working deployed application; it does not establish accessibility certification, broad device coverage, independent validation, or real-world defect-detection accuracy.

The numerical reproduction gate remains open (3 of 4 scores within the original tolerance; 4 of 4 review flags unchanged). The system remains a human-supervised, noncertified research inspection tool.
