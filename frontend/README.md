# ForgeSight AI · Inspection Studio

Local React + TypeScript + Vite interface for the existing M3.15 FastAPI service.
The Python environment, frozen preprocessing, model, banks, thresholds, and scientific gate are unchanged.

## Run locally

From the repository root, start one CPU service worker:

```bash
cd /data/Projects/Project_05
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m uvicorn forgesight.api.app:app --host 127.0.0.1 --port 8000 --workers 1
```

In a second terminal (Node.js 24 or newer):

```bash
cd /data/Projects/Project_05/frontend
npm ci
npm run dev
```

Open http://127.0.0.1:5173/. The Vite development server proxies `/api` to the local FastAPI process on port 8000. Both processes are needed. An unavailable service gives a reconnect action; no fallback results are generated. Model loading remains lazy on the first valid inspection. The original verified model cache and frozen artifact directory must remain available.

For a local production-build preview:

```bash
npm run build
npm run preview
```

Open http://127.0.0.1:4173/. Preview also proxies `/api` to port 8000. The `dist` directory alone requires equivalent same-origin API routing if hosted elsewhere. This change does not publish a service or configure public ingress/authentication.

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
cd /data/Projects/Project_05/frontend
npm test
npm run build
cd /data/Projects/Project_05
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -W error tests/test_service_api.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -W error
# With the local API running; synthetic input only:
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/m315_real_http_smoke.py --category all
```

The frontend's Node tests cover raw-coordinate mapping through letterboxing and zoom, padding eligibility, API response/provenance validation, multipart uploads, service failures, and SHA-256 calculation. Transport fixtures in these tests are invented solely for tests and are never loaded into the UI. They are not scientific results.

Service tests use the already-installed HTTPX AsyncClient + ASGITransport, following [FastAPI's async testing guidance](https://fastapi.tiangolo.com/advanced/async-tests/). This avoids importing the deprecated Starlette TestClient fallback. No Python package was upgraded, installed, or silenced. Older commands that still import TestClient will continue to warn. There are no lifespan handlers in the current app; ASGITransport does not trigger startup/shutdown if those are introduced later.

## Scientific interpretation

- The CPU cross-platform score gate remains open. Three of four tracked M3.14B scores met the original absolute tolerance; the UI reproduces the tracked note, not a fresh benchmark.
- Scores are cosine-distance statistics, not probabilities. Review flags are research indicators. Threshold non-exceedance does not establish manufacturing acceptance.
- The 48×48 map is letterboxed patch evidence. Patches are eligible by their centres; padding zeros are not low-distance measurements. Views map clicks through the exact response geometry.
- Heatmaps are interpolated for display, not pixel-level defect masks. Color scaling is per image; hues cannot be quantitatively compared across inputs. The existing scientific visualization palette is retained.
- Background/boundary sensitivity remains a limitation. No foreground filtering, threshold tuning, automatic manufacturing decision, sealed-image browsing, or new accuracy claim is introduced.

Research view sources are the tracked `docs/engineering/M314B_CPU_REPRODUCTION_NOTE.md`, `M314C2_FOUR_PCB_VISUAL_AUDIT.md`, and frozen inference code. It loads no case images. Bank fingerprints are not supplied by this API and are not invented in the UI.

## Verification limits for this delivery

The production build, Node tests, 10 API tests, 227-test Python regression, direct real-model HTTP checks for all four categories, and the Vite API proxy were verified. Browser preview access was denied by the browser approval gate, so responsive visual inspection, pointer/keyboard interaction in a real browser, and screen-reader testing remain outstanding. The layout and interactions are implemented, but browser QA is not claimed.

The execution host also failed to wake an asyncio selector after AnyIO worker completion. Python test validation used a temporary external runner that bounds selector waits to 0.05 seconds. That runner is not application code and does not alter warnings, model math, test assertions, dependencies, or the API. The same runner reproduced the original baseline at 227 passed / one warning, and the updated suite at 227 passed with warnings treated as errors. Ordinary local pytest commands above are the intended workflow on the user's machine.
