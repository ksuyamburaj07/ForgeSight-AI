import { useRef, useState } from "react";
import type { CSSProperties, KeyboardEvent, PointerEvent } from "react";
import type { InspectionResponse } from "./types";
import {
  boundPan,
  clamp,
  patchBounds,
  screenToSource,
  sourceToPatch,
  zoomAround,
} from "./geometry";
import type { Patch, ViewTransform } from "./geometry";
import "./InspectionViewer.css";

const INITIAL_VIEW: ViewTransform = { zoom: 1, pan: { x: 0, y: 0 } };
const PANELS = [
  { key: "original", number: "01", label: "Original", detail: "Source image" },
  {
    key: "heatmap",
    number: "02",
    label: "Distance map",
    detail: "Interpolated evidence",
  },
  {
    key: "overlay",
    number: "03",
    label: "Overlay",
    detail: "Source-aligned view",
  },
] as const;

interface Drag {
  pointerId: number;
  x: number;
  y: number;
  width: number;
  height: number;
  pan: { x: number; y: number };
  moved: boolean;
}

export function InspectionViewer({
  result,
}: {
  result: InspectionResponse | null;
}) {
  const identity = result
    ? `${result.inspection.source_sha256}:${result.inspection.category}:${result.inspection.processing_ms}`
    : "empty";
  return <InspectionCanvas key={identity} result={result} />;
}

export default InspectionViewer;

function InspectionCanvas({ result }: { result: InspectionResponse | null }) {
  const [view, setView] = useState<ViewTransform>(INITIAL_VIEW);
  const [blend, setBlend] = useState(100);
  const [selected, setSelected] = useState<Patch | null>(null);
  const [dragging, setDragging] = useState(false);
  const drag = useRef<Drag | null>(null);
  const geometry = result?.inspection.geometry;
  const ratio = geometry ? geometry[4] / geometry[5] : 4 / 3;
  const fitStyle: CSSProperties = {
    width: ratio >= 4 / 3 ? "100%" : `${(ratio / (4 / 3)) * 100}%`,
    height: ratio >= 4 / 3 ? `${(4 / 3 / ratio) * 100}%` : "100%",
  };
  const bounds = selected && geometry ? patchBounds(selected, geometry) : null;
  const valid =
    selected && result
      ? result.patch_evidence.valid_mask[selected.row]?.[selected.column]
      : null;
  const distance =
    selected && result
      ? result.patch_evidence.distances[selected.row]?.[selected.column]
      : null;

  function pointerDown(event: PointerEvent<HTMLDivElement>) {
    if (!result || event.button !== 0 || drag.current) return;
    const rect = event.currentTarget.getBoundingClientRect();
    drag.current = {
      pointerId: event.pointerId,
      x: event.clientX,
      y: event.clientY,
      width: rect.width,
      height: rect.height,
      pan: view.pan,
      moved: false,
    };
    event.currentTarget.setPointerCapture(event.pointerId);
    setDragging(true);
  }

  function pointerMove(event: PointerEvent<HTMLDivElement>) {
    const start = drag.current;
    if (!start || start.pointerId !== event.pointerId) return;
    const dx = event.clientX - start.x;
    const dy = event.clientY - start.y;
    if (Math.hypot(dx, dy) > 4) start.moved = true;
    if (start.moved) {
      setView((current) => ({
        ...current,
        pan: boundPan(
          {
            x: start.pan.x + dx / start.width,
            y: start.pan.y + dy / start.height,
          },
          current.zoom,
        ),
      }));
    }
  }

  function pointerUp(event: PointerEvent<HTMLDivElement>) {
    const start = drag.current;
    if (!start || start.pointerId !== event.pointerId) return;
    if (!start.moved && geometry) {
      const rect = event.currentTarget.getBoundingClientRect();
      const point = screenToSource(
        {
          x: (event.clientX - rect.left) / rect.width,
          y: (event.clientY - rect.top) / rect.height,
        },
        view,
      );
      setSelected(sourceToPatch(point, geometry));
    }
    drag.current = null;
    setDragging(false);
  }

  function cancelPointer() {
    drag.current = null;
    setDragging(false);
  }
  function viewKeyDown(event: KeyboardEvent<HTMLDivElement>) {
    const step = event.shiftKey ? 0.12 : 0.04;
    const directions: Record<string, { x: number; y: number }> = {
      ArrowLeft: { x: step, y: 0 },
      ArrowRight: { x: -step, y: 0 },
      ArrowUp: { x: 0, y: step },
      ArrowDown: { x: 0, y: -step },
    };
    if (directions[event.key]) {
      event.preventDefault();
      const delta = directions[event.key];
      setView((current) => ({
        ...current,
        pan: boundPan(
          { x: current.pan.x + delta.x, y: current.pan.y + delta.y },
          current.zoom,
        ),
      }));
    } else if (event.key === "+" || event.key === "=") {
      event.preventDefault();
      setView((current) => zoomAround(current, current.zoom + 0.25));
    } else if (event.key === "-") {
      event.preventDefault();
      setView((current) => zoomAround(current, current.zoom - 0.25));
    } else if (event.key === "0") {
      event.preventDefault();
      setView(INITIAL_VIEW);
    } else if ((event.key === "Enter" || event.key === " ") && geometry) {
      event.preventDefault();
      setSelected(
        sourceToPatch(screenToSource({ x: 0.5, y: 0.5 }, view), geometry),
      );
    }
  }
  function selectAxis(axis: "row" | "column", value: string) {
    if (value === "") return;
    setSelected((current) => ({
      ...(current ?? { row: 24, column: 24 }),
      [axis]: clamp(Math.round(Number(value)), 0, 47),
    }));
  }

  return (
    <section className="inspection-viewer" aria-labelledby="viewer-title">
      <div className="viewer-heading">
        <div>
          <span className="viewer-eyebrow">INSPECTION WORKSPACE</span>
          <h2 id="viewer-title">Three views. One source.</h2>
        </div>
        <span className={`viewer-linked ${result ? "is-active" : ""}`}>
          <span aria-hidden="true">↔</span> Views linked
        </span>
      </div>

      <div className="viewer-toolbar">
        <div className="viewer-zoom" aria-label="Synchronized zoom controls">
          <button
            type="button"
            disabled={!result || view.zoom <= 1}
            onClick={() => setView((v) => zoomAround(v, v.zoom - 0.25))}
            aria-label="Zoom out"
          >
            −
          </button>
          <label>
            <span className="viewer-sr-only">Zoom all views</span>
            <input
              type="range"
              min="1"
              max="5"
              step="0.05"
              value={view.zoom}
              disabled={!result}
              onChange={(e) =>
                setView((v) => zoomAround(v, Number(e.target.value)))
              }
              aria-valuetext={`${Math.round(view.zoom * 100)} percent`}
            />
          </label>
          <button
            type="button"
            disabled={!result || view.zoom >= 5}
            onClick={() => setView((v) => zoomAround(v, v.zoom + 0.25))}
            aria-label="Zoom in"
          >
            +
          </button>
          <output>{Math.round(view.zoom * 100)}%</output>
          <button
            className="viewer-fit"
            type="button"
            disabled={!result}
            onClick={() => setView(INITIAL_VIEW)}
          >
            Fit
          </button>
        </div>
        <p className="viewer-gesture">
          <span aria-hidden="true">⌖</span> Click to inspect · Drag to pan
        </p>
      </div>

      <div className="viewer-panels">
        {PANELS.map((panel) => (
          <figure className="viewer-panel" key={panel.key}>
            <figcaption>
              <div>
                <strong>{panel.label}</strong>
              </div>
              <span>{panel.detail}</span>
            </figcaption>
            <div className={`viewer-stage ${result ? "has-image" : ""}`}>
              {result ? (
                <div
                  className={`viewer-image-window ${dragging ? "is-dragging" : ""}`}
                  style={fitStyle}
                  onPointerDown={pointerDown}
                  onPointerMove={pointerMove}
                  onPointerUp={pointerUp}
                  onPointerCancel={cancelPointer}
                  onLostPointerCapture={cancelPointer}
                  onKeyDown={viewKeyDown}
                  tabIndex={0}
                  role="group"
                  aria-label={`${panel.label} for ${result.inspection.category.toUpperCase()}. Arrow keys pan; plus and minus zoom; zero resets; Enter inspects the centre. Or select raw patch coordinates below.`}
                >
                  <div
                    className="viewer-image-plane"
                    style={{
                      transform: `translate(${view.pan.x * 100}%, ${view.pan.y * 100}%) scale(${view.zoom})`,
                    }}
                  >
                    {panel.key === "overlay" && (
                      <img
                        src={result.visualizations.original}
                        alt=""
                        draggable={false}
                      />
                    )}
                    <img
                      src={result.visualizations[panel.key]}
                      alt=""
                      draggable={false}
                      style={
                        panel.key === "overlay"
                          ? { opacity: blend / 100 }
                          : undefined
                      }
                    />
                    {bounds && (
                      <span
                        className={`viewer-patch-box ${valid ? "" : "is-excluded"}`}
                        aria-hidden="true"
                        style={
                          {
                            left: `${bounds.x * 100}%`,
                            top: `${bounds.y * 100}%`,
                            width: `${bounds.width * 100}%`,
                            height: `${bounds.height * 100}%`,
                            "--inverse-zoom": 1 / view.zoom,
                          } as CSSProperties
                        }
                      >
                        <i />
                      </span>
                    )}
                  </div>
                </div>
              ) : (
                <div className="viewer-empty">
                  <div className="viewer-reticle" aria-hidden="true">
                    <span />
                    <i />
                  </div>
                  <span className="viewer-empty-label">
                    {panel.key === "original"
                      ? "Awaiting image"
                      : panel.key === "heatmap"
                        ? "Evidence begins here"
                        : "See it in context"}
                  </span>
                  <small>
                    {panel.key === "original"
                      ? "Choose a PCB and upload an image"
                      : "Available after a real inspection"}
                  </small>
                </div>
              )}
              <span
                className="viewer-corner viewer-corner-tl"
                aria-hidden="true"
              />
              <span
                className="viewer-corner viewer-corner-br"
                aria-hidden="true"
              />
            </div>
            <div className="viewer-panel-footer">
              {panel.key === "original" ? (
                <>
                  <span>
                    {geometry
                      ? `${geometry[4]} × ${geometry[5]} source px`
                      : "Orientation-corrected source"}
                  </span>
                  <span>INPUT</span>
                </>
              ) : panel.key === "heatmap" ? (
                <>
                  <span className="viewer-legend" aria-hidden="true" />
                  <span>Per-image scale</span>
                </>
              ) : (
                <label className="viewer-blend">
                  <span>Blend</span>
                  <input
                    type="range"
                    min="0"
                    max="100"
                    step="1"
                    value={blend}
                    disabled={!result}
                    onChange={(e) => setBlend(Number(e.target.value))}
                    aria-label="Blend original with service overlay"
                  />
                  <output>{blend}%</output>
                </label>
              )}
            </div>
          </figure>
        ))}
      </div>

      <div
        className={`viewer-patch-inspector ${selected ? "has-selection" : ""}`}
      >
        <div className="viewer-patch-intro">
          <span className="viewer-crosshair" aria-hidden="true">
            ⌖
          </span>
          <div>
            <strong>Raw patch inspector</strong>
            <p>
              {result
                ? "Select any view or enter grid coordinates."
                : "Raw measurements appear after inspection."}
            </p>
          </div>
        </div>
        <div className="viewer-patch-coordinates">
          <label>
            Row{" "}
            <input
              aria-label="Raw patch row, zero based"
              type="number"
              min="0"
              max="47"
              step="1"
              disabled={!result}
              value={selected?.row ?? ""}
              placeholder="—"
              onChange={(e) => selectAxis("row", e.target.value)}
            />
          </label>
          <label>
            Column{" "}
            <input
              aria-label="Raw patch column, zero based"
              type="number"
              min="0"
              max="47"
              step="1"
              disabled={!result}
              value={selected?.column ?? ""}
              placeholder="—"
              onChange={(e) => selectAxis("column", e.target.value)}
            />
          </label>
        </div>
        <div className="viewer-patch-reading" aria-live="polite">
          <span>Cosine NN distance</span>
          <strong>
            {selected && valid && distance != null ? distance.toFixed(8) : "—"}
          </strong>
          <small>
            {selected
              ? valid
                ? "Eligible patch centre · zero-based indices"
                : "Excluded patch centre · padding is not measured evidence"
              : "48 × 48 grid · select a patch"}
          </small>
        </div>
      </div>
      <p className="viewer-method-note">
        <span aria-hidden="true">↳</span> Heatmaps show interpolated patch
        evidence, not pixel-level defect segmentation. Colors use a separate
        scale for each image. Blend 100% shows the service overlay; viewing
        controls do not change scores.
      </p>
    </section>
  );
}
