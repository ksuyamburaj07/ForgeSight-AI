import {
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";

import type {
  KeyboardEvent as ReactKeyboardEvent,
  PointerEvent as ReactPointerEvent,
} from "react";

import type { InspectionResponse } from "./types";
import type { Patch } from "./geometry";

import {
  buildSurface,
  projectSample,
} from "./spatialMath";

import {
  fitProjectedPoints,
  createSurfaceTiles,
  pickSurfacePatch,
} from "./spatialRendering";
import type { SurfaceTile } from "./spatialRendering";

import "./SpatialLandscape.css";

interface Props {
  response: InspectionResponse;
  selectedPatch: Patch | null;
  onSelectPatch: (patch: Patch | null) => void;
}

interface Camera {
  yaw: number;
  elevation: number;
  zoom: number;
}

const INITIAL_CAMERA: Camera = {
  yaw: -35,
  elevation: 48,
  zoom: 1,
};

const clamp = (v: number, min: number, max: number) =>
  Math.max(min, Math.min(max, v));

export function SpatialLandscape({
  response,
  selectedPatch,
  onSelectPatch,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const visibleTilesRef = useRef<SurfaceTile[]>([]);

  const dragRef = useRef<{
    x: number;
    y: number;
    lastX: number;
    lastY: number;
    moved: boolean;
  } | null>(null);

  const surface = useMemo(
    () => buildSurface(response.patch_evidence),
    [response.patch_evidence],
  );

  const [camera, setCamera] =
    useState<Camera>(INITIAL_CAMERA);

  const selected = selectedPatch
    ? surface.points.find(
        (p) =>
          p.row === selectedPatch.row &&
          p.col === selectedPatch.column,
      ) ?? null
    : null;


  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const draw = () => {
      const rect = canvas.getBoundingClientRect();
      const width = rect.width;
      const height = rect.height;

      if (width <= 0 || height <= 0) return;

      const dpr = Math.min(
        window.devicePixelRatio || 1,
        2,
      );

      canvas.width = Math.round(width * dpr);
      canvas.height = Math.round(height * dpr);

      const ctx = canvas.getContext("2d");
      if (!ctx) return;

      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.fillStyle = "#1E2522";
      ctx.fillRect(0, 0, width, height);

      // Fit at neutral zoom, then apply the user's zoom.
      const baseProjected = surface.points.map(
        (sample) => projectSample(
          sample,
          { ...camera, zoom: 1 },
          width,
          height,
          surface.maxDistance,
        ),
      );

      const projected = fitProjectedPoints(
        baseProjected,
        width,
        height,
        camera.zoom,
      );

      const tiles = createSurfaceTiles(projected);
      visibleTilesRef.current = tiles;

      for (const tile of tiles) {
        const normalized = surface.maxDistance > 0
          ? clamp(
              tile.distance / surface.maxDistance,
              0, 1,
            )
          : 0;

        const hue = 185 - 165 * normalized;
        const lightness = 37 + 20 * normalized;

        ctx.beginPath();
        ctx.moveTo(
          tile.corners[0].x,
          tile.corners[0].y,
        );

        for (let i = 1; i < 4; i++) {
          ctx.lineTo(
            tile.corners[i].x,
            tile.corners[i].y,
          );
        }

        ctx.closePath();
        ctx.fillStyle =
          `hsl(${hue} 72% ${lightness}%)`;
        ctx.fill();

        ctx.strokeStyle =
          "rgba(237, 248, 228, 0.13)";
        ctx.lineWidth = 0.45;
        ctx.stroke();
      }

      const marker = projected.find(
        (p) =>
          selected !== null &&
          p.row === selected.row &&
          p.col === selected.col,
      );

      if (marker) {
        ctx.beginPath();
        ctx.arc(
          marker.x,
          marker.y,
          6,
          0,
          Math.PI * 2,
        );
        ctx.fillStyle = "#C7F04B";
        ctx.fill();
        ctx.strokeStyle = "#15211B";
        ctx.lineWidth = 2;
        ctx.stroke();

        ctx.beginPath();
        ctx.moveTo(marker.x, marker.y - 13);
        ctx.lineTo(marker.x, marker.y - 24);
        ctx.strokeStyle = "#C7F04B";
        ctx.lineWidth = 1.5;
        ctx.stroke();
      }
    };

    draw();

    const resizeObserver = new ResizeObserver(draw);
    resizeObserver.observe(canvas);

    return () => resizeObserver.disconnect();
  }, [surface, camera, selected]);

  function onPointerDown(
    event: ReactPointerEvent<HTMLCanvasElement>,
  ) {
    event.currentTarget.setPointerCapture(
      event.pointerId,
    );

    dragRef.current = {
      x: event.clientX,
      y: event.clientY,
      lastX: event.clientX,
      lastY: event.clientY,
      moved: false,
    };
  }

  function onPointerMove(
    event: ReactPointerEvent<HTMLCanvasElement>,
  ) {
    const drag = dragRef.current;
    if (!drag) return;

    const dx = event.clientX - drag.lastX;
    const dy = event.clientY - drag.lastY;

    if (
      Math.hypot(
        event.clientX - drag.x,
        event.clientY - drag.y,
      ) > 4
    ) {
      drag.moved = true;
    }

    drag.lastX = event.clientX;
    drag.lastY = event.clientY;

    if (!drag.moved) return;

    setCamera((current) => ({
      ...current,
      yaw: current.yaw + dx * 0.45,
      elevation: clamp(
        current.elevation + dy * 0.35,
        15,
        77,
      ),
    }));
  }

  function onPointerUp(
    event: ReactPointerEvent<HTMLCanvasElement>,
  ) {
    const drag = dragRef.current;
    dragRef.current = null;

    if (
      event.currentTarget.hasPointerCapture(
        event.pointerId,
      )
    ) {
      event.currentTarget.releasePointerCapture(
        event.pointerId,
      );
    }

    if (!drag || drag.moved) return;

    const rect =
      event.currentTarget.getBoundingClientRect();

    const chosen = pickSurfacePatch(
      visibleTilesRef.current,
      event.clientX - rect.left,
      event.clientY - rect.top,
    );

    if (chosen) {
      onSelectPatch({
        row: chosen.row,
        column: chosen.col,
      });
    }
  }

  function onKeyDown(
    event: ReactKeyboardEvent<HTMLCanvasElement>,
  ) {
    const moves: Record<string, [number, number]> = {
      ArrowUp: [-1, 0],
      ArrowDown: [1, 0],
      ArrowLeft: [0, -1],
      ArrowRight: [0, 1],
    };

    const move = moves[event.key];
    if (!move) return;

    event.preventDefault();

    if (!selected) return;

    const [dr, dc] = move;
    let row = selected.row + dr;
    let col = selected.col + dc;

    while (
      row >= 0 && row < 48 &&
      col >= 0 && col < 48
    ) {
      const found = surface.points.find(
        (p) => p.row === row && p.col === col,
      );

      if (found) {
        onSelectPatch({
          row: found.row,
          column: found.col,
        });
        break;
      }

      row += dr;
      col += dc;
    }
  }

  const selectedDistance = selected?.distance;

  return (
    <section
      className="fs-spatial"
      aria-labelledby="fs-spatial-heading"
    >
      <div className="fs-spatial-heading">
        <div>
          <span className="fs-spatial-eyebrow">
            SPATIAL ANALYSIS / LIVE EVIDENCE
          </span>
          <h2 id="fs-spatial-heading">
            Anomaly distance landscape.
          </h2>
          <p>
            Orbit the measured 48×48 feature-distance
            field. Height represents relative patch
            distance, not physical PCB geometry.
          </p>
        </div>

        <div className="fs-spatial-counter">
          <strong>{surface.validCount}</strong>
          <span>eligible patches</span>
        </div>
      </div>

      <div className="fs-spatial-controls">
        <button
          type="button"
          onClick={() =>
            setCamera((c) => ({
              ...c,
              yaw: c.yaw - 20,
            }))
          }
        >
          Rotate left
        </button>

        <button
          type="button"
          onClick={() =>
            setCamera((c) => ({
              ...c,
              yaw: c.yaw + 20,
            }))
          }
        >
          Rotate right
        </button>

        <button
          type="button"
          onClick={() =>
            setCamera((c) => ({
              ...c,
              zoom: clamp(c.zoom / 1.2, 0.6, 2),
            }))
          }
        >
          − Zoom
        </button>

        <button
          type="button"
          onClick={() =>
            setCamera((c) => ({
              ...c,
              zoom: clamp(c.zoom * 1.2, 0.6, 2),
            }))
          }
        >
          + Zoom
        </button>

        <button
          type="button"
          onClick={() => setCamera(INITIAL_CAMERA)}
        >
          Reset camera
        </button>

        <button
          type="button"
          onClick={() =>
            setCamera({
              yaw: 0,
              elevation: 90,
              zoom: 1,
            })
          }
        >
          Top view
        </button>
      </div>

      <canvas
        ref={canvasRef}
        className="fs-spatial-canvas"
        tabIndex={0}
        aria-label={
          "Interactive spatial distance surface. " +
          "Drag to rotate, click to inspect a patch, " +
          "or use arrow keys when focused."
        }
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerCancel={() => {
          dragRef.current = null;
        }}
        onKeyDown={onKeyDown}
      >
        Spatial distances are also available in
        the inspection evidence JSON.
      </canvas>

      <div className="fs-spatial-footer">
        <div className="fs-spatial-readout" aria-live="polite">
          <span>Selected measurement</span>
          <strong>
            {selectedPatch
              ? `Row ${selectedPatch.row}, column ${selectedPatch.column}`
              : "No patch selected"}
          </strong>
          <code>
            {selectedDistance !== undefined
              ? selectedDistance.toFixed(8)
              : selectedPatch
                ? "Excluded — no measured distance"
                : "—"}
          </code>
        </div>

        <div className="fs-spatial-legend">
          <span>Relative colour scale</span>
          <div className="fs-spatial-gradient" />
          <small>
            Low measured distance → high measured distance
          </small>
        </div>
      </div>

      <p className="fs-spatial-disclaimer">
        Values come from the current inspection's raw
        cosine nearest-neighbor distances. Colours and
        heights are scaled within this image only.
        Excluded letterbox patches are omitted.
        This is neither defect segmentation nor a
        physical 3D reconstruction.
      </p>
    </section>
  );
}
