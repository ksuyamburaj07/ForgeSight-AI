import type { Geometry } from "./types";

export interface Point {
  x: number;
  y: number;
}
export interface ViewTransform {
  zoom: number;
  pan: Point;
}
export interface Patch {
  row: number;
  column: number;
}

export const clamp = (value: number, min: number, max: number) =>
  Math.min(max, Math.max(min, value));

/** Pan is expressed in fractions of the unscaled, source-aligned image. */
export function boundPan(pan: Point, zoom: number): Point {
  const limit = Math.max(0, (zoom - 1) / 2);
  if (limit === 0) return { x: 0, y: 0 };
  return { x: clamp(pan.x, -limit, limit), y: clamp(pan.y, -limit, limit) };
}

export function screenToSource(point: Point, view: ViewTransform): Point {
  return {
    x: (point.x - 0.5 - view.pan.x) / view.zoom + 0.5,
    y: (point.y - 0.5 - view.pan.y) / view.zoom + 0.5,
  };
}

export function zoomAround(
  view: ViewTransform,
  nextZoom: number,
  anchor: Point = { x: 0.5, y: 0.5 },
): ViewTransform {
  const zoom = clamp(nextZoom, 1, 5);
  const source = screenToSource(anchor, view);
  return {
    zoom,
    pan: boundPan(
      {
        x: anchor.x - 0.5 - (source.x - 0.5) * zoom,
        y: anchor.y - 0.5 - (source.y - 0.5) * zoom,
      },
      zoom,
    ),
  };
}

/** Map source coordinates into the frozen 672px letterbox, never stretch the grid. */
export function sourceToPatch(point: Point, geometry: Geometry): Patch {
  const [left, top, width, height] = geometry;
  const x = left + clamp(point.x, 0, 1 - Number.EPSILON) * width;
  const y = top + clamp(point.y, 0, 1 - Number.EPSILON) * height;
  return {
    row: clamp(Math.floor(y / 14), 0, 47),
    column: clamp(Math.floor(x / 14), 0, 47),
  };
}

export function patchBounds(patch: Patch, geometry: Geometry) {
  const [left, top, width, height] = geometry;
  return {
    x: (patch.column * 14 - left) / width,
    y: (patch.row * 14 - top) / height,
    width: 14 / width,
    height: 14 / height,
  };
}

/** Detect unsupported thin inputs before upload, without altering frozen preprocessing. */
export function hasEligiblePatchCenters(
  width: number,
  height: number,
): boolean {
  if (width <= 0 || height <= 0) return false;
  const scale = Math.min(672 / width, 672 / height);
  const roundEven = (value: number) => {
    const floor = Math.floor(value);
    return value - floor === 0.5 ? floor + (floor % 2) : Math.round(value);
  };
  const resizedWidth = Math.min(672, roundEven(width * scale));
  const resizedHeight = Math.min(672, roundEven(height * scale));
  const left = Math.floor((672 - resizedWidth) / 2);
  const top = Math.floor((672 - resizedHeight) / 2);
  const centres = Array.from({ length: 48 }, (_, i) => (i + 0.5) * 14);
  return (
    centres.some((x) => x >= left && x < left + resizedWidth) &&
    centres.some((y) => y >= top && y < top + resizedHeight)
  );
}
