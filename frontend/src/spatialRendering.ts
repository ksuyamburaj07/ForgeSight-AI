import type { Projected } from "./spatialMath.ts";

export interface SurfaceTile {
  corners: [Projected, Projected, Projected, Projected];
  depth: number;
  distance: number;
}

export function fitProjectedPoints(
  points: Projected[],
  width: number,
  height: number,
  zoom = 1,
): Projected[] {
  if (!points.length) return [];

  const xs = points.map((p) => p.x);
  const ys = points.map((p) => p.y);

  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);

  const spanX = Math.max(maxX - minX, 1e-6);
  const spanY = Math.max(maxY - minY, 1e-6);

  const scale = Math.min(
    width * 0.86 / spanX,
    height * 0.80 / spanY,
  );

  const centreX = (minX + maxX) / 2;
  const centreY = (minY + maxY) / 2;

  return points.map((p) => ({
    ...p,
    x: width / 2 + (p.x - centreX) * scale * zoom,
    y: height / 2 + (p.y - centreY) * scale * zoom,
  }));
}

export function createSurfaceTiles(
  points: Projected[],
): SurfaceTile[] {
  const grid: (Projected | null)[][] =
    Array.from(
      { length: 48 },
      () => Array<Projected | null>(48).fill(null),
    );

  for (const point of points) {
    if (
      Number.isInteger(point.row) &&
      Number.isInteger(point.col) &&
      point.row >= 0 &&
      point.row < 48 &&
      point.col >= 0 &&
      point.col < 48
    ) {
      grid[point.row][point.col] = point;
    }
  }

  const tiles: SurfaceTile[] = [];

  for (let row = 0; row < 47; row++) {
    for (let col = 0; col < 47; col++) {
      const a = grid[row][col];
      const b = grid[row][col + 1];
      const c = grid[row + 1][col + 1];
      const d = grid[row + 1][col];

      // Do not create surfaces across excluded padding.
      if (!a || !b || !c || !d) continue;

      const corners: SurfaceTile["corners"] =
        [a, b, c, d];

      tiles.push({
        corners,
        depth: corners.reduce(
          (sum, p) => sum + p.depth,
          0,
        ) / 4,
        distance: corners.reduce(
          (sum, p) => sum + p.distance,
          0,
        ) / 4,
      });
    }
  }

  return tiles.sort((a, b) => a.depth - b.depth);
}

function insideTriangle(
  x: number,
  y: number,
  a: Projected,
  b: Projected,
  c: Projected,
): boolean {
  const denominator =
    (b.y - c.y) * (a.x - c.x) +
    (c.x - b.x) * (a.y - c.y);

  if (Math.abs(denominator) < 1e-10) {
    return false;
  }

  const u = (
    (b.y - c.y) * (x - c.x) +
    (c.x - b.x) * (y - c.y)
  ) / denominator;

  const v = (
    (c.y - a.y) * (x - c.x) +
    (a.x - c.x) * (y - c.y)
  ) / denominator;

  const w = 1 - u - v;
  const epsilon = 1e-7;

  return (
    u >= -epsilon &&
    v >= -epsilon &&
    w >= -epsilon
  );
}

export function pickSurfacePatch(
  tiles: SurfaceTile[],
  x: number,
  y: number,
): Projected | null {
  // Reverse the painter's drawing order: the last
  // rendered overlapping tile is inspected first.
  for (let i = tiles.length - 1; i >= 0; i--) {
    const corners = tiles[i].corners;
    const [a, b, c, d] = corners;

    if (
      !insideTriangle(x, y, a, b, c) &&
      !insideTriangle(x, y, a, c, d)
    ) {
      continue;
    }

    return [...corners].sort((p, q) => {
      const pd = (p.x - x) ** 2 + (p.y - y) ** 2;
      const qd = (q.x - x) ** 2 + (q.y - y) ** 2;
      return pd - qd;
    })[0];
  }

  return null;
}
