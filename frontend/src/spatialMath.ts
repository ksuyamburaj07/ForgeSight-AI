import type { InspectionResponse } from "./types";

type Evidence = InspectionResponse["patch_evidence"];

export interface Sample {
  row: number;
  col: number;
  distance: number;
}

export interface Projected extends Sample {
  x: number;
  y: number;
  depth: number;
}

export interface Surface {
  points: Sample[];
  maxDistance: number;
  validCount: number;
}

const SIZE = 48;

export function buildSurface(evidence: Evidence): Surface {
  if (
    evidence.grid[0] !== SIZE ||
    evidence.grid[1] !== SIZE
  ) {
    throw new Error("Expected frozen 48x48 patch evidence");
  }

  const points: Sample[] = [];
  let maxDistance = 0;

  for (let row = 0; row < SIZE; row++) {
    const distances = evidence.distances[row];
    const mask = evidence.valid_mask[row];

    if (
      !Array.isArray(distances) ||
      !Array.isArray(mask) ||
      distances.length !== SIZE ||
      mask.length !== SIZE
    ) {
      throw new Error("Malformed patch evidence grid");
    }

    for (let col = 0; col < SIZE; col++) {
      if (typeof mask[col] !== "boolean") {
        throw new Error("Non-boolean patch eligibility");
      }

      if (!mask[col]) continue;

      const distance = distances[col];

      if (
        !Number.isFinite(distance) ||
        distance < 0
      ) {
        throw new Error("Invalid eligible patch distance");
      }

      points.push({ row, col, distance });
      maxDistance = Math.max(maxDistance, distance);
    }
  }

  if (points.length === 0) {
    throw new Error("No eligible model patches");
  }

  return {
    points,
    maxDistance,
    validCount: points.length,
  };
}

export function projectSample(
  sample: Sample,
  camera: {
    yaw: number;
    elevation: number;
    zoom: number;
  },
  width: number,
  height: number,
  maxDistance: number,
): Projected {
  const x = (sample.col - 23.5) / 23.5;
  const y = (sample.row - 23.5) / 23.5;

  const heightValue = maxDistance > 0
    ? sample.distance / maxDistance
    : 0;

  const yaw = camera.yaw * Math.PI / 180;
  const elevation = camera.elevation * Math.PI / 180;

  const rotatedX =
    x * Math.cos(yaw) - y * Math.sin(yaw);

  const rotatedY =
    x * Math.sin(yaw) + y * Math.cos(yaw);

  const z = heightValue * 1.15;
  const scale = Math.min(width / 3, height / 3) *
    camera.zoom;

  return {
    ...sample,
    x: width / 2 + rotatedX * scale,
    y: height * 0.60 +
      (rotatedY * Math.sin(elevation) -
       z * Math.cos(elevation)) * scale,
    depth:
      rotatedY * Math.cos(elevation) +
      z * Math.sin(elevation),
  };
}

export function nearestProjected(
  points: Projected[],
  x: number,
  y: number,
  radius = 24,
): Projected | null {
  let best: Projected | null = null;
  let bestSquared = radius * radius;

  for (const point of points) {
    const dx = point.x - x;
    const dy = point.y - y;
    const squared = dx * dx + dy * dy;

    if (
      squared < bestSquared ||
      (squared === bestSquared &&
       best !== null &&
       point.depth > best.depth)
    ) {
      best = point;
      bestSquared = squared;
    }
  }

  return best;
}
