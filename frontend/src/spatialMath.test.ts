import test from "node:test";
import assert from "node:assert/strict";

import {
  buildSurface,
  nearestProjected,
  projectSample,
} from "./spatialMath.ts";

import type { InspectionResponse } from "./types";

type Evidence = InspectionResponse["patch_evidence"];

function evidence(): Evidence {
  return {
    grid: [48, 48],
    distances: Array.from(
      { length: 48 },
      () => Array(48).fill(0.12),
    ),
    valid_mask: Array.from(
      { length: 48 },
      () => Array(48).fill(true),
    ),
    units: "cosine_nearest_neighbor_distance",
  };
}

test("all eligible patches are retained", () => {
  const result = buildSurface(evidence());

  assert.equal(result.validCount, 2304);
  assert.equal(result.maxDistance, 0.12);
});

test("excluded padding never becomes a surface sample", () => {
  const data = evidence();

  for (let col = 0; col < 48; col++) {
    data.valid_mask[0][col] = false;
    data.valid_mask[47][col] = false;
  }

  const result = buildSurface(data);

  assert.equal(result.validCount, 2208);
  assert.ok(
    result.points.every(
      (p) => p.row !== 0 && p.row !== 47,
    ),
  );
});

test("invalid eligible distances are rejected", () => {
  const data = evidence();
  data.distances[10][10] = Number.NaN;

  assert.throws(
    () => buildSurface(data),
    /Invalid eligible patch distance/,
  );
});

test("perspective projection stays finite", () => {
  const p = projectSample(
    { row: 24, col: 24, distance: 0.25 },
    { yaw: -35, elevation: 48, zoom: 1 },
    800,
    500,
    0.5,
  );

  assert.ok(Number.isFinite(p.x));
  assert.ok(Number.isFinite(p.y));
  assert.ok(Number.isFinite(p.depth));
});

test("pointer lookup selects the nearest eligible sample", () => {
  const p = projectSample(
    { row: 12, col: 18, distance: 0.2 },
    { yaw: 0, elevation: 45, zoom: 1 },
    800,
    500,
    0.5,
  );

  const found = nearestProjected(
    [p],
    p.x + 2,
    p.y + 2,
  );

  assert.equal(found?.row, 12);
  assert.equal(found?.col, 18);

  assert.equal(
    nearestProjected([p], 0, 0, 4),
    null,
  );
});
