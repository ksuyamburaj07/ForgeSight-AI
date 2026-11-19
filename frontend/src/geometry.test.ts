import { strict as assert } from "node:assert";
import test from "node:test";
import {
  hasEligiblePatchCenters,
  boundPan,
  patchBounds,
  screenToSource,
  sourceToPatch,
  zoomAround,
} from "./geometry.ts";
import type { Geometry } from "./types.ts";

test("square source maps corners and centre onto the frozen grid", () => {
  const geometry: Geometry = [0, 0, 672, 672, 224, 224];
  assert.deepEqual(sourceToPatch({ x: 0, y: 0 }, geometry), {
    row: 0,
    column: 0,
  });
  assert.deepEqual(sourceToPatch({ x: 0.5, y: 0.5 }, geometry), {
    row: 24,
    column: 24,
  });
  assert.deepEqual(sourceToPatch({ x: 1, y: 1 }, geometry), {
    row: 47,
    column: 47,
  });
});

test("landscape and portrait map the image crop rather than stretching all 48 rows", () => {
  assert.deepEqual(sourceToPatch({ x: 0, y: 0 }, [0, 84, 672, 504, 640, 480]), {
    row: 6,
    column: 0,
  });
  assert.deepEqual(sourceToPatch({ x: 1, y: 1 }, [0, 84, 672, 504, 640, 480]), {
    row: 41,
    column: 47,
  });
  assert.deepEqual(sourceToPatch({ x: 0, y: 0 }, [84, 0, 504, 672, 480, 640]), {
    row: 0,
    column: 6,
  });
  assert.deepEqual(sourceToPatch({ x: 1, y: 1 }, [84, 0, 504, 672, 480, 640]), {
    row: 47,
    column: 41,
  });
});

test("a boundary point can map to an excluded patch centre; retain that cell", () => {
  const geometry: Geometry = [0, 94, 672, 483, 640, 460];
  const patch = sourceToPatch({ x: 0.5, y: 0 }, geometry);
  assert.equal(patch.row, 6);
  assert.ok((patch.row + 0.5) * 14 < geometry[1]);
  assert.ok(patchBounds(patch, geometry).y < 0);
});

test("inverse zoom/pan gives source coordinates for a click", () => {
  const point = screenToSource(
    { x: 0.75, y: 0.25 },
    { zoom: 2, pan: { x: 0.1, y: -0.1 } },
  );
  assert.equal(point.x, 0.575);
  assert.equal(point.y, 0.425);
  assert.deepEqual(boundPan({ x: 99, y: -99 }, 2), { x: 0.5, y: -0.5 });
});

test("zoom preserves its source anchor and reset removes pan", () => {
  const anchor = { x: 0.7, y: 0.3 };
  const view = { zoom: 2, pan: { x: 0.1, y: -0.1 } };
  const before = screenToSource(anchor, view);
  const after = screenToSource(anchor, zoomAround(view, 3, anchor));
  assert.ok(Math.abs(before.x - after.x) < 1e-12);
  assert.ok(Math.abs(before.y - after.y) < 1e-12);
  assert.deepEqual(zoomAround(view, 1), { zoom: 1, pan: { x: 0, y: 0 } });
});

test("thin images with no eligible patch centres are rejected before inference", () => {
  assert.equal(hasEligiblePatchCenters(4096, 1), false);
  assert.equal(hasEligiblePatchCenters(1344, 1), false);
  assert.equal(hasEligiblePatchCenters(672, 12), false);
  assert.equal(hasEligiblePatchCenters(672, 13), true);
  assert.equal(hasEligiblePatchCenters(12, 672), false);
  assert.equal(hasEligiblePatchCenters(13, 672), true);
  assert.equal(hasEligiblePatchCenters(640, 480), true);
});
