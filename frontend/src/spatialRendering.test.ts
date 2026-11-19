import test from "node:test";
import assert from "node:assert/strict";

import type { Projected } from "./spatialMath.ts";

import {
  fitProjectedPoints,
  createSurfaceTiles,
  pickSurfacePatch,
} from "./spatialRendering.ts";

function point(
  row: number,
  col: number,
  x: number,
  y: number,
  depth = 0,
): Projected {
  return {
    row,
    col,
    x,
    y,
    depth,
    distance: 0.25,
  };
}

test("surface is fitted within the canvas", () => {
  const points = [
    point(0, 0, -100, -50),
    point(0, 47, 200, -50),
    point(47, 0, -100, 50),
    point(47, 47, 200, 50),
  ];

  const result = fitProjectedPoints(
    points, 800, 500,
  );

  assert.ok(
    result.every(
      (p) =>
        p.x >= 0 && p.x <= 800 &&
        p.y >= 0 && p.y <= 500,
    ),
  );
});

test("zoom enlarges the fitted projection", () => {
  const points = [
    point(0, 0, 0, 0),
    point(1, 1, 100, 100),
  ];

  const normal = fitProjectedPoints(
    points, 800, 500, 1,
  );

  const zoomed = fitProjectedPoints(
    points, 800, 500, 2,
  );

  assert.ok(
    Math.abs(
      zoomed[1].x - zoomed[0].x,
    ) >
    Math.abs(
      normal[1].x - normal[0].x,
    ),
  );
});

test("tiles do not bridge missing patches", () => {
  const points = [
    point(0, 0, 0, 0),
    point(0, 1, 10, 0),
    point(1, 1, 10, 10),
    point(1, 0, 0, 10),
  ];

  assert.equal(
    createSurfaceTiles(points).length,
    1,
  );

  assert.equal(
    createSurfaceTiles(points.slice(0, 3)).length,
    0,
  );
});

test("overlapping surfaces pick the front drawn tile", () => {
  const back = [
    point(0, 0, 0, 0, -1),
    point(0, 1, 10, 0, -1),
    point(1, 1, 10, 10, -1),
    point(1, 0, 0, 10, -1),
  ] as [
    Projected,
    Projected,
    Projected,
    Projected,
  ];

  const front = [
    point(10, 10, 0, 0, 1),
    point(10, 11, 10, 0, 1),
    point(11, 11, 10, 10, 1),
    point(11, 10, 0, 10, 1),
  ] as typeof back;

  const tiles = [
    { corners: back, depth: -1, distance: 0.2 },
    { corners: front, depth: 1, distance: 0.5 },
  ];

  const selected = pickSurfacePatch(tiles, 2, 2);

  assert.equal(selected?.row, 10);
  assert.equal(
    pickSurfacePatch(tiles, 40, 40),
    null,
  );
});
