import test from "node:test";
import assert from "node:assert/strict";

import {
  validFrameDimensions,
  cameraErrorMessage,
} from "./cameraFrame.ts";

test("accepts valid camera frame dimensions", () => {
  assert.equal(validFrameDimensions(1280, 720), true);
  assert.equal(validFrameDimensions(1920, 1080), true);
  assert.equal(validFrameDimensions(640, 480), true);
});

test("rejects absent or unreasonable frame dimensions", () => {
  for (const [width, height] of [
    [0, 720],
    [1280, 0],
    [-1, 200],
    [Infinity, 720],
    [640.5, 480],
    [10000, 10000],
  ]) {
    assert.equal(
      validFrameDimensions(width, height),
      false,
    );
  }
});

test("camera permission errors have useful fallbacks", () => {
  assert.match(
    cameraErrorMessage({ name: "NotAllowedError" }),
    /permission was denied/,
  );
  assert.match(
    cameraErrorMessage({ name: "NotFoundError" }),
    /No camera was found/,
  );
});

test("busy camera errors are explained", () => {
  assert.match(
    cameraErrorMessage({ name: "NotReadableError" }),
    /another application/,
  );
});
