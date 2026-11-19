import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

import {
  parseDatasetRegistry,
} from "./datasetRegistry.ts";

import {
  galleryPreviewPath,
  letterboxImageBounds,
} from "./datasetVisual.ts";

const inventory = parseDatasetRegistry(
  JSON.parse(
    readFileSync(
      new URL(
        "../public/dataset/inventory.json",
        import.meta.url,
      ),
      "utf8",
    ),
  ),
);

test("gallery preview paths match the 16 published names", () => {
  for (const pcb of [
    "pcb1", "pcb2", "pcb3", "pcb4",
  ] as const) {
    for (let index = 0; index < 4; index++) {
      assert.equal(
        galleryPreviewPath(pcb, index),
        `/dataset/gallery/${pcb}_${String(index + 1)
          .padStart(2, "0")}.jpg`,
      );
    }
  }
});

test("invalid preview selections cannot escape the gallery", () => {
  assert.throws(
    () => galleryPreviewPath("pcb1", -1),
    /Invalid gallery/,
  );

  assert.throws(
    () => galleryPreviewPath("pcb2", 4),
    /Invalid gallery/,
  );
});

test("source photographs fit inside the frozen input frame", () => {
  for (const category of Object.values(inventory.categories)) {
    const r = letterboxImageBounds(category.letterbox);

    assert.ok(r.x >= 0 && r.y >= 0);
    assert.ok(r.x + r.width <= 480);
    assert.ok(r.y + r.height <= 480);
    assert.ok(Math.abs(r.x + r.width / 2 - 240) <= 1);
    assert.ok(Math.abs(r.y + r.height / 2 - 240) <= 1);
  }
});

test("illustration geometry scales consistently", () => {
  const box = inventory.categories.pcb1.letterbox;
  const small = letterboxImageBounds(box, 240);
  const large = letterboxImageBounds(box, 480);

  assert.equal(large.x, small.x * 2);
  assert.equal(large.y, small.y * 2);
  assert.equal(large.width, small.width * 2);
  assert.equal(large.height, small.height * 2);
});
