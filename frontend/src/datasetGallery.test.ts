import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { createHash } from "node:crypto";

import {
  parseDatasetGallery,
} from "./datasetGallery.ts";

const raw = JSON.parse(
  readFileSync(
    new URL("../public/dataset/gallery.json", import.meta.url),
    "utf8",
  ),
);

test("gallery contains exactly sixteen normal-training images", () => {
  const manifest = parseDatasetGallery(raw);

  assert.deepEqual(
    Object.values(manifest.items).map((items) => items.length),
    [4, 4, 4, 4],
  );

  for (const entries of Object.values(manifest.items)) {
    for (const entry of entries) {
      assert.match(
        entry.source_path,
        /\/Data\/Images\/Normal\//,
      );
    }
  }
});

test("gallery includes dataset attribution and transformation", () => {
  const manifest = parseDatasetGallery(raw);

  assert.equal(manifest.license, "CC BY 4.0");
  assert.match(manifest.official_source, /amazon-science/);
  assert.match(manifest.transformation, /resize/i);
});

test("all sixteen published thumbnails match their SHA-256", () => {
  const manifest = parseDatasetGallery(raw);
  let verified = 0;

  for (const entries of Object.values(manifest.items)) {
    for (const entry of entries) {
      const path = new URL(
        `../public${entry.src}`,
        import.meta.url,
      );

      const bytes = readFileSync(path);
      const actual = createHash("sha256")
        .update(bytes)
        .digest("hex");

      assert.equal(
        actual,
        entry.derived_sha256,
        entry.id,
      );

      verified++;
    }
  }

  assert.equal(verified, 16);
});

test("evaluation image paths cannot enter the gallery", () => {
  const changed = structuredClone(raw);
  changed.items.pcb1[0].source_path =
    "pcb1/Data/Images/Anomaly/0001.JPG";

  assert.throws(
    () => parseDatasetGallery(changed),
    /Invalid gallery image/,
  );
});

test("unexpected public gallery paths are rejected", () => {
  const changed = structuredClone(raw);
  changed.items.pcb2[1].src =
    "/dataset/gallery/../../private.jpg";

  assert.throws(
    () => parseDatasetGallery(changed),
    /Invalid gallery image/,
  );
});

test("incorrect source geometry is rejected", () => {
  const changed = structuredClone(raw);
  changed.items.pcb4[0].original_height = 1070;

  assert.throws(
    () => parseDatasetGallery(changed),
    /Invalid gallery image/,
  );
});
