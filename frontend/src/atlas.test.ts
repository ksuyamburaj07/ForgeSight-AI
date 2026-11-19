import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { createHash as importedCreateHash } from "node:crypto";

import { parseAtlas } from "./atlas.ts";

const raw = JSON.parse(
  readFileSync(
    new URL("../public/research-atlas/cases.json", import.meta.url),
    "utf-8",
  ),
);

test("four archived cases match the expected identities", () => {
  const atlas = parseAtlas(raw);

  assert.deepEqual(
    atlas.cases.map((c) => c.id),
    ["B3-01", "B3-09", "B3-17", "B3-23"],
  );
});

test("retrospective decisions remain unchanged", () => {
  const atlas = parseAtlas(raw);

  assert.deepEqual(
    atlas.cases.map((c) => c.review_flag),
    [false, false, true, true],
  );

  assert.deepEqual(
    atlas.cases.map((c) => c.reference_label),
    ["anomaly", "anomaly", "normal", "normal"],
  );
});

test("numerical reproduction remains strictly 3 of 4", () => {
  const atlas = parseAtlas(raw);

  assert.deepEqual(
    atlas.cases.map(
      (c) => c.reproduction.within_original_tolerance
    ),
    [false, true, true, true],
  );
});

test("Atlas rejects an altered frozen score comparison", () => {
  const changed = structuredClone(raw);
  changed.cases[0].reproduction.frozen_score += 0.01;

  assert.throws(
    () => parseAtlas(changed),
    /Score comparison mismatch/,
  );
});

test("Atlas rejects unexpected image paths", () => {
  const changed = structuredClone(raw);
  changed.cases[0].triptych.url =
    "https://external.example/image.png";

  assert.throws(
    () => parseAtlas(changed),
    /Unexpected image path/,
  );
});

test("all four cases retain provenance and attribution", () => {
  const atlas = parseAtlas(raw);

  assert.equal(atlas.license.dataset_license, "CC BY 4.0");

  for (const item of atlas.cases) {
    assert.match(
      item.provenance.source_sha256_reported,
      /^[0-9a-f]{64}$/,
    );

    assert.match(item.triptych.sha256, /^[0-9a-f]{64}$/);
    assert.equal(item.cpu.geometry.length, 6);
  }
});

test("published triptych files match manifest SHA-256", () => {
  const { createHash } = requireCrypto();
  const atlas = parseAtlas(raw);

  for (const item of atlas.cases) {
    const image = readFileSync(
      new URL(
        `../public${item.triptych.url}`,
        import.meta.url,
      ),
    );

    const actual = createHash("sha256")
      .update(image)
      .digest("hex");

    assert.equal(
      actual,
      item.triptych.sha256,
      `${item.id} published image differs from its manifest`,
    );
  }
});

function requireCrypto() {
  return { createHash: importedCreateHash };
}
