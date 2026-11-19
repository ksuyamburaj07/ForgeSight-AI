import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

import { parseAtlas } from "./atlas.ts";
import {
  parseHistoricalPatch,
  rankPatches,
} from "./historicalPatch.ts";

const manifest = parseAtlas(JSON.parse(
  readFileSync(
    new URL("../public/research-atlas/cases.json", import.meta.url),
    "utf-8",
  ),
));

function source(id: string) {
  return JSON.parse(readFileSync(
    new URL(
      `../public/research-atlas/${id}_patch_field.json`,
      import.meta.url,
    ),
    "utf-8",
  ));
}

test("all four historical fields match the case records", () => {
  for (const item of manifest.cases) {
    const field = parseHistoricalPatch(source(item.id), item);

    assert.equal(field.valid_count, item.cpu.valid_patches);
    assert.equal(rankPatches(field).length, field.valid_count);

    assert.ok(
      Math.abs(
        field.archived_top_k_mean -
        item.reproduction.frozen_score
      ) < 1e-6,
    );
  }
});

test("excluded cells remain unmeasured", () => {
  const item = manifest.cases[0];
  const field = parseHistoricalPatch(source(item.id), item);

  let excluded = 0;

  for (let row = 0; row < 48; row++) {
    for (let col = 0; col < 48; col++) {
      if (!field.valid_mask[row][col]) {
        assert.equal(field.distances[row][col], null);
        excluded++;
      }
    }
  }

  assert.equal(excluded, 2304 - item.cpu.valid_patches);
});

test("corrupted eligible measurements are rejected", () => {
  const item = manifest.cases[0];
  const corrupted = source(item.id);

  const row = corrupted.valid_mask.findIndex(
    (values: boolean[]) => values.includes(true),
  );
  const col = corrupted.valid_mask[row].indexOf(true);

  corrupted.distances[row][col] = null;

  assert.throws(
    () => parseHistoricalPatch(corrupted, item),
    /Invalid measured patch distance/,
  );
});

test("false provenance claims are rejected", () => {
  const item = manifest.cases[0];
  const corrupted = source(item.id);

  corrupted.execution_origin_independently_verified = true;

  assert.throws(
    () => parseHistoricalPatch(corrupted, item),
    /Historical patch provenance mismatch/,
  );
});
