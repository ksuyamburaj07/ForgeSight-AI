import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

import {
  isEligible,
  parseDatasetRegistry,
} from "./datasetRegistry.ts";

const source = JSON.parse(
  readFileSync(
    new URL("../public/dataset/inventory.json", import.meta.url),
    "utf-8",
  ),
);

test("original dataset and evaluation totals remain distinct", () => {
  const data = parseDatasetRegistry(source);

  assert.equal(data.totals.images, 4416);
  assert.equal(data.totals.train_normal, 3614);
  assert.equal(data.totals.evaluation_normal, 402);
  assert.equal(data.totals.evaluation_anomaly, 400);
});

test("all category identities and original counts agree", () => {
  const data = parseDatasetRegistry(source);

  assert.deepEqual(
    Object.values(data.categories).map((c) => c.total),
    [1104, 1101, 1106, 1105],
  );
});

test("letterbox eligibility agrees with historical geometry", () => {
  const data = parseDatasetRegistry(source);

  assert.deepEqual(
    Object.values(data.categories).map(
      (c) => c.letterbox.eligible_patch_centres
    ),
    [1728, 1728, 1440, 1872],
  );

  assert.equal(
    isEligible(data.categories.pcb1.letterbox, 0, 24),
    false,
  );
  assert.equal(
    isEligible(data.categories.pcb1.letterbox, 24, 24),
    true,
  );
});

test("reversing PCB4 class counts is rejected", () => {
  const changed = structuredClone(source);

  changed.categories.pcb4.evaluation_normal = 100;
  changed.categories.pcb4.evaluation_anomaly = 101;

  assert.throws(
    () => parseDatasetRegistry(changed),
    /Category inventory mismatch/,
  );
});

test("tampered geometry and provenance are rejected", () => {
  const changedGeometry = structuredClone(source);
  changedGeometry.categories.pcb3.letterbox.top += 1;

  assert.throws(
    () => parseDatasetRegistry(changedGeometry),
    /Letterbox geometry mismatch/,
  );

  const changedSource = structuredClone(source);
  changedSource.provenance.inventory_sha256 = "0".repeat(64);

  assert.throws(
    () => parseDatasetRegistry(changedSource),
    /Dataset source provenance mismatch/,
  );
});
