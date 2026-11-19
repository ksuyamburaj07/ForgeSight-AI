import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

import {
  metricValue,
  parseModelRegistry,
} from "./modelRegistry.ts";

const raw = JSON.parse(
  readFileSync(
    new URL(
      "../public/benchmark/model-registry.json",
      import.meta.url,
    ),
    "utf8",
  ),
);

test("six real configurations and current model are preserved", () => {
  const registry = parseModelRegistry(raw);

  assert.deepEqual(
    registry.models.map((model) => model.id),
    [
      "dino_full", "dino_tiled", "resnet_coreset",
      "padim", "student_teacher", "dino_672",
    ],
  );

  assert.equal(registry.current_inspection_id, "dino_672");
});

test("six-model aggregate counts use the 802-image cohort", () => {
  const registry = parseModelRegistry(raw);

  for (const model of registry.models) {
    const c = model.aggregate;
    assert.equal(c.TP + c.FN, 400);
    assert.equal(c.FP + c.TN, 402);
  }
});

test("tiled ranks higher but 672 px flags more anomalies", () => {
  const registry = parseModelRegistry(raw);
  const tiled = registry.models.find((m) => m.id === "dino_tiled")!;
  const high = registry.models.find((m) => m.id === "dino_672")!;

  assert.ok(tiled.aggregate.macro_auroc >
    high.aggregate.macro_auroc);

  assert.equal(high.aggregate.TP - tiled.aggregate.TP, 70);
  assert.equal(high.aggregate.FP - tiled.aggregate.FP, 44);

  assert.equal(metricValue(high, "all", "recall"), 255 / 400);
});

test("category-level strength is not universally shared", () => {
  const registry = parseModelRegistry(raw);
  const student = registry.models.find(
    (m) => m.id === "student_teacher",
  )!;
  const high = registry.models.find((m) => m.id === "dino_672")!;

  assert.ok(
    metricValue(student, "pcb2", "auroc") >
    metricValue(high, "pcb2", "auroc"),
  );
});

test("tampered confusion counts are rejected", () => {
  const changed = structuredClone(raw);
  changed.models[0].per_category.pcb1.TP += 1;

  assert.throws(
    () => parseModelRegistry(changed),
    /Confusion cohort mismatch/,
  );
});
