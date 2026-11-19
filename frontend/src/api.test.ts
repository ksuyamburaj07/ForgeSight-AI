import { strict as assert } from "node:assert";
import test from "node:test";
import type { TestContext } from "node:test";
import { connect, fingerprint, inspect } from "./api.ts";

// Fabricated transport fixtures only. These are not scientific measurements or
// inference results and are never imported by the application.
const healthFixture = {
  status: "ok",
  runtime_device: "cpu",
  loading_policy: "on_first_valid_inspection",
  scientific_status: "cross_platform_score_gate_open",
};
const capabilityFixture = {
  categories: ["pcb1", "pcb2", "pcb3", "pcb4"],
  accepted_media_types: ["image/jpeg", "image/png", "image/webp"],
  max_upload_bytes: 8 * 1024 * 1024,
  max_image_pixels: 8_000_000,
  max_image_edge: 4096,
  image_resolution: 672,
  patch_grid: [48, 48],
  result_type: "research_anomaly_evidence",
  automatic_manufacturing_decision: false,
};
function evidenceFixture() {
  return {
    inspection: {
      inference_performed: true,
      category: "pcb1",
      score: 0.25,
      threshold: 0.15,
      review_flag: true,
      result_type: "research_anomaly_evidence",
      valid_patches: 2304,
      processing_ms: 100,
      runtime_device: "cpu",
      input_resolution: 672,
      patch_grid: [48, 48],
      model_id: "facebook/dinov2-small",
      model_weights_sha256: "a".repeat(64),
      local_model_snapshot: "synthetic-test-snapshot",
      source_sha256: "b".repeat(64),
      frozen_scoring_method:
        "mean_top_1_percent_of_valid_cosine_nearest_neighbor_patch_distances",
      validation_status: "cpu_cross_platform_score_gate_open",
      geometry: [0, 0, 672, 672, 224, 224],
      visualization: {
        display_scale_max: 0.25,
        scaling_method: "synthetic-test-scale",
        interpretation: "Synthetic transport fixture only",
      },
    },
    // Encoded markers test URI shape checks, not image decoding.
    visualizations: {
      original: "data:image/jpeg;base64,/9j/2Q==",
      heatmap: "data:image/jpeg;base64,/9j/2Q==",
      overlay: "data:image/jpeg;base64,/9j/2Q==",
    },
    patch_evidence: {
      grid: [48, 48],
      distances: Array.from({ length: 48 }, () => Array(48).fill(0.25)),
      valid_mask: Array.from({ length: 48 }, () => Array(48).fill(true)),
      units: "cosine_nearest_neighbor_distance",
    },
  };
}
const file = () =>
  new File(["synthetic transport test bytes"], "transport-test.png", {
    type: "image/png",
  });
function respond(t: TestContext, payload: unknown, status = 200) {
  t.mock.method(
    globalThis,
    "fetch",
    async () =>
      new Response(JSON.stringify(payload), {
        status,
        headers: { "Content-Type": "application/json" },
      }),
  );
}
function connection(
  t: TestContext,
  health: unknown = healthFixture,
  capabilities: unknown = capabilityFixture,
) {
  t.mock.method(
    globalThis,
    "fetch",
    async (path: string) =>
      new Response(
        JSON.stringify(path.endsWith("/health") ? health : capabilities),
        { headers: { "Content-Type": "application/json" } },
      ),
  );
}

test("connect accepts the current CPU research contract", async (t) => {
  connection(t);
  assert.deepEqual(await connect(new AbortController().signal), {
    health: healthFixture,
    capabilities: capabilityFixture,
  });
});

test("connect rejects incompatible or malformed health and capabilities", async (t) => {
  const invalid: [unknown, unknown][] = [
    [null, capabilityFixture],
    [{ ...healthFixture, runtime_device: "cuda" }, capabilityFixture],
    [{ ...healthFixture, scientific_status: "validated" }, capabilityFixture],
    [healthFixture, null],
    [
      healthFixture,
      { ...capabilityFixture, automatic_manufacturing_decision: true },
    ],
    [
      healthFixture,
      { ...capabilityFixture, categories: ["pcb1", "pcb2", "pcb3", "pcb3"] },
    ],
    [
      healthFixture,
      { ...capabilityFixture, categories: ["pcb1", "pcb2", "pcb3", "pcb5"] },
    ],
    [healthFixture, { ...capabilityFixture, accepted_media_types: [null] }],
    [healthFixture, { ...capabilityFixture, max_upload_bytes: 0 }],
    [healthFixture, { ...capabilityFixture, image_resolution: 224 }],
    [healthFixture, { ...capabilityFixture, patch_grid: { 0: 48, 1: 48 } }],
  ];
  for (const [health, caps] of invalid) {
    connection(t, health, caps);
    await assert.rejects(
      connect(new AbortController().signal),
      /service contract does not match/,
    );
    t.mock.restoreAll();
  }
});

test("inspection sends multipart input and preserves validated response", async (t) => {
  const fixture = evidenceFixture();
  t.mock.method(
    globalThis,
    "fetch",
    async (path: string, options: RequestInit) => {
      assert.equal(path, "/api/v1/inspect");
      assert.equal(options.method, "POST");
      assert.ok(options.body instanceof FormData);
      assert.equal(options.body.get("category"), "pcb1");
      assert.equal(
        (options.body.get("file") as File).name,
        "transport-test.png",
      );
      assert.equal(options.headers, undefined); // Browser supplies multipart boundary.
      return new Response(JSON.stringify(fixture));
    },
  );
  assert.deepEqual(await inspect(file(), "pcb1"), fixture);
});

test("inspection accepts source-aligned rectangular geometry and excluded padding", async (t) => {
  const fixture = evidenceFixture();
  fixture.inspection.geometry = [0, 84, 672, 504, 640, 480];
  fixture.inspection.valid_patches = 1728;
  fixture.patch_evidence.valid_mask = Array.from({ length: 48 }, (_, row) =>
    Array(48).fill(row >= 6 && row <= 41),
  );
  respond(t, fixture);
  assert.deepEqual(
    (await inspect(file(), "pcb1")).inspection.geometry,
    [0, 84, 672, 504, 640, 480],
  );
});

test("inspection rejects malformed nested evidence without method TypeErrors", async (t) => {
  const invalid: unknown[] = [
    null,
    [],
    { inspection: null },
    { ...evidenceFixture(), visualizations: [] },
  ];
  const changes: [(string | number)[], unknown][] = [
    [["inspection", "runtime_device"], null],
    [["inspection", "visualization"], null],
    [["inspection", "review_flag"], "false"],
    [["inspection", "source_sha256"], "not-a-sha"],
    [["inspection", "geometry", 0], -1],
    [["inspection", "geometry", 2], 671.5],
    [["inspection", "geometry", 4], 5000],
    [["inspection", "processing_ms"], -1],
    [["inspection", "validation_status"], "fully_validated"],
    [["inspection", "valid_patches"], 1],
    [["patch_evidence", "distances", 0], null],
    [["patch_evidence", "distances", 1], { length: 48 }],
    [["patch_evidence", "distances", 0, 0], 3],
    [["patch_evidence", "valid_mask", 0, 0], "true"],
    [["visualizations", "overlay"], 123],
    [
      ["visualizations", "heatmap"],
      "https://example.invalid/not-service-evidence.jpg",
    ],
  ];
  for (const [path, value] of changes) {
    const fixture = evidenceFixture();
    let parent = fixture as unknown as Record<string, unknown>;
    for (const key of path.slice(0, -1))
      parent = parent[String(key)] as Record<string, unknown>;
    parent[String(path.at(-1))] = value;
    invalid.push(fixture);
  }
  for (const payload of invalid) {
    respond(t, payload);
    await assert.rejects(
      inspect(file(), "pcb1"),
      (error) =>
        error instanceof Error &&
        error.message ===
          "The service returned incomplete or incompatible evidence. No result has been displayed.",
    );
    t.mock.restoreAll();
  }
});

test("inspection refuses another category in the response", async (t) => {
  respond(t, evidenceFixture());
  await assert.rejects(
    inspect(file(), "pcb2"),
    /incomplete or incompatible evidence/,
  );
  await assert.rejects(inspect(file(), "pcb5"), /Choose PCB1/);
});

test("HTTP busy, server detail, and non-string validation errors are understandable", async (t) => {
  for (const [status, detail, message] of [
    [429, "Inference is busy", /CPU service is busy/],
    [
      503,
      "Inspection temporarily unavailable",
      /Inspection temporarily unavailable/,
    ],
    [422, [{ msg: "Invalid input" }], /Request failed \(HTTP 422\)/],
  ] as const) {
    respond(t, { detail }, status);
    await assert.rejects(inspect(file(), "pcb1"), message);
    t.mock.restoreAll();
  }
});

test("unreadable and unreachable services report an error without evidence", async (t) => {
  t.mock.method(
    globalThis,
    "fetch",
    async () => new Response("<html>unavailable</html>", { status: 502 }),
  );
  await assert.rejects(
    inspect(file(), "pcb1"),
    /unreadable response \(HTTP 502\)/,
  );
  t.mock.restoreAll();
  t.mock.method(globalThis, "fetch", async () => {
    throw new TypeError("network disconnected");
  });
  await assert.rejects(
    inspect(file(), "pcb1"),
    /Cannot reach the local inspection service/,
  );
});

test("aborted connection preserves abort error for the deadline handler", async (t) => {
  const aborted = new DOMException("Stopped", "AbortError");
  t.mock.method(globalThis, "fetch", async () => {
    throw aborted;
  });
  await assert.rejects(
    connect(new AbortController().signal),
    (error) => error === aborted,
  );
});

test("fingerprint computes SHA-256 of original file bytes", async () => {
  assert.equal(
    await fingerprint(new File(["abc"], "known-bytes.txt")),
    "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
  );
});
