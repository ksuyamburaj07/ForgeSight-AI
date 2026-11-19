import type { Category } from "./types";

export interface AtlasCase {
  id: string;
  title: string;
  finding: string;
  category: Category;
  reference_label: "normal" | "anomaly";
  review_flag: boolean;
  audit_observation: string;
  cpu: {
    score: number;
    threshold: number;
    valid_patches: number;
    geometry: number[];
    validation_status: string;
  };
  reproduction: {
    frozen_score: number;
    absolute_delta: number;
    original_tolerance: number;
    within_original_tolerance: boolean;
  };
  provenance: {
    source_sha256_reported: string;
    model_id: string;
    model_weights_sha256: string;
    local_model_snapshot: string;
    record_source: string;
  };
  triptych: {
    url: string;
    sha256: string;
  };
}

export interface AtlasManifest {
  version: string;
  cases: AtlasCase[];
  scientific_gate: string;
  license: {
    dataset: string;
    dataset_creator: string;
    dataset_license: string;
    source: string;
    license_url: string;
    modifications: string;
  };
}

type Data = Record<string, unknown>;

const EXPECTED = {
  "B3-01": { category: "pcb1", label: "anomaly", flag: false },
  "B3-09": { category: "pcb2", label: "anomaly", flag: false },
  "B3-17": { category: "pcb3", label: "normal", flag: true },
  "B3-23": { category: "pcb4", label: "normal", flag: true },
} as const;

function object(value: unknown, field: string): Data {
  if (
    value === null ||
    typeof value !== "object" ||
    Array.isArray(value)
  ) {
    throw new Error(`Invalid Atlas object: ${field}`);
  }
  return value as Data;
}

function text(value: unknown, field: string): string {
  if (typeof value !== "string" || !value.trim()) {
    throw new Error(`Invalid Atlas text: ${field}`);
  }
  return value;
}

function number(value: unknown, field: string): number {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new Error(`Invalid Atlas number: ${field}`);
  }
  return value;
}

function sha(value: unknown, field: string): string {
  const result = text(value, field);
  if (!/^[0-9a-f]{64}$/.test(result)) {
    throw new Error(`Invalid SHA-256: ${field}`);
  }
  return result;
}

export function parseAtlas(input: unknown): AtlasManifest {
  const root = object(input, "manifest");

  if (root.version !== "1.0") {
    throw new Error("Unsupported Research Atlas version");
  }

  if (
    root.scientific_gate !==
    "cpu_cross_platform_score_gate_open"
  ) {
    throw new Error("Unexpected scientific validation status");
  }

  if (!Array.isArray(root.cases) || root.cases.length !== 4) {
    throw new Error("Expected four retrospective cases");
  }

  const cases: AtlasCase[] = root.cases.map((entry) => {
    const item = object(entry, "case");
    const id = text(item.id, "case ID");
    const expected = EXPECTED[id as keyof typeof EXPECTED];

    if (!expected) {
      throw new Error(`Unknown research case: ${id}`);
    }

    if (
      item.category !== expected.category ||
      item.reference_label !== expected.label ||
      item.review_flag !== expected.flag
    ) {
      throw new Error(`Case identity mismatch: ${id}`);
    }

    const cpu = object(item.cpu, `${id}.cpu`);
    const repro = object(item.reproduction, `${id}.reproduction`);
    const provenance = object(item.provenance, `${id}.provenance`);
    const triptych = object(item.triptych, `${id}.triptych`);

    const score = number(cpu.score, `${id}.score`);
    const threshold = number(cpu.threshold, `${id}.threshold`);
    const frozen = number(repro.frozen_score, `${id}.frozen`);
    const delta = number(repro.absolute_delta, `${id}.delta`);
    const tolerance = number(
      repro.original_tolerance, `${id}.tolerance`
    );

    const validPatches = number(
      cpu.valid_patches, `${id}.valid_patches`
    );

    if (
      score < 0 ||
      threshold <= 0 ||
      frozen < 0 ||
      delta < 0 ||
      tolerance !== 0.00005 ||
      !Number.isInteger(validPatches) ||
      validPatches < 1 ||
      validPatches > 2304
    ) {
      throw new Error(`Invalid measurements: ${id}`);
    }

    const actualDelta = Math.abs(frozen - score);
    if (Math.abs(actualDelta - delta) > 1e-7) {
      throw new Error(`Score comparison mismatch: ${id}`);
    }

    if (
      repro.within_original_tolerance !==
      (delta <= tolerance)
    ) {
      throw new Error(`Reproduction gate mismatch: ${id}`);
    }

    if (
      cpu.validation_status !==
      "cpu_cross_platform_score_gate_open"
    ) {
      throw new Error(`Missing CPU limitation: ${id}`);
    }

    if (
      !Array.isArray(cpu.geometry) ||
      cpu.geometry.length !== 6 ||
      !cpu.geometry.every(
        (v) => Number.isInteger(v) && v >= 0
      )
    ) {
      throw new Error(`Invalid case geometry: ${id}`);
    }

    const url = text(triptych.url, `${id}.triptych.url`);
    if (url !== `/research-atlas/${id}_triptych.png`) {
      throw new Error(`Unexpected image path: ${id}`);
    }

    return {
      id,
      title: text(item.title, `${id}.title`),
      finding: text(item.finding, `${id}.finding`),
      category: expected.category,
      reference_label: expected.label,
      review_flag: expected.flag,
      audit_observation: text(
        item.audit_observation, `${id}.audit_observation`
      ),
      cpu: {
        score,
        threshold,
        valid_patches: validPatches,
        geometry: cpu.geometry as number[],
        validation_status: text(
          cpu.validation_status, `${id}.validation_status`
        ),
      },
      reproduction: {
        frozen_score: frozen,
        absolute_delta: delta,
        original_tolerance: tolerance,
        within_original_tolerance:
          repro.within_original_tolerance as boolean,
      },
      provenance: {
        source_sha256_reported: sha(
          provenance.source_sha256_reported,
          `${id}.source_sha256_reported`
        ),
        model_id: text(provenance.model_id, `${id}.model_id`),
        model_weights_sha256: sha(
          provenance.model_weights_sha256,
          `${id}.model_weights_sha256`
        ),
        local_model_snapshot: text(
          provenance.local_model_snapshot,
          `${id}.local_model_snapshot`
        ),
        record_source: text(
          provenance.record_source,
          `${id}.record_source`
        ),
      },
      triptych: {
        url,
        sha256: sha(triptych.sha256, `${id}.triptych.sha256`),
      },
    };
  });

  if (new Set(cases.map((c) => c.id)).size !== 4) {
    throw new Error("Duplicate Atlas case identities");
  }

  const license = object(root.license, "license");
  if (
    license.dataset !== "VisA" ||
    license.dataset_license !== "CC BY 4.0"
  ) {
    throw new Error("Unexpected Atlas dataset attribution");
  }

  return {
    version: "1.0",
    cases,
    scientific_gate: root.scientific_gate,
    license: {
      dataset: "VisA",
      dataset_creator: text(
        license.dataset_creator, "dataset_creator"
      ),
      dataset_license: "CC BY 4.0",
      source: text(license.source, "license.source"),
      license_url: text(
        license.license_url, "license.license_url"
      ),
      modifications: text(
        license.modifications, "license.modifications"
      ),
    },
  };
}

export async function loadAtlas(
  signal: AbortSignal
): Promise<AtlasManifest> {
  const response = await fetch("/research-atlas/cases.json", {
    signal,
    cache: "no-store",
  });

  if (!response.ok) {
    throw new Error(
      `Research Atlas unavailable (HTTP ${response.status})`
    );
  }

  return parseAtlas(await response.json());
}
