import type { AtlasCase } from "./atlas";

export interface HistoricalPatch {
  case_id: string;
  archive_sha256: string;
  valid_count: number;
  top_k_count: number;
  archived_top_k_mean: number;
  frozen_reference_score: number;
  distances: (number | null)[][];
  valid_mask: boolean[][];
  execution_origin_independently_verified: false;
}

export interface RankedPatch {
  row: number;
  col: number;
  distance: number;
}

const ARCHIVE_SHA =
  "cbb58d6aafbfd9e62570f67a9050f449a64de106f58872a3d3459c426aa21e3f";

function asObject(value: unknown): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new Error("Historical evidence is not an object");
  }
  return value as Record<string, unknown>;
}

function finite(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

export function parseHistoricalPatch(
  input: unknown,
  item: AtlasCase,
): HistoricalPatch {
  const data = asObject(input);

  if (
    data.schema_version !== 1 ||
    data.case_id !== item.id ||
    data.archive_sha256 !== ARCHIVE_SHA ||
    data.evidence_source !==
      "retained_m313c_diagnostic_array_archive" ||
    data.execution_origin_independently_verified !== false
  ) {
    throw new Error("Historical patch provenance mismatch");
  }

  if (
    !Array.isArray(data.grid) ||
    data.grid[0] !== 48 ||
    data.grid[1] !== 48 ||
    data.grid.length !== 2 ||
    !Array.isArray(data.distances) ||
    !Array.isArray(data.valid_mask) ||
    data.distances.length !== 48 ||
    data.valid_mask.length !== 48
  ) {
    throw new Error("Invalid historical patch grid");
  }

  const distances: (number | null)[][] = [];
  const masks: boolean[][] = [];
  const validValues: number[] = [];

  for (let row = 0; row < 48; row++) {
    const values: unknown = data.distances[row];
    const masksRow: unknown = data.valid_mask[row];

    if (
      !Array.isArray(values) ||
      !Array.isArray(masksRow) ||
      values.length !== 48 ||
      masksRow.length !== 48
    ) {
      throw new Error("Malformed historical patch row");
    }

    const normalized: (number | null)[] = [];
    const normalizedMask: boolean[] = [];

    for (let col = 0; col < 48; col++) {
      const eligible = masksRow[col];
      const value = values[col];

      if (typeof eligible !== "boolean") {
        throw new Error("Invalid historical validity mask");
      }

      if (eligible) {
        if (!finite(value) || value < 0) {
          throw new Error("Invalid measured patch distance");
        }
        validValues.push(value);
        normalized.push(value);
      } else {
        if (value !== null) {
          throw new Error("Excluded patch must be null");
        }
        normalized.push(null);
      }

      normalizedMask.push(eligible);
    }

    distances.push(normalized);
    masks.push(normalizedMask);
  }

  const expectedCount = item.cpu.valid_patches;
  const k = Math.ceil(expectedCount * 0.01);

  if (
    validValues.length !== expectedCount ||
    data.valid_count !== expectedCount ||
    data.top_k_count !== k ||
    !finite(data.archived_top_k_mean) ||
    !finite(data.frozen_reference_score)
  ) {
    throw new Error("Historical patch counts or score invalid");
  }

  const ranked = [...validValues].sort((a, b) => b - a);
  const calculated =
    ranked.slice(0, k).reduce((sum, n) => sum + n, 0) / k;

  if (
    Math.abs(calculated - data.archived_top_k_mean) > 1e-8 ||
    Math.abs(
      data.frozen_reference_score -
      item.reproduction.frozen_score
    ) > 1e-9 ||
    Math.abs(
      calculated - item.reproduction.frozen_score
    ) > 1e-6
  ) {
    throw new Error("Historical score consistency failed");
  }

  return {
    case_id: item.id,
    archive_sha256: ARCHIVE_SHA,
    valid_count: expectedCount,
    top_k_count: k,
    archived_top_k_mean: calculated,
    frozen_reference_score: item.reproduction.frozen_score,
    distances,
    valid_mask: masks,
    execution_origin_independently_verified: false,
  };
}

export function rankPatches(field: HistoricalPatch): RankedPatch[] {
  const result: RankedPatch[] = [];

  for (let row = 0; row < 48; row++) {
    for (let col = 0; col < 48; col++) {
      const distance = field.distances[row][col];
      if (distance !== null) {
        result.push({ row, col, distance });
      }
    }
  }

  return result.sort(
    (a, b) =>
      b.distance - a.distance ||
      a.row - b.row ||
      a.col - b.col,
  );
}

export async function loadHistoricalPatch(
  item: AtlasCase,
  signal: AbortSignal,
): Promise<HistoricalPatch> {
  const response = await fetch(
    `/research-atlas/${item.id}_patch_field.json`,
    { signal, cache: "no-store" },
  );

  if (!response.ok) {
    throw new Error(
      `Historical measurements unavailable (HTTP ${response.status})`,
    );
  }

  return parseHistoricalPatch(await response.json(), item);
}
