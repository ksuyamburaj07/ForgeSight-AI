export type PCB = "pcb1" | "pcb2" | "pcb3" | "pcb4";

export interface Letterbox {
  input_size: number;
  patch_size: number;
  grid_size: number;
  resized_width: number;
  resized_height: number;
  left: number;
  top: number;
  eligible_patch_centres: number;
  excluded_patch_centres: number;
}

export interface DatasetCategory {
  total: number;
  train_normal: number;
  evaluation_normal: number;
  evaluation_anomaly: number;
  source_width: number;
  source_height: number;
  letterbox: Letterbox;
}

export interface DatasetRegistry {
  dataset: string;
  totals: {
    images: number;
    train_normal: number;
    evaluation_normal: number;
    evaluation_anomaly: number;
  };
  reference_images_per_category: number;
  categories: Record<PCB, DatasetCategory>;
  provenance: {
    inventory_path: string;
    inventory_sha256: string;
    original_manifest_sha256: string;
    paired_audit_sha256: string;
    reference_policy_path: string;
  };
  limitations: string[];
}

const PCBS: PCB[] = ["pcb1", "pcb2", "pcb3", "pcb4"];

const EXPECTED: Record<PCB, number[]> = {
  pcb1: [904, 100, 100, 1404, 1070, 1728],
  pcb2: [901, 100, 100, 1404, 1070, 1728],
  pcb3: [905, 101, 100, 1562, 960, 1440],
  pcb4: [904, 101, 100, 1358, 1104, 1872],
};

const ORIGINAL_SHA =
  "8835e63225bd75794c99fc1684563e2a8820054542ebc9e96219f0725d6f83eb";

const PAIRED_SHA =
  "474e763a65d0148965ffad186ec4722cb5f3cf141376443dcf570398aaa62df2";

function object(value: unknown, field: string):
  Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new Error(`Invalid dataset object: ${field}`);
  }
  return value as Record<string, unknown>;
}

function number(value: unknown, field: string): number {
  if (
    typeof value !== "number" ||
    !Number.isSafeInteger(value) ||
    value < 0
  ) {
    throw new Error(`Invalid dataset count: ${field}`);
  }
  return value;
}

function sha(value: unknown, field: string): string {
  if (
    typeof value !== "string" ||
    !/^[0-9a-f]{64}$/.test(value)
  ) {
    throw new Error(`Invalid dataset SHA: ${field}`);
  }
  return value;
}

export function isEligible(
  box: Letterbox,
  row: number,
  col: number,
): boolean {
  if (
    !Number.isInteger(row) || !Number.isInteger(col) ||
    row < 0 || row >= 48 || col < 0 || col >= 48
  ) return false;

  const x = (col + 0.5) * box.patch_size;
  const y = (row + 0.5) * box.patch_size;

  return x >= box.left &&
    x < box.left + box.resized_width &&
    y >= box.top &&
    y < box.top + box.resized_height;
}

export function parseDatasetRegistry(input: unknown):
  DatasetRegistry {
  const root = object(input, "root");
  const categories = object(root.categories, "categories");
  const totals = object(root.totals, "totals");
  const provenance = object(root.provenance, "provenance");

  if (
    root.schema_version !== 1 ||
    root.study_type !==
      "original_inventory_and_reused_evaluation" ||
    root.dataset !== "Amazon Science VisA" ||
    root.reference_images_per_category !== 8 ||
    Object.keys(categories).length !== 4 ||
    PCBS.some((pcb) => !(pcb in categories))
  ) {
    throw new Error("Unexpected dataset research contract");
  }

  const result = {} as Record<PCB, DatasetCategory>;
  let totalImages = 0;
  let totalTrain = 0;
  let totalNormal = 0;
  let totalAnomaly = 0;

  for (const pcb of PCBS) {
    const item = object(categories[pcb], pcb);
    const box = object(item.letterbox, `${pcb}.letterbox`);

    const train = number(item.train_normal, `${pcb}.train`);
    const normal = number(item.evaluation_normal, `${pcb}.normal`);
    const anomaly = number(item.evaluation_anomaly, `${pcb}.anomaly`);
    const width = number(item.source_width, `${pcb}.width`);
    const height = number(item.source_height, `${pcb}.height`);
    const total = number(item.total, `${pcb}.total`);

    const expected = EXPECTED[pcb];
    const actual = [train, normal, anomaly, width, height];

    if (
      actual.some((n, i) => n !== expected[i]) ||
      total !== train + normal + anomaly
    ) {
      throw new Error(`Category inventory mismatch: ${pcb}`);
    }

    const geometry: Letterbox = {
      input_size: number(box.input_size, "input_size"),
      patch_size: number(box.patch_size, "patch_size"),
      grid_size: number(box.grid_size, "grid_size"),
      resized_width: number(box.resized_width, "resized_width"),
      resized_height: number(box.resized_height, "resized_height"),
      left: number(box.left, "left"),
      top: number(box.top, "top"),
      eligible_patch_centres: number(
        box.eligible_patch_centres, "eligible_patch_centres"
      ),
      excluded_patch_centres: number(
        box.excluded_patch_centres, "excluded_patch_centres"
      ),
    };

    const scale = Math.min(672 / width, 672 / height);
    const rw = Math.round(width * scale);
    const rh = Math.round(height * scale);
    const left = Math.floor((672 - rw) / 2);
    const top = Math.floor((672 - rh) / 2);

    let eligible = 0;
    for (let row = 0; row < 48; row++) {
      for (let col = 0; col < 48; col++) {
        if (isEligible(geometry, row, col)) eligible++;
      }
    }

    if (
      geometry.input_size !== 672 ||
      geometry.patch_size !== 14 ||
      geometry.grid_size !== 48 ||
      geometry.resized_width !== rw ||
      geometry.resized_height !== rh ||
      geometry.left !== left ||
      geometry.top !== top ||
      eligible !== expected[5] ||
      geometry.eligible_patch_centres !== eligible ||
      geometry.excluded_patch_centres !== 2304 - eligible
    ) {
      throw new Error(`Letterbox geometry mismatch: ${pcb}`);
    }

    result[pcb] = {
      total,
      train_normal: train,
      evaluation_normal: normal,
      evaluation_anomaly: anomaly,
      source_width: width,
      source_height: height,
      letterbox: geometry,
    };

    totalImages += total;
    totalTrain += train;
    totalNormal += normal;
    totalAnomaly += anomaly;
  }

  if (
    number(totals.images, "images") !== totalImages ||
    number(totals.train_normal, "train_normal") !== totalTrain ||
    number(totals.evaluation_normal, "evaluation_normal")
      !== totalNormal ||
    number(totals.evaluation_anomaly, "evaluation_anomaly")
      !== totalAnomaly ||
    totalImages !== 4416 ||
    totalTrain !== 3614 ||
    totalNormal !== 402 ||
    totalAnomaly !== 400
  ) {
    throw new Error("Aggregate inventory mismatch");
  }

  const inventorySha = sha(
    provenance.inventory_sha256, "inventory_sha256"
  );
  const pairedSha = sha(
    provenance.paired_audit_sha256, "paired_audit_sha256"
  );

  if (
    inventorySha !== ORIGINAL_SHA ||
    pairedSha !== PAIRED_SHA
  ) {
    throw new Error("Dataset source provenance mismatch");
  }

  if (
    !Array.isArray(root.limitations) ||
    root.limitations.length < 4 ||
    !root.limitations.every(
      (v: unknown) => typeof v === "string" && !!v.trim()
    )
  ) {
    throw new Error("Dataset limitations missing");
  }

  return {
    dataset: "Amazon Science VisA",
    totals: {
      images: totalImages,
      train_normal: totalTrain,
      evaluation_normal: totalNormal,
      evaluation_anomaly: totalAnomaly,
    },
    reference_images_per_category: 8,
    categories: result,
    provenance: {
      inventory_path: String(provenance.inventory_path),
      inventory_sha256: inventorySha,
      original_manifest_sha256: sha(
        provenance.original_manifest_sha256,
        "original_manifest_sha256",
      ),
      paired_audit_sha256: pairedSha,
      reference_policy_path: String(
        provenance.reference_policy_path,
      ),
    },
    limitations: root.limitations as string[],
  };
}

export async function loadDatasetRegistry(
  signal: AbortSignal,
): Promise<DatasetRegistry> {
  const response = await fetch("/dataset/inventory.json", {
    signal,
    cache: "no-store",
  });

  if (!response.ok) {
    throw new Error(
      `Dataset evidence unavailable (HTTP ${response.status})`,
    );
  }

  return parseDatasetRegistry(await response.json());
}
