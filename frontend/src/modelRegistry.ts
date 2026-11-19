export type PCB = "pcb1" | "pcb2" | "pcb3" | "pcb4";
export type Scope = "all" | PCB;
export type Metric = "auroc" | "ap" | "recall" | "review";

const PCBS: PCB[] = ["pcb1", "pcb2", "pcb3", "pcb4"];
const IDS = [
  "dino_full",
  "dino_tiled",
  "resnet_coreset",
  "padim",
  "student_teacher",
  "dino_672",
];

export interface Counts {
  TP: number;
  FN: number;
  FP: number;
  TN: number;
}

export interface CategoryResult extends Counts {
  auroc: number;
  average_precision: number;
}

export interface ModelRecord {
  id: string;
  name: string;
  family: string;
  phase: string;
  research_question: string;
  current_inspection: boolean;
  evaluation_status: string;
  per_category: Record<PCB, CategoryResult>;
  aggregate: Counts & {
    macro_auroc: number;
    macro_average_precision: number;
    defect_recall: number;
    normal_review_rate: number;
  };
  source: {
    path: string;
    sha256: string;
    experiment: string;
  };
}

export interface ModelRegistry {
  schema_version: 1;
  study_type: string;
  current_inspection_id: string;
  cohort: {
    total: number;
    anomaly_labelled: number;
    normal_labelled: number;
  };
  models: ModelRecord[];
  methodology: {
    threshold_note: string;
    selection_note: string;
    limitations: string[];
  };
}

function obj(value: unknown, name: string): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new Error(`Invalid model registry object: ${name}`);
  }
  return value as Record<string, unknown>;
}

function str(value: unknown, name: string): string {
  if (typeof value !== "string" || !value.trim()) {
    throw new Error(`Invalid model registry text: ${name}`);
  }
  return value;
}

function number(value: unknown, name: string): number {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new Error(`Invalid model registry number: ${name}`);
  }
  return value;
}

function integer(value: unknown, name: string): number {
  const result = number(value, name);
  if (!Number.isInteger(result) || result < 0) {
    throw new Error(`Invalid model registry count: ${name}`);
  }
  return result;
}

function rate(value: unknown, name: string): number {
  const result = number(value, name);
  if (result < 0 || result > 1) {
    throw new Error(`Invalid model registry rate: ${name}`);
  }
  return result;
}

function counts(
  source: Record<string, unknown>,
  name: string,
): Counts {
  return {
    TP: integer(source.TP, `${name}.TP`),
    FN: integer(source.FN, `${name}.FN`),
    FP: integer(source.FP, `${name}.FP`),
    TN: integer(source.TN, `${name}.TN`),
  };
}

function close(a: number, b: number): boolean {
  return Math.abs(a - b) < 1e-10;
}

export function parseModelRegistry(input: unknown): ModelRegistry {
  const root = obj(input, "root");
  const cohort = obj(root.cohort, "cohort");
  const methodology = obj(root.methodology, "methodology");

  if (
    root.schema_version !== 1 ||
    root.study_type !==
      "historical_six_configuration_comparison" ||
    root.current_inspection_id !== "dino_672" ||
    root.independent_external_validation !== false ||
    cohort.total !== 802 ||
    cohort.anomaly_labelled !== 400 ||
    cohort.normal_labelled !== 402 ||
    !Array.isArray(root.models) ||
    root.models.length !== 6
  ) {
    throw new Error("Unexpected six-model research contract");
  }

  const parsed: ModelRecord[] = root.models.map((raw, index) => {
    const model = obj(raw, `models[${index}]`);
    const id = str(model.id, "model.id");

    if (id !== IDS[index]) {
      throw new Error("Unexpected model identities or order");
    }

    if (model.current_inspection !== (id === "dino_672")) {
      throw new Error("Current inspection identity mismatch");
    }

    const categoryObj = obj(
      model.per_category,
      `${id}.per_category`,
    );

    if (
      PCBS.some((pcb) => !(pcb in categoryObj)) ||
      Object.keys(categoryObj).length !== 4
    ) {
      throw new Error(`Unexpected categories for ${id}`);
    }

    const categoryRows = {} as Record<PCB, CategoryResult>;
    const totals: Counts = { TP: 0, FN: 0, FP: 0, TN: 0 };
    let aucSum = 0;
    let apSum = 0;

    PCBS.forEach((pcb) => {
      const rawCategory = obj(categoryObj[pcb], `${id}/${pcb}`);
      const rowCounts = counts(rawCategory, `${id}/${pcb}`);

      if (
        rowCounts.TP + rowCounts.FN !== 100 ||
        rowCounts.FP + rowCounts.TN !==
          (pcb === "pcb1" || pcb === "pcb2" ? 100 : 101)
      ) {
        throw new Error(`Confusion cohort mismatch: ${id}/${pcb}`);
      }

      const auroc = rate(rawCategory.auroc, `${id}/${pcb}.auroc`);
      const ap = rate(
        rawCategory.average_precision,
        `${id}/${pcb}.average_precision`,
      );

      categoryRows[pcb] = {
        ...rowCounts,
        auroc,
        average_precision: ap,
      };

      aucSum += auroc;
      apSum += ap;

      (Object.keys(totals) as (keyof Counts)[]).forEach(
        (key) => { totals[key] += rowCounts[key]; },
      );
    });

    const aggregate = obj(model.aggregate, `${id}.aggregate`);
    const reported = counts(aggregate, `${id}.aggregate`);

    if (
      (Object.keys(totals) as (keyof Counts)[]).some(
        (key) => totals[key] !== reported[key],
      ) ||
      !close(
        number(aggregate.macro_auroc, `${id}.macro_auroc`),
        aucSum / 4,
      ) ||
      !close(
        number(
          aggregate.macro_average_precision,
          `${id}.macro_average_precision`,
        ),
        apSum / 4,
      ) ||
      !close(
        number(aggregate.defect_recall, `${id}.defect_recall`),
        totals.TP / 400,
      ) ||
      !close(
        number(
          aggregate.normal_review_rate,
          `${id}.normal_review_rate`,
        ),
        totals.FP / 402,
      )
    ) {
      throw new Error(`Aggregate metric mismatch: ${id}`);
    }

    const source = obj(model.source, `${id}.source`);
    const digest = str(source.sha256, `${id}.source.sha256`);

    if (!/^[0-9a-f]{64}$/.test(digest)) {
      throw new Error(`Invalid model evidence hash: ${id}`);
    }

    const status = str(model.evaluation_status, `${id}.status`);

    if (
      status !== (
        id === "dino_full" || id === "dino_tiled"
          ? "ORIGINAL_FROZEN_EVALUATION"
          : "RETROSPECTIVE_BENCHMARK_NOT_INDEPENDENT"
      )
    ) {
      throw new Error(`Unexpected research status: ${id}`);
    }

    return {
      id,
      name: str(model.name, `${id}.name`),
      family: str(model.family, `${id}.family`),
      phase: str(model.phase, `${id}.phase`),
      research_question: str(
        model.research_question, `${id}.research_question`,
      ),
      current_inspection: id === "dino_672",
      evaluation_status: status,
      per_category: categoryRows,
      aggregate: {
        ...reported,
        macro_auroc: aucSum / 4,
        macro_average_precision: apSum / 4,
        defect_recall: totals.TP / 400,
        normal_review_rate: totals.FP / 402,
      },
      source: {
        path: str(source.path, `${id}.source.path`),
        sha256: digest,
        experiment: str(source.experiment, `${id}.source.experiment`),
      },
    };
  });

  if (
    !Array.isArray(methodology.limitations) ||
    !methodology.limitations.every(
      (v: unknown) => typeof v === "string" && v.length > 0
    )
  ) {
    throw new Error("Research limitations are missing");
  }

  return {
    schema_version: 1,
    study_type: str(root.study_type, "study_type"),
    current_inspection_id: "dino_672",
    cohort: {
      total: 802,
      anomaly_labelled: 400,
      normal_labelled: 402,
    },
    models: parsed,
    methodology: {
      threshold_note: str(
        methodology.threshold_note, "threshold_note",
      ),
      selection_note: str(
        methodology.selection_note, "selection_note",
      ),
      limitations: methodology.limitations as string[],
    },
  };
}

export function metricValue(
  model: ModelRecord,
  scope: Scope,
  metric: Metric,
): number {
  if (scope === "all") {
    switch (metric) {
      case "auroc": return model.aggregate.macro_auroc;
      case "ap": return model.aggregate.macro_average_precision;
      case "recall": return model.aggregate.defect_recall;
      case "review": return model.aggregate.normal_review_rate;
    }
  }

  const row = model.per_category[scope];

  switch (metric) {
    case "auroc": return row.auroc;
    case "ap": return row.average_precision;
    case "recall": return row.TP / (row.TP + row.FN);
    case "review": return row.FP / (row.FP + row.TN);
  }
}

export async function loadModelRegistry(
  signal: AbortSignal,
): Promise<ModelRegistry> {
  const response = await fetch("/benchmark/model-registry.json", {
    signal,
    cache: "no-store",
  });

  if (!response.ok) {
    throw new Error(
      `Model research evidence unavailable (HTTP ${response.status})`,
    );
  }

  return parseModelRegistry(await response.json());
}
