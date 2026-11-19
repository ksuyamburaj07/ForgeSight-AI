import { useEffect, useMemo, useState } from "react";

import {
  loadModelRegistry,
  metricValue,
} from "./modelRegistry";

import type {
  Metric,
  ModelRecord,
  ModelRegistry,
  PCB,
  Scope,
} from "./modelRegistry";

import "./ModelExplorer.css";

const METRICS: {
  id: Metric;
  title: string;
  note: string;
  lowerBetter: boolean;
}[] = [
  {
    id: "auroc",
    title: "Image AUROC",
    note: "Ranking performance. Higher is better.",
    lowerBetter: false,
  },
  {
    id: "ap",
    title: "Image average precision",
    note: "Precision–recall ranking summary. Higher is better.",
    lowerBetter: false,
  },
  {
    id: "recall",
    title: "Anomaly recall",
    note: "Defect-labelled images flagged at each model's own threshold.",
    lowerBetter: false,
  },
  {
    id: "review",
    title: "Normal false-review rate",
    note: "Normal-labelled images flagged. Lower is better.",
    lowerBetter: true,
  },
];

const PCBS: PCB[] = ["pcb1", "pcb2", "pcb3", "pcb4"];

function formatted(value: number, metric: Metric) {
  return metric === "auroc" || metric === "ap"
    ? value.toFixed(4)
    : `${(value * 100).toFixed(2)}%`;
}

function ResultsTable({ model }: { model: ModelRecord }) {
  return (
    <div className="me-table-scroll">
      <table className="me-breakdown">
        <caption>
          Per-category results for {model.name}. Counts use the
          original model-specific frozen operating threshold.
        </caption>
        <thead>
          <tr>
            <th scope="col">Category</th>
            <th scope="col">AUROC</th>
            <th scope="col">Image AP</th>
            <th scope="col">TP</th>
            <th scope="col">FN</th>
            <th scope="col">FP</th>
            <th scope="col">TN</th>
          </tr>
        </thead>
        <tbody>
          {PCBS.map((pcb) => {
            const row = model.per_category[pcb];

            return (
              <tr key={pcb}>
                <th scope="row">{pcb.toUpperCase()}</th>
                <td>{row.auroc.toFixed(4)}</td>
                <td>{row.average_precision.toFixed(4)}</td>
                <td>{row.TP}</td>
                <td>{row.FN}</td>
                <td>{row.FP}</td>
                <td>{row.TN}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

export function ModelExplorer() {
  const [registry, setRegistry] =
    useState<ModelRegistry | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [retry, setRetry] = useState(0);

  const [scope, setScope] = useState<Scope>("all");
  const [metric, setMetric] = useState<Metric>("auroc");
  const [selectedId, setSelectedId] = useState("dino_672");

  useEffect(() => {
    const controller = new AbortController();

    setLoading(true);
    setError("");

    loadModelRegistry(controller.signal)
      .then((data) => {
        if (!controller.signal.aborted) {
          setRegistry(data);
        }
      })
      .catch((cause: unknown) => {
        if (!controller.signal.aborted) {
          setError(
            cause instanceof Error
              ? cause.message
              : "Research evidence could not be loaded.",
          );
        }
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false);
      });

    return () => controller.abort();
  }, [retry]);

  const selected = registry?.models.find(
    (model) => model.id === selectedId,
  ) ?? registry?.models[5];

  const definition = METRICS.find(
    (item) => item.id === metric,
  )!;

  const compared = useMemo(() => {
    if (!registry) return [];

    return registry.models
      .map((model) => ({
        model,
        value: metricValue(model, scope, metric),
      }))
      .sort((a, b) =>
        definition.lowerBetter
          ? a.value - b.value
          : b.value - a.value,
      );
  }, [registry, scope, metric, definition.lowerBetter]);

  if (loading) {
    return (
      <section className="me-explorer" role="status">
        Loading six verified research configurations…
      </section>
    );
  }

  if (!registry || error || !selected) {
    return (
      <section className="me-explorer">
        <h2>Model evidence unavailable</h2>
        <p role="alert">
          {error || "Model research registry could not be loaded."}
        </p>
        <button
          type="button"
          onClick={() => setRetry((n) => n + 1)}
        >
          Retry
        </button>
      </section>
    );
  }

  const tiled = registry.models.find(
    (model) => model.id === "dino_tiled",
  )!;
  const high = registry.models.find(
    (model) => model.id === "dino_672",
  )!;

  return (
    <section
      className="me-explorer"
      id="model-explorer"
      aria-labelledby="model-explorer-title"
    >
      <header className="me-intro">
        <div>
          <span className="me-overline">
            Research / Model Explorer
          </span>
          <h2 id="model-explorer-title">
            Six experiments. No universal winner.
          </h2>
          <p>
            Explore the verified model configurations behind
            ForgeSight. Their strengths differ across ranking,
            anomaly detection and normal-review workload.
          </p>
        </div>
        <div className="me-cohort">
          <strong>802</strong>
          <span>reused evaluation images</span>
          <small>400 anomaly · 402 normal</small>
        </div>
      </header>

      <div className="me-section-title">
        <div>
          <span className="me-overline">
            Experiment chronology
          </span>
          <h3>How we arrived here.</h3>
        </div>
        <p>
          Select an experiment to inspect the underlying
          evidence. Chronology is not a performance ranking.
        </p>
      </div>

      <div
        className="me-timeline"
        role="group"
        aria-label="Six evaluated research configurations"
      >
        {registry.models.map((model, index) => (
          <button
            key={model.id}
            type="button"
            aria-pressed={selected.id === model.id}
            className={
              "me-timeline-item " +
              (selected.id === model.id ? "is-selected " : "") +
              (model.current_inspection ? "is-current" : "")
            }
            onClick={() => setSelectedId(model.id)}
          >
            <span className="me-step">
              {String(index + 1).padStart(2, "0")}
            </span>
            <strong>{model.name}</strong>
            <span>{model.phase}</span>
            {model.current_inspection && (
              <small>Current research inspection</small>
            )}
          </button>
        ))}
      </div>

      <div className="me-section-title me-compare-heading">
        <div>
          <span className="me-overline">
            Comparative evidence
          </span>
          <h3>Change the question. Change the ranking.</h3>
        </div>
        <p>
          AUROC and average precision summarize ranking.
          Recall and false-review rates reflect different
          frozen thresholds.
        </p>
      </div>

      <div className="me-controls">
        <label>
          Category
          <select
            value={scope}
            onChange={(event) =>
              setScope(event.currentTarget.value as Scope)
            }
          >
            <option value="all">All four PCBs</option>
            {PCBS.map((pcb) => (
              <option key={pcb} value={pcb}>
                {pcb.toUpperCase()}
              </option>
            ))}
          </select>
        </label>

        <div
          className="me-metric-buttons"
          role="group"
          aria-label="Metric to compare"
        >
          {METRICS.map((option) => (
            <button
              key={option.id}
              type="button"
              aria-pressed={metric === option.id}
              onClick={() => setMetric(option.id)}
            >
              {option.title}
            </button>
          ))}
        </div>
      </div>

      <div className="me-chart" aria-label="Six-model comparison">
        <div className="me-chart-heading">
          <strong>
            {definition.title}
            {scope === "all" ? " · all categories" :
              ` · ${scope.toUpperCase()}`}
          </strong>
          <span>{definition.note}</span>
        </div>

        <div className="me-ranked-bars">
          {compared.map(({ model, value }) => (
            <button
              key={model.id}
              type="button"
              className={
                "me-bar-row " +
                (selected.id === model.id ? "is-active" : "")
              }
              onClick={() => setSelectedId(model.id)}
              aria-pressed={selected.id === model.id}
              aria-label={
                `${model.name}: ${formatted(value, metric)}. ` +
                "Select model details."
              }
            >
              <span className="me-bar-name">
                {model.name}
                {model.current_inspection && (
                  <small>Active inspection</small>
                )}
              </span>
              <span className="me-bar-track">
                <span
                  className={
                    "me-bar-fill " +
                    (model.current_inspection ? "is-current" : "")
                  }
                  style={{ width: `${value * 100}%` }}
                />
              </span>
              <strong className="me-bar-value">
                {formatted(value, metric)}
              </strong>
            </button>
          ))}
        </div>

        <p className="me-chart-footnote">
          {scope === "all"
            ? metric === "auroc" || metric === "ap"
              ? "Unweighted mean of four PCB category metrics."
              : "Rate pooled across the corresponding labelled images."
            : "Metric computed for the selected PCB category."}
          {" "}Bars use a common 0–1 scale. This is retrospective
          research, not independent external validation.
        </p>
      </div>

      <article className="me-model-detail">
        <div className="me-detail-header">
          <div>
            <span className="me-overline">
              Selected research configuration
            </span>
            <h3>{selected.name}</h3>
            <p>{selected.research_question}</p>
          </div>
          <span className="me-detail-tag">
            {selected.current_inspection
              ? "Current inspection"
              : selected.phase}
          </span>
        </div>

        <div className="me-detail-kpis">
          <div>
            <span>Macro image AUROC</span>
            <strong>
              {selected.aggregate.macro_auroc.toFixed(4)}
            </strong>
          </div>
          <div>
            <span>Macro image AP</span>
            <strong>
              {selected.aggregate.macro_average_precision.toFixed(4)}
            </strong>
          </div>
          <div>
            <span>Anomalies flagged</span>
            <strong>
              {selected.aggregate.TP} / 400
            </strong>
          </div>
          <div>
            <span>Normal reviews</span>
            <strong>
              {selected.aggregate.FP} / 402
            </strong>
          </div>
        </div>

        <ResultsTable model={selected} />

        <details className="me-provenance">
          <summary>Experiment source and provenance</summary>
          <dl>
            <dt>Model family</dt>
            <dd>{selected.family}</dd>
            <dt>Research experiment</dt>
            <dd>{selected.source.experiment}</dd>
            <dt>Evaluation status</dt>
            <dd>{selected.evaluation_status}</dd>
            <dt>Local source report</dt>
            <dd><code>{selected.source.path}</code></dd>
            <dt>Report SHA-256</dt>
            <dd><code>{selected.source.sha256}</code></dd>
          </dl>
        </details>
      </article>

      <section className="me-decision">
        <span className="me-overline">
          Engineering decision / Interpretation
        </span>
        <h3>Why high-resolution DINOv2?</h3>

        <div className="me-decision-grid">
          <div>
            <span>Higher reported macro image AUROC</span>
            <strong>
              {tiled.aggregate.macro_auroc.toFixed(4)}
            </strong>
            <small>{tiled.name}</small>
          </div>
          <div>
            <span>Current inspection configuration</span>
            <strong>
              {high.aggregate.macro_auroc.toFixed(4)}
            </strong>
            <small>{high.name}</small>
          </div>
        </div>

        <p>
          The earlier tiled configuration ranked images slightly
          better overall. The 672-pixel configuration flagged{" "}
          <strong>
            {high.aggregate.TP - tiled.aggregate.TP} more
            anomaly-labelled images
          </strong>{" "}
          at its frozen thresholds, while also generating{" "}
          <strong>
            {high.aggregate.FP - tiled.aggregate.FP} additional
            normal-image review flags
          </strong>.
        </p>

        <p>
          The current ForgeSight studio uses the 672-pixel
          configuration to explore dense, source-aligned patch
          evidence in a human-supervised workflow. This is an
          engineering and research-demonstration choice—not proof
          of independent model superiority or factory readiness.
        </p>
      </section>

      <footer className="me-methodology">
        <strong>Scientific interpretation</strong>
        <p>{registry.methodology.threshold_note}</p>
        <p>{registry.methodology.selection_note}</p>
        <details>
          <summary>Additional research limitations</summary>
          <ul>
            {registry.methodology.limitations.map((note) => (
              <li key={note}>{note}</li>
            ))}
          </ul>
        </details>
      </footer>
    </section>
  );
}
