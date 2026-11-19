import { useEffect, useState } from "react";
import type { KeyboardEvent, MouseEvent } from "react";

import {
  isEligible,
  loadDatasetRegistry,
} from "./datasetRegistry";
import type {
  DatasetRegistry,
  PCB,
} from "./datasetRegistry";

import { DatasetGallery } from "./DatasetGallery";
import {
  galleryPreviewPath,
  letterboxImageBounds,
} from "./datasetVisual";
import type { DatasetViewMode } from "./datasetVisual";

import "./DatasetExplorer.css";

const PCBS: PCB[] = ["pcb1", "pcb2", "pcb3", "pcb4"];

const clamp = (n: number) =>
  Math.max(0, Math.min(47, Math.floor(n)));

function percentage(value: number, total: number) {
  return `${(100 * value / total).toFixed(1)}%`;
}

export function DatasetExplorer() {
  const [registry, setRegistry] =
    useState<DatasetRegistry | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [retry, setRetry] = useState(0);

  const [pcb, setPCB] = useState<PCB>("pcb1");
  const [galleryIndex, setGalleryIndex] = useState(0);
  const [viewMode, setViewMode] =
    useState<DatasetViewMode>("overlay");
  const [selected, setSelected] = useState({
    row: 24,
    col: 24,
  });

  useEffect(() => {
    const controller = new AbortController();

    setLoading(true);
    setError("");

    loadDatasetRegistry(controller.signal)
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
              : "Dataset inventory could not be loaded.",
          );
        }
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false);
      });

    return () => controller.abort();
  }, [retry]);

  if (loading) {
    return (
      <section className="ds-explorer" role="status">
        Loading verified dataset inventory…
      </section>
    );
  }

  if (!registry || error) {
    return (
      <section className="ds-explorer">
        <p role="alert">
          {error || "Dataset inventory unavailable."}
        </p>
        <button
          type="button"
          onClick={() => setRetry((n) => n + 1)}
        >
          Retry loading
        </button>
      </section>
    );
  }

  const category = registry.categories[pcb];
  const box = category.letterbox;
  const imageBounds = letterboxImageBounds(box);
  const imageSrc = galleryPreviewPath(pcb, galleryIndex);
  const eligible = isEligible(
    box, selected.row, selected.col,
  );

  function pick(event: MouseEvent<SVGSVGElement>) {
    const rect = event.currentTarget.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) return;

    setSelected({
      row: clamp(
        (event.clientY - rect.top) / rect.height * 48,
      ),
      col: clamp(
        (event.clientX - rect.left) / rect.width * 48,
      ),
    });
  }

  function keydown(event: KeyboardEvent<SVGSVGElement>) {
    const steps: Record<string, [number, number]> = {
      ArrowUp: [-1, 0],
      ArrowDown: [1, 0],
      ArrowLeft: [0, -1],
      ArrowRight: [0, 1],
    };

    const step = steps[event.key];
    if (!step) return;

    event.preventDefault();

    setSelected((current) => ({
      row: clamp(current.row + step[0]),
      col: clamp(current.col + step[1]),
    }));
  }

  function changeAxis(
    axis: "row" | "col",
    value: string,
  ) {
    if (!value.trim()) return;

    const number = Number(value);
    if (!Number.isFinite(number)) return;

    setSelected((current) => ({
      ...current,
      [axis]: clamp(number),
    }));
  }

  const points = Array.from({ length: 48 * 48 }, (_, i) => {
    const row = Math.floor(i / 48);
    const col = i % 48;
    return { row, col, valid: isEligible(box, row, col) };
  });

  return (
    <section
      className="ds-explorer"
      id="dataset-explorer"
      aria-labelledby="dataset-explorer-title"
    >
      <header className="ds-intro">
        <div>
          <span className="ds-overline">
            Research / Dataset Explorer
          </span>
          <h2 id="dataset-explorer-title">
            Understand the data behind every decision.
          </h2>
          <p>
            Explore the original VisA PCB inventory, the
            evaluation cohort and the source geometry used
            by the current inspection configuration.
          </p>
        </div>
        <div className="ds-total">
          <strong>4,416</strong>
          <span>Original PCB images</span>
          <small>Four VisA categories</small>
        </div>
      </header>

      <div className="ds-summary">
        <div>
          <span>Normal-training inventory</span>
          <strong>{registry.totals.train_normal.toLocaleString()}</strong>
          <small>Source training partition</small>
        </div>
        <div>
          <span>Historical evaluation</span>
          <strong>
            {(registry.totals.evaluation_normal +
              registry.totals.evaluation_anomaly).toLocaleString()}
          </strong>
          <small>Reused, not independent per experiment</small>
        </div>
        <div>
          <span>Current frozen reference policy</span>
          <strong>{registry.reference_images_per_category} / PCB</strong>
          <small>Normal reference images, not all training images</small>
        </div>
      </div>

      <div className="ds-section-heading">
        <div>
          <span className="ds-overline">
            Original dataset inventory
          </span>
          <h3>Four categories. Distinct source populations.</h3>
        </div>
        <p>
          These counts describe the original dataset partitions,
          not model predictions or additional experiments.
        </p>
      </div>

      <div className="ds-categories">
        {PCBS.map((id) => {
          const c = registry.categories[id];
          return (
            <button
              key={id}
              type="button"
              aria-pressed={id === pcb}
              className={id === pcb ? "is-selected" : ""}
              onClick={() => {
                setPCB(id);
                setGalleryIndex(0);
              }}
            >
              <span>{id.toUpperCase()}</span>
              <strong>{c.total.toLocaleString()}</strong>
              <small>images</small>
            </button>
          );
        })}
      </div>

      <DatasetGallery
        pcb={pcb}
        selectedIndex={galleryIndex}
        onSelectIndex={setGalleryIndex}
      />

      <div className="ds-distribution">
        <div className="ds-distribution-heading">
          <strong>{pcb.toUpperCase()} · Original inventory</strong>
          <span>{category.total.toLocaleString()} images</span>
        </div>

        <div
          className="ds-distribution-bar"
          role="img"
          aria-label={
            `${category.train_normal} normal training images, ` +
            `${category.evaluation_normal} normal evaluation images, ` +
            `${category.evaluation_anomaly} anomalous evaluation images`
          }
        >
          <span
            className="ds-train"
            style={{
              width: percentage(category.train_normal, category.total),
            }}
          />
          <span
            className="ds-normal"
            style={{
              width: percentage(
                category.evaluation_normal, category.total
              ),
            }}
          />
          <span
            className="ds-anomaly"
            style={{
              width: percentage(
                category.evaluation_anomaly, category.total
              ),
            }}
          />
        </div>

        <div className="ds-legend">
          <span><i className="ds-train" /> Normal training: {category.train_normal}</span>
          <span><i className="ds-normal" /> Evaluation normal: {category.evaluation_normal}</span>
          <span><i className="ds-anomaly" /> Evaluation anomaly: {category.evaluation_anomaly}</span>
        </div>
      </div>

      <div className="ds-section-heading ds-geometry-title">
        <div>
          <span className="ds-overline">
            Interactive geometry / Frozen inspection input
          </span>
          <h3>What the model actually measures.</h3>
        </div>
        <p>
          Explore patch-centre eligibility for the selected
          category's original source dimensions.
        </p>
      </div>

      <div className="ds-geometry-layout">
        <div className="ds-map">
          <div className="ds-map-heading">
            <strong>{pcb.toUpperCase()} · Patch eligibility</strong>
            <span>48×48 grid</span>
          </div>

          <div
            className="ds-view-modes"
            role="group"
            aria-label="Source image and patch geometry display"
          >
            {([
              ["source", "Source photograph"],
              ["overlay", "Patch overlay"],
              ["mask", "Eligibility mask"],
            ] as const).map(([id, label]) => (
              <button
                key={id}
                type="button"
                aria-pressed={viewMode === id}
                onClick={() => setViewMode(id)}
              >
                {label}
              </button>
            ))}
          </div>

          <svg
            viewBox="0 0 480 480"
            role="group"
            tabIndex={0}
            aria-label={
              `Patch eligibility diagram for ${pcb.toUpperCase()}. ` +
              "Click a cell or use arrow keys to inspect its geometry."
            }
            onClick={pick}
            onKeyDown={keydown}
          >
            {viewMode !== "mask" && (
              <image
                href={imageSrc}
                x={imageBounds.x}
                y={imageBounds.y}
                width={imageBounds.width}
                height={imageBounds.height}
                preserveAspectRatio="none"
                pointerEvents="none"
              />
            )}

            {viewMode !== "source" && points.map((point) => (
              <rect
                key={`${point.row}-${point.col}`}
                x={point.col * 10}
                y={point.row * 10}
                width="9.5"
                height="9.5"
                fill={
                  viewMode === "mask"
                    ? point.valid
                      ? "#79A58B"
                      : "#5C6560"
                    : point.valid
                      ? "rgba(118, 197, 139, 0.13)"
                      : "#5C6560"
                }
                stroke={
                  viewMode === "overlay"
                    ? "rgba(248, 252, 243, 0.33)"
                    : "#304036"
                }
                strokeWidth={0.4}
              />
            ))}

            <rect
              x={selected.col * 10}
              y={selected.row * 10}
              width="10"
              height="10"
              fill="none"
              stroke="#FFE58E"
              strokeWidth="2.5"
              pointerEvents="none"
            />
          </svg>

          <p className="ds-illustration-note">
            Showing normal-training example {galleryIndex + 1}
            of 4, geometrically fitted into the 672×672
            input frame. This is an instructional illustration,
            not an encoder output or anomaly heatmap.
          </p>

          <div className="ds-map-legend">
            <span><i className="ds-valid" /> Eligible patch centre</span>
            <span><i className="ds-excluded" /> Excluded padding</span>
          </div>
        </div>

        <div className="ds-geometry-details">
          <div className="ds-measurements">
            <div>
              <span>Original image shape</span>
              <strong>
                {category.source_width} × {category.source_height}
              </strong>
              <small>Source pixels</small>
            </div>
            <div>
              <span>Resized image area</span>
              <strong>
                {box.resized_width} × {box.resized_height}
              </strong>
              <small>Inside the 672×672 model input</small>
            </div>
            <div>
              <span>Eligible centres</span>
              <strong>{box.eligible_patch_centres.toLocaleString()}</strong>
              <small>Out of 2,304 patch positions</small>
            </div>
            <div>
              <span>Excluded centres</span>
              <strong>{box.excluded_patch_centres.toLocaleString()}</strong>
              <small>Image padding, not evidence</small>
            </div>
          </div>

          <div className="ds-cell-reading">
            <span className="ds-overline">Inspect a patch centre</span>

            <div className="ds-cell-inputs">
              <label>
                Row
                <input
                  type="number"
                  min={0}
                  max={47}
                  value={selected.row}
                  onChange={(e) =>
                    changeAxis("row", e.target.value)
                  }
                />
              </label>
              <label>
                Column
                <input
                  type="number"
                  min={0}
                  max={47}
                  value={selected.col}
                  onChange={(e) =>
                    changeAxis("col", e.target.value)
                  }
                />
              </label>
            </div>

            <strong aria-live="polite">
              {eligible
                ? "Eligible image evidence"
                : "Excluded padding"}
            </strong>
            <p>
              {eligible
                ? "This patch centre falls within the resized image."
                : "This patch centre falls in letterbox padding. It is not a measured anomaly distance."}
            </p>
          </div>
        </div>
      </div>

      <div className="ds-lineage">
        <span className="ds-overline">
          Current inspection / Data-to-evidence lineage
        </span>
        <div>
          <span>Original image</span>
          <span>672×672 letterbox</span>
          <span>48×48 patch features</span>
          <span>Nearest-neighbour distances</span>
          <span>Human review</span>
        </div>
        <p>
          This pipeline describes the active high-resolution
          DINOv2 configuration only. Earlier evaluated models
          used different preprocessing, representations and
          scoring rules.
        </p>
      </div>

      <details className="ds-provenance">
        <summary>Dataset provenance and limitations</summary>
        <dl>
          <dt>Dataset</dt>
          <dd>{registry.dataset}</dd>
          <dt>Original inventory record</dt>
          <dd><code>{registry.provenance.inventory_path}</code></dd>
          <dt>Inventory SHA-256</dt>
          <dd><code>{registry.provenance.inventory_sha256}</code></dd>
          <dt>Original manifest SHA-256</dt>
          <dd><code>{registry.provenance.original_manifest_sha256}</code></dd>
          <dt>Paired audit SHA-256</dt>
          <dd><code>{registry.provenance.paired_audit_sha256}</code></dd>
        </dl>

        <ul>
          {registry.limitations.map((note) => (
            <li key={note}>{note}</li>
          ))}
        </ul>
      </details>

      <footer className="ds-footer">
        Original imagery: Amazon Science VisA. This explorer
        publishes aggregate inventory and geometric explanations;
        it does not expose individual sealed evaluation images
        or perform new inference.
      </footer>
    </section>
  );
}
