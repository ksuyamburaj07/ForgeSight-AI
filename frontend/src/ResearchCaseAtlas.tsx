import { useEffect, useRef, useState } from "react";
import type { KeyboardEvent, PointerEvent } from "react";

import { loadAtlas } from "./atlas";
import { HistoricalPatchExplorer } from "./HistoricalPatchExplorer";
import type { AtlasCase, AtlasManifest } from "./atlas";
import "./ResearchCaseAtlas.css";

function clamp(value: number, low: number, high: number) {
  return Math.max(low, Math.min(high, value));
}

function decimal(value: number) {
  return value.toFixed(8);
}

function conclusion(item: AtlasCase): string {
  if (item.reference_label === "anomaly") {
    return item.review_flag
      ? "Anomaly labelled · review flagged"
      : "Anomaly labelled · review not flagged";
  }

  return item.review_flag
    ? "Normal labelled · review flagged"
    : "Normal labelled · review not flagged";
}

function TriptychViewer({ item }: { item: AtlasCase }) {
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });

  const pointer = useRef<{
    id: number;
    x: number;
    y: number;
    panX: number;
    panY: number;
  } | null>(null);

  function changeZoom(value: number) {
    const next = clamp(value, 1, 3);
    setZoom(next);

    if (next === 1) {
      setPan({ x: 0, y: 0 });
    }
  }

  function boundPan(x: number, y: number, element: HTMLElement) {
    const rect = element.getBoundingClientRect();

    return {
      x: clamp(
        x,
        -rect.width * (zoom - 1) / 2,
        rect.width * (zoom - 1) / 2,
      ),
      y: clamp(
        y,
        -rect.height * (zoom - 1) / 2,
        rect.height * (zoom - 1) / 2,
      ),
    };
  }

  function pointerDown(event: PointerEvent<HTMLDivElement>) {
    if (
      zoom <= 1 ||
      pointer.current ||
      (event.pointerType === "mouse" && event.button !== 0)
    ) {
      return;
    }

    pointer.current = {
      id: event.pointerId,
      x: event.clientX,
      y: event.clientY,
      panX: pan.x,
      panY: pan.y,
    };

    event.currentTarget.setPointerCapture(event.pointerId);
  }

  function pointerMove(event: PointerEvent<HTMLDivElement>) {
    const start = pointer.current;

    if (!start || start.id !== event.pointerId) return;

    setPan(
      boundPan(
        start.panX + event.clientX - start.x,
        start.panY + event.clientY - start.y,
        event.currentTarget,
      ),
    );
  }

  function pointerEnd(event: PointerEvent<HTMLDivElement>) {
    if (pointer.current?.id !== event.pointerId) return;

    pointer.current = null;

    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
  }

  function keyboard(event: KeyboardEvent<HTMLDivElement>) {
    if (zoom <= 1) return;

    const steps: Record<string, [number, number]> = {
      ArrowLeft: [-30, 0],
      ArrowRight: [30, 0],
      ArrowUp: [0, -30],
      ArrowDown: [0, 30],
    };

    if (event.key === "0") {
      event.preventDefault();
      changeZoom(1);
      return;
    }

    const movement = steps[event.key];
    if (!movement) return;

    event.preventDefault();

    setPan((current) =>
      boundPan(
        current.x + movement[0],
        current.y + movement[1],
        event.currentTarget,
      ),
    );
  }

  return (
    <figure className="atlas-triptych">
      <div className="atlas-image-topline">
        <div>
          <span className="atlas-eyebrow">Historical visual evidence</span>
          <strong>Original · Distance map · Overlay</strong>
        </div>

        <div className="atlas-image-tools">
          <button
            type="button"
            onClick={() => changeZoom(zoom - 0.25)}
            disabled={zoom <= 1}
            aria-label="Zoom historical image out"
          >
            −
          </button>

          <label>
            <span className="atlas-sr-only">
              Historical image zoom
            </span>
            <input
              aria-label="Historical image zoom"
              type="range"
              min="1"
              max="3"
              step="0.25"
              value={zoom}
              onChange={(e) =>
                changeZoom(Number(e.target.value))
              }
            />
          </label>

          <button
            type="button"
            onClick={() => changeZoom(zoom + 0.25)}
            disabled={zoom >= 3}
            aria-label="Zoom historical image in"
          >
            +
          </button>

          <button
            type="button"
            onClick={() => {
              setPan({ x: 0, y: 0 });
              setZoom(1);
            }}
          >
            Fit
          </button>
        </div>
      </div>

      <div
        className="atlas-image-stage"
        tabIndex={0}
        role="group"
        aria-label={
          `${item.id} historical triptych. ` +
          "Zoom with controls. At magnification, drag or use " +
          "arrow keys to pan. Press zero to reset."
        }
        onPointerDown={pointerDown}
        onPointerMove={pointerMove}
        onPointerUp={pointerEnd}
        onPointerCancel={pointerEnd}
        onLostPointerCapture={() => {
          pointer.current = null;
        }}
        onKeyDown={keyboard}
      >
        <img
          src={item.triptych.url}
          alt={
            `${item.id} historical research triptych showing ` +
            "the original PCB image, interpolated patch-distance " +
            "heatmap, and source-aligned overlay."
          }
          draggable={false}
          loading="lazy"
          style={{
            transform:
              `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`,
          }}
        />
      </div>

      <figcaption>
        CPU-generated retrospective visualization. Heatmap colour
        intensity is scaled for this image; it is not a defect mask.
        This triptych is a combined research export, not a new
        inspection performed by your browser.
      </figcaption>
    </figure>
  );
}

function CaseDetail({ item }: { item: AtlasCase }) {
  const [evidenceMode, setEvidenceMode] =
    useState<"triptych" | "patch">("triptych");
  const high = Math.max(item.cpu.score, item.cpu.threshold) * 1.15;
  const scorePosition = item.cpu.score / high * 100;
  const thresholdPosition = item.cpu.threshold / high * 100;

  return (
    <article className="atlas-detail" aria-live="polite">
      <header className="atlas-detail-header">
        <div>
          <span className="atlas-eyebrow">
            {item.category.toUpperCase()} / Retrospective investigation
          </span>
          <h3>{item.title}</h3>
          <p>{item.finding}</p>
        </div>

        <span
          className={
            "atlas-status " +
            (item.review_flag ? "is-flagged" : "is-unflagged")
          }
        >
          {item.review_flag
            ? "Review flagged"
            : "Review not flagged"}
        </span>
      </header>

      <div
        className="atlas-evidence-modes"
        role="group"
        aria-label="Historical evidence display"
      >
        <button
          type="button"
          aria-pressed={evidenceMode === "triptych"}
          onClick={() => setEvidenceMode("triptych")}
        >
          Image evidence
        </button>
        <button
          type="button"
          aria-pressed={evidenceMode === "patch"}
          onClick={() => setEvidenceMode("patch")}
        >
          Raw patch field
        </button>
      </div>

      {evidenceMode === "triptych" ? (
        <TriptychViewer key={item.id} item={item} />
      ) : (
        <HistoricalPatchExplorer item={item} />
      )}

      <div className="atlas-measurement-grid">
        <div>
          <span>CPU anomaly score</span>
          <strong>{decimal(item.cpu.score)}</strong>
          <small>Cosine distance statistic</small>
        </div>
        <div>
          <span>Frozen threshold</span>
          <strong>{decimal(item.cpu.threshold)}</strong>
          <small>{item.category.toUpperCase()} operating threshold</small>
        </div>
        <div>
          <span>Eligible patches</span>
          <strong>{item.cpu.valid_patches.toLocaleString()}</strong>
          <small>Of 2,304 model patch positions</small>
        </div>
      </div>

      <section
        className="atlas-threshold"
        aria-label="Anomaly score and frozen threshold comparison"
      >
        <div className="atlas-threshold-title">
          <strong>Score relative to the frozen threshold</strong>
          <span>
            {conclusion(item)}
          </span>
        </div>

        <div className="atlas-track">
          <div
            className="atlas-score-bar"
            style={{ width: `${scorePosition}%` }}
          />
          <span
            className="atlas-threshold-marker"
            style={{ left: `${thresholdPosition}%` }}
          />
        </div>

        <div className="atlas-track-labels">
          <span>0</span>
          <span>
            Orange bar: CPU score · marker: frozen threshold
          </span>
          <span>{high.toFixed(3)}</span>
        </div>

        <p>
          This is a distance comparison, not a probability or a
          calibrated confidence percentage. The threshold was not
          adjusted for this case.
        </p>
      </section>

      <section className="atlas-interpretation">
        <div>
          <span className="atlas-eyebrow">
            What the visual audit observed
          </span>
          <h4>Reading the evidence</h4>
          <p>{item.audit_observation}</p>
          <p>
            The source image was labelled{" "}
            <strong>{item.reference_label}</strong>; the frozen
            research operating rule{" "}
            <strong>
              {item.review_flag
                ? "flagged it for review."
                : "did not flag it for review."}
            </strong>
          </p>
        </div>

        <div>
          <span className="atlas-eyebrow">
            Numerical reproduction
          </span>
          <h4>
            {item.reproduction.within_original_tolerance
              ? "Within the original tolerance"
              : "Outside the original tolerance"}
          </h4>
          <p>
            Frozen reference score:{" "}
            <strong>
              {item.reproduction.frozen_score.toFixed(10)}
            </strong>
          </p>
          <p>
            Absolute CPU difference:{" "}
            <strong>
              {item.reproduction.absolute_delta.toFixed(10)}
            </strong>
          </p>
          <p>
            Original tolerance:{" "}
            {item.reproduction.original_tolerance.toFixed(5)}.
            The strict four-case gate remains open.
          </p>
        </div>
      </section>

      <details className="atlas-provenance">
        <summary>Inspect case provenance and limitations</summary>

        <dl>
          <dt>Research case</dt>
          <dd>{item.id}</dd>
          <dt>Model</dt>
          <dd>{item.provenance.model_id}</dd>
          <dt>Source SHA-256 (reported in CPU record)</dt>
          <dd><code>{item.provenance.source_sha256_reported}</code></dd>
          <dt>Model weights SHA-256</dt>
          <dd><code>{item.provenance.model_weights_sha256}</code></dd>
          <dt>Source record</dt>
          <dd><code>{item.provenance.record_source}</code></dd>
          <dt>Triptych SHA-256 (export manifest)</dt>
          <dd><code>{item.triptych.sha256}</code></dd>
          <dt>Validation status</dt>
          <dd><code>{item.cpu.validation_status}</code></dd>
        </dl>

        <p>
          These cases were previously selected for retrospective
          diagnostics. They are not independent validation, causal
          localization proof or manufacturing acceptance evidence.
        </p>
      </details>
    </article>
  );
}

export function ResearchCaseAtlas() {
  const [manifest, setManifest] = useState<AtlasManifest | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);
  const [retry, setRetry] = useState(0);
  const [active, setActive] = useState("B3-01");

  useEffect(() => {
    const controller = new AbortController();

    setLoading(true);
    setError("");

    loadAtlas(controller.signal)
      .then((data) => {
        if (!controller.signal.aborted) {
          setManifest(data);
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

  if (loading) {
    return (
      <section className="fs-atlas" role="status">
        <p>Loading the historical research evidence…</p>
      </section>
    );
  }

  if (error || !manifest) {
    return (
      <section className="fs-atlas">
        <h2>Research evidence unavailable</h2>
        <p role="alert">
          {error || "The Atlas manifest was not loaded."}
        </p>
        <button type="button" onClick={() => setRetry((v) => v + 1)}>
          Retry loading
        </button>
      </section>
    );
  }

  const current =
    manifest.cases.find((item) => item.id === active) ??
    manifest.cases[0];

  return (
    <section className="fs-atlas" aria-labelledby="fs-atlas-title">
      <header className="fs-atlas-intro">
        <div>
          <span className="atlas-eyebrow">
            Research Case Atlas / Four investigations
          </span>
          <h2 id="fs-atlas-title">
            Where the model meets its limits.
          </h2>
          <p>
            Explore four real retrospective investigations—two
            missed anomalies and two false review flags.
            Every displayed measurement comes from retained
            research records.
          </p>
        </div>
        <div className="fs-atlas-number">
          <strong>04</strong>
          <span>research cases</span>
        </div>
      </header>

      <div className="fs-atlas-layout">
        <nav
          className="fs-atlas-nav"
          aria-label="Historical research cases"
        >
          {manifest.cases.map((item, index) => (
            <button
              key={item.id}
              type="button"
              className={
                "fs-atlas-case " +
                (current.id === item.id ? "is-active" : "")
              }
              aria-pressed={current.id === item.id}
              onClick={() => setActive(item.id)}
            >
              <span className="fs-atlas-case-number">
                {String(index + 1).padStart(2, "0")}
                <small>{item.category.toUpperCase()}</small>
              </span>
              <span className="fs-atlas-case-info">
                <strong>{item.title}</strong>
                <small>
                  {item.reference_label === "anomaly"
                    ? "Missed anomaly"
                    : "False review flag"}
                </small>
              </span>
            </button>
          ))}
          <p>
            Cases are deliberately selected investigations,
            not a random evaluation sample.
          </p>
        </nav>

        <CaseDetail key={current.id} item={current} />
      </div>

      <footer className="fs-atlas-attribution">
        <strong>Dataset and research attribution</strong>
        <p>
          Original VisA imagery: {manifest.license.dataset_creator}.
          Dataset licence:{" "}
          <a
            href={manifest.license.license_url}
            target="_blank"
            rel="noopener noreferrer"
          >
            {manifest.license.dataset_license}
          </a>.
          {" "}{manifest.license.modifications}
        </p>
        <p>
          Four reused diagnostic cases; no new independent
          benchmark or segmentation evaluation. The CPU
          cross-platform score-reproduction gate remains open.
        </p>
      </footer>
    </section>
  );
}
