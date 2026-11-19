import { ResearchWorkspace } from "./ResearchWorkspace";
import { useEffect, useRef, useState } from "react";
import type { ChangeEvent, DragEvent } from "react";
import type {
  Capabilities,
  Category,
  Health,
  InspectionResponse,
} from "./types";
import {
  connect,
  fingerprint,
  inspect,
  syntheticInput,
  validateFile,
} from "./api";
import { LinkedInspection } from "./LinkedInspection";



type View = "studio" | "research" | "evidence";
const categories: Category[] = ["pcb1", "pcb2", "pcb3", "pcb4"];
function Icon({
  name,
  size = 20,
}: {
  name: "scan" | "upload" | "layers" | "book" | "check" | "download" | "plus";
  size?: number;
}) {
  const paths = {
    scan: (
      <>
        <path d="M8 3H3v5m13-5h5v5M3 16v5h5m13-5v5h-5M7 12h10M12 7v10" />
      </>
    ),
    upload: (
      <>
        <path d="M12 16V4m-4 4 4-4 4 4M4 16v5h16v-5" />
      </>
    ),
    layers: (
      <>
        <path d="m3 8 9-5 9 5-9 5-9-5Zm0 5 9 5 9-5M3 18l9 5 9-5" />
      </>
    ),
    book: (
      <>
        <path d="M12 5v16M3 3c4 0 7 0 9 2 2-2 5-2 9-2v16c-4 0-7 0-9 2-2-2-5-2-9-2V3Z" />
      </>
    ),
    check: <path d="m5 12 4 4L19 6" />,
    download: <path d="M12 3v12m-4-4 4 4 4-4M4 17v4h16v-4" />,
    plus: <path d="M12 4v16M4 12h16" />,
  };
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.5"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      {paths[name]}
    </svg>
  );
}
function download(data: InspectionResponse) {
  const url = URL.createObjectURL(
    new Blob([JSON.stringify(data, null, 2)], { type: "application/json" }),
  );
  const a = document.createElement("a");
  a.href = url;
  a.download = `forgesight-${data.inspection.category}-${data.inspection.source_sha256.slice(0, 12)}.json`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export default function App() {
  const [view, setView] = useState<View>("studio");
  const [health, setHealth] = useState<Health | null>(null);
  const [caps, setCaps] = useState<Capabilities | null>(null);
  const [connectionError, setConnectionError] = useState("");
  const [connecting, setConnecting] = useState(true);
  const [retry, setRetry] = useState(0);
  const [file, setFile] = useState<File | null>(null);
  const [category, setCategory] = useState<Category>("pcb1");
  const [result, setResult] = useState<InspectionResponse | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [validating, setValidating] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [verified, setVerified] = useState(false);
  const input = useRef<HTMLInputElement>(null);
  const selection = useRef(0);

  useEffect(() => {
    const controller = new AbortController();
    let disposed = false;
    const timer = setTimeout(() => controller.abort(), 10000);
    setConnecting(true);
    setConnectionError("");
    connect(controller.signal)
      .then((data) => {
        if (!disposed) {
          setHealth(data.health);
          setCaps(data.capabilities);
        }
      })
      .catch((e) => {
        if (!disposed)
          setConnectionError(
            controller.signal.aborted
              ? "Connection timed out. Start the local API, then reconnect."
              : e instanceof Error
                ? e.message
                : "Service unavailable.",
          );
      })
      .finally(() => {
        clearTimeout(timer);
        if (!disposed) setConnecting(false);
      });
    return () => {
      disposed = true;
      controller.abort();
      clearTimeout(timer);
    };
  }, [retry]);
  useEffect(() => {
    if (!busy) return;
    const started = Date.now();
    const timer = setInterval(
      () => setElapsed(Math.floor((Date.now() - started) / 1000)),
      1000,
    );
    return () => clearInterval(timer);
  }, [busy]);

  async function choose(next: File | undefined) {
    if (!next || busy || !caps) return;
    const id = ++selection.current;
    setFile(null);
    setResult(null);
    setVerified(false);
    setError("");
    setValidating(true);
    try {
      await validateFile(next, caps);
      if (id === selection.current) setFile(next);
    } catch (e) {
      if (id === selection.current)
        setError(e instanceof Error ? e.message : "Unable to load image.");
    } finally {
      if (id === selection.current) setValidating(false);
    }
  }
  function onInput(e: ChangeEvent<HTMLInputElement>) {
    void choose(e.target.files?.[0]);
    e.target.value = "";
  }
  function drop(e: DragEvent) {
    e.preventDefault();
    setDragging(false);
    if (e.dataTransfer.files.length !== 1) {
      setError("Choose one image at a time.");
      return;
    }
    void choose(e.dataTransfer.files[0]);
  }
  async function run() {
    if (!file || busy) return;
    setBusy(true);
    setError("");
    setResult(null);
    setVerified(false);
    setElapsed(0);
    try {
      const hash = await fingerprint(file);
      const data = await inspect(file, category);
      if (hash !== data.inspection.source_sha256)
        throw new Error(
          "Source fingerprint mismatch. The returned evidence has not been displayed.",
        );
      setResult(data);
      setVerified(true);
    } catch (e) {
      setError(
        e instanceof Error ? e.message : "Inspection failed. Please retry.",
      );
    } finally {
      setBusy(false);
    }
  }
  async function loadSynthetic() {
    try {
      await choose(await syntheticInput());
    } catch (e) {
      setError(
        e instanceof Error ? e.message : "Cannot generate the synthetic input.",
      );
    }
  }
  const r = result?.inspection;
  const connected = Boolean(health && caps && !connectionError);

  return (
    <div className="app-shell">
      <a className="skip-link" href="#main">
        Skip to workspace
      </a>
      <aside className="rail" aria-label="Workspace navigation">
        <a className="brand-symbol" href="#main" aria-label="ForgeSight home">
          <Icon name="scan" size={28} />
        </a>
        <div className="rail-links">
          {(
            [
              ["studio", "scan", "Inspection studio"],
              ["research", "book", "Research lab"],
              ["evidence", "layers", "Evidence explorer"],
            ] as const
          ).map(([id, icon, label]) => (
            <button
              key={id}
              className={view === id ? "rail-button active" : "rail-button"}
              onClick={() => setView(id)}
              aria-label={label}
              aria-pressed={view === id}
              title={label}
            >
              <Icon name={icon} size={23} />
            </button>
          ))}
        </div>
        <span className="rail-caption">LOCAL / CPU</span>
        <span className="rail-marker">
          F<span>05</span>
        </span>
      </aside>
      <div className="page">
        <header className="topbar">
          <div className="wordmark">
            FORGESIGHT<span>AI</span>
            <span className="brand-divider" />
            <small>MATERIAL INTELLIGENCE</small>
          </div>
          <div className="top-status">
            <span className={connected ? "status-dot live" : "status-dot"} />
            <span>
              {connecting
                ? "Connecting to local service"
                : connected
                  ? "Local CPU connected"
                  : "Service disconnected"}
            </span>
            <span className="version-tag">RESEARCH PREVIEW</span>
          </div>
        </header>
        <main id="main">
          <div className="workspace-label">
            <span>INDUSTRIAL VISION</span>
            <span className="slash">/</span> VISUAL ANOMALY RESEARCH
          </div>
          <section className="intro">
            <div>
              <div className="eyebrow">
                <span className="orange-tick" /> EVIDENCE BEFORE DECISIONS
              </div>
              <h1>
                {view === "studio" ? (
                  <>
                    See beyond
                    <br />
                    the <em>surface.</em>
                  </>
                ) : view === "research" ? (
                  <>
                    Keep the
                    <br />
                    <em>evidence</em> in view.
                  </>
                ) : (
                  <>
                    Every result.
                    <br />A traceable <em>source.</em>
                  </>
                )}
              </h1>
              <p>
                {view === "studio"
                  ? "A closer look at what matters. Inspect PCB imagery with frozen-model evidence, source-aligned views, and room for human judgment."
                  : view === "research"
                    ? "The engineering record behind the studio. Verified findings, retained limitations, and a scientific gate that stays open."
                    : "Live measurements and provenance, together. Explore the current response and take its complete patch evidence with you."}
              </p>
            </div>
            <div className="protocol-card">
              <span className="eyebrow">THE FROZEN PROTOCOL</span>
              <div className="protocol-grid">
                <span>
                  672<small>INPUT / PX</small>
                </span>
                <span>
                  48×48<small>PATCH GRID</small>
                </span>
              </div>
              <div className="protocol-bottom">
                <span>DINOv2–Small</span>
                <span className="lime-label">HUMAN REVIEW</span>
              </div>
            </div>
          </section>
          <div className="workspace-tabs" aria-label="Workspace views">
            {(["studio", "research", "evidence"] as const).map((v) => (
              <button
                className={view === v ? "tab selected" : "tab"}
                onClick={() => setView(v)}
                aria-pressed={view === v}
                key={v}
              >
                {v === "studio"
                  ? "Inspection studio"
                  : v === "research"
                    ? "Research lab"
                    : "Evidence explorer"}
                {v === "studio" && <span className="tiny-square" />}
              </button>
            ))}
            <span className="workspace-note">FROZEN MODEL / LOCAL CPU</span>
          </div>
          {connectionError && (
            <div className="connection-banner" role="status">
              <div>
                <strong>Local service unavailable</strong>
                <span>{connectionError}</span>
              </div>
              <button
                className="secondary-button"
                onClick={() => {
                  setHealth(null);
                  setCaps(null);
                  setRetry((n) => n + 1);
                }}
                disabled={connecting}
              >
                Reconnect
              </button>
            </div>
          )}
          {view === "studio" ? (
            <>
              <section className="input-strip" aria-labelledby="input-title">
                <div className="input-heading">
                  <span className="section-index">
                    <Icon name="upload" />
                  </span>
                  <div>
                    <h2 id="input-title">Start with an image</h2>
                    <p>One board. One traceable inspection.</p>
                  </div>
                </div>
                <fieldset
                  className="category-control"
                  disabled={busy || validating}
                >
                  <legend>PCB REFERENCE BANK</legend>
                  <div>
                    {categories.map((c) => (
                      <button
                        key={c}
                        type="button"
                        className={
                          c === category ? "category selected" : "category"
                        }
                        aria-pressed={c === category}
                        disabled={Boolean(caps && !caps.categories.includes(c))}
                        onClick={() => {
                          ++selection.current;
                          setCategory(c);
                          setResult(null);
                          setVerified(false);
                          setError("");
                        }}
                      >
                        {c.toUpperCase()}
                      </button>
                    ))}
                  </div>
                </fieldset>
                <input
                  ref={input}
                  className="visually-hidden"
                  type="file"
                  accept="image/png,image/jpeg,image/webp"
                  onChange={onInput}
                  disabled={busy || !connected}
                  aria-label="Upload PCB image"
                />
                <button
                  className="primary-button"
                  onClick={() => void run()}
                  disabled={!file || !connected || busy || validating}
                >
                  {busy ? (
                    <>
                      <span className="spinner" /> Inspecting · {elapsed}s
                    </>
                  ) : (
                    <>
                      <Icon name="scan" /> Run inspection
                    </>
                  )}
                </button>
              </section>
              <div
                className={`dropzone ${dragging ? "dragging" : ""} ${file ? "has-file" : ""}`}
                onDragOver={(e) => {
                  e.preventDefault();
                  if (!busy) setDragging(true);
                }}
                onDragLeave={() => setDragging(false)}
                onDrop={drop}
              >
                <button
                  className="file-picker"
                  disabled={busy || !connected || validating}
                  onClick={() => input.current?.click()}
                >
                  <span className="upload-glyph">
                    <Icon name={file ? "check" : "plus"} />
                  </span>
                  <span>
                    <strong>
                      {validating
                        ? "Checking image…"
                        : file
                          ? file.name
                          : "Drop your PCB image here, or browse files"}{" "}
                    </strong>
                    <small>
                      {file
                        ? `${(file.size / 1024).toFixed(1)} KiB · ready for ${category.toUpperCase()}${file.name === "synthetic-integration-input.png" ? " · SYNTHETIC INPUT" : ""}`
                        : `PNG, JPEG, WebP · ${caps ? (caps.max_upload_bytes / 1048576).toFixed(0) : "8"} MiB maximum · single frame`}
                    </small>
                  </span>
                </button>
                <div className="drop-actions">
                  {file ? (
                    <button
                      className="text-button"
                      disabled={busy}
                      onClick={() => {
                        ++selection.current;
                        setFile(null);
                        setResult(null);
                        setError("");
                        setVerified(false);
                      }}
                    >
                      Clear image
                    </button>
                  ) : (
                    <button
                      className="text-button"
                      disabled={!connected || busy || validating}
                      onClick={() => void loadSynthetic()}
                    >
                      Use synthetic input
                    </button>
                  )}
                  <span className="local-note">
                    Uploads are not retained by the API
                  </span>
                </div>
              </div>
              {error && (
                <div className="error-banner" role="alert">
                  <strong>Inspection needs attention.</strong> {error}
                </div>
              )}
              {busy && (
                <div className="progress-message" role="status">
                  <span className="spinner" /> Running genuine CPU inference.
                  First use loads the verified model and category bank; this can
                  take a moment. Keep this page open.
                </div>
              )}
              <LinkedInspection
                key={r ? `${r.category}:${r.source_sha256}:${r.processing_ms}` : "empty"}
                result={result}
              />
              <section
                className="result-strip"
                aria-label="Current inspection measurements"
              >
                <div>
                  <span className="eyebrow">ANOMALY SCORE</span>
                  <strong>
                    {r ? r.score.toFixed(8) : "—"}
                    <small>cosine distance</small>
                  </strong>
                </div>
                <div>
                  <span className="eyebrow">FROZEN THRESHOLD</span>
                  <strong>
                    {r ? r.threshold.toFixed(8) : "—"}
                    <small>
                      {r ? r.category.toUpperCase() : "Awaiting inspection"}
                    </small>
                  </strong>
                </div>
                <div>
                  <span className="eyebrow">RESEARCH REVIEW</span>
                  <strong className="review-value">
                    {r
                      ? r.review_flag
                        ? "Review flagged"
                        : "Threshold not exceeded"
                      : "Awaiting evidence"}
                    <small>
                      {r
                        ? "Human interpretation required"
                        : "No inference has been performed"}
                    </small>
                  </strong>
                </div>
                <button
                  className="secondary-button"
                  disabled={!result}
                  onClick={() => setView("evidence")}
                >
                  Explore evidence <Icon name="layers" size={17} />
                </button>
              </section>
            </>
          ) : view === "research" ? (
            <Research />
          ) : (
            <Evidence
              result={result}
              verified={verified}
              onUpload={() => setView("studio")}
            />
          )}
          <section className="science-banner">
            <span className="gate-icon">!</span>
            <div>
              <strong>Scientific gate remains open</strong>
              <p>
                CPU cross-platform score reproduction is unresolved. This studio
                presents research anomaly evidence for human review. It does not
                make manufacturing acceptance or rejection decisions.
              </p>
              <code>
                {r?.validation_status ??
                  health?.scientific_status ??
                  "cross_platform_score_gate_open"}
              </code>
            </div>
            <button className="text-button" onClick={() => setView("research")}>
              Read the record
            </button>
          </section>
          <footer>
            <span>
              FORGESIGHT AI <span className="slash">/</span> INDUSTRIAL VISION
            </span>
            <span>Research with context. Decisions with people.</span>
            <span>LOCAL WORKSPACE</span>
          </footer>
        </main>
      </div>
    </div>
  );
}

function Research() {
  return <ResearchWorkspace record={<ResearchRecord />} />;
}

function ResearchRecord() {
  return (
    <section className="research-view">
      <div className="research-lead">
        <div>
          <span className="eyebrow">VERIFIED ENGINEERING RECORD</span>
          <h2>
            Precision includes
            <br />
            knowing the limits.
          </h2>
          <p>
            These figures come from the tracked M3.14B reproduction note. They
            reuse four retrospective cases and are not a new independent
            benchmark.
          </p>
        </div>
        <div className="gate-stat">
          <strong>
            3<span>/4</span>
          </strong>
          <p>
            Frozen scores within the original
            <br />
            absolute tolerance of 0.00005
          </p>
          <span className="orange-label">STRICT GATE OPEN</span>
        </div>
      </div>
      <div className="table-wrap">
        <table>
          <caption>CPU / frozen-score diagnostic · M3.14B</caption>
          <thead>
            <tr>
              <th>Case</th>
              <th>Frozen score</th>
              <th>CPU SDPA score</th>
              <th>Absolute delta</th>
              <th>Tolerance</th>
            </tr>
          </thead>
          <tbody>
            {[
              [
                "B3-01",
                "0.1453762800",
                "0.1453241557",
                "0.0000521243",
                "Outside",
              ],
              [
                "B3-09",
                "0.1426969320",
                "0.1427421272",
                "0.0000451952",
                "Within",
              ],
              [
                "B3-17",
                "0.2334015220",
                "0.2334035933",
                "0.0000020713",
                "Within",
              ],
              [
                "B3-23",
                "0.2991513014",
                "0.2991603315",
                "0.0000090301",
                "Within",
              ],
            ].map((row) => (
              <tr key={row[0]}>
                {row.map((cell, i) => (
                  <td key={i} className={cell === "Outside" ? "outside" : ""}>
                    {cell}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="research-notes">
        <article>
          <span className="eyebrow">WHAT WAS PRESERVED</span>
          <h3>The original scoring contract.</h3>
          <p>
            672×672 letterboxing, 48×48 DINOv2–Small patch embeddings, frozen
            category banks, and the mean of the top 1% of valid cosine
            nearest-neighbor patch distances. No recalibration or foreground
            filtering.
          </p>
        </article>
        <article>
          <span className="eyebrow">WHAT THE VIEWS MEAN</span>
          <h3>Localization is evidence.</h3>
          <p>
            Interpolated heatmaps are not pixel-level defect masks. Colors are
            scaled per image and cannot be compared as absolute values across
            images. Background and PCB boundary sensitivity remain limitations.
          </p>
        </article>
        <article>
          <span className="eyebrow">WHAT REMAINS OPEN</span>
          <h3>Agreement has a boundary.</h3>
          <p>
            All four frozen review flags agreed. One score narrowly exceeded the
            tolerance. Switching from SDPA to eager attention did not resolve
            the mismatch; the cause has not been isolated.
          </p>
        </article>
      </div>
      <p className="source-note">
        Sources: docs/engineering/M314B_CPU_REPRODUCTION_NOTE.md ·
        docs/engineering/M314C2_FOUR_PCB_VISUAL_AUDIT.md · frozen inference
        code. No benchmark images are loaded in this view.
      </p>
    </section>
  );
}
function Evidence({
  result,
  verified,
  onUpload,
}: {
  result: InspectionResponse | null;
  verified: boolean;
  onUpload: () => void;
}) {
  if (!result)
    return (
      <section className="evidence-empty">
        <Icon name="layers" size={44} />
        <span className="eyebrow">EVIDENCE EXPLORER</span>
        <h2>A record begins with a real inspection.</h2>
        <p>
          Run an image through the local API to see its score, source
          fingerprint, runtime, model identity, and complete raw patch evidence
          here.
        </p>
        <button className="primary-button" onClick={onUpload}>
          Open inspection studio
        </button>
      </section>
    );
  const r = result.inspection;
  return (
    <section className="evidence-view">
      <div className="section-title">
        <div>
          <span className="eyebrow">
            CURRENT RESPONSE / {r.category.toUpperCase()}
          </span>
          <h2>Trace the evidence.</h2>
        </div>
        <button className="primary-button" onClick={() => download(result)}>
          <Icon name="download" size={18} /> Download full JSON
        </button>
      </div>
      <div className="evidence-layout">
        <div className="evidence-summary">
          <span className="lime-label">
            {verified ? "SOURCE SHA VERIFIED" : "SOURCE SHA UNVERIFIED"}
          </span>
          <h3>{r.score.toFixed(8)}</h3>
          <p>
            Anomaly score · cosine distance
            <br />A distance statistic, not a probability.
          </p>
          <div className="evidence-rule" />
          <strong>
            {r.review_flag ? "Research review flag" : "Threshold not exceeded"}
          </strong>
          <p>
            Threshold: {r.threshold.toPrecision(12)}
            <br />
            Human interpretation remains necessary.
          </p>
        </div>
        <dl className="provenance">
          {[
            ["Source SHA-256", r.source_sha256],
            ["Model", r.model_id],
            ["Weights SHA-256", r.model_weights_sha256],
            ["Local snapshot", r.local_model_snapshot],
            [
              "Runtime",
              `${r.runtime_device.toUpperCase()} · ${(r.processing_ms / 1000).toFixed(2)} s engine time`,
            ],
            ["Geometry", r.geometry.join(", ")],
            ["Valid patches", `${r.valid_patches} / 2304`],
            ["Scoring method", r.frozen_scoring_method],
            ["Validation status", r.validation_status],
            [
              "Display scaling",
              `${r.visualization.scaling_method} · maximum ${r.visualization.display_scale_max}`,
            ],
          ].map(([label, value]) => (
            <div key={label}>
              <dt>{label}</dt>
              <dd>{value}</dd>
            </div>
          ))}
        </dl>
      </div>
      <p className="source-note">
        The downloaded JSON is the actual API response, including all 48×48 raw
        distances, the eligibility mask, previews, and provenance. This browser
        keeps only the current inspection in memory; reload clears it. Bank
        fingerprints are not provided by this API.
      </p>
    </section>
  );
}
