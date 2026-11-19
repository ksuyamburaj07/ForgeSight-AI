import { useEffect, useMemo, useState } from "react";
import type { KeyboardEvent, MouseEvent } from "react";

import type { AtlasCase } from "./atlas";
import {
  loadHistoricalPatch,
  rankPatches,
} from "./historicalPatch";
import type { HistoricalPatch } from "./historicalPatch";

import "./HistoricalPatchExplorer.css";

type Selection = { row: number; col: number };

const bound = (n: number) =>
  Math.max(0, Math.min(47, Math.round(n)));

function Profile({
  label,
  values,
  selected,
}: {
  label: string;
  values: (number | null)[];
  selected: number;
}) {
  const max = Math.max(
    0.000001,
    ...values.filter((n): n is number => n !== null),
  );

  let drawing = false;

  const path = values.map((value, index) => {
    if (value === null) {
      drawing = false;
      return "";
    }

    const x = 12 + index * (456 / 47);
    const y = 110 - (value / max) * 94;
    const command = drawing ? "L" : "M";
    drawing = true;

    return `${command}${x.toFixed(2)} ${y.toFixed(2)}`;
  }).join(" ");

  const selectedValue = values[selected];

  return (
    <figure className="hp-profile">
      <figcaption>{label}</figcaption>
      <svg viewBox="0 0 480 135" role="img"
        aria-label={`${label} raw patch-distance profile`}
      >
        <path d="M12 110 H468" className="hp-axis" />
        <path d={path} className="hp-profile-line" />
        {selectedValue !== null && (
          <circle
            cx={12 + selected * (456 / 47)}
            cy={110 - (selectedValue / max) * 94}
            r="5"
            className="hp-profile-marker"
          />
        )}
        <text x="12" y="128">0</text>
        <text x="450" y="128">47</text>
      </svg>
    </figure>
  );
}

export function HistoricalPatchExplorer({
  item,
}: {
  item: AtlasCase;
}) {
  const [field, setField] =
    useState<HistoricalPatch | null>(null);
  const [selected, setSelected] =
    useState<Selection | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [retry, setRetry] = useState(0);

  useEffect(() => {
    const controller = new AbortController();

    setField(null);
    setSelected(null);
    setLoading(true);
    setError("");

    loadHistoricalPatch(item, controller.signal)
      .then((data) => {
        if (controller.signal.aborted) return;

        setField(data);
        const highest = rankPatches(data)[0];

        setSelected(
          highest
            ? { row: highest.row, col: highest.col }
            : null,
        );
      })
      .catch((cause: unknown) => {
        if (controller.signal.aborted) return;

        setError(
          cause instanceof Error
            ? cause.message
            : "Historical evidence could not be loaded.",
        );
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false);
      });

    return () => controller.abort();
  }, [item, retry]);

  const ranked = useMemo(
    () => field ? rankPatches(field) : [],
    [field],
  );

  if (loading) {
    return (
      <section className="hp-explorer" role="status">
        Loading verified patch measurements…
      </section>
    );
  }

  if (!field || error) {
    return (
      <section className="hp-explorer">
        <p role="alert">
          {error || "Historical patch evidence unavailable."}
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

  const max = ranked[0]?.distance ?? 1;
  const distance = selected
    ? field.distances[selected.row][selected.col]
    : null;

  function chooseFromPointer(
    event: MouseEvent<SVGSVGElement>,
  ) {
    const rect = event.currentTarget.getBoundingClientRect();

    if (rect.width <= 0 || rect.height <= 0) return;

    const col = bound(
      ((event.clientX - rect.left) / rect.width) * 48,
    );
    const row = bound(
      ((event.clientY - rect.top) / rect.height) * 48,
    );

    setSelected({ row, col });
  }

  function keyboard(event: KeyboardEvent<SVGSVGElement>) {
    const movement: Record<string, [number, number]> = {
      ArrowUp: [-1, 0],
      ArrowDown: [1, 0],
      ArrowLeft: [0, -1],
      ArrowRight: [0, 1],
    };

    const step = movement[event.key];
    if (!step) return;

    event.preventDefault();

    setSelected((current) => ({
      row: bound((current?.row ?? 24) + step[0]),
      col: bound((current?.col ?? 24) + step[1]),
    }));
  }

  function changeAxis(
    axis: "row" | "col",
    raw: string,
  ) {
    if (raw.trim() === "") return;

    const value = Number(raw);
    if (!Number.isFinite(value)) return;

    setSelected((current) => ({
      ...(current ?? { row: 24, col: 24 }),
      [axis]: bound(value),
    }));
  }

  const rowProfile = selected
    ? field.distances[selected.row]
    : [];

  const colProfile = selected
    ? field.distances.map((row) => row[selected.col])
    : [];

  return (
    <section className="hp-explorer"
      aria-label={`${item.id} historical patch measurements`}
    >
      <div className="hp-heading">
        <div>
          <span className="hp-eyebrow">
            Archived numerical evidence
          </span>
          <h4>Inside the distance field.</h4>
          <p>
            Select a patch to inspect its retained raw cosine
            distance. Colour is relative to the maximum eligible
            distance in this historical array.
          </p>
        </div>

        <div className="hp-quantity">
          <strong>{field.valid_count.toLocaleString()}</strong>
          <span>eligible cells</span>
        </div>
      </div>

      <div className="hp-content">
        <div className="hp-map-column">
          <svg
            className="hp-grid"
            viewBox="0 0 480 480"
            tabIndex={0}
            role="group"
            aria-label={
              "Historical 48 by 48 measurement grid. " +
              "Click to select a cell or use arrow keys."
            }
            onClick={chooseFromPointer}
            onKeyDown={keyboard}
          >
            {field.distances.map((row, r) =>
              row.map((value, c) => {
                const ratio = value === null || max === 0
                  ? 0
                  : Math.min(1, value / max);

                const fill = value === null
                  ? "#4B5550"
                  : `hsl(${185 - ratio * 165} 73% ${
                      33 + ratio * 25
                    }%)`;

                return (
                  <rect
                    key={`${r}-${c}`}
                    x={c * 10}
                    y={r * 10}
                    width="10"
                    height="10"
                    fill={fill}
                  />
                );
              }),
            )}

            {selected && (
              <rect
                x={selected.col * 10}
                y={selected.row * 10}
                width="10"
                height="10"
                fill="none"
                stroke="#F6F9EF"
                strokeWidth="2"
                pointerEvents="none"
              />
            )}
          </svg>

          <div className="hp-legend">
            <span>Lower distance</span>
            <span className="hp-gradient" />
            <span>Higher distance</span>
          </div>
          <p className="hp-map-note">
            Grey cells are excluded image-padding regions,
            not measurements of low anomaly distance.
          </p>
        </div>

        <div className="hp-analysis">
          <div className="hp-reading" aria-live="polite">
            <span>Selected patch</span>
            <div className="hp-coordinates">
              <label>
                Row
                <input
                  type="number"
                  min="0"
                  max="47"
                  aria-label="Historical patch row"
                  value={selected?.row ?? ""}
                  onChange={(e) =>
                    changeAxis("row", e.target.value)
                  }
                />
              </label>
              <label>
                Column
                <input
                  type="number"
                  min="0"
                  max="47"
                  aria-label="Historical patch column"
                  value={selected?.col ?? ""}
                  onChange={(e) =>
                    changeAxis("col", e.target.value)
                  }
                />
              </label>
            </div>

            <strong>
              {distance === null
                ? "Not measured"
                : distance.toFixed(8)}
            </strong>
            <small>
              {distance === null
                ? "Excluded patch centre"
                : "Raw cosine nearest-neighbour distance"}
            </small>
          </div>

          <div className="hp-top">
            <span className="hp-eyebrow">
              Highest-distance patches
            </span>
            <p>
              Top 1% comprises {field.top_k_count} eligible
              patches. Select a position to investigate;
              these are not confirmed defects.
            </p>

            <div className="hp-rank-list">
              {ranked.slice(0, 6).map((patch, i) => (
                <button
                  key={`${patch.row}:${patch.col}`}
                  type="button"
                  aria-pressed={
                    selected?.row === patch.row &&
                    selected?.col === patch.col
                  }
                  onClick={() => setSelected({
                    row: patch.row,
                    col: patch.col,
                  })}
                >
                  <span>{String(i + 1).padStart(2, "0")}</span>
                  <span>R{patch.row} · C{patch.col}</span>
                  <strong>{patch.distance.toFixed(6)}</strong>
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>

      {selected && (
        <div className="hp-profiles">
          <Profile
            label={`Row ${selected.row} · 48 positions`}
            values={rowProfile}
            selected={selected.col}
          />
          <Profile
            label={`Column ${selected.col} · 48 positions`}
            values={colProfile}
            selected={selected.row}
          />
        </div>
      )}

      <footer className="hp-disclosure">
        These measurements come from the retained M3.13C
        diagnostic-array archive. Their top-1% scores closely
        reproduce the frozen references, but execution origin
        has not been independently established. They are not
        the Dell CPU distances in the historical triptych.
        No new inference or defect segmentation is performed.
      </footer>
    </section>
  );
}
