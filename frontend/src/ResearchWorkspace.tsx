import { useState } from "react";
import type { ReactNode } from "react";

import { ModelExplorer } from "./ModelExplorer";
import { DatasetExplorer } from "./DatasetExplorer";
import { ResearchCaseAtlas } from "./ResearchCaseAtlas";

import "./ResearchWorkspace.css";

type Section = "models" | "dataset" | "cases" | "record";

const sections: {
  id: Section;
  label: string;
  description: string;
}[] = [
  {
    id: "models",
    label: "Model Explorer",
    description:
      "Six evaluated configurations, per-PCB metrics " +
      "and research-selection tradeoffs.",
  },
  {
    id: "dataset",
    label: "Dataset Explorer",
    description:
      "Original inventory, cohort structure, frozen "
      + "letterbox geometry and data lineage.",
  },
  {
    id: "cases",
    label: "Case Atlas",
    description:
      "Four retrospective investigations with historical " +
      "images and raw patch-distance evidence.",
  },
  {
    id: "record",
    label: "Engineering Record",
    description:
      "Frozen methodology, numerical reproduction " +
      "and documented scientific limitations.",
  },
];

export function ResearchWorkspace({
  record,
}: {
  record: ReactNode;
}) {
  const [active, setActive] =
    useState<Section>("models");

  const [visited, setVisited] = useState<
    Record<Section, boolean>
  >({
    models: true,
    dataset: false,
    cases: false,
    record: false,
  });

  const current = sections.find(
    (section) => section.id === active,
  )!;

  function activate(section: Section) {
    setVisited((previous) => ({
      ...previous,
      [section]: true,
    }));
    setActive(section);
  }

  return (
    <div className="research-workspace">
      <nav
        className="rw-navigation"
        aria-label="Research workspace sections"
      >
        <div className="rw-navigation-heading">
          <span>Research workspace</span>
          <p>
            Explore the experiments, their failure cases,
            and the verified engineering record.
          </p>
        </div>

        <div className="rw-navigation-buttons">
          {sections.map((section, index) => (
            <button
              key={section.id}
              id={`rw-button-${section.id}`}
              type="button"
              aria-pressed={active === section.id}
              aria-controls={`rw-panel-${section.id}`}
              onClick={() => activate(section.id)}
            >
              <span className="rw-navigation-number">
                {String(index + 1).padStart(2, "0")}
              </span>
              <span>{section.label}</span>
            </button>
          ))}
        </div>

        <p className="rw-current-description">
          {current.description}
        </p>
      </nav>

      <section
        id="rw-panel-models"
        className="rw-panel"
        hidden={active !== "models"}
        aria-labelledby="rw-button-models"
      >
        {visited.models && <ModelExplorer />}
      </section>

      <section
        id="rw-panel-dataset"
        className="rw-panel"
        hidden={active !== "dataset"}
        aria-labelledby="rw-button-dataset"
      >
        {visited.dataset && <DatasetExplorer />}
      </section>

      <section
        id="rw-panel-cases"
        className="rw-panel"
        hidden={active !== "cases"}
        aria-labelledby="rw-button-cases"
      >
        {visited.cases && <ResearchCaseAtlas />}
      </section>

      <section
        id="rw-panel-record"
        className="rw-panel"
        hidden={active !== "record"}
        aria-labelledby="rw-button-record"
      >
        {visited.record && record}
      </section>
    </div>
  );
}
