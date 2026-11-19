import { useState } from "react";

import type { InspectionResponse } from "./types";
import type { Patch } from "./geometry";

import { InspectionViewer } from "./InspectionViewer";
import { SpatialLandscape } from "./SpatialLandscape";

export function LinkedInspection({
  result,
}: {
  result: InspectionResponse | null;
}) {
  const [selectedPatch, setSelectedPatch] =
    useState<Patch | null>(null);

  return (
    <>
      <InspectionViewer
        result={result}
        selectedPatch={selectedPatch}
        onSelectPatch={setSelectedPatch}
      />

      {result && (
        <SpatialLandscape
          response={result}
          selectedPatch={selectedPatch}
          onSelectPatch={setSelectedPatch}
        />
      )}
    </>
  );
}
