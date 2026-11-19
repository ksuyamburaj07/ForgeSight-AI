import type { PCB } from "./datasetRegistry";

export interface GalleryImage {
  id: string;
  src: string;
  source_path: string;
  source_sha256: string;
  derived_sha256: string;
  original_width: number;
  original_height: number;
  preview_width: number;
  preview_height: number;
}

export interface GalleryManifest {
  dataset: string;
  license: string;
  license_url: string;
  official_source: string;
  transformation: string;
  items: Record<PCB, GalleryImage[]>;
}

const CATEGORIES: PCB[] =
  ["pcb1", "pcb2", "pcb3", "pcb4"];

const AUDIT_SHA =
  "8835e63225bd75794c99fc1684563e2a8820054542ebc9e96219f0725d6f83eb";

const DIMENSIONS: Record<PCB, [number, number]> = {
  pcb1: [1404, 1070],
  pcb2: [1404, 1070],
  pcb3: [1562, 960],
  pcb4: [1358, 1104],
};

function object(value: unknown): Record<string, unknown> {
  if (
    value === null ||
    typeof value !== "object" ||
    Array.isArray(value)
  ) {
    throw new Error("Invalid gallery object");
  }
  return value as Record<string, unknown>;
}

function validHash(value: unknown): value is string {
  return (
    typeof value === "string" &&
    /^[0-9a-f]{64}$/.test(value)
  );
}

function positiveInteger(value: unknown): value is number {
  return (
    typeof value === "number" &&
    Number.isSafeInteger(value) &&
    value > 0
  );
}

export function parseDatasetGallery(input: unknown):
  GalleryManifest {
  const raw = object(input);
  const items = object(raw.items);

  if (
    raw.schema_version !== 1 ||
    raw.dataset !== "Amazon Science VisA" ||
    raw.source_partition !== "normal_training" ||
    raw.inventory_audit_sha256 !== AUDIT_SHA ||
    raw.gallery_reference_bank_membership_verified !== false ||
    raw.license !== "CC BY 4.0" ||
    raw.license_url !==
      "https://creativecommons.org/licenses/by/4.0/" ||
    raw.official_source !==
      "https://github.com/amazon-science/spot-diff" ||
    typeof raw.transformation !== "string" ||
    !raw.transformation.trim() ||
    Object.keys(items).length !== 4 ||
    CATEGORIES.some((pcb) => !(pcb in items))
  ) {
    throw new Error("Gallery source contract mismatch");
  }

  const parsed = {} as Record<PCB, GalleryImage[]>;
  const usedSources = new Set<string>();

  for (const pcb of CATEGORIES) {
    const entries = items[pcb];

    if (!Array.isArray(entries) || entries.length !== 4) {
      throw new Error(`Invalid gallery count: ${pcb}`);
    }

    parsed[pcb] = entries.map((entry, index) => {
      const image = object(entry);
      const id = `${pcb}_${String(index + 1).padStart(2, "0")}`;
      const src = `/dataset/gallery/${id}.jpg`;
      const [width, height] = DIMENSIONS[pcb];

      if (
        image.id !== id ||
        image.src !== src ||
        typeof image.source_path !== "string" ||
        !new RegExp(
          `^${pcb}/Data/Images/Normal/[0-9]+\\.JPG$`,
        ).test(image.source_path) ||
        !validHash(image.source_sha256) ||
        !validHash(image.derived_sha256) ||
        image.original_width !== width ||
        image.original_height !== height ||
        !positiveInteger(image.preview_width) ||
        !positiveInteger(image.preview_height) ||
        image.preview_width > 1000 ||
        image.preview_height > 1000
      ) {
        throw new Error(`Invalid gallery image: ${id}`);
      }

      const originalRatio = width / height;
      const previewRatio =
        image.preview_width / image.preview_height;

      if (Math.abs(originalRatio - previewRatio) > 0.005) {
        throw new Error(`Gallery aspect ratio mismatch: ${id}`);
      }

      if (usedSources.has(image.source_path)) {
        throw new Error(`Repeated gallery source: ${id}`);
      }

      usedSources.add(image.source_path);

      return image as unknown as GalleryImage;
    });
  }

  return {
    dataset: "Amazon Science VisA",
    license: "CC BY 4.0",
    official_source:
      "https://github.com/amazon-science/spot-diff",
    license_url:
      "https://creativecommons.org/licenses/by/4.0/",
    transformation: raw.transformation as string,
    items: parsed,
  };
}

export async function loadDatasetGallery(
  signal: AbortSignal,
): Promise<GalleryManifest> {
  const response = await fetch("/dataset/gallery.json", {
    signal,
    cache: "no-store",
  });

  if (!response.ok) {
    throw new Error(
      `PCB image gallery unavailable (HTTP ${response.status})`,
    );
  }

  return parseDatasetGallery(await response.json());
}
