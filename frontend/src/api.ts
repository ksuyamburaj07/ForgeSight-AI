import { hasEligiblePatchCenters } from "./geometry.ts";
import type { Capabilities, Health, InspectionResponse } from "./types";

async function request(path: string, init: RequestInit = {}): Promise<unknown> {
  let response: Response;
  try {
    response = await fetch(`/api/v1/${path}`, init);
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError")
      throw error;
    throw new Error(
      "Cannot reach the local inspection service. Start FastAPI on 127.0.0.1:8000, then reconnect.",
    );
  }
  let payload: unknown;
  try {
    payload = await response.json();
  } catch {
    throw new Error(
      `Service returned an unreadable response (HTTP ${response.status}). Check the local API.`,
    );
  }
  if (!response.ok) {
    const detail = (payload as { detail?: unknown })?.detail;
    const message =
      typeof detail === "string"
        ? detail
        : `Request failed (HTTP ${response.status}).`;
    throw new Error(
      response.status === 429
        ? "The CPU service is busy. Wait for the current inspection to finish, then retry."
        : message,
    );
  }
  return payload;
}

const CATEGORIES = ["pcb1", "pcb2", "pcb3", "pcb4"];
const MEDIA_TYPES = ["image/jpeg", "image/png", "image/webp"];
const CONTRACT_ERROR =
  "The service contract does not match this research studio. Check the API version.";
const EVIDENCE_ERROR =
  "The service returned incomplete or incompatible evidence. No result has been displayed.";

function record(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
function finite(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}
function positiveInteger(value: unknown): value is number {
  return typeof value === "number" && Number.isSafeInteger(value) && value > 0;
}
function text(value: unknown): value is string {
  return typeof value === "string" && value.trim().length > 0;
}
function sha256(value: unknown): value is string {
  return typeof value === "string" && /^[a-f0-9]{64}$/.test(value);
}
function exactStrings(value: unknown, expected: string[]): boolean {
  return (
    Array.isArray(value) &&
    value.length === expected.length &&
    new Set(value).size === expected.length &&
    value.every((item) => typeof item === "string" && expected.includes(item))
  );
}
function grid(value: unknown): boolean {
  return (
    Array.isArray(value) &&
    value.length === 2 &&
    value[0] === 48 &&
    value[1] === 48
  );
}
function matrix<T>(
  value: unknown,
  cell: (item: unknown) => item is T,
): value is T[][] {
  return (
    Array.isArray(value) &&
    value.length === 48 &&
    value.every(
      (row) => Array.isArray(row) && row.length === 48 && row.every(cell),
    )
  );
}
function distance(value: unknown): value is number {
  return finite(value) && value >= 0 && value <= 2;
}
function boolean(value: unknown): value is boolean {
  return typeof value === "boolean";
}
function geometry(value: unknown): boolean {
  if (
    !Array.isArray(value) ||
    value.length !== 6 ||
    !value.every(Number.isSafeInteger)
  )
    return false;
  const [left, top, width, height, sourceWidth, sourceHeight] =
    value as number[];
  return (
    left >= 0 &&
    top >= 0 &&
    width > 0 &&
    height > 0 &&
    left + width <= 672 &&
    top + height <= 672 &&
    Math.max(width, height) === 672 &&
    left === Math.floor((672 - width) / 2) &&
    top === Math.floor((672 - height) / 2) &&
    sourceWidth > 0 &&
    sourceHeight > 0 &&
    sourceWidth <= 4096 &&
    sourceHeight <= 4096 &&
    sourceWidth * sourceHeight <= 8_000_000
  );
}
function jpegPreview(value: unknown): boolean {
  if (typeof value !== "string" || !value.startsWith("data:image/jpeg;base64,"))
    return false;
  const encoded = value.slice("data:image/jpeg;base64,".length);
  return (
    encoded.length > 0 &&
    encoded.length % 4 === 0 &&
    /^[A-Za-z0-9+/]+={0,2}$/.test(encoded)
  );
}

export async function connect(
  signal: AbortSignal,
): Promise<{ health: Health; capabilities: Capabilities }> {
  const [health, capabilities] = await Promise.all([
    request("health", { signal }),
    request("capabilities", { signal }),
  ]);
  if (
    !record(health) ||
    !record(capabilities) ||
    health.status !== "ok" ||
    health.runtime_device !== "cpu" ||
    health.loading_policy !== "on_first_valid_inspection" ||
    health.scientific_status !== "cross_platform_score_gate_open" ||
    !exactStrings(capabilities.categories, CATEGORIES) ||
    !exactStrings(capabilities.accepted_media_types, MEDIA_TYPES) ||
    capabilities.automatic_manufacturing_decision !== false ||
    capabilities.result_type !== "research_anomaly_evidence" ||
    !positiveInteger(capabilities.max_upload_bytes) ||
    !positiveInteger(capabilities.max_image_pixels) ||
    !positiveInteger(capabilities.max_image_edge) ||
    capabilities.image_resolution !== 672 ||
    !grid(capabilities.patch_grid)
  ) {
    throw new Error(CONTRACT_ERROR);
  }
  return {
    health: health as unknown as Health,
    capabilities: capabilities as unknown as Capabilities,
  };
}

export async function inspect(
  file: File,
  category: string,
): Promise<InspectionResponse> {
  if (!CATEGORIES.includes(category))
    throw new Error("Choose PCB1, PCB2, PCB3, or PCB4 before inspection.");
  const body = new FormData();
  body.set("category", category);
  body.set("file", file);
  const data = await request("inspect", { method: "POST", body });
  if (
    !record(data) ||
    !record(data.inspection) ||
    !record(data.patch_evidence) ||
    !record(data.visualizations)
  ) {
    throw new Error(EVIDENCE_ERROR);
  }
  const r = data.inspection,
    p = data.patch_evidence,
    previews = data.visualizations;
  if (
    r.inference_performed !== true ||
    !distance(r.score) ||
    !distance(r.threshold) ||
    r.category !== category ||
    typeof r.review_flag !== "boolean" ||
    r.result_type !== "research_anomaly_evidence" ||
    !geometry(r.geometry) ||
    !positiveInteger(r.valid_patches) ||
    r.valid_patches > 2304 ||
    !finite(r.processing_ms) ||
    r.processing_ms < 0 ||
    r.runtime_device !== "cpu" ||
    r.input_resolution !== 672 ||
    !grid(r.patch_grid) ||
    r.model_id !== "facebook/dinov2-small" ||
    !sha256(r.model_weights_sha256) ||
    !sha256(r.source_sha256) ||
    !text(r.local_model_snapshot) ||
    r.frozen_scoring_method !==
      "mean_top_1_percent_of_valid_cosine_nearest_neighbor_patch_distances" ||
    r.validation_status !== "cpu_cross_platform_score_gate_open" ||
    !record(r.visualization) ||
    !finite(r.visualization.display_scale_max) ||
    r.visualization.display_scale_max <= 0 ||
    !text(r.visualization.scaling_method) ||
    !text(r.visualization.interpretation) ||
    !grid(p.grid) ||
    !matrix(p.distances, distance) ||
    !matrix(p.valid_mask, boolean) ||
    p.units !== "cosine_nearest_neighbor_distance" ||
    !["original", "heatmap", "overlay"].every((key) =>
      jpegPreview(previews[key]),
    )
  ) {
    throw new Error(EVIDENCE_ERROR);
  }
  if (
    p.valid_mask.reduce(
      (count, row) => count + row.filter(Boolean).length,
      0,
    ) !== r.valid_patches
  ) {
    throw new Error(EVIDENCE_ERROR);
  }
  // These are transport/shape checks. Source bytes are independently SHA-verified by App.
  return data as unknown as InspectionResponse;
}

export async function fingerprint(file: File): Promise<string> {
  const bytes = await file.arrayBuffer();
  const digest = await crypto.subtle.digest("SHA-256", bytes);
  return Array.from(new Uint8Array(digest), (n) =>
    n.toString(16).padStart(2, "0"),
  ).join("");
}

export async function validateFile(
  file: File,
  caps: Capabilities,
): Promise<void> {
  if (!caps.accepted_media_types.includes(file.type))
    throw new Error(
      "Choose a PNG, JPEG, or WebP image. Animated images are not supported.",
    );
  if (!file.size || file.size > caps.max_upload_bytes)
    throw new Error(
      `Choose an image under ${(caps.max_upload_bytes / 1048576).toFixed(0)} MiB.`,
    );
  const url = URL.createObjectURL(file);
  try {
    const image = new Image();
    image.src = url;
    await image.decode();
    const w = image.naturalWidth,
      h = image.naturalHeight;
    if (
      w > caps.max_image_edge ||
      h > caps.max_image_edge ||
      w * h > caps.max_image_pixels
    ) {
      throw new Error(
        `Image exceeds the service limits: ${caps.max_image_edge}px per edge and ${(caps.max_image_pixels / 1000000).toFixed(0)} megapixels.`,
      );
    }
    if (!hasEligiblePatchCenters(w, h)) {
      throw new Error(
        "This aspect ratio is too thin to contain eligible patch centres in the frozen preprocessing. Choose a less extreme image.",
      );
    }
  } catch (error) {
    if (error instanceof Error && error.name !== "EncodingError") throw error;
    throw new Error(
      "This image cannot be decoded. Choose a valid PNG, JPEG, or WebP.",
    );
  } finally {
    URL.revokeObjectURL(url);
  }
}

export async function syntheticInput(): Promise<File> {
  const canvas = document.createElement("canvas");
  canvas.width = canvas.height = 224;
  const ctx = canvas.getContext("2d")!;
  ctx.fillStyle = "#b8beb4";
  ctx.fillRect(0, 0, 224, 224);
  ctx.fillStyle = "#3c6952";
  ctx.fillRect(32, 28, 160, 168);
  ctx.strokeStyle = "#182f26";
  ctx.lineWidth = 3;
  ctx.strokeRect(32, 28, 160, 168);
  ctx.strokeStyle = "#b6ad64";
  for (const x of [52, 78, 104, 130, 156]) {
    ctx.beginPath();
    ctx.moveTo(x, 42);
    ctx.lineTo(x, 180);
    ctx.stroke();
  }
  ctx.fillStyle = "#2b3130";
  ctx.fillRect(84, 85, 59, 55);
  const blob = await new Promise<Blob>((resolve, reject) =>
    canvas.toBlob(
      (b) =>
        b
          ? resolve(b)
          : reject(new Error("Synthetic input could not be generated.")),
      "image/png",
    ),
  );
  return new File([blob], "synthetic-integration-input.png", {
    type: "image/png",
  });
}
