import type { Letterbox, PCB } from "./datasetRegistry";

export type DatasetViewMode = "source" | "overlay" | "mask";

const PCBS: PCB[] = ["pcb1", "pcb2", "pcb3", "pcb4"];

export function galleryPreviewPath(
  pcb: PCB,
  index: number,
): string {
  if (
    !PCBS.includes(pcb) ||
    !Number.isInteger(index) ||
    index < 0 ||
    index > 3
  ) {
    throw new Error("Invalid gallery preview selection");
  }

  return (
    `/dataset/gallery/${pcb}_` +
    `${String(index + 1).padStart(2, "0")}.jpg`
  );
}

export function letterboxImageBounds(
  geometry: Letterbox,
  canvasSize = 480,
) {
  if (
    !Number.isFinite(canvasSize) ||
    canvasSize <= 0 ||
    geometry.input_size !== 672
  ) {
    throw new Error("Invalid illustration geometry");
  }

  const scale = canvasSize / geometry.input_size;

  return {
    x: geometry.left * scale,
    y: geometry.top * scale,
    width: geometry.resized_width * scale,
    height: geometry.resized_height * scale,
  };
}
