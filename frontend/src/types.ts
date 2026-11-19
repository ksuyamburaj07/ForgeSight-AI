export type Category = "pcb1" | "pcb2" | "pcb3" | "pcb4";
export type Geometry = [number, number, number, number, number, number];
export interface Health {
  status: string;
  runtime_device: string;
  loading_policy: string;
  scientific_status: string;
}
export interface Capabilities {
  categories: Category[];
  accepted_media_types: string[];
  max_upload_bytes: number;
  max_image_pixels: number;
  max_image_edge: number;
  image_resolution: number;
  patch_grid: [number, number];
  automatic_manufacturing_decision: boolean;
  result_type: string;
}
export interface InspectionResponse {
  inspection: {
    inference_performed: boolean;
    category: Category;
    score: number;
    threshold: number;
    review_flag: boolean;
    result_type: string;
    valid_patches: number;
    processing_ms: number;
    runtime_device: string;
    input_resolution: number;
    patch_grid: [number, number];
    model_id: string;
    model_weights_sha256: string;
    local_model_snapshot: string;
    frozen_scoring_method: string;
    validation_status: string;
    geometry: Geometry;
    source_sha256: string;
    visualization: {
      display_scale_max: number;
      scaling_method: string;
      interpretation: string;
    };
  };
  visualizations: { original: string; heatmap: string; overlay: string };
  patch_evidence: {
    grid: [number, number];
    distances: number[][];
    valid_mask: boolean[][];
    units: string;
  };
}
