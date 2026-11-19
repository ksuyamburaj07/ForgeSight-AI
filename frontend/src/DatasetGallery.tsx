import { useEffect, useState } from "react";

import type { PCB } from "./datasetRegistry";
import {
  loadDatasetGallery,
} from "./datasetGallery";
import type {
  GalleryManifest,
} from "./datasetGallery";

import "./DatasetGallery.css";

export function DatasetGallery({
  pcb,
  selectedIndex,
  onSelectIndex,
}: {
  pcb: PCB;
  selectedIndex: number;
  onSelectIndex: (index: number) => void;
}) {
  const [gallery, setGallery] =
    useState<GalleryManifest | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);
  const [retry, setRetry] = useState(0);

  const [imageError, setImageError] = useState(false);

  useEffect(() => {
    const controller = new AbortController();

    setLoading(true);
    setError("");

    loadDatasetGallery(controller.signal)
      .then((data) => {
        if (!controller.signal.aborted) setGallery(data);
      })
      .catch((cause: unknown) => {
        if (!controller.signal.aborted) {
          setError(
            cause instanceof Error
              ? cause.message
              : "Dataset photographs are unavailable.",
          );
        }
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false);
      });

    return () => controller.abort();
  }, [retry]);

  useEffect(() => {
    setImageError(false);
  }, [pcb, selectedIndex]);

  if (loading) {
    return (
      <section className="dg-gallery" role="status">
        Loading the audited PCB image gallery…
      </section>
    );
  }

  if (!gallery || error) {
    return (
      <section className="dg-gallery">
        <p role="alert">
          {error || "The image gallery is unavailable."}
        </p>
        <button
          type="button"
          onClick={() => setRetry((n) => n + 1)}
        >
          Retry gallery
        </button>
      </section>
    );
  }

  const images = gallery.items[pcb];
  const selected = images[selectedIndex] ?? images[0];
  const filename = selected.source_path.split("/").at(-1);

  function selectImage(index: number) {
    onSelectIndex(index);
    setImageError(false);
  }

  return (
    <section
      className="dg-gallery"
      aria-label={`${pcb.toUpperCase()} source image gallery`}
    >
      <header className="dg-heading">
        <div>
          <span className="dg-kicker">
            Source imagery / Normal training partition
          </span>
          <h3>Meet the material.</h3>
          <p>
            Four original normal-training examples from the
            selected VisA category. Explore the actual visual
            material behind the research.
          </p>
        </div>
        <span className="dg-count">
          {pcb.toUpperCase()} / 04 examples
        </span>
      </header>

      <div className="dg-layout">
        <figure className="dg-feature">
          <div className="dg-image-stage">
            {imageError ? (
              <div role="alert" className="dg-image-error">
                Image unavailable.
                The rest of the research record remains accessible.
              </div>
            ) : (
              <img
                key={selected.id}
                src={selected.src}
                alt={
                  `${pcb.toUpperCase()} normal-training PCB ` +
                  `photograph, example ${selectedIndex + 1}`
                }
                decoding="async"
                onError={() => setImageError(true)}
              />
            )}
            <span className="dg-image-index">
              {String(selectedIndex + 1).padStart(2, "0")}
              <span> / 04</span>
            </span>
          </div>

          <figcaption className="dg-caption">
            <div>
              <span>VisA / {pcb.toUpperCase()}</span>
              <strong>{filename}</strong>
            </div>
            <div>
              <span>Original resolution</span>
              <strong>
                {selected.original_width} ×
                {" "}{selected.original_height}
              </strong>
            </div>
            <div>
              <span>Source partition</span>
              <strong>Normal training</strong>
            </div>
          </figcaption>
        </figure>

        <div className="dg-thumbnails">
          <div className="dg-thumbnails-heading">
            <strong>Explore the collection</strong>
            <span>SELECT AN IMAGE</span>
          </div>

          <div className="dg-thumbnails-grid">
            {images.map((image, index) => (
              <button
                key={image.id}
                type="button"
                aria-pressed={selectedIndex === index}
                onClick={() => selectImage(index)}
                className={
                  "dg-thumb " +
                  (selectedIndex === index
                    ? "is-selected"
                    : "")
                }
              >
                <img
                  src={image.src}
                  alt={
                    `${pcb.toUpperCase()} example ${index + 1}`
                  }
                  loading="lazy"
                  decoding="async"
                />
                <span>
                  {String(index + 1).padStart(2, "0")}
                </span>
              </button>
            ))}
          </div>

          <div className="dg-context">
            <span className="dg-kicker">Research context</span>
            <strong>These are normal-labelled examples.</strong>
            <p>
              They help illustrate the category's source appearance.
              Their inclusion does not prove frozen reference-bank
              membership, defect detection or model accuracy.
            </p>
          </div>
        </div>
      </div>

      <footer className="dg-attribution">
        <p>
          Photographs: Amazon Science VisA.
          Resized, recompressed derivatives with original
          aspect ratios retained.
        </p>
        <div>
          <a
            href={gallery.official_source}
            target="_blank"
            rel="noopener noreferrer"
          >
            Original dataset
          </a>
          <a
            href={gallery.license_url}
            target="_blank"
            rel="noopener noreferrer"
          >
            {gallery.license} license
          </a>
          <a
            href="/dataset/gallery/ATTRIBUTION.md"
            target="_blank"
            rel="noopener noreferrer"
          >
            Full attribution
          </a>
        </div>
      </footer>
    </section>
  );
}
