import { useEffect, useRef, useState } from "react";

import {
  cameraErrorMessage,
  captureCameraFrame,
} from "./cameraFrame";

import type { CameraFacing } from "./cameraFrame";
import { createPacedSampler } from "./pacedVideo";
import type { PacedSampler } from "./pacedVideo";

import "./CameraStation.css";

type Mode = "photo" | "video";
type Phase = "idle" | "requesting" | "live" | "review";

interface Props {
  available: boolean;
  busy: boolean;
  onSelectSnapshot: (frame: File) => void;
  onAnalyzeFrame: (frame: File) => Promise<boolean>;
}

export function CameraStation({
  available,
  busy,
  onSelectSnapshot,
  onAnalyzeFrame,
}: Props) {
  const [expanded, setExpanded] = useState(false);
  const [mode, setMode] = useState<Mode>("photo");
  const [phase, setPhase] = useState<Phase>("idle");
  const [facing, setFacing] =
    useState<CameraFacing>("environment");
  const [error, setError] = useState("");
  const [captured, setCaptured] = useState<File | null>(null);
  const [preview, setPreview] = useState("");
  const [sampling, setSampling] = useState(false);
  const [lastSample, setLastSample] = useState("");
  const [autoEnabled, setAutoEnabled] = useState(false);
  const [cooldownMs, setCooldownMs] = useState(15000);
  const [completedFrames, setCompletedFrames] = useState(0);
  const [lastLatencyMs, setLastLatencyMs] =
    useState<number | null>(null);
  const [lastFrameTime, setLastFrameTime] = useState("");

  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const previewRef = useRef("");
  const generation = useRef(0);
  const sampleInFlight = useRef(false);
  const samplerRef = useRef<PacedSampler | null>(null);
  const sampleFnRef =
    useRef<() => Promise<boolean>>(async () => false);
  const canAutoRef = useRef(false);

  canAutoRef.current =
    mode === "video" &&
    phase === "live" &&
    available &&
    !busy &&
    !sampling;

  function releaseVideo() {
    const stream = streamRef.current;
    streamRef.current = null;

    if (stream) {
      for (const track of stream.getTracks()) {
        track.stop();
      }
    }

    if (videoRef.current) {
      videoRef.current.pause();
      videoRef.current.srcObject = null;
    }
  }

  function clearReview() {
    if (previewRef.current) {
      URL.revokeObjectURL(previewRef.current);
      previewRef.current = "";
    }
    setPreview("");
    setCaptured(null);
  }

  function stopAutoSampling() {
    samplerRef.current?.stop();
    samplerRef.current = null;
    setAutoEnabled(false);
  }

  function stopCamera() {
    stopAutoSampling();
    ++generation.current;
    releaseVideo();
    setPhase("idle");
  }

  function closeWorkspace() {
    stopCamera();
    clearReview();
    setExpanded(false);
    setError("");
    setLastSample("");
  }

  useEffect(() => {
    const onVisibilityChange = () => {
      if (document.hidden) {
        stopCamera();
      }
    };

    document.addEventListener(
      "visibilitychange",
      onVisibilityChange,
    );

    return () => {
      document.removeEventListener(
        "visibilitychange",
        onVisibilityChange,
      );
      samplerRef.current?.stop();
      samplerRef.current = null;
      ++generation.current;
      releaseVideo();

      if (previewRef.current) {
        URL.revokeObjectURL(previewRef.current);
        previewRef.current = "";
      }
    };
  }, []);

  useEffect(() => {
    if (!available && phase !== "idle") {
      stopCamera();
    }
  }, [available, phase]);

  async function openCamera() {
    if (!available || busy) return;

    setError("");
    setLastSample("");
    setCompletedFrames(0);
    setLastLatencyMs(null);
    setLastFrameTime("");
    clearReview();

    if (
      !window.isSecureContext ||
      !navigator.mediaDevices?.getUserMedia
    ) {
      setError(
        "Camera access requires HTTPS or localhost and a compatible browser. Image upload remains available.",
      );
      return;
    }

    const request = ++generation.current;
    setPhase("requesting");

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: {
          facingMode: { ideal: facing },
          width: { ideal: 1280 },
          height: { ideal: 720 },
        },
      });

      // Permission might have resolved after the user closed
      // the camera. Never retain such a stale stream.
      if (request !== generation.current) {
        stream.getTracks().forEach((track) => track.stop());
        return;
      }

      const video = videoRef.current;

      if (!video) {
        stream.getTracks().forEach((track) => track.stop());
        throw new Error("Camera preview is unavailable.");
      }

      streamRef.current = stream;
      video.srcObject = stream;

      const track = stream.getVideoTracks()[0];

      track?.addEventListener("ended", () => {
        if (request !== generation.current) return;
        stopCamera();
        setError(
          "The camera stream ended. Reconnect the camera to continue.",
        );
      }, { once: true });

      await video.play();

      if (request !== generation.current) {
        if (streamRef.current === stream) {
          releaseVideo();
        } else {
          stream.getTracks().forEach((track) => track.stop());
        }
        return;
      }

      setPhase("live");
    } catch (cause) {
      if (request !== generation.current) return;

      releaseVideo();
      setPhase("idle");
      setError(cameraErrorMessage(cause));
    }
  }

  function changeMode(next: Mode) {
    if (busy || sampling || mode === next) return;

    stopCamera();
    clearReview();
    setError("");
    setLastSample("");
    setMode(next);
  }

  async function capturePhoto() {
    if (
      phase !== "live" ||
      busy ||
      sampling ||
      !videoRef.current
    ) return;

    setSampling(true);
    setError("");

    const captureGeneration = generation.current;

    try {
      const frame = await captureCameraFrame(videoRef.current);

      if (captureGeneration !== generation.current) return;

      // A still capture should release the camera immediately.
      stopCamera();
      clearReview();

      const url = URL.createObjectURL(frame);
      previewRef.current = url;

      setCaptured(frame);
      setPreview(url);
      setPhase("review");
    } catch (cause) {
      setError(cameraErrorMessage(cause));
    } finally {
      setSampling(false);
    }
  }

  async function analyzeCurrentFrame(): Promise<boolean> {
    if (
      mode !== "video" ||
      phase !== "live" ||
      !available ||
      busy ||
      sampleInFlight.current ||
      !videoRef.current
    ) return false;

    sampleInFlight.current = true;
    setSampling(true);
    setError("");

    const request = generation.current;

    try {
      const frame = await captureCameraFrame(videoRef.current);

      // A stopped or replaced camera must not submit a
      // snapshot that finished encoding late.
      if (request !== generation.current) return false;

      const started = performance.now();
      const success = await onAnalyzeFrame(frame);

      if (request === generation.current) {
        setLastFrameTime(
          new Date(frame.lastModified).toLocaleTimeString()
        );
        setLastLatencyMs(
          Math.round(performance.now() - started)
        );
        if (success) {
          setCompletedFrames((count) => count + 1);
        }
        setLastSample(
          success
            ? "Frame analyzed. The latest result is shown below."
            : "Frame analysis failed. Check the inspection error."
        );
      }

      return success;
    } catch (cause) {
      if (request === generation.current) {
        setError(cameraErrorMessage(cause));
      }
      return false;
    } finally {
      sampleInFlight.current = false;
      setSampling(false);
    }
  }

  // The scheduler always invokes the latest React callback.
  sampleFnRef.current = analyzeCurrentFrame;

  function startAutoSampling() {
    if (
      samplerRef.current ||
      mode !== "video" ||
      phase !== "live" ||
      !available ||
      busy ||
      sampling ||
      document.hidden
    ) return;

    const sampler = createPacedSampler({
      cooldownMs,
      canSample: () =>
        canAutoRef.current && !document.hidden,
      sample: () => sampleFnRef.current(),
      onHalt: (reason) => {
        samplerRef.current = null;
        setAutoEnabled(false);
        setLastSample(
          reason === "frame-failed"
            ? "Paced sampling stopped after an unsuccessful frame."
            : "Paced sampling stopped because the camera or service became unavailable."
        );
      },
      onError: (cause) => {
        setError(cameraErrorMessage(cause));
      },
    });

    samplerRef.current = sampler;
    sampler.start();
    setAutoEnabled(true);
    setLastSample(
      "Paced sampling started. The first frame will be analyzed after the selected cooldown."
    );
  }

  return (
    <section
      className="camera-station"
      aria-label="Camera inspection input"
    >
      <header className="camera-station-header">
        <div>
          <span className="camera-kicker">
            Additional inspection input
          </span>
          <h3>Capture a frame. Examine the evidence.</h3>
          <p>
            Use a camera photograph or inspect a selected frame
            from a live video preview.
          </p>
        </div>
        <button
          type="button"
          className="camera-open-button"
          aria-expanded={expanded}
          onClick={() =>
            expanded
              ? closeWorkspace()
              : setExpanded(true)
          }
        >
          {expanded ? "Close camera" : "Open camera studio"}
        </button>
      </header>

      {expanded && (
        <div className="camera-station-body">
          <div
            className="camera-mode-switch"
            role="group"
            aria-label="Camera inspection mode"
          >
            <button
              type="button"
              aria-pressed={mode === "photo"}
              onClick={() => changeMode("photo")}
              disabled={busy || sampling}
            >
              Photo capture
            </button>
            <button
              type="button"
              aria-pressed={mode === "video"}
              onClick={() => changeMode("video")}
              disabled={busy || sampling}
            >
              Guided video
            </button>
          </div>

          <div className="camera-layout">
            <div className="camera-visual">
              <div className="camera-viewport">
                <video
                  ref={videoRef}
                  autoPlay
                  muted
                  playsInline
                  className={
                    phase === "live"
                      ? "camera-video"
                      : "camera-video camera-hidden"
                  }
                  aria-label="Live camera preview"
                />

                {phase === "review" && preview && (
                  <img
                    src={preview}
                    alt="Captured camera frame awaiting selection"
                    className="camera-review-image"
                  />
                )}

                {(phase === "idle" ||
                  phase === "requesting") && (
                  <div className="camera-placeholder">
                    <span className="camera-reticle">▣</span>
                    <strong>
                      {phase === "requesting"
                        ? "Requesting camera access…"
                        : "Camera ready to connect"}
                    </strong>
                    <small>
                      The preview remains local to your browser
                      until you choose to analyze a frame.
                    </small>
                  </div>
                )}

                {phase === "live" && (
                  <span className="camera-live-indicator">
                    ● LIVE CAMERA
                  </span>
                )}
              </div>

              <div className="camera-guidance">
                <span>
                  {mode === "photo"
                    ? "STILL IMAGE / USER-TRIGGERED"
                    : "VIDEO PREVIEW / FRAME-BY-FRAME"}
                </span>
                <span>
                  {phase === "live"
                    ? "Preview active"
                    : phase === "review"
                      ? "Captured image"
                      : "No camera stream"}
                </span>
              </div>
            </div>

            <div className="camera-controls">
              <span className="camera-kicker">
                Capture controls
              </span>

              <label className="camera-facing">
                Preferred camera
                <select
                  value={facing}
                  disabled={
                    phase !== "idle" || busy || sampling
                  }
                  onChange={(event) =>
                    setFacing(
                      event.target.value as CameraFacing,
                    )
                  }
                >
                  <option value="environment">
                    Rear / external preferred
                  </option>
                  <option value="user">
                    Front camera preferred
                  </option>
                </select>
              </label>

              {phase === "idle" && (
                <button
                  type="button"
                  className="camera-primary"
                  onClick={() => void openCamera()}
                  disabled={!available || busy}
                >
                  Start camera
                </button>
              )}

              {phase === "requesting" && (
                <button
                  type="button"
                  className="camera-secondary"
                  onClick={stopCamera}
                >
                  Cancel camera request
                </button>
              )}

              {phase === "live" && (
                <>
                  {mode === "photo" ? (
                    <button
                      type="button"
                      className="camera-primary"
                      onClick={() => void capturePhoto()}
                      disabled={busy || sampling}
                    >
                      Capture photograph
                    </button>
                  ) : (
                    <button
                      type="button"
                      className="camera-primary"
                      onClick={() =>
                        void analyzeCurrentFrame()
                      }
                      disabled={!available || busy || sampling || autoEnabled}
                    >
                      {busy || sampling
                        ? "Analyzing frame…"
                        : "Analyze current frame"}
                    </button>
                  )}

                  {mode === "video" && (
                    <div className="camera-pacing">
                      <label>
                        Cooldown after each completed analysis
                        <select
                          value={cooldownMs}
                          disabled={autoEnabled || busy || sampling}
                          onChange={(event) =>
                            setCooldownMs(Number(event.target.value))
                          }
                        >
                          <option value={10000}>10 seconds</option>
                          <option value={15000}>15 seconds</option>
                          <option value={30000}>30 seconds</option>
                        </select>
                      </label>

                      {autoEnabled ? (
                        <button
                          type="button"
                          className="camera-pacing-stop"
                          onClick={() => {
                            stopAutoSampling();
                            setLastSample(
                              "Paced sampling stopped. An already-submitted frame may still finish."
                            );
                          }}
                        >
                          Stop paced sampling
                        </button>
                      ) : (
                        <button
                          type="button"
                          className="camera-pacing-start"
                          disabled={!available || busy || sampling}
                          onClick={startAutoSampling}
                        >
                          Start paced sampling
                        </button>
                      )}

                      <small>
                        One request at a time. The cooldown
                        starts after the previous response.
                        No queued video frames.
                      </small>
                    </div>
                  )}

                  <button
                    type="button"
                    className="camera-secondary"
                    onClick={stopCamera}
                  >
                    Stop camera
                  </button>
                </>
              )}

              {phase === "review" && captured && (
                <>
                  <button
                    type="button"
                    className="camera-primary"
                    disabled={!available || busy}
                    onClick={() => {
                      onSelectSnapshot(captured);
                      closeWorkspace();
                    }}
                  >
                    Use captured photograph
                  </button>
                  <button
                    type="button"
                    className="camera-secondary"
                    disabled={busy}
                    onClick={() =>
                      void openCamera()
                    }
                  >
                    Retake photograph
                  </button>
                </>
              )}

              <div
                className="camera-feedback"
                aria-live="polite"
              >
                {error && (
                  <p className="camera-error" role="alert">
                    {error}
                  </p>
                )}
                {lastSample && <p>{lastSample}</p>}
              </div>

              {mode === "video" && (
                <div
                  className="camera-telemetry"
                  aria-label="Video analysis telemetry"
                >
                  <div>
                    <span>Completed frame analyses</span>
                    <strong>{completedFrames}</strong>
                  </div>
                  <div>
                    <span>Last frame captured</span>
                    <strong>{lastFrameTime || "—"}</strong>
                  </div>
                  <div>
                    <span>Analysis round trip</span>
                    <strong>
                      {lastLatencyMs === null
                        ? "—"
                        : `${(lastLatencyMs / 1000).toFixed(2)} s`}
                    </strong>
                  </div>
                </div>
              )}

              <p className="camera-scientific-note">
                Camera imagery differs from the VisA
                benchmark environment. A strong response can
                arise from reflections, displays, background or
                lighting. This is experimental human-supervised
                evidence, not manufacturing defect confirmation.
              </p>
            </div>
          </div>
        </div>
      )}
    </section>
  );
}
