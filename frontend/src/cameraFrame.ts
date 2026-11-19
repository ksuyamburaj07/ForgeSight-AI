export type CameraFacing = "environment" | "user";

export function validFrameDimensions(
  width: number,
  height: number,
): boolean {
  return (
    Number.isSafeInteger(width) &&
    Number.isSafeInteger(height) &&
    width > 0 &&
    height > 0 &&
    width <= 8192 &&
    height <= 8192
  );
}

export function cameraErrorMessage(cause: unknown): string {
  const name =
    typeof cause === "object" && cause !== null &&
    "name" in cause
      ? String(cause.name)
      : "";

  switch (name) {
    case "NotAllowedError":
    case "PermissionDeniedError":
      return "Camera permission was denied. Check browser site permissions or use image upload.";
    case "NotFoundError":
    case "DevicesNotFoundError":
      return "No camera was found. Connect a camera or use image upload.";
    case "NotReadableError":
    case "TrackStartError":
      return "The camera is unavailable or is being used by another application.";
    case "OverconstrainedError":
      return "The selected camera cannot provide the requested video settings.";
    case "SecurityError":
      return "Camera access is blocked by browser or site security settings.";
    default:
      return cause instanceof Error
        ? cause.message
        : "The camera could not be opened.";
  }
}

export async function captureCameraFrame(
  video: HTMLVideoElement,
): Promise<File> {
  const width = video.videoWidth;
  const height = video.videoHeight;

  if (
    video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA ||
    !validFrameDimensions(width, height)
  ) {
    throw new Error(
      "The camera has not produced a usable frame yet. Try again.",
    );
  }

  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;

  const context = canvas.getContext("2d");

  if (!context) {
    throw new Error("Camera frame encoding is unavailable.");
  }

  // Preserve the physical frame orientation and aspect ratio.
  // No decorative guide lines or mirrored preview are captured.
  context.drawImage(video, 0, 0, width, height);

  const blob = await new Promise<Blob>((resolve, reject) => {
    canvas.toBlob(
      (value) => value
        ? resolve(value)
        : reject(new Error("Unable to encode the camera frame.")),
      "image/jpeg",
      0.92,
    );
  });

  const capturedAt = Date.now();

  return new File(
    [blob],
    `forgesight-camera-${capturedAt}.jpg`,
    {
      type: "image/jpeg",
      lastModified: capturedAt,
    },
  );
}
