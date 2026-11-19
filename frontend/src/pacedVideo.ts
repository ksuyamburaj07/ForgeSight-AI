/**
 * Serial frame-sampling scheduler.
 *
 * Waits for each sample Promise to settle, then starts
 * another cooldown. No concurrent samples or frame queue.
 */

export type HaltReason = "unavailable" | "frame-failed";

export interface SamplingClock {
  schedule(callback: () => void, delayMs: number): unknown;
  clear(handle: unknown): void;
}

export interface PacedSampler {
  start(): void;
  stop(): void;
  isRunning(): boolean;
  isBusy(): boolean;
}

export interface SamplingOptions {
  cooldownMs: number;
  canSample(): boolean;
  sample(): Promise<boolean>;
  onHalt?(reason: HaltReason): void;
  onError?(cause: unknown): void;
}

const realClock: SamplingClock = {
  schedule: (callback, delayMs) =>
    setTimeout(callback, delayMs),
  clear: (handle) =>
    clearTimeout(handle as ReturnType<typeof setTimeout>),
};

export function createPacedSampler(
  options: SamplingOptions,
  clock: SamplingClock = realClock,
): PacedSampler {
  if (
    !Number.isSafeInteger(options.cooldownMs) ||
    options.cooldownMs < 5_000 ||
    options.cooldownMs > 120_000
  ) {
    throw new Error(
      "Video cooldown must be between 5 and 120 seconds.",
    );
  }

  let running = false;
  let inFlight = false;
  let pending: unknown = null;

  function stop() {
    running = false;

    if (pending !== null) {
      clock.clear(pending);
      pending = null;
    }
  }

  function next() {
    if (!running || inFlight || pending !== null) return;

    pending = clock.schedule(() => {
      pending = null;

      if (!running) return;

      if (!options.canSample()) {
        stop();
        options.onHalt?.("unavailable");
        return;
      }

      inFlight = true;

      void (async () => {
        try {
          const success = await options.sample();

          if (!success && running) {
            stop();
            options.onHalt?.("frame-failed");
          }
        } catch (cause) {
          if (running) {
            stop();
            options.onHalt?.("frame-failed");
          }
          options.onError?.(cause);
        } finally {
          inFlight = false;
          next();
        }
      })();
    }, options.cooldownMs);
  }

  return {
    start() {
      if (running) return;
      running = true;
      next();
    },
    stop,
    isRunning: () => running,
    isBusy: () => inFlight,
  };
}
