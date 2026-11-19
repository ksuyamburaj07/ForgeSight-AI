import test from "node:test";
import assert from "node:assert/strict";

import {
  createPacedSampler,
} from "./pacedVideo.ts";

import type {
  SamplingClock,
  HaltReason,
} from "./pacedVideo.ts";

function fakeClock() {
  let nextId = 1;
  const tasks = new Map<number, () => void>();
  const delays: number[] = [];

  const clock: SamplingClock = {
    schedule(callback, delayMs) {
      const id = nextId++;
      delays.push(delayMs);
      tasks.set(id, callback);
      return id;
    },
    clear(handle) {
      tasks.delete(handle as number);
    },
  };

  return {
    clock,
    delays,
    pending: () => tasks.size,
    fire() {
      const entry = tasks.entries().next().value;

      if (!entry) throw new Error("No scheduled callback");

      const [id, callback] = entry;
      tasks.delete(id);
      callback();
    },
  };
}

const settle = () =>
  new Promise<void>((resolve) => setImmediate(resolve));

test("rejects invalid video sampling intervals", () => {
  for (const cooldownMs of [0, 1000, 3000, Infinity, 120001]) {
    assert.throws(
      () => createPacedSampler({
        cooldownMs,
        canSample: () => true,
        sample: async () => true,
      }),
      /cooldown/,
    );
  }
});

test("starts only after the selected cooldown", async () => {
  const clock = fakeClock();
  let samples = 0;

  const sampler = createPacedSampler({
    cooldownMs: 10000,
    canSample: () => true,
    sample: async () => {
      samples++;
      return true;
    },
  }, clock.clock);

  sampler.start();
  assert.equal(samples, 0);
  assert.deepEqual(clock.delays, [10000]);

  clock.fire();
  await settle();

  assert.equal(samples, 1);
  assert.equal(clock.pending(), 1);
  assert.deepEqual(clock.delays, [10000, 10000]);

  sampler.stop();
});

test("never overlaps an unresolved inference", async () => {
  const clock = fakeClock();
  let complete!: (success: boolean) => void;
  let samples = 0;

  const sampler = createPacedSampler({
    cooldownMs: 10000,
    canSample: () => true,
    sample: () => {
      samples++;
      return new Promise<boolean>((resolve) => {
        complete = resolve;
      });
    },
  }, clock.clock);

  sampler.start();
  clock.fire();

  assert.equal(samples, 1);
  assert.equal(sampler.isBusy(), true);
  assert.equal(clock.pending(), 0);

  sampler.start();
  assert.equal(clock.pending(), 0);

  complete(true);
  await settle();

  assert.equal(sampler.isBusy(), false);
  assert.equal(clock.pending(), 1);

  sampler.stop();
});

test("stopping cancels the next planned sample", () => {
  const clock = fakeClock();
  let samples = 0;

  const sampler = createPacedSampler({
    cooldownMs: 10000,
    canSample: () => true,
    sample: async () => {
      samples++;
      return true;
    },
  }, clock.clock);

  sampler.start();
  assert.equal(clock.pending(), 1);

  sampler.stop();
  assert.equal(clock.pending(), 0);
  assert.equal(samples, 0);
  assert.equal(sampler.isRunning(), false);
});

test("stopping during inference prevents another sample", async () => {
  const clock = fakeClock();
  let complete!: (success: boolean) => void;

  const sampler = createPacedSampler({
    cooldownMs: 10000,
    canSample: () => true,
    sample: () =>
      new Promise<boolean>((resolve) => {
        complete = resolve;
      }),
  }, clock.clock);

  sampler.start();
  clock.fire();
  sampler.stop();

  complete(true);
  await settle();

  assert.equal(clock.pending(), 0);
  assert.equal(sampler.isRunning(), false);
});

test("unsuccessful inference halts automatic sampling", async () => {
  const clock = fakeClock();
  const reasons: HaltReason[] = [];

  const sampler = createPacedSampler({
    cooldownMs: 10000,
    canSample: () => true,
    sample: async () => false,
    onHalt: (reason) => reasons.push(reason),
  }, clock.clock);

  sampler.start();
  clock.fire();
  await settle();

  assert.deepEqual(reasons, ["frame-failed"]);
  assert.equal(clock.pending(), 0);
  assert.equal(sampler.isRunning(), false);
});

test("unavailable camera halts before frame submission", () => {
  const clock = fakeClock();
  const reasons: HaltReason[] = [];
  let samples = 0;

  const sampler = createPacedSampler({
    cooldownMs: 10000,
    canSample: () => false,
    sample: async () => {
      samples++;
      return true;
    },
    onHalt: (reason) => reasons.push(reason),
  }, clock.clock);

  sampler.start();
  clock.fire();

  assert.equal(samples, 0);
  assert.deepEqual(reasons, ["unavailable"]);
  assert.equal(clock.pending(), 0);
});
