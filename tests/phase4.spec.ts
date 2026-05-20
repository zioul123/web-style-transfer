import { expect, test } from "@playwright/test";
import type { WorkerRequest, WorkerResponse } from "../src/types";
import { gotoStableApp } from "./helpers/appPage";

type BackwardFixtureEntry = {
  shape?: [number, number, number, number];
  input?: [number, number, number, number];
  gradOut?: [number, number, number, number];
  inputShape?: [number, number, number, number];
  gradOutShape?: [number, number, number, number];
  weightShape?: [number, number, number, number];
  std?: number[];
  inputValues: number[];
  gradOutValues?: number[];
  targetValues?: number[];
  weightValues?: number[];
  expectedGradIn: number[];
};

type Phase4Fixture = {
  relu: BackwardFixtureEntry;
  pool: BackwardFixtureEntry;
  normalize: BackwardFixtureEntry;
  convBackwardInput: BackwardFixtureEntry;
  gramBackward: BackwardFixtureEntry;
  contentLossBackward: BackwardFixtureEntry;
  styleLossBackward: BackwardFixtureEntry;
};

test("phase 4 backward parity by layer + tiny e2e input gradient", async ({
  page,
}) => {
  test.setTimeout(120000);
  await gotoStableApp(page);

  const result = await page.evaluate(async () => {
    const loadJson = async <T>(url: string): Promise<T> => {
      const res = await fetch(url);
      if (!res.ok) throw new Error(`Failed ${url}`);
      return (await res.json()) as T;
    };
    const fixture = await loadJson<Phase4Fixture>(
      "/phase4-backprop/phase4_backprop_fixture.json",
    );
    const worker = new Worker(
      new URL("/src/styleTransfer.worker.ts", window.location.origin),
      { type: "module" },
    );
    const ask = (payload: WorkerRequest): Promise<WorkerResponse> =>
      new Promise((resolve) => {
        const handler = (event: MessageEvent<WorkerResponse>): void => {
          if (event.data.id === payload.id) {
            worker.removeEventListener("message", handler);
            resolve(event.data);
          }
        };
        worker.addEventListener("message", handler);
        worker.postMessage(payload);
      });
    const init = await ask({ type: "init-webgpu", id: "p4-init" });
    const relu = fixture.relu;
    const reluBackward = await ask({
      type: "tensor-op",
      id: "p4-relu",
      op: "relu-backward",
      input: { shape: relu.input, values: relu.inputValues },
      gradOut: { shape: relu.input, values: relu.gradOutValues },
    });
    const pool = fixture.pool;
    const poolBackward = await ask({
      type: "tensor-op",
      id: "p4-pool",
      op: "maxpool2d-backward",
      input: { shape: pool.input, values: pool.inputValues },
      gradOut: { shape: pool.gradOut, values: pool.gradOutValues },
    });
    const norm = fixture.normalize;
    const normBackward = await ask({
      type: "tensor-op",
      id: "p4-norm",
      op: "normalize-backward",
      input: { shape: norm.shape, values: norm.inputValues },
      gradOut: { shape: norm.shape, values: norm.gradOutValues },
      std: norm.std,
    });
    const conv = fixture.convBackwardInput;
    const convBackward = await ask({
      type: "tensor-op",
      id: "p4-conv",
      op: "conv2d-backward-input",
      inputShape: conv.inputShape,
      gradOut: { shape: conv.gradOutShape, values: conv.gradOutValues },
      weight: { shape: conv.weightShape, values: conv.weightValues },
    });
    const gram = fixture.gramBackward;
    const gramBackward = await ask({
      type: "tensor-op",
      id: "p4-gram",
      op: "gram-backward",
      input: { shape: gram.inputShape, values: gram.inputValues },
      gradOut: { shape: gram.gradOutShape, values: gram.gradOutValues },
    });
    const content = fixture.contentLossBackward;
    const contentBackward = await ask({
      type: "tensor-op",
      id: "p4-content",
      op: "content-loss-backward",
      input: { shape: content.shape, values: content.inputValues },
      target: { shape: content.shape, values: content.targetValues },
    });
    const weightedContentBackward = await ask({
      type: "tensor-op",
      id: "p4-content-weighted",
      op: "content-loss-backward",
      input: { shape: content.shape, values: content.inputValues },
      target: { shape: content.shape, values: content.targetValues },
      contentWeight: 2.5,
    });
    const style = fixture.styleLossBackward;
    const styleBackward = await ask({
      type: "tensor-op",
      id: "p4-style",
      op: "style-loss-backward",
      input: { shape: style.shape, values: style.inputValues },
      target: { shape: style.shape, values: style.targetValues },
    });
    const weightedStyleBackward = await ask({
      type: "tensor-op",
      id: "p4-style-weighted",
      op: "style-loss-backward",
      input: { shape: style.shape, values: style.inputValues },
      target: { shape: style.shape, values: style.targetValues },
      styleWeight: 7.0,
    });

    worker.terminate();
    return {
      init,
      fixture,
      reluBackward,
      poolBackward,
      normBackward,
      convBackward,
      gramBackward,
      contentBackward,
      weightedContentBackward,
      styleBackward,
      weightedStyleBackward,
    };
  });

  expect(result.init.type).toBe("webgpu-init-result");
  const checks: Array<[WorkerResponse, number[]]> = [
    [result.reluBackward, result.fixture.relu.expectedGradIn],
    [result.poolBackward, result.fixture.pool.expectedGradIn],
    [result.normBackward, result.fixture.normalize.expectedGradIn],
    [result.convBackward, result.fixture.convBackwardInput.expectedGradIn],
    [result.gramBackward, result.fixture.gramBackward.expectedGradIn],
    [result.contentBackward, result.fixture.contentLossBackward.expectedGradIn],
    [
      result.weightedContentBackward,
      result.fixture.contentLossBackward.expectedGradIn.map(
        (value) => value * 2.5,
      ),
    ],
    [result.styleBackward, result.fixture.styleLossBackward.expectedGradIn],
    [
      result.weightedStyleBackward,
      result.fixture.styleLossBackward.expectedGradIn.map(
        (value) => value * 7.0,
      ),
    ],
  ];
  for (const [response, expected] of checks) {
    expect(response.type).toBe("tensor-op-result");
    if (
      response.type !== "tensor-op-result" ||
      !response.ok ||
      !("values" in response)
    )
      throw new Error("bad response");
    expected.forEach((value, index) =>
      expect(response.values[index]).toBeCloseTo(value, 4),
    );
  }
});
