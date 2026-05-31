import { test } from "@playwright/test";
import type { WorkerResponse } from "../src/types";
import { gotoStableApp } from "./helpers/appPage";
import { expectTensorCloseTo } from "./helpers/tensorAssertions";
import {
  expectWebGpuInitOk,
  expectWorkerTensorVectorResponse,
} from "./helpers/workerClient";

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

test("backward kernels match layer and tiny e2e input-gradient references", async ({
  page,
}) => {
  test.setTimeout(120000);
  await gotoStableApp(page);

  const result = await page.evaluate(async () => {
    const { fetchJsonOrNull } = await import("/tests/helpers/fixtures.ts");
    const { createBrowserStyleTransferWorkerClient } =
      await import("/tests/helpers/browserWorkerClient.ts");
    const fixture = await fetchJsonOrNull<Phase4Fixture>(
      "/phase4-backprop/phase4_backprop_fixture.json",
    );
    if (fixture === null) {
      return { ok: false as const, reason: "missing-fixture" as const };
    }
    const workerClient = createBrowserStyleTransferWorkerClient();
    try {
      const init = await workerClient.initWebGpu("p4-init");
      const relu = fixture.relu;
      const reluBackward = await workerClient.ask({
        type: "tensor-op",
        id: "p4-relu",
        op: "relu-backward",
        input: { shape: relu.input, values: relu.inputValues },
        gradOut: { shape: relu.input, values: relu.gradOutValues },
      });
      const pool = fixture.pool;
      const poolBackward = await workerClient.ask({
        type: "tensor-op",
        id: "p4-pool",
        op: "maxpool2d-backward",
        input: { shape: pool.input, values: pool.inputValues },
        gradOut: { shape: pool.gradOut, values: pool.gradOutValues },
      });
      const norm = fixture.normalize;
      const normBackward = await workerClient.ask({
        type: "tensor-op",
        id: "p4-norm",
        op: "normalize-backward",
        input: { shape: norm.shape, values: norm.inputValues },
        gradOut: { shape: norm.shape, values: norm.gradOutValues },
        std: norm.std,
      });
      const conv = fixture.convBackwardInput;
      const convBackward = await workerClient.ask({
        type: "tensor-op",
        id: "p4-conv",
        op: "conv2d-backward-input",
        inputShape: conv.inputShape,
        gradOut: { shape: conv.gradOutShape, values: conv.gradOutValues },
        weight: { shape: conv.weightShape, values: conv.weightValues },
      });
      const gram = fixture.gramBackward;
      const gramBackward = await workerClient.ask({
        type: "tensor-op",
        id: "p4-gram",
        op: "gram-backward",
        input: { shape: gram.inputShape, values: gram.inputValues },
        gradOut: { shape: gram.gradOutShape, values: gram.gradOutValues },
      });
      const content = fixture.contentLossBackward;
      const contentBackward = await workerClient.ask({
        type: "tensor-op",
        id: "p4-content",
        op: "content-loss-backward",
        input: { shape: content.shape, values: content.inputValues },
        target: { shape: content.shape, values: content.targetValues },
      });
      const weightedContentBackward = await workerClient.ask({
        type: "tensor-op",
        id: "p4-content-weighted",
        op: "content-loss-backward",
        input: { shape: content.shape, values: content.inputValues },
        target: { shape: content.shape, values: content.targetValues },
        contentWeight: 2.5,
      });
      const style = fixture.styleLossBackward;
      const styleBackward = await workerClient.ask({
        type: "tensor-op",
        id: "p4-style",
        op: "style-loss-backward",
        input: { shape: style.shape, values: style.inputValues },
        target: { shape: style.shape, values: style.targetValues },
      });
      const weightedStyleBackward = await workerClient.ask({
        type: "tensor-op",
        id: "p4-style-weighted",
        op: "style-loss-backward",
        input: { shape: style.shape, values: style.inputValues },
        target: { shape: style.shape, values: style.targetValues },
        styleWeight: 7.0,
      });

      return {
        ok: true as const,
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
    } finally {
      workerClient.dispose();
    }
  });

  test.skip(
    !result.ok && result.reason === "missing-fixture",
    "Missing phase 4 fixture. Run python-reference/export_phase4_backprop_fixtures.py.",
  );
  if (!result.ok) return;

  expectWebGpuInitOk(result.init);
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
    const vectorResponse = expectWorkerTensorVectorResponse(response);
    expectTensorCloseTo(vectorResponse.values, expected, 4);
  }
});
