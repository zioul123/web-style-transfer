import { test } from "@playwright/test";
import { gotoStableApp } from "./helpers/appPage";
import {
  firstPoolFixtureMissingMessage,
  type VggFirstPoolArtifactsResult,
  type VggFirstPoolCaseFixture,
} from "./helpers/fixtures";
import { expectTensorCloseTo } from "./helpers/tensorAssertions";
import {
  createStyleTransferWorkerClient,
  expectWebGpuInitOk,
  expectWorkerTensorVectorResponse,
} from "./helpers/workerClient";

type VggFirstPoolParityCaseFixture = VggFirstPoolCaseFixture & {
  normalizedValues: number[];
  expectedValues: number[];
};

const isVggFirstPoolParityCaseFixture = (
  caseData: VggFirstPoolCaseFixture,
): caseData is VggFirstPoolParityCaseFixture =>
  Array.isArray(caseData.normalizedValues) &&
  Array.isArray(caseData.expectedValues);

test("vgg19 truncated forward pass matches pytorch through first pool", async ({
  page,
}) => {
  test.setTimeout(120000);
  await gotoStableApp(page);
  const artifact: VggFirstPoolArtifactsResult = await page.evaluate(
    async () => {
      const { loadVggFirstPoolArtifacts } =
        await import("/tests/helpers/fixtures.ts");
      return loadVggFirstPoolArtifacts();
    },
  );

  test.skip(!artifact.ok, firstPoolFixtureMissingMessage);
  if (!artifact.ok) return;

  const { weights, caseData } = artifact;
  if (!isVggFirstPoolParityCaseFixture(caseData)) {
    throw new Error(
      "First-pool parity case fixture is missing expected values.",
    );
  }
  const workerClient = await createStyleTransferWorkerClient(page);

  try {
    const init = await workerClient.initWebGpu();
    const norm = await workerClient.ask({
      type: "tensor-op",
      id: "phase2-vgg-norm",
      op: "normalize-forward",
      input: { shape: caseData.inputShape, values: caseData.inputValues },
      mean: caseData.mean,
      std: caseData.std,
    });
    const conv1 = await workerClient.ask({
      type: "tensor-op",
      id: "phase2-vgg-conv1",
      op: "conv2d-forward",
      input: { shape: caseData.inputShape, values: caseData.normalizedValues },
      weight: {
        shape: weights.conv1WeightShape,
        values: weights.conv1WeightValues,
      },
      bias: weights.conv1BiasValues,
    });
    const conv1Values = expectWorkerTensorVectorResponse(conv1).values;
    const relu1 = await workerClient.ask({
      type: "tensor-op",
      id: "phase2-vgg-relu1",
      op: "relu-forward",
      input: { shape: [1, 64, 16, 16], values: conv1Values },
    });
    const relu1Values = expectWorkerTensorVectorResponse(relu1).values;
    const conv2 = await workerClient.ask({
      type: "tensor-op",
      id: "phase2-vgg-conv2",
      op: "conv2d-forward",
      input: { shape: [1, 64, 16, 16], values: relu1Values },
      weight: {
        shape: weights.conv2WeightShape,
        values: weights.conv2WeightValues,
      },
      bias: weights.conv2BiasValues,
    });
    const conv2Values = expectWorkerTensorVectorResponse(conv2).values;
    const relu2 = await workerClient.ask({
      type: "tensor-op",
      id: "phase2-vgg-relu2",
      op: "relu-forward",
      input: { shape: [1, 64, 16, 16], values: conv2Values },
    });
    const relu2Values = expectWorkerTensorVectorResponse(relu2).values;
    const pool1 = await workerClient.ask({
      type: "tensor-op",
      id: "phase2-vgg-pool1",
      op: "maxpool2d-forward",
      input: { shape: [1, 64, 16, 16], values: relu2Values },
    });
    const pool1Values = expectWorkerTensorVectorResponse(pool1).values;
    const conv3 = await workerClient.ask({
      type: "tensor-op",
      id: "phase2-vgg-conv3",
      op: "conv2d-forward",
      input: { shape: [1, 64, 8, 8], values: pool1Values },
      weight: {
        shape: weights.conv3WeightShape,
        values: weights.conv3WeightValues,
      },
      bias: weights.conv3BiasValues,
    });
    const conv3Values = expectWorkerTensorVectorResponse(conv3).values;
    const relu3 = await workerClient.ask({
      type: "tensor-op",
      id: "phase2-vgg-relu3",
      op: "relu-forward",
      input: { shape: [1, 128, 8, 8], values: conv3Values },
    });

    expectWebGpuInitOk(init);
    expectWorkerTensorVectorResponse(norm);
    expectTensorCloseTo(
      expectWorkerTensorVectorResponse(relu3).values,
      caseData.expectedValues,
      4,
    );
  } finally {
    await workerClient.dispose();
  }
});
