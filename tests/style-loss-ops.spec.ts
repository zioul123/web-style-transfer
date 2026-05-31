import { expect, test } from "./helpers/coverage";
import { gotoStableApp } from "./helpers/appPage";
import {
  expectScalarCloseTo,
  expectTensorCloseTo,
} from "./helpers/tensorAssertions";
import {
  createStyleTransferWorkerClient,
  expectWebGpuInitOk,
  expectWorkerTensorScalarResponse,
  expectWorkerTensorVectorResponse,
} from "./helpers/workerClient";

type Phase3Fixture = {
  inputShape: [number, number, number, number];
  inputValues: number[];
  targetShape: [number, number, number, number];
  targetValues: number[];
  expectedGram: number[];
  expectedContentLoss: number;
  expectedStyleLoss: number;
};

const fixture: Phase3Fixture = {
  inputShape: [1, 2, 2, 2],
  inputValues: [1, 2, 3, 4, 5, 6, 7, 8],
  targetShape: [1, 2, 2, 2],
  targetValues: [1.5, 1, 2.5, 3, 4, 5, 6, 7],
  expectedGram: [3.75, 8.75, 8.75, 21.75],
  expectedContentLoss: 0.8125,
  expectedStyleLoss: 13.6494140625,
};

test("style loss ops match reshape gram content and style references", async ({
  page,
}) => {
  await gotoStableApp(page);
  const workerClient = await createStyleTransferWorkerClient(page);

  try {
    const init = await workerClient.initWebGpu();
    const reshape = await workerClient.ask({
      type: "tensor-op",
      id: "phase3-reshape",
      op: "reshape-chw-flatten",
      input: { shape: fixture.inputShape, values: fixture.inputValues },
    });
    const gram = await workerClient.ask({
      type: "tensor-op",
      id: "phase3-gram",
      op: "gram-matrix",
      input: { shape: fixture.inputShape, values: fixture.inputValues },
    });
    const contentLoss = await workerClient.ask({
      type: "tensor-op",
      id: "phase3-content-loss",
      op: "content-loss",
      input: { shape: fixture.inputShape, values: fixture.inputValues },
      target: { shape: fixture.targetShape, values: fixture.targetValues },
    });
    const styleLoss = await workerClient.ask({
      type: "tensor-op",
      id: "phase3-style-loss",
      op: "style-loss",
      input: { shape: fixture.inputShape, values: fixture.inputValues },
      target: { shape: fixture.targetShape, values: fixture.targetValues },
    });
    const weightedContentLoss = await workerClient.ask({
      type: "tensor-op",
      id: "phase3-content-loss-weighted",
      op: "content-loss",
      input: { shape: fixture.inputShape, values: fixture.inputValues },
      target: { shape: fixture.targetShape, values: fixture.targetValues },
      contentWeight: 3.0,
    });
    const weightedStyleLoss = await workerClient.ask({
      type: "tensor-op",
      id: "phase3-style-loss-weighted",
      op: "style-loss",
      input: { shape: fixture.inputShape, values: fixture.inputValues },
      target: { shape: fixture.targetShape, values: fixture.targetValues },
      styleWeight: 5.0,
    });

    expectWebGpuInitOk(init);

    const reshapeResponse = expectWorkerTensorVectorResponse(reshape);
    expect(reshapeResponse.values).toEqual(fixture.inputValues);

    const gramResponse = expectWorkerTensorVectorResponse(gram);
    expectTensorCloseTo(gramResponse.values, fixture.expectedGram, 5);

    const contentLossResponse = expectWorkerTensorScalarResponse(contentLoss);
    expectScalarCloseTo(
      contentLossResponse.scalar,
      fixture.expectedContentLoss,
      6,
    );

    const styleLossResponse = expectWorkerTensorScalarResponse(styleLoss);
    expectScalarCloseTo(styleLossResponse.scalar, fixture.expectedStyleLoss, 6);

    const weightedContentLossResponse =
      expectWorkerTensorScalarResponse(weightedContentLoss);
    expectScalarCloseTo(
      weightedContentLossResponse.scalar,
      fixture.expectedContentLoss * 3.0,
      6,
    );

    const weightedStyleLossResponse =
      expectWorkerTensorScalarResponse(weightedStyleLoss);
    expectScalarCloseTo(
      weightedStyleLossResponse.scalar,
      fixture.expectedStyleLoss * 5.0,
      6,
    );
  } finally {
    await workerClient.dispose();
  }
});
