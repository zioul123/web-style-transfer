import { expect, test } from "./helpers/coverage";
import {
  VGG19_POOL_LAYER_INDICES_UP_TO_CONV5_1,
  VGG19_RELU_LAYER_INDICES_UP_TO_CONV5_1,
} from "../src/ml/constants/vgg19";
import { gotoStableApp } from "./helpers/appPage";

test("vgg19 full pass matches conv5_1 style and relu4_2 content losses", async ({
  page,
}) => {
  test.setTimeout(300000);
  await gotoStableApp(page);

  const result = await page.evaluate(
    async ({ reluLayers, poolLayers }) => {
      const { loadFullPassArtifacts } =
        await import("/tests/helpers/fullPassArtifacts.ts");
      const {
        createBrowserStyleTransferWorkerClient,
        getTensorScalar,
        getTensorValues,
      } = await import("/tests/helpers/browserWorkerClient.ts");
      const artifactsResult = await loadFullPassArtifacts();
      if (!artifactsResult.ok) return artifactsResult;
      const { fixture, weights, modelSource, lossPrecision } =
        artifactsResult.artifacts;

      const workerClient = createBrowserStyleTransferWorkerClient();
      try {
        const init = await workerClient.initWebGpu("phase3-full-init");
        const normalizedInput = await workerClient.ask({
          type: "tensor-op",
          id: "phase3-full-norm-input",
          op: "normalize-forward",
          input: {
            shape: fixture.inputShape,
            values: fixture.inputImageValues,
          },
          mean: fixture.mean,
          std: fixture.std,
        });
        const normalizedContent = await workerClient.ask({
          type: "tensor-op",
          id: "phase3-full-norm-content",
          op: "normalize-forward",
          input: {
            shape: fixture.inputShape,
            values: fixture.contentImageValues,
          },
          mean: fixture.mean,
          std: fixture.std,
        });
        const normalizedStyle = await workerClient.ask({
          type: "tensor-op",
          id: "phase3-full-norm-style",
          op: "normalize-forward",
          input: {
            shape: fixture.inputShape,
            values: fixture.styleImageValues,
          },
          mean: fixture.mean,
          std: fixture.std,
        });

        let currentContentShape: [number, number, number, number] =
          fixture.inputShape;
        let currentInputShape: [number, number, number, number] =
          fixture.inputShape;
        let currentStyleShape: [number, number, number, number] =
          fixture.inputShape;
        let currentContentValues: number[] = getTensorValues(normalizedContent);
        let currentInputValues: number[] = getTensorValues(normalizedInput);
        let currentStyleValues: number[] = getTensorValues(normalizedStyle);
        const styleLossByLayer: Record<string, number> = {};
        let contentLoss = 0;

        for (let layerIndex = 0; layerIndex <= 29; layerIndex += 1) {
          const weightShape = weights[`conv${layerIndex}.weightShape`];
          const weightValues = weights[`conv${layerIndex}.weightValues`];
          const biasValues = weights[`conv${layerIndex}.biasValues`];

          if (
            Array.isArray(weightValues) &&
            Array.isArray(biasValues) &&
            Array.isArray(weightShape)
          ) {
            const convContent = await workerClient.ask({
              type: "tensor-op",
              id: `conv-c-${layerIndex}`,
              op: "conv2d-forward",
              input: {
                shape: currentContentShape,
                values: currentContentValues,
              },
              weight: {
                shape: weightShape as [number, number, number, number],
                values: weightValues,
              },
              bias: biasValues,
            });
            const convInput = await workerClient.ask({
              type: "tensor-op",
              id: `conv-i-${layerIndex}`,
              op: "conv2d-forward",
              input: { shape: currentInputShape, values: currentInputValues },
              weight: {
                shape: weightShape as [number, number, number, number],
                values: weightValues,
              },
              bias: biasValues,
            });
            const convStyle = await workerClient.ask({
              type: "tensor-op",
              id: `conv-s-${layerIndex}`,
              op: "conv2d-forward",
              input: { shape: currentStyleShape, values: currentStyleValues },
              weight: {
                shape: weightShape as [number, number, number, number],
                values: weightValues,
              },
              bias: biasValues,
            });
            currentContentValues = getTensorValues(convContent);
            currentInputValues = getTensorValues(convInput);
            currentStyleValues = getTensorValues(convStyle);
            currentContentShape = [
              1,
              weightShape[0],
              currentContentShape[2],
              currentContentShape[3],
            ];
            currentInputShape = [
              1,
              weightShape[0],
              currentInputShape[2],
              currentInputShape[3],
            ];
            currentStyleShape = [
              1,
              weightShape[0],
              currentStyleShape[2],
              currentStyleShape[3],
            ];
          }
          if (reluLayers.includes(layerIndex)) {
            currentContentValues = getTensorValues(
              await workerClient.ask({
                type: "tensor-op",
                id: `relu-c-${layerIndex}`,
                op: "relu-forward",
                input: {
                  shape: currentContentShape,
                  values: currentContentValues,
                },
              }),
            );
            currentInputValues = getTensorValues(
              await workerClient.ask({
                type: "tensor-op",
                id: `relu-i-${layerIndex}`,
                op: "relu-forward",
                input: { shape: currentInputShape, values: currentInputValues },
              }),
            );
            currentStyleValues = getTensorValues(
              await workerClient.ask({
                type: "tensor-op",
                id: `relu-s-${layerIndex}`,
                op: "relu-forward",
                input: { shape: currentStyleShape, values: currentStyleValues },
              }),
            );
          }
          if (layerIndex === fixture.contentLayerIndex) {
            contentLoss = getTensorScalar(
              await workerClient.ask({
                type: "tensor-op",
                id: `content-${layerIndex}`,
                op: "content-loss",
                input: { shape: currentInputShape, values: currentInputValues },
                target: {
                  shape: currentContentShape,
                  values: currentContentValues,
                },
              }),
            );
          }
          if (fixture.styleLayerIndices.includes(layerIndex)) {
            styleLossByLayer[`relu${layerIndex}`] = getTensorScalar(
              await workerClient.ask({
                type: "tensor-op",
                id: `style-${layerIndex}`,
                op: "style-loss",
                input: { shape: currentInputShape, values: currentInputValues },
                target: {
                  shape: currentStyleShape,
                  values: currentStyleValues,
                },
              }),
            );
          }
          if (poolLayers.includes(layerIndex)) {
            currentContentValues = getTensorValues(
              await workerClient.ask({
                type: "tensor-op",
                id: `pool-c-${layerIndex}`,
                op: "maxpool2d-forward",
                input: {
                  shape: currentContentShape,
                  values: currentContentValues,
                },
              }),
            );
            currentInputValues = getTensorValues(
              await workerClient.ask({
                type: "tensor-op",
                id: `pool-i-${layerIndex}`,
                op: "maxpool2d-forward",
                input: { shape: currentInputShape, values: currentInputValues },
              }),
            );
            currentStyleValues = getTensorValues(
              await workerClient.ask({
                type: "tensor-op",
                id: `pool-s-${layerIndex}`,
                op: "maxpool2d-forward",
                input: { shape: currentStyleShape, values: currentStyleValues },
              }),
            );
            currentContentShape = [
              1,
              currentContentShape[1],
              Math.floor(currentContentShape[2] / 2),
              Math.floor(currentContentShape[3] / 2),
            ];
            currentInputShape = [
              1,
              currentInputShape[1],
              Math.floor(currentInputShape[2] / 2),
              Math.floor(currentInputShape[3] / 2),
            ];
            currentStyleShape = [
              1,
              currentStyleShape[1],
              Math.floor(currentStyleShape[2] / 2),
              Math.floor(currentStyleShape[3] / 2),
            ];
          }
        }

        const styleTotal = Object.values(styleLossByLayer).reduce(
          (sum, value) => sum + value,
          0,
        );
        return {
          ok: true as const,
          init,
          styleLossByLayer,
          styleTotal,
          contentLoss,
          totalLoss: styleTotal + contentLoss,
          expected: fixture,
          modelSource,
          lossPrecision,
        };
      } finally {
        workerClient.dispose();
      }
    },
    {
      reluLayers: VGG19_RELU_LAYER_INDICES_UP_TO_CONV5_1,
      poolLayers: VGG19_POOL_LAYER_INDICES_UP_TO_CONV5_1,
    },
  );

  test.skip(
    !result.ok && result.reason === "missing-fixtures",
    result.ok ? "" : result.message,
  );
  if (!result.ok) return;

  const styleLayerPairs = Object.entries(
    result.expected.expectedStyleLossByLayer,
  ).map(([layerName, expectedLoss]) => ({
    layerName,
    expectedLoss,
    actualLoss: result.styleLossByLayer[layerName],
    diff: (result.styleLossByLayer[layerName] ?? Number.NaN) - expectedLoss,
  }));
  console.log(
    "phase3 full-pass style-loss pairs",
    JSON.stringify(styleLayerPairs),
  );

  expect(result.init.type).toBe("webgpu-init-result");
  if (result.init.type !== "webgpu-init-result")
    throw new Error("Expected init result");
  expect(result.init.ok).toBeTruthy();
  for (const [layerName, expectedLoss] of Object.entries(
    result.expected.expectedStyleLossByLayer,
  ))
    expect(result.styleLossByLayer[layerName]).toBeCloseTo(
      expectedLoss,
      result.lossPrecision,
    );
  expect(result.styleTotal).toBeCloseTo(
    result.expected.expectedStyleLossTotal,
    result.lossPrecision,
  );
  expect(result.contentLoss).toBeCloseTo(
    result.expected.expectedContentLoss,
    result.lossPrecision,
  );
  expect(result.totalLoss).toBeCloseTo(
    result.expected.expectedTotalLoss,
    result.lossPrecision,
  );
});
