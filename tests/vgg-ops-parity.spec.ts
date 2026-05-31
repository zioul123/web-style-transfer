import { test } from "./helpers/coverage";
import type { WorkerResponse } from "../src/types";
import { gotoStableApp } from "./helpers/appPage";
import { expectTensorCloseTo } from "./helpers/tensorAssertions";
import {
  createStyleTransferWorkerClient,
  expectWebGpuInitOk,
  expectWorkerTensorVectorResponse,
} from "./helpers/workerClient";

const inputShape = [1, 2, 4, 4] as const;
const inputValues = [
  1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 0.5, -1, 1.5, 0, -0.5,
  2, -2.5, 1, 3, -3.5, 4, -4.5, 5, -5.5, 6, -6.5,
];
const weightShape = [2, 2, 3, 3] as const;
const weightValues = [
  1, 0, -1, 0.5, 0, -0.5, 1, 0, -1, 0.25, 0, -0.25, 0.5, 0, -0.5, 0.25, 0,
  -0.25, 0, 1, 0, 1, -4, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, -1, -1, -1,
];
const bias = [0.1, -0.2];

const convExpected = [
  -6.900000095367432, -2.9000000953674316, -3.1500000953674316,
  8.725000381469727, -14.774999618530273, -4.400000095367432,
  -4.400000095367432, 17.725000381469727, -22.274999618530273,
  -5.150000095367432, -3.9000000953674316, 30.475000381469727,
  -13.274999618530273, -3.6500000953674316, -2.1500000953674316,
  22.600000381469727, 1.2999999523162842, 2.799999952316284,
  0.30000001192092896, -3.700000047683716, -4.199999809265137,
  -2.700000047683716, 4.300000190734863, -7.199999809265137, -6.199999809265137,
  -6.699999809265137, 6.300000190734863, -14.199999809265137,
  -29.700000762939453, -14.699999809265137, -23.200000762939453,
  -37.70000076293945,
];
const reluExpected = [
  0, 0, 0, 8.725000381469727, 0, 0, 0, 17.725000381469727, 0, 0, 0,
  30.475000381469727, 0, 0, 0, 22.600000381469727, 1.2999999523162842,
  2.799999952316284, 0.30000001192092896, 0, 0, 0, 4.300000190734863, 0, 0, 0,
  6.300000190734863, 0, 0, 0, 0, 0,
];
const poolExpected = [
  0, 17.725000381469727, 0, 30.475000381469727, 2.799999952316284,
  4.300000190734863, 0, 6.300000190734863,
];
const normExpected = [
  2.248908281326294, 6.615720272064209, 10.982532501220703, 15.349344253540039,
  19.716156005859375, 24.08296775817871, 28.44978141784668, 32.816593170166016,
  37.183406829833984, 41.55022048950195, 45.917030334472656, 50.283843994140625,
  54.650657653808594, 59.0174674987793, 63.384281158447266, 67.75109100341797,
  0.1964285671710968, -6.499999523162842, 4.660714149475098,
  -2.0357141494750977, -4.267857074737549, 6.892857074737549,
  -13.196428298950195, 2.4285714626312256, 11.357142448425293,
  -17.66071319580078, 15.821427345275879, -22.124998092651367,
  20.285715103149414, -26.589284896850586, 24.75, -31.053569793701172,
];

test("vgg ops match pytorch references for conv/relu/pool/normalize", async ({
  page,
}) => {
  await gotoStableApp(page);
  const workerClient = await createStyleTransferWorkerClient(page);

  try {
    const init = await workerClient.initWebGpu();
    const conv = await workerClient.ask({
      type: "tensor-op",
      id: "phase2-conv",
      op: "conv2d-forward",
      input: { shape: inputShape, values: inputValues },
      weight: { shape: weightShape, values: weightValues },
      bias,
    });
    const relu = await workerClient.ask({
      type: "tensor-op",
      id: "phase2-relu",
      op: "relu-forward",
      input: { shape: [1, 2, 4, 4], values: convExpected },
    });
    const pool = await workerClient.ask({
      type: "tensor-op",
      id: "phase2-pool",
      op: "maxpool2d-forward",
      input: { shape: [1, 2, 4, 4], values: reluExpected },
    });
    const norm = await workerClient.ask({
      type: "tensor-op",
      id: "phase2-norm",
      op: "normalize-forward",
      input: { shape: inputShape, values: inputValues },
      mean: [0.485, 0.456],
      std: [0.229, 0.224],
    });

    expectWebGpuInitOk(init);

    const checks: Array<[WorkerResponse, number[]]> = [
      [conv, convExpected],
      [relu, reluExpected],
      [pool, poolExpected],
      [norm, normExpected],
    ];
    for (const [response, expectedValues] of checks) {
      const vectorResponse = expectWorkerTensorVectorResponse(response);
      expectTensorCloseTo(vectorResponse.values, expectedValues, 5);
    }
  } finally {
    await workerClient.dispose();
  }
});
