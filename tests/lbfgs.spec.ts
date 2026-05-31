import { expect, test } from "./helpers/coverage";
import { gotoStableApp } from "./helpers/appPage";

type LbfgsFixtureCase = {
  name: string;
  initial: number[];
  target: number[];
  matrix: number[][];
  lr: number;
  steps: number;
  historySize: number;
  trajectory: {
    step: number;
    lossBefore: number;
    valuesAfter: number[];
  }[];
};

type LbfgsFixture = {
  torchVersion: string;
  cases: LbfgsFixtureCase[];
};

test("input LBFGS matches PyTorch max_iter=1 fixture trajectories", async ({
  page,
}) => {
  await gotoStableApp(page);
  const result = await page.evaluate(async () => {
    const { fetchJsonOrNull } = await import("/tests/helpers/fixtures.ts");
    const fixture = await fetchJsonOrNull<LbfgsFixture>(
      "/lbfgs/lbfgs_fixture.json",
    );
    if (fixture === null)
      return { ok: false as const, reason: "missing-fixture" as const };
    const { createInputOptimizer } =
      await import("/src/ml/worker/pipelines/optimization/input-optimizer/createInputOptimizer.ts");
    const { createCpuVectorOps } =
      await import("/src/ml/worker/pipelines/optimization/input-optimizer/cpuVectorOps.ts");
    const multiplyMatrix = (
      matrix: number[][],
      values: Float32Array,
      target: number[],
    ): Float32Array => {
      const out = new Float32Array(values.length);
      for (let row = 0; row < values.length; row += 1) {
        let total = 0;
        for (let col = 0; col < values.length; col += 1)
          total += matrix[row][col] * (values[col] - target[col]);
        out[row] = total;
      }
      return out;
    };
    const caseResults = [];
    for (const testCase of fixture.cases) {
      let input = new Float32Array(testCase.initial);
      const optimizer = createInputOptimizer(
        {
          optimizer: "lbfgs",
          count: input.length,
          learningRate: testCase.lr,
          adamBeta1: 0.9,
          adamBeta2: 0.999,
          adamEpsilon: 1e-8,
          lbfgsMemory: testCase.historySize,
          lbfgsEpsilon: 1e-9,
        },
        createCpuVectorOps(input.length),
      );
      let maxDiff = 0;
      for (const expected of testCase.trajectory) {
        const grad = multiplyMatrix(testCase.matrix, input, testCase.target);
        input = await optimizer.step(input, grad, expected.step);
        for (let i = 0; i < input.length; i += 1) {
          maxDiff = Math.max(
            maxDiff,
            Math.abs(input[i] - expected.valuesAfter[i]),
          );
        }
      }
      optimizer.dispose();
      caseResults.push({ name: testCase.name, maxDiff });
    }
    return { ok: true as const, caseResults };
  });
  test.skip(
    !result.ok && result.reason === "missing-fixture",
    "Missing LBFGS fixture. Run python-reference/export_lbfgs_fixtures.py.",
  );
  if (!result.ok) return;
  for (const caseResult of result.caseResults) {
    expect(caseResult.maxDiff, caseResult.name).toBeLessThan(2e-5);
  }
});
