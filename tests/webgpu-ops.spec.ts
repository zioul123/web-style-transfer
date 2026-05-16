import { expect, test } from '@playwright/test';

type OpComparisonResult = {
  addCpuData: number[];
  addGpuData: number[];
  addMeta?: { backend?: string };
  firstAddMismatch?: { cpu: number; gpu: number; index: number };
  mseCpu: number;
  mseGpu: number;
  mseMeta?: { backend?: string };
  mseTolerance: number;
};

test('webgpu ops execute on webgpu backend with swiftshader', async ({ page }) => {
  await page.goto('/');

  const result = await page.evaluate(async (): Promise<OpComparisonResult> => {
    const { Tensor } = await import('/src/ml/core/tensor.ts');
    const { cpuOps } = await import('/src/ml/ops/cpu.ts');
    const { gpuOps, getLastGPUExecutionMeta } = await import('/src/ml/ops/gpu.ts');

    const shape: [number, number, number, number] = [1, 1, 4, 4];
    const a = new Tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], shape);
    const b = new Tensor([16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1], shape);

    const addGpu = await gpuOps.add(a, b);
    const addCpu = cpuOps.add(a, b);
    const addMeta = getLastGPUExecutionMeta();

    const mseGpu = await gpuOps.mse(a, b);
    const mseCpu = cpuOps.mse(a, b);
    const mseMeta = getLastGPUExecutionMeta();

    const addGpuData: number[] = Array.from(addGpu.data);
    const addCpuData: number[] = Array.from(addCpu.data);
    const firstAddMismatch = addGpuData.reduce<{ cpu: number; gpu: number; index: number } | undefined>(
      (mismatch, gpuValue, index) => {
        if (mismatch) {
          return mismatch;
        }
        const cpuValue = addCpuData[index];
        return gpuValue === cpuValue
          ? undefined
          : {
              cpu: cpuValue,
              gpu: gpuValue,
              index
            };
      },
      undefined
    );

    return {
      addCpuData,
      addGpuData,
      addMeta,
      firstAddMismatch,
      mseCpu,
      mseGpu,
      mseMeta,
      mseTolerance: 1e-6
    };
  });

  expect(
    result.firstAddMismatch,
    result.firstAddMismatch
      ? `GPU add mismatch at index ${result.firstAddMismatch.index}: gpu=${result.firstAddMismatch.gpu}, cpu=${result.firstAddMismatch.cpu}. GPU output=${JSON.stringify(result.addGpuData)}, CPU output=${JSON.stringify(result.addCpuData)}`
      : `GPU add unexpectedly reported mismatch without mismatch detail. GPU output=${JSON.stringify(result.addGpuData)}, CPU output=${JSON.stringify(result.addCpuData)}`
  ).toBeUndefined();

  const mseDelta = Math.abs(result.mseGpu - result.mseCpu);
  expect(
    mseDelta,
    `GPU mse mismatch: gpu=${result.mseGpu}, cpu=${result.mseCpu}, |delta|=${mseDelta}, tolerance=${result.mseTolerance}`
  ).toBeLessThanOrEqual(result.mseTolerance);

  expect(result.addMeta?.backend).toBe('webgpu');
  expect(result.mseMeta?.backend).toBe('webgpu');
});
