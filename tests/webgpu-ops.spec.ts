import { expect, test } from '@playwright/test';

test('webgpu ops execute on webgpu backend with swiftshader', async ({ page }) => {
  await page.goto('/');

  const result = await page.evaluate(async () => {
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

    return {
      addEqual: Array.from(addGpu.data).every((value, index) => value === addCpu.data[index]),
      addMeta,
      mseEqual: Math.abs(mseGpu - mseCpu) < 1e-6,
      mseMeta
    };
  });

  expect(result.addEqual).toBe(true);
  expect(result.mseEqual).toBe(true);
  expect(result.addMeta?.backend).toBe('webgpu');
  expect(result.mseMeta?.backend).toBe('webgpu');
});
