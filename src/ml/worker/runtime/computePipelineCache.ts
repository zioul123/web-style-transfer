const pipelineCacheByDevice = new WeakMap<
  GPUDevice,
  Map<string, GPUComputePipeline>
>();

// WebGPU pipeline creation is expensive. We keep one cache per device and
// key by operation signature so hot math kernels can be reused across steps.
export const getOrCreateComputePipeline = (
  gpuDevice: GPUDevice,
  key: string,
  code: string,
): GPUComputePipeline => {
  let cache = pipelineCacheByDevice.get(gpuDevice);
  if (cache === undefined) {
    cache = new Map<string, GPUComputePipeline>();
    pipelineCacheByDevice.set(gpuDevice, cache);
  }
  const cached = cache.get(key);
  if (cached !== undefined) return cached;
  const created = gpuDevice.createComputePipeline({
    layout: "auto",
    compute: {
      module: gpuDevice.createShaderModule({ code }),
      entryPoint: "main",
    },
  });
  cache.set(key, created);
  return created;
};
