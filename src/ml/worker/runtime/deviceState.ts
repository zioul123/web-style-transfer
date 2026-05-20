let gpuDevice: GPUDevice | null = null;

export const getGpuDevice = (): GPUDevice | null => gpuDevice;

export const setGpuDevice = (device: GPUDevice): void => {
  gpuDevice = device;
};

export const clearGpuDevice = (): void => {
  gpuDevice = null;
};

export const requireGpuDevice = (): GPUDevice => {
  if (gpuDevice === null) {
    throw new Error("WebGPU is not initialized.");
  }
  return gpuDevice;
};
