import { reusableBufferPool } from "../runtime/bufferPool";
import { clearGpuDevice, setGpuDevice } from "../runtime/deviceState";
import { sendWebGpuInitResult } from "./responses";

export const initWebGpu = async (id: string): Promise<void> => {
  if (!("gpu" in navigator)) {
    clearGpuDevice();
    sendWebGpuInitResult(
      id,
      false,
      "navigator.gpu is unavailable in this worker context.",
    );
    return;
  }

  const adapter: GPUAdapter | null = await navigator.gpu.requestAdapter();
  if (adapter === null) {
    clearGpuDevice();
    sendWebGpuInitResult(id, false, "WebGPU adapter request returned null.");
    return;
  }

  const device = await adapter.requestDevice();
  setGpuDevice(device);
  reusableBufferPool.clear();
  sendWebGpuInitResult(
    id,
    true,
    `WebGPU device initialized: ${device.label || "unnamed-device"}`,
  );
};
