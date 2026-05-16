export interface WebGPUStatus {
  supported: boolean;
  adapterName?: string;
  reason: string;
}

export async function checkWebGPUSupport(): Promise<WebGPUStatus> {
  if (!('gpu' in navigator)) {
    return { supported: false, reason: 'WebGPU not available in this browser.' };
  }
  const gpu = (navigator as Navigator & { gpu?: GPU }).gpu;
  if (!gpu) {
    return { supported: false, reason: 'WebGPU not available in this browser.' };
  }
  const adapter = await gpu.requestAdapter();
  if (!adapter) {
    return { supported: false, reason: 'No WebGPU adapter found.' };
  }
  const device = await adapter.requestDevice();
  device.destroy();
  return { supported: true, adapterName: adapter.info.vendor || 'unknown vendor', reason: 'ok' };
}
