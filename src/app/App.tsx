import { useEffect, useMemo, useState } from 'react';
import { createStyleTransferWorker } from '../features/style-transfer/workers/client';
import { checkWebGPUSupport } from '../ml/webgpu/adapter';

export function App() {
  const [gpuStatus, setGpuStatus] = useState<string>('Checking WebGPU support...');
  const [workerStatus, setWorkerStatus] = useState<string>('Worker not started');
  const worker = useMemo(() => createStyleTransferWorker(), []);

  useEffect(() => {
    let disposed = false;

    checkWebGPUSupport()
      .then((result) => {
        if (disposed) return;
        setGpuStatus(result.supported ? `WebGPU ready (${result.adapterName ?? 'adapter found'})` : result.reason);
      })
      .catch((err: unknown) => {
        if (disposed) return;
        setGpuStatus(`WebGPU check failed: ${String(err)}`);
      });

    worker
      .ping()
      .then((msg) => {
        if (disposed) return;
        setWorkerStatus(msg);
      })
      .catch((err: unknown) => {
        if (disposed) return;
        setWorkerStatus(`Worker error: ${String(err)}`);
      });

    return () => {
      disposed = true;
      worker.dispose();
    };
  }, [worker]);

  return (
    <main style={{ fontFamily: 'sans-serif', padding: 16 }}>
      <h1>Web Style Transfer</h1>
      <p><strong>Phase 0:</strong> {gpuStatus}</p>
      <p><strong>Worker:</strong> {workerStatus}</p>
    </main>
  );
}
