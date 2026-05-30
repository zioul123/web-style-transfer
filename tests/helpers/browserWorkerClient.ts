import type { WorkerRequest, WorkerResponse } from "../../src/types";

export type BrowserStyleTransferWorkerClient = {
  initWebGpu: (id?: string) => Promise<WorkerResponse>;
  ask: (
    payload: WorkerRequest | Record<string, unknown>,
  ) => Promise<WorkerResponse>;
  dispose: () => void;
};

const makeRequestId = (prefix: string): string =>
  `${prefix}-${Date.now()}-${Math.random().toString(36).slice(2)}`;

export const createBrowserStyleTransferWorkerClient =
  (): BrowserStyleTransferWorkerClient => {
    const worker = new Worker(
      new URL("/src/styleTransfer.worker.ts", window.location.origin),
      { type: "module" },
    );

    const ask = (
      payload: WorkerRequest | Record<string, unknown>,
    ): Promise<WorkerResponse> =>
      new Promise((resolve) => {
        const expectedId =
          typeof payload.id === "string" ? payload.id : undefined;
        const handler = (event: MessageEvent<WorkerResponse>): void => {
          if (
            expectedId === undefined ||
            event.data.id === expectedId ||
            event.data.type === "error"
          ) {
            worker.removeEventListener("message", handler);
            resolve(event.data);
          }
        };
        worker.addEventListener("message", handler);
        worker.postMessage(payload);
      });

    const initWebGpu = (
      id = makeRequestId("init-webgpu"),
    ): Promise<WorkerResponse> => ask({ type: "init-webgpu", id });

    const dispose = (): void => {
      worker.terminate();
    };

    return { initWebGpu, ask, dispose };
  };

export const getTensorValues = (
  response: WorkerResponse,
  label = "tensor op",
): number[] => {
  if (
    response.type !== "tensor-op-result" ||
    !response.ok ||
    !("values" in response)
  ) {
    throw new Error(`${label} did not return tensor values.`);
  }
  return response.values;
};

export const getTensorScalar = (
  response: WorkerResponse,
  label = "tensor op",
): number => {
  if (
    response.type !== "tensor-op-result" ||
    !response.ok ||
    !("scalar" in response)
  ) {
    throw new Error(`${label} did not return a tensor scalar.`);
  }
  return response.scalar;
};
