import type { WorkerRequest, WorkerResponse } from "../../src/types";

export type BrowserStyleTransferWorkerClient = {
  initWebGpu: (id?: string) => Promise<WorkerResponse>;
  ask: (
    payload: WorkerRequest | Record<string, unknown>,
  ) => Promise<WorkerResponse>;
  dispose: () => Promise<void>;
};

type WorkerGpuDispatchCoverageRecord = {
  label: string;
  workgroups: readonly [number, number, number];
  count: number;
};

type WorkerGpuDispatchCoverageMessage = {
  type: "__gpu-dispatch-coverage";
  label: string;
  workgroups: readonly [number, number, number];
};

type BrowserWindowWithDispatchCoverage = Window & {
  __styleTransferDispatchCoverage?: WorkerGpuDispatchCoverageRecord[];
};

const recordGpuDispatchCoverage = (
  message: WorkerGpuDispatchCoverageMessage,
): void => {
  const browserWindow: BrowserWindowWithDispatchCoverage = window;
  browserWindow.__styleTransferDispatchCoverage ??= [];
  const existing = browserWindow.__styleTransferDispatchCoverage.find(
    (record): boolean =>
      record.label === message.label &&
      record.workgroups.join("x") === message.workgroups.join("x"),
  );
  if (existing !== undefined) {
    existing.count += 1;
    return;
  }
  browserWindow.__styleTransferDispatchCoverage.push({
    label: message.label,
    workgroups: message.workgroups,
    count: 1,
  });
};

const isGpuDispatchCoverageMessage = (
  message: WorkerResponse | WorkerGpuDispatchCoverageMessage,
): message is WorkerGpuDispatchCoverageMessage =>
  message.type === "__gpu-dispatch-coverage";

const makeRequestId = (prefix: string): string =>
  `${prefix}-${Date.now()}-${Math.random().toString(36).slice(2)}`;

export const createBrowserStyleTransferWorkerClient =
  (): BrowserStyleTransferWorkerClient => {
    const worker = new Worker(
      new URL(
        "/src/styleTransfer.worker.ts?dispatchCoverage=1",
        window.location.origin,
      ),
      { type: "module" },
    );

    worker.addEventListener(
      "message",
      (
        event: MessageEvent<WorkerResponse | WorkerGpuDispatchCoverageMessage>,
      ): void => {
        if (isGpuDispatchCoverageMessage(event.data)) {
          recordGpuDispatchCoverage(event.data);
        }
      },
    );

    const ask = (
      payload: WorkerRequest | Record<string, unknown>,
    ): Promise<WorkerResponse> =>
      new Promise((resolve) => {
        const expectedId =
          typeof payload.id === "string" ? payload.id : undefined;
        const handler = (
          event: MessageEvent<
            WorkerResponse | WorkerGpuDispatchCoverageMessage
          >,
        ): void => {
          if (isGpuDispatchCoverageMessage(event.data)) return;
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

    const dispose = async (): Promise<void> => {
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

export type RunStyleTransferRequest = Extract<
  WorkerRequest,
  { type: "run-style-transfer" }
>;

export type RunStyleTransferSuccessResponse = Extract<
  WorkerResponse,
  { type: "run-style-transfer-result"; ok: true }
>;

export type RunStyleTransferResult =
  | { ok: true; response: RunStyleTransferSuccessResponse }
  | { ok: false; reason: "wrong-response" }
  | { ok: false; reason: "worker-failed"; message: string };

export const validateRunStyleTransferResult = (
  response: WorkerResponse,
): RunStyleTransferResult => {
  if (response.type !== "run-style-transfer-result") {
    return { ok: false, reason: "wrong-response" };
  }
  if (!response.ok) {
    return {
      ok: false,
      reason: "worker-failed",
      message: response.message,
    };
  }
  return { ok: true, response };
};

export const askRunStyleTransfer = async (
  workerClient: BrowserStyleTransferWorkerClient,
  payload: RunStyleTransferRequest,
): Promise<RunStyleTransferResult> =>
  validateRunStyleTransferResult(await workerClient.ask(payload));

export const withBrowserStyleTransferWorkerClient = async <T>(
  callback: (workerClient: BrowserStyleTransferWorkerClient) => Promise<T>,
): Promise<T> => {
  const workerClient = createBrowserStyleTransferWorkerClient();
  try {
    return await callback(workerClient);
  } finally {
    await workerClient.dispose();
  }
};
