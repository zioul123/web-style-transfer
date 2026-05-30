import { expect, type Page } from "@playwright/test";
import {
  isWorkerTensorScalarOpResponse,
  isWorkerTensorVectorOpResponse,
  type WorkerRequest,
  type WorkerResponse,
  type WorkerTensorOpErrorResponse,
  type WorkerTensorScalarOpResponse,
  type WorkerTensorVectorOpResponse,
} from "../../src/types";

type BrowserWorkerClientEntry = {
  worker: Worker;
};

type BrowserWindowWithWorkerClients = Window & {
  __styleTransferWorkerClients?: Record<string, BrowserWorkerClientEntry>;
};

export type StyleTransferWorkerClient = {
  initWebGpu: () => Promise<WorkerResponse>;
  ask: (
    payload: WorkerRequest | Record<string, unknown>,
  ) => Promise<WorkerResponse>;
  dispose: () => Promise<void>;
};

const workerClientRegistryName = "__styleTransferWorkerClients";

const makeRequestId = (prefix: string): string =>
  `${prefix}-${Date.now()}-${Math.random().toString(36).slice(2)}`;

export const createStyleTransferWorkerClient = async (
  page: Page,
): Promise<StyleTransferWorkerClient> => {
  const clientId: string = makeRequestId("style-transfer-worker-client");

  await page.evaluate(
    ({ clientIdArg, registryName }) => {
      const browserWindow: BrowserWindowWithWorkerClients = window;
      browserWindow[registryName] ??= {};
      browserWindow[registryName][clientIdArg] = {
        worker: new Worker(
          new URL("/src/styleTransfer.worker.ts", window.location.origin),
          { type: "module" },
        ),
      };
    },
    { clientIdArg: clientId, registryName: workerClientRegistryName },
  );

  const ask = async (
    payload: WorkerRequest | Record<string, unknown>,
  ): Promise<WorkerResponse> =>
    page.evaluate(
      ({ clientIdArg, payloadArg, registryName }) => {
        const browserWindow: BrowserWindowWithWorkerClients = window;
        const entry = browserWindow[registryName]?.[clientIdArg];
        if (entry === undefined) {
          throw new Error(`Missing worker client: ${clientIdArg}`);
        }

        return new Promise<WorkerResponse>((resolve) => {
          const expectedId =
            typeof payloadArg.id === "string" ? payloadArg.id : undefined;
          const handler = (event: MessageEvent<WorkerResponse>): void => {
            if (
              expectedId === undefined ||
              event.data.id === expectedId ||
              event.data.type === "error"
            ) {
              entry.worker.removeEventListener("message", handler);
              resolve(event.data);
            }
          };
          entry.worker.addEventListener("message", handler);
          entry.worker.postMessage(payloadArg);
        });
      },
      {
        clientIdArg: clientId,
        payloadArg: payload,
        registryName: workerClientRegistryName,
      },
    );

  const initWebGpu = (): Promise<WorkerResponse> =>
    ask({ type: "init-webgpu", id: makeRequestId("init-webgpu") });

  const dispose = async (): Promise<void> => {
    await page.evaluate(
      ({ clientIdArg, registryName }) => {
        const browserWindow: BrowserWindowWithWorkerClients = window;
        const entry = browserWindow[registryName]?.[clientIdArg];
        if (entry === undefined) return;
        entry.worker.terminate();
        delete browserWindow[registryName]?.[clientIdArg];
      },
      { clientIdArg: clientId, registryName: workerClientRegistryName },
    );
  };

  return { initWebGpu, ask, dispose };
};

export const expectWebGpuInitOk = (response: WorkerResponse): void => {
  expect(response.type).toBe("webgpu-init-result");
  if (response.type !== "webgpu-init-result") {
    throw new Error("Expected webgpu-init-result response.");
  }
  expect(response.ok).toBeTruthy();
};

export const expectWorkerTensorVectorResponse = (
  response: WorkerResponse,
): WorkerTensorVectorOpResponse => {
  expect(response.type).toBe("tensor-op-result");
  if (response.type !== "tensor-op-result") {
    throw new Error("Expected tensor-op-result response.");
  }
  expect(isWorkerTensorVectorOpResponse(response)).toBeTruthy();
  if (!isWorkerTensorVectorOpResponse(response)) {
    throw new Error("Expected tensor-op vector response.");
  }
  return response;
};

export const expectWorkerTensorScalarResponse = (
  response: WorkerResponse,
): WorkerTensorScalarOpResponse => {
  expect(response.type).toBe("tensor-op-result");
  if (response.type !== "tensor-op-result") {
    throw new Error("Expected tensor-op-result response.");
  }
  expect(isWorkerTensorScalarOpResponse(response)).toBeTruthy();
  if (!isWorkerTensorScalarOpResponse(response)) {
    throw new Error("Expected tensor-op scalar response.");
  }
  return response;
};

export const expectWorkerTensorErrorResponse = (
  response: WorkerResponse,
): WorkerTensorOpErrorResponse => {
  expect(response.type).toBe("tensor-op-result");
  if (response.type !== "tensor-op-result") {
    throw new Error("Expected tensor-op-result response.");
  }
  expect(response.ok).toBeFalsy();
  if (response.ok) {
    throw new Error("Expected tensor-op error response.");
  }
  return response;
};
