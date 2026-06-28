import { expect, test, type Page } from "@playwright/test";
import { gotoStableApp } from "./helpers/appPage";

type BackendMockMode =
  | "healthy"
  | "unhealthy"
  | "foreign-health"
  | "run-fails"
  | "malformed-run";

const backendSettingsKey = "web-style-transfer.backend-settings.v1";
const defaultBackendUrl = "http://127.0.0.1:8000";

const installBackendFetchMock = async (
  page: Page,
  mode: BackendMockMode,
): Promise<void> => {
  await page.addInitScript(
    ({ backendUrl, mockMode }) => {
      const originalFetch = window.fetch.bind(window);
      const state: {
        healthCalls: number;
        runCalls: number;
        runBodies: unknown[];
      } = {
        healthCalls: 0,
        runCalls: 0,
        runBodies: [],
      };
      (
        window as Window & {
          __styleTransferBackendMock?: typeof state;
        }
      ).__styleTransferBackendMock = state;

      window.fetch = async (...args: Parameters<typeof fetch>) => {
        const input = args[0];
        const url =
          typeof input === "string"
            ? input
            : input instanceof Request
              ? input.url
              : String(input);
        if (!url.startsWith(backendUrl)) return originalFetch(...args);

        if (url.endsWith("/health")) {
          state.healthCalls += 1;
          if (mockMode === "foreign-health") {
            return new Response(
              JSON.stringify({
                ok: true,
                backend: "other",
                engine: "other",
                device: "mock-device",
                modelReady: true,
                message: "foreign service",
              }),
              { headers: { "Content-Type": "application/json" } },
            );
          }
          if (mockMode === "unhealthy") {
            return new Response(
              JSON.stringify({
                ok: false,
                backend: "fastapi",
                engine: "pytorch",
                device: "unknown",
                modelReady: false,
                message: "mock backend offline",
              }),
              { headers: { "Content-Type": "application/json" } },
            );
          }
          return new Response(
            JSON.stringify({
              ok: true,
              backend: "fastapi",
              engine: "pytorch",
              device: "mock-device",
              modelReady: true,
              message: "mock backend ready",
            }),
            { headers: { "Content-Type": "application/json" } },
          );
        }

        if (url.endsWith("/style-transfer/run")) {
          state.runCalls += 1;
          const body = JSON.parse(String(args[1]?.body ?? "{}")) as {
            inputImageValues?: number[];
            steps?: number;
          };
          state.runBodies.push(body);
          if (mockMode === "run-fails") {
            return new Response(JSON.stringify({ ok: false }), {
              status: 503,
              headers: { "Content-Type": "application/json" },
            });
          }
          if (mockMode === "malformed-run") {
            return new Response(
              JSON.stringify({
                ok: true,
                losses: [42.25],
                finalValues: [0.5],
                stats: {
                  elapsedMs: 8,
                  avgStepMs: 8,
                  forwardMs: 2,
                  backwardMs: 2,
                  lossMs: 2,
                  updateMs: 2,
                  steps: body.steps ?? 1,
                },
              }),
              { headers: { "Content-Type": "application/json" } },
            );
          }
          const values = body.inputImageValues ?? [];
          return new Response(
            JSON.stringify({
              ok: true,
              losses: [42.25],
              finalValues: values.map((value) => Math.min(1, value + 0.01)),
              stats: {
                elapsedMs: 8,
                avgStepMs: 8,
                forwardMs: 2,
                backwardMs: 2,
                lossMs: 2,
                updateMs: 2,
                steps: body.steps ?? 1,
              },
            }),
            { headers: { "Content-Type": "application/json" } },
          );
        }

        if (url.endsWith("/style-transfer/session/clear")) {
          return new Response(JSON.stringify({ ok: true }), {
            headers: { "Content-Type": "application/json" },
          });
        }

        return originalFetch(...args);
      };
    },
    { backendUrl: defaultBackendUrl, mockMode: mode },
  );
};

const installWorkerPostTracker = async (page: Page): Promise<void> => {
  await page.addInitScript(() => {
    const OriginalWorker = window.Worker;
    const state = { runMessages: 0 };
    (
      window as Window & {
        __styleTransferWorkerPosts?: typeof state;
      }
    ).__styleTransferWorkerPosts = state;

    const WorkerReplacement = function (
      ...args: ConstructorParameters<typeof Worker>
    ): Worker {
      const worker = new OriginalWorker(...args);
      const originalPostMessage = worker.postMessage.bind(worker);
      worker.postMessage = (
        message: unknown,
        transfer?: StructuredSerializeOptions | Transferable[],
      ): void => {
        if (
          typeof message === "object" &&
          message !== null &&
          "type" in message &&
          message.type === "run-style-transfer"
        ) {
          state.runMessages += 1;
        }
        if (transfer === undefined) {
          originalPostMessage(message);
        } else {
          originalPostMessage(message, transfer);
        }
      };
      return worker;
    } as unknown as typeof Worker;
    window.Worker = WorkerReplacement;
  });
};

test("auto backend uses FastAPI run endpoint when health succeeds", async ({
  page,
}) => {
  await installBackendFetchMock(page, "healthy");
  await gotoStableApp(page);

  await expect(page.getByText(/Active backend: fastapi/i)).toBeVisible();
  await page.getByRole("button", { name: /Play/i }).click();
  await expect
    .poll(async () =>
      page.evaluate(
        () =>
          (
            window as Window & {
              __styleTransferBackendMock?: { runCalls: number };
            }
          ).__styleTransferBackendMock?.runCalls ?? 0,
      ),
    )
    .toBeGreaterThan(0);
  await page.getByRole("button", { name: /Pause/i }).click();

  await expect(page.getByText(/42\.2500/)).toBeVisible();
  await page.getByText("View stats").click();
  await expect(page.getByText(/Backend: fastapi\/pytorch/i)).toBeVisible();

  const firstRunBody = await page.evaluate(
    () =>
      (
        window as Window & {
          __styleTransferBackendMock?: { runBodies: Record<string, unknown>[] };
        }
      ).__styleTransferBackendMock?.runBodies[0],
  );
  expect(firstRunBody).toBeTruthy();
  expect(firstRunBody?.weights).toBeUndefined();
  expect(firstRunBody?.kernelFlags).toBeUndefined();
});

test("auto backend falls back to WebGPU status when FastAPI is unhealthy", async ({
  page,
}) => {
  await installBackendFetchMock(page, "unhealthy");
  await gotoStableApp(page);

  await expect(page.getByText(/Active backend: webgpu/i)).toBeVisible();
  await expect(page.getByText("FastAPI: mock backend offline")).toBeVisible();
  await expect(page.getByText(/using WebGPU worker/i)).toBeVisible();
});

test("auto backend rejects foreign health responses before running", async ({
  page,
}) => {
  await installBackendFetchMock(page, "foreign-health");
  await gotoStableApp(page);

  await expect(page.getByText(/Active backend: webgpu/i)).toBeVisible();
  await expect(
    page.getByText(
      "FastAPI: Health response did not identify FastAPI/PyTorch.",
    ),
  ).toBeVisible();
  const runCalls = await page.evaluate(
    () =>
      (
        window as Window & {
          __styleTransferBackendMock?: { runCalls: number };
        }
      ).__styleTransferBackendMock?.runCalls ?? 0,
  );
  expect(runCalls).toBe(0);
});

test("auto backend refuses non-loopback URLs before fetching", async ({
  page,
}) => {
  await page.addInitScript((storageKey) => {
    localStorage.setItem(
      storageKey,
      JSON.stringify({
        backendPreference: "auto",
        backendUrl: "https://example.com",
      }),
    );
    const originalFetch = window.fetch.bind(window);
    const state = { exampleFetches: 0 };
    (
      window as Window & {
        __nonLoopbackFetchMock?: typeof state;
      }
    ).__nonLoopbackFetchMock = state;
    window.fetch = async (...args: Parameters<typeof fetch>) => {
      const input = args[0];
      const url =
        typeof input === "string"
          ? input
          : input instanceof Request
            ? input.url
            : String(input);
      if (url.startsWith("https://example.com")) {
        state.exampleFetches += 1;
        return new Response(
          JSON.stringify({
            ok: true,
            backend: "fastapi",
            engine: "pytorch",
            device: "should-not-fetch",
            modelReady: true,
            message: "should not fetch",
          }),
          { headers: { "Content-Type": "application/json" } },
        );
      }
      return originalFetch(...args);
    };
  }, backendSettingsKey);
  await gotoStableApp(page);

  await expect(page.getByText(/Active backend: webgpu/i)).toBeVisible();
  await expect(
    page.getByText(
      "FastAPI: Backend URL must point to localhost or a loopback address.",
    ),
  ).toBeVisible();
  const exampleFetches = await page.evaluate(
    () =>
      (
        window as Window & {
          __nonLoopbackFetchMock?: { exampleFetches: number };
        }
      ).__nonLoopbackFetchMock?.exampleFetches ?? 0,
  );
  expect(exampleFetches).toBe(0);
});

test("auto backend falls back to WebGPU when run request fails", async ({
  page,
}) => {
  await installWorkerPostTracker(page);
  await installBackendFetchMock(page, "run-fails");
  await gotoStableApp(page);
  await expect
    .poll(async () => await page.locator("main").textContent(), {
      timeout: 15000,
    })
    .toMatch(/Loaded VGG19 weights pack/i);

  await page.getByRole("button", { name: /Play/i }).click();
  await expect
    .poll(async () =>
      page.evaluate(
        () =>
          (
            window as Window & {
              __styleTransferWorkerPosts?: { runMessages: number };
            }
          ).__styleTransferWorkerPosts?.runMessages ?? 0,
      ),
    )
    .toBeGreaterThan(0);
  await page.getByRole("button", { name: /Pause/i }).click();
  await expect(page.getByText(/Run failed with 503/i)).toBeVisible();
  await expect(page.getByText(/using WebGPU worker/i)).toBeVisible();
});

test("auto backend rejects malformed successful run responses", async ({
  page,
}) => {
  await installWorkerPostTracker(page);
  await installBackendFetchMock(page, "malformed-run");
  await gotoStableApp(page);
  await expect
    .poll(async () => await page.locator("main").textContent(), {
      timeout: 15000,
    })
    .toMatch(/Loaded VGG19 weights pack/i);

  await page.getByRole("button", { name: /Play/i }).click();
  await expect
    .poll(async () =>
      page.evaluate(
        () =>
          (
            window as Window & {
              __styleTransferWorkerPosts?: { runMessages: number };
            }
          ).__styleTransferWorkerPosts?.runMessages ?? 0,
      ),
    )
    .toBeGreaterThan(0);
  await page.getByRole("button", { name: /Pause/i }).click();
  await expect(
    page.getByText(/Backend run response was malformed/i),
  ).toBeVisible();
  await expect(page.getByText(/using WebGPU worker/i)).toBeVisible();
});

test("backend settings parse defaults and persist user choices", async ({
  page,
}) => {
  await installBackendFetchMock(page, "healthy");
  await page.addInitScript((storageKey) => {
    const seededKey = `${storageKey}.seeded`;
    if (sessionStorage.getItem(seededKey) === "true") return;
    localStorage.setItem(
      storageKey,
      JSON.stringify({ backendPreference: "bad", backendUrl: "" }),
    );
    sessionStorage.setItem(seededKey, "true");
  }, backendSettingsKey);
  await gotoStableApp(page);

  await page.getByRole("button", { name: "Show options" }).click();
  const backendSelect = page.getByLabel("Backend", { exact: true });
  const backendUrlInput = page.getByLabel("Backend URL");
  await expect(backendSelect).toHaveValue("auto");
  await expect(backendUrlInput).toHaveValue(defaultBackendUrl);

  await backendSelect.selectOption("webgpu");
  await backendUrlInput.fill("http://127.0.0.1:8999");
  await page.reload({ waitUntil: "load" });
  await page.getByRole("button", { name: "Show options" }).click();
  await expect(page.getByLabel("Backend", { exact: true })).toHaveValue(
    "webgpu",
  );
  await expect(page.getByLabel("Backend URL")).toHaveValue(
    "http://127.0.0.1:8999",
  );
});
