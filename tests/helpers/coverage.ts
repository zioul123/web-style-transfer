import { expect, test as base } from "@playwright/test";
import fs from "node:fs/promises";
import path from "node:path";
import type { WorkerGpuDispatchCoverageRecord } from "../../src/types";

type CoverageWorkerClientEntry = {
  worker: Worker;
};

type CoverageBrowserWindow = Window & {
  __styleTransferWorkerClients?: Record<string, CoverageWorkerClientEntry>;
  __styleTransferDispatchCoverage?: WorkerGpuDispatchCoverageRecord[];
};

type ChromiumJsCoverageEntry = Awaited<
  ReturnType<import("@playwright/test").Page["coverage"]["stopJSCoverage"]>
>[number];

type TestCoverageArtifact = {
  testId: string;
  title: string;
  status: string | undefined;
  js: ChromiumJsCoverageEntry[];
  gpuDispatch: WorkerGpuDispatchCoverageRecord[];
};

const coverageRoot = path.resolve(process.cwd(), "coverage");
const rawCoverageDir = path.join(coverageRoot, "raw");

const sanitizeSegment = (value: string): string =>
  value
    .replace(/[^a-zA-Z0-9._-]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 160) || "test";

const getArtifactPath = (testId: string): string =>
  path.join(rawCoverageDir, `${sanitizeSegment(testId)}.json`);

const collectRegisteredWorkerDispatchCoverage = async (
  page: import("@playwright/test").Page,
): Promise<WorkerGpuDispatchCoverageRecord[]> =>
  page.evaluate(async () => {
    const browserWindow: CoverageBrowserWindow = window;
    const records: WorkerGpuDispatchCoverageRecord[] = [
      ...(browserWindow.__styleTransferDispatchCoverage ?? []),
    ];
    const entries = Object.entries(
      browserWindow.__styleTransferWorkerClients ?? {},
    );
    await Promise.all(
      entries.map(
        async ([clientId, entry]): Promise<void> =>
          new Promise<void>((resolve) => {
            const requestId = `coverage-${clientId}-${Date.now()}`;
            const timeoutId = window.setTimeout(resolve, 500);
            const handler = (event: MessageEvent): void => {
              if (
                event.data?.type !== "gpu-dispatch-coverage-result" ||
                event.data.id !== requestId
              ) {
                return;
              }
              window.clearTimeout(timeoutId);
              entry.worker.removeEventListener("message", handler);
              if (Array.isArray(event.data.records)) {
                records.push(...event.data.records);
              }
              resolve();
            };
            entry.worker.addEventListener("message", handler);
            entry.worker.postMessage({
              type: "get-gpu-dispatch-coverage",
              id: requestId,
            });
          }),
      ),
    );
    return records;
  });

export const test = base.extend<{ page: import("@playwright/test").Page }>({
  page: async ({ page }, applyPage, testInfo) => {
    await fs.mkdir(rawCoverageDir, { recursive: true });
    await page.coverage.startJSCoverage({ resetOnNavigation: false });
    try {
      await applyPage(page);
    } finally {
      const [js, gpuDispatch] = await Promise.all([
        page.coverage.stopJSCoverage(),
        collectRegisteredWorkerDispatchCoverage(page),
      ]);
      const artifact: TestCoverageArtifact = {
        testId: testInfo.testId,
        title: testInfo.titlePath.join(" › "),
        status: testInfo.status,
        js,
        gpuDispatch,
      };
      await fs.writeFile(
        getArtifactPath(testInfo.testId),
        JSON.stringify(artifact, null, 2),
      );
    }
  },
});

export { expect };
