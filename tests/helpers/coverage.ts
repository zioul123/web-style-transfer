import { expect, test as base } from "@playwright/test";
import fs from "node:fs/promises";
import path from "node:path";
type WorkerGpuDispatchCoverageRecord = {
  label: string;
  workgroups: readonly [number, number, number];
  count: number;
};

type CoverageBrowserWindow = Window & {
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

const collectDispatchCoverage = async (
  page: import("@playwright/test").Page,
): Promise<WorkerGpuDispatchCoverageRecord[]> =>
  page.evaluate(() => {
    const browserWindow: CoverageBrowserWindow = window;
    return browserWindow.__styleTransferDispatchCoverage ?? [];
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
        collectDispatchCoverage(page),
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
