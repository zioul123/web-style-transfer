import { CoverageReport } from "monocart-coverage-reports";
import fs from "node:fs/promises";
import path from "node:path";

const coverageRoot = path.resolve(process.cwd(), "coverage");
const rawDir = path.join(coverageRoot, "raw");
const jsCoverageDir = path.join(coverageRoot, "js");
const rootIndexPath = path.join(coverageRoot, "index.html");
const summaryMarkdownPath = path.join(coverageRoot, "summary.md");
const dispatchJsonPath = path.join(coverageRoot, "dispatch-summary.json");
const dispatchMarkdownPath = path.join(coverageRoot, "dispatch-summary.md");

const readRawArtifacts = async () => {
  try {
    const names = await fs.readdir(rawDir);
    return Promise.all(
      names
        .filter((name) => name.endsWith(".json"))
        .map(async (name) =>
          JSON.parse(await fs.readFile(path.join(rawDir, name), "utf8")),
        ),
    );
  } catch (error) {
    if (error?.code === "ENOENT") return [];
    throw error;
  }
};

const toSourcePath = (url) => {
  const srcMatch = url.match(/\/src\/([^?#]+)/);
  return srcMatch === null ? url : `src/${srcMatch[1]}`;
};

const collectJsCoverage = (artifacts) =>
  artifacts
    .flatMap((artifact) => artifact.js ?? [])
    .filter((entry) => entry.source && entry.url.includes("/src/"));

const summarizeDispatchCoverage = (artifacts) => {
  const byKey = new Map();
  for (const artifact of artifacts) {
    for (const record of artifact.gpuDispatch ?? []) {
      const workgroups = record.workgroups ?? [1, 1, 1];
      const key = `${record.label}|${workgroups.join("x")}`;
      const existing = byKey.get(key) ?? {
        label: record.label,
        workgroups,
        count: 0,
        tests: new Set(),
      };
      existing.count += record.count ?? 0;
      existing.tests.add(artifact.title);
      byKey.set(key, existing);
    }
  }
  return Array.from(byKey.values())
    .map((record) => ({
      label: record.label,
      workgroups: record.workgroups,
      count: record.count,
      tests: Array.from(record.tests).sort(),
    }))
    .sort((a, b) => a.label.localeCompare(b.label));
};

const renderDispatchMarkdown = (records) =>
  `# GPU shader/path dispatch coverage\n\nThis report is test-only dispatch coverage. It intentionally does **not** claim WGSL line coverage; browser WebGPU APIs do not expose shader source-line execution coverage.\n\n| Pipeline/path label | Workgroups | Dispatches | Tests |\n| --- | --- | ---: | ---: |\n${records.map((record) => `| ${record.label} | ${record.workgroups.join(" × ")} | ${record.count} | ${record.tests.length} |`).join("\n")}\n`;

const renderIndexHtml = () => `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>Coverage reports</title>
<style>body { font-family: system-ui, sans-serif; margin: 2rem; line-height: 1.5; }</style>
</head>
<body>
<h1>Coverage reports</h1>
<ul>
<li><a href="./js/index.html">Chromium/V8 JavaScript coverage</a></li>
<li><a href="./dispatch-summary.md">GPU shader/path dispatch coverage</a></li>
</ul>
<p>GPU dispatch coverage is not WGSL line coverage.</p>
</body>
</html>`;

const writeSummaryMarkdown = async () => {
  const jsSummaryPath = path.join(jsCoverageDir, "summary.md");
  const jsSummary = await fs.readFile(jsSummaryPath, "utf8").catch(() => "");
  await fs.writeFile(
    summaryMarkdownPath,
    `${jsSummary.trimEnd()}\n\n## GPU shader/path dispatch coverage\n\nSee \`coverage/dispatch-summary.md\` and \`coverage/dispatch-summary.json\`. This is dispatch/path coverage only, not WGSL line coverage.\n`,
  );
};

const copyJsSummaryJson = async () => {
  await fs
    .copyFile(
      path.join(jsCoverageDir, "coverage-summary.json"),
      path.join(coverageRoot, "coverage-summary.json"),
    )
    .catch(() => undefined);
};

const artifacts = await readRawArtifacts();
await fs.mkdir(coverageRoot, { recursive: true });

const coverageReport = new CoverageReport({
  name: "Playwright Chromium JS Coverage",
  outputDir: jsCoverageDir,
  reports: [
    ["v8"],
    ["json-summary", { file: "coverage-summary.json" }],
    ["markdown-summary", { outputFile: "summary.md" }],
    ["lcovonly", { file: "lcov.info" }],
  ],
  entryFilter: (entry) => entry.url.includes("/src/"),
  sourcePath: (filePath) => toSourcePath(filePath),
});

await coverageReport.add(collectJsCoverage(artifacts));
await coverageReport.generate();

const dispatchSummary = summarizeDispatchCoverage(artifacts);
await fs.writeFile(dispatchJsonPath, JSON.stringify(dispatchSummary, null, 2));
await fs.writeFile(
  dispatchMarkdownPath,
  renderDispatchMarkdown(dispatchSummary),
);
await writeSummaryMarkdown();
await copyJsSummaryJson();
await fs.writeFile(rootIndexPath, renderIndexHtml());

console.log(`Wrote coverage artifacts to ${coverageRoot}`);
