import fs from "node:fs/promises";
import path from "node:path";

const coverageRoot = path.resolve(process.cwd(), "coverage");
const rawDir = path.join(coverageRoot, "raw");
const summaryJsonPath = path.join(coverageRoot, "coverage-summary.json");
const summaryMarkdownPath = path.join(coverageRoot, "summary.md");
const summaryHtmlPath = path.join(coverageRoot, "index.html");

const readRawArtifacts = async () => {
  try {
    const names = await fs.readdir(rawDir);
    const artifacts = await Promise.all(
      names
        .filter((name) => name.endsWith(".json"))
        .map(async (name) =>
          JSON.parse(await fs.readFile(path.join(rawDir, name), "utf8")),
        ),
    );
    return artifacts;
  } catch (error) {
    if (error?.code === "ENOENT") return [];
    throw error;
  }
};

const getLineOffsets = (source) => {
  const offsets = [0];
  for (let index = 0; index < source.length; index += 1) {
    if (source[index] === "\n") offsets.push(index + 1);
  }
  return offsets;
};

const offsetToLine = (offsets, offset) => {
  let low = 0;
  let high = offsets.length - 1;
  while (low <= high) {
    const mid = Math.floor((low + high) / 2);
    if (offsets[mid] <= offset) low = mid + 1;
    else high = mid - 1;
  }
  return Math.max(0, high);
};

const sourcePathFromUrl = (url) => {
  const srcMatch = url.match(/\/src\/([^?#]+)/);
  if (srcMatch) return `src/${srcMatch[1]}`;
  const testsMatch = url.match(/\/tests\/([^?#]+)/);
  if (testsMatch) return `tests/${testsMatch[1]}`;
  return url;
};

const summarizeJsCoverage = (artifacts) => {
  const byUrl = new Map();
  for (const artifact of artifacts) {
    for (const entry of artifact.js ?? []) {
      if (!entry.source || !entry.url.includes("/src/")) continue;
      const key = sourcePathFromUrl(entry.url);
      const offsets = getLineOffsets(entry.source);
      const existing = byUrl.get(key) ?? {
        path: key,
        totalLines: offsets.length,
        coveredLines: new Set(),
      };
      existing.totalLines = Math.max(existing.totalLines, offsets.length);
      const lineHits = new Array(offsets.length).fill(false);
      const ranges = (entry.functions ?? [])
        .flatMap((fn) => fn.ranges ?? [])
        .sort(
          (a, b) => b.endOffset - b.startOffset - (a.endOffset - a.startOffset),
        );
      for (const range of ranges) {
        const startLine = offsetToLine(offsets, range.startOffset);
        const endLine = offsetToLine(
          offsets,
          Math.max(range.startOffset, range.endOffset - 1),
        );
        for (let line = startLine; line <= endLine; line += 1) {
          lineHits[line] = range.count > 0;
        }
      }
      lineHits.forEach((covered, index) => {
        if (covered) existing.coveredLines.add(index + 1);
      });
      byUrl.set(key, existing);
    }
  }
  const files = Array.from(byUrl.values())
    .map((item) => ({
      path: item.path,
      coveredLines: item.coveredLines.size,
      totalLines: item.totalLines,
      percent:
        item.totalLines === 0
          ? 100
          : Number(
              ((item.coveredLines.size / item.totalLines) * 100).toFixed(2),
            ),
    }))
    .sort((a, b) => a.path.localeCompare(b.path));
  const totals = files.reduce(
    (acc, file) => ({
      coveredLines: acc.coveredLines + file.coveredLines,
      totalLines: acc.totalLines + file.totalLines,
    }),
    { coveredLines: 0, totalLines: 0 },
  );
  return {
    files,
    totals: {
      ...totals,
      percent:
        totals.totalLines === 0
          ? 100
          : Number(
              ((totals.coveredLines / totals.totalLines) * 100).toFixed(2),
            ),
    },
  };
};

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

const escapeHtml = (value) =>
  String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");

const makeMarkdown = (summary) =>
  `# Coverage summary\n\nGenerated from Playwright Chromium JS coverage plus test-only GPU dispatch recording. This report intentionally does not claim WGSL line coverage; shader coverage is represented as dispatched pipeline/path labels.\n\n## JavaScript/TypeScript line coverage\n\n- Covered lines: ${summary.js.totals.coveredLines}\n- Total lines: ${summary.js.totals.totalLines}\n- Line coverage: ${summary.js.totals.percent}%\n\n| File | Covered | Total | % |\n| --- | ---: | ---: | ---: |\n${summary.js.files.map((file) => `| ${file.path} | ${file.coveredLines} | ${file.totalLines} | ${file.percent}% |`).join("\n")}\n\n## GPU shader/path dispatch coverage\n\n| Pipeline/path label | Workgroups | Dispatches | Tests |\n| --- | --- | ---: | --- |\n${summary.gpuDispatch.map((record) => `| ${record.label} | ${record.workgroups.join(" × ")} | ${record.count} | ${record.tests.length} |`).join("\n")}\n`;

const makeHtml = (summary) => `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>Coverage summary</title>
<style>
body { font-family: system-ui, sans-serif; margin: 2rem; line-height: 1.45; }
table { border-collapse: collapse; width: 100%; margin: 1rem 0 2rem; }
th, td { border: 1px solid #ddd; padding: 0.4rem 0.55rem; text-align: left; }
th { background: #f3f4f6; }
.metric { font-size: 1.25rem; font-weight: 700; }
.note { background: #fff8db; border: 1px solid #f0d56a; padding: 0.75rem; }
</style>
</head>
<body>
<h1>Coverage summary</h1>
<p class="note">Generated from Playwright Chromium JS coverage plus test-only GPU dispatch recording. This report intentionally does not claim WGSL line coverage; shader coverage is represented as dispatched pipeline/path labels.</p>
<h2>JavaScript/TypeScript line coverage</h2>
<p class="metric">${summary.js.totals.percent}% (${summary.js.totals.coveredLines}/${summary.js.totals.totalLines} lines)</p>
<table><thead><tr><th>File</th><th>Covered</th><th>Total</th><th>%</th></tr></thead><tbody>
${summary.js.files.map((file) => `<tr><td>${escapeHtml(file.path)}</td><td>${file.coveredLines}</td><td>${file.totalLines}</td><td>${file.percent}%</td></tr>`).join("\n")}
</tbody></table>
<h2>GPU shader/path dispatch coverage</h2>
<table><thead><tr><th>Pipeline/path label</th><th>Workgroups</th><th>Dispatches</th><th>Tests</th></tr></thead><tbody>
${summary.gpuDispatch.map((record) => `<tr><td>${escapeHtml(record.label)}</td><td>${record.workgroups.join(" × ")}</td><td>${record.count}</td><td>${record.tests.length}</td></tr>`).join("\n")}
</tbody></table>
</body>
</html>`;

const artifacts = await readRawArtifacts();
await fs.mkdir(coverageRoot, { recursive: true });
const summary = {
  generatedAt: new Date().toISOString(),
  source: "Playwright Chromium JS coverage and WebGPU dispatch recording",
  limitations: [
    "No WGSL line coverage is reported.",
    "GPU shader/path coverage means a compute pipeline/path was dispatched by a test.",
    "Playwright coverage is Chromium/V8-specific and may differ from production browser behavior.",
  ],
  testCount: artifacts.length,
  js: summarizeJsCoverage(artifacts),
  gpuDispatch: summarizeDispatchCoverage(artifacts),
};
await fs.writeFile(summaryJsonPath, JSON.stringify(summary, null, 2));
await fs.writeFile(summaryMarkdownPath, makeMarkdown(summary));
await fs.writeFile(summaryHtmlPath, makeHtml(summary));
console.log(`Wrote coverage artifacts to ${coverageRoot}`);
