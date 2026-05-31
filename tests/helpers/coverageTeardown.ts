import { execFile } from "node:child_process";
import { promisify } from "node:util";

const execFileAsync = promisify(execFile);

export default async function globalTeardown(): Promise<void> {
  await execFileAsync(
    process.execPath,
    ["scripts/generate-coverage-report.mjs"],
    {
      cwd: process.cwd(),
    },
  );
}
