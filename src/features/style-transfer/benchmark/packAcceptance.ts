import type { VggPackName } from "../modelPacks";

export type PackComparisonRow = {
  pack: VggPackName;
  elapsedMs: number;
  avgStepMs: number;
  finalLoss: number;
  downloadSizeBytes: number;
};

export type PackAcceptanceRow = {
  pack: VggPackName;
  verdict: "pass" | "fail";
  reason: string;
};

export const buildPackAcceptanceRows = (
  comparisonRows: readonly PackComparisonRow[],
): PackAcceptanceRow[] => {
  const fp32Row = comparisonRows.find((row) => row.pack === "fp32");
  if (fp32Row === undefined) return [];
  return comparisonRows.map((row) => {
    if (row.pack === "fp32") {
      return { pack: row.pack, verdict: "pass", reason: "baseline" };
    }
    const delta = Math.abs(row.finalLoss - fp32Row.finalLoss);
    if (row.pack === "fp16") {
      const pass = delta < 1e-3;
      return {
        pack: row.pack,
        verdict: pass ? "pass" : "fail",
        reason: `|Δloss|=${delta.toExponential(2)} < 1e-3`,
      };
    }
    const thresholdByPack: Record<Exclude<VggPackName, "fp32" | "fp16">, number> = {
      "int8-per-channel": 5e-2,
      "int8log-per-channel": 7.5e-2,
      "int4-experimental": 1e-1,
      "int4log-experimental": 1.5e-1,
    };
    const threshold = thresholdByPack[row.pack];
    const pass = delta < threshold;
    return {
      pack: row.pack,
      verdict: pass ? "pass" : "fail",
      reason: `|Δloss|=${delta.toExponential(2)} < ${threshold.toExponential(1)}`,
    };
  });
};
