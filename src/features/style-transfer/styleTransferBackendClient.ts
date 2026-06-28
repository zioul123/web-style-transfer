import type {
  WorkerRunStats,
  WorkerRunStyleTransferRequest,
} from "../../types/worker-protocol/pipelines";

export type StyleTransferRuntimeBackend = "fastapi" | "webgpu";

export type FastApiHealthStatus = {
  readonly ok: boolean;
  readonly backend: "fastapi";
  readonly engine: "pytorch";
  readonly device: string;
  readonly modelReady: boolean;
  readonly message: string;
};

export type FastApiStyleTransferRunRequest = Omit<
  WorkerRunStyleTransferRequest,
  | "type"
  | "id"
  | "weights"
  | "kernelFlags"
  | "synchronizePhaseTimings"
  | "collectKernelStats"
>;

export type StyleTransferBackendRunResult =
  | {
      readonly ok: true;
      readonly losses: number[];
      readonly finalValues: number[];
      readonly stats: WorkerRunStats;
    }
  | { readonly ok: false; readonly message: string };

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value);

const fallbackHealth = (message: string): FastApiHealthStatus => ({
  ok: false,
  backend: "fastapi",
  engine: "pytorch",
  device: "unknown",
  modelReady: false,
  message,
});

type BackendEndpointResult =
  | { readonly ok: true; readonly url: string }
  | { readonly ok: false; readonly message: string };

const isIpv4Loopback = (hostname: string): boolean => {
  const parts = hostname.split(".");
  if (parts.length !== 4 || parts[0] !== "127") return false;
  return parts.every((part) => {
    if (!/^\d+$/.test(part)) return false;
    const value = Number(part);
    return value >= 0 && value <= 255;
  });
};

const isLoopbackHostname = (hostname: string): boolean =>
  hostname === "localhost" ||
  hostname === "::1" ||
  hostname === "[::1]" ||
  isIpv4Loopback(hostname);

const resolveBackendEndpoint = (
  baseUrl: string,
  path: string,
): BackendEndpointResult => {
  try {
    const base = new URL(baseUrl.trim());
    if (base.protocol !== "http:" && base.protocol !== "https:") {
      return {
        ok: false,
        message: "Backend URL must use http or https.",
      };
    }
    if (base.username.length > 0 || base.password.length > 0) {
      return {
        ok: false,
        message: "Backend URL must not include credentials.",
      };
    }
    if (!isLoopbackHostname(base.hostname)) {
      return {
        ok: false,
        message: "Backend URL must point to localhost or a loopback address.",
      };
    }
    const normalizedBase = base.href.endsWith("/")
      ? base.href
      : `${base.href}/`;
    return {
      ok: true,
      url: new URL(path.replace(/^\/+/, ""), normalizedBase).toString(),
    };
  } catch {
    return { ok: false, message: "Backend URL is invalid." };
  }
};

const readString = (
  record: Record<string, unknown>,
  key: string,
  fallback: string,
): string => {
  const value = record[key];
  return typeof value === "string" ? value : fallback;
};

const readBoolean = (
  record: Record<string, unknown>,
  key: string,
  fallback: boolean,
): boolean => {
  const value = record[key];
  return typeof value === "boolean" ? value : fallback;
};

const parseHealth = (value: unknown): FastApiHealthStatus => {
  if (!isRecord(value)) return fallbackHealth("Invalid health response.");
  if (value.backend !== "fastapi" || value.engine !== "pytorch") {
    return fallbackHealth("Health response did not identify FastAPI/PyTorch.");
  }
  return {
    ok: readBoolean(value, "ok", false),
    backend: "fastapi",
    engine: "pytorch",
    device: readString(value, "device", "unknown"),
    modelReady: readBoolean(value, "modelReady", false),
    message: readString(value, "message", "No backend health message."),
  };
};

const toFiniteNumberArray = (value: unknown): number[] | null => {
  if (!Array.isArray(value)) return null;
  if (
    !value.every((entry) => typeof entry === "number" && Number.isFinite(entry))
  ) {
    return null;
  }
  return value;
};

const parseStats = (
  value: unknown,
  expectedSteps: number,
): WorkerRunStats | null => {
  if (!isRecord(value)) return null;
  const elapsedMs = value.elapsedMs;
  const avgStepMs = value.avgStepMs;
  const forwardMs = value.forwardMs;
  const backwardMs = value.backwardMs;
  const lossMs = value.lossMs;
  const updateMs = value.updateMs;
  const steps = value.steps;
  if (
    typeof elapsedMs !== "number" ||
    typeof avgStepMs !== "number" ||
    typeof forwardMs !== "number" ||
    typeof backwardMs !== "number" ||
    typeof lossMs !== "number" ||
    typeof updateMs !== "number" ||
    typeof steps !== "number" ||
    ![
      elapsedMs,
      avgStepMs,
      forwardMs,
      backwardMs,
      lossMs,
      updateMs,
      steps,
    ].every(Number.isFinite) ||
    steps !== expectedSteps
  ) {
    return null;
  }
  return {
    elapsedMs,
    avgStepMs,
    forwardMs,
    backwardMs,
    lossMs,
    updateMs,
    steps,
  };
};

const tensorValueCount = (
  shape: FastApiStyleTransferRunRequest["inputShape"],
): number => shape.reduce((acc, dimension) => acc * dimension, 1);

const parseRunResponse = (
  value: unknown,
  request: FastApiStyleTransferRunRequest,
): StyleTransferBackendRunResult => {
  if (!isRecord(value)) return { ok: false, message: "Invalid run response." };
  if (value.ok === false) {
    return {
      ok: false,
      message: readString(value, "message", "Backend run failed."),
    };
  }
  if (value.ok !== true) {
    return { ok: false, message: "Backend response omitted ok=true." };
  }
  const losses = toFiniteNumberArray(value.losses);
  const finalValues = toFiniteNumberArray(value.finalValues);
  const stats = parseStats(value.stats, request.steps);
  if (
    losses === null ||
    finalValues === null ||
    finalValues.length !== tensorValueCount(request.inputShape) ||
    stats === null
  ) {
    return { ok: false, message: "Backend run response was malformed." };
  }
  return { ok: true, losses, finalValues, stats };
};

export const toFastApiStyleTransferRequest = ({
  sessionId,
  optimizer,
  adamBeta1,
  adamBeta2,
  adamEpsilon,
  lbfgsMemory,
  lbfgsEpsilon,
  inputShape,
  contentShape,
  styleShape,
  inputImageValues,
  contentImageValues,
  styleImageValues,
  mean,
  std,
  styleLayerIndices,
  contentLayerIndex,
  contentWeight,
  styleWeight,
  learningRate,
  steps,
  lossReadbackInterval,
}: WorkerRunStyleTransferRequest): FastApiStyleTransferRunRequest => ({
  sessionId,
  optimizer,
  adamBeta1,
  adamBeta2,
  adamEpsilon,
  lbfgsMemory,
  lbfgsEpsilon,
  inputShape,
  contentShape,
  styleShape,
  inputImageValues,
  contentImageValues,
  styleImageValues,
  mean,
  std,
  styleLayerIndices,
  contentLayerIndex,
  contentWeight,
  styleWeight,
  learningRate,
  steps,
  lossReadbackInterval,
});

export const probeFastApiBackend = async (
  baseUrl: string,
  timeoutMs = 1200,
): Promise<FastApiHealthStatus> => {
  const endpoint = resolveBackendEndpoint(baseUrl, "health");
  if (!endpoint.ok) return fallbackHealth(endpoint.message);
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(endpoint.url, {
      signal: controller.signal,
    });
    if (!response.ok) {
      return fallbackHealth(`Health check failed with ${response.status}.`);
    }
    return parseHealth(await response.json());
  } catch (error) {
    return fallbackHealth(
      error instanceof Error ? error.message : String(error),
    );
  } finally {
    clearTimeout(timeout);
  }
};

export const runFastApiStyleTransfer = async (
  baseUrl: string,
  request: FastApiStyleTransferRunRequest,
): Promise<StyleTransferBackendRunResult> => {
  const endpoint = resolveBackendEndpoint(baseUrl, "style-transfer/run");
  if (!endpoint.ok) return { ok: false, message: endpoint.message };
  try {
    const response = await fetch(endpoint.url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(request),
    });
    if (!response.ok) {
      return { ok: false, message: `Run failed with ${response.status}.` };
    }
    return parseRunResponse(await response.json(), request);
  } catch (error) {
    return {
      ok: false,
      message: error instanceof Error ? error.message : String(error),
    };
  }
};

export const clearFastApiStyleTransferSession = async (
  baseUrl: string,
  sessionId: string,
): Promise<void> => {
  const endpoint = resolveBackendEndpoint(
    baseUrl,
    "style-transfer/session/clear",
  );
  if (!endpoint.ok) return;
  try {
    await fetch(endpoint.url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sessionId }),
    });
  } catch {
    // Session clearing is best-effort; the next run can create fresh state.
  }
};
