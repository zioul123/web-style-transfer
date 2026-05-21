/// <reference lib="webworker" />

import { getGpuDevice } from "./deviceState";
import { runUnaryShader } from "./shaderRunner";

/**
 * Runtime-level compute helper for unary shader execution.
 *
 * This lives under runtime so worker pipelines and protocol handlers can
 * share the same device-bound shader execution context without introducing
 * pipeline ownership coupling.
 */
export const runUnary = (
  code: string,
  input: Float32Array,
  outCount: number,
  extraEntries: GPUBindGroupEntry[] = [],
): Promise<Float32Array> =>
  runUnaryShader(getGpuDevice(), code, input, outCount, extraEntries);
