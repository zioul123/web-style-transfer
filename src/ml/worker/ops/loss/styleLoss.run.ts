import { runGramMatrix, runGramBackward } from "../gram/gram.run";
import { runContentLossBackward } from "./contentLoss.run";

export const runStyleLossBackward = async (
  gpuDevice: GPUDevice | null,
  runUnaryShader: (
    code: string,
    input: Float32Array,
    outCount: number,
    extraEntries?: GPUBindGroupEntry[],
  ) => Promise<Float32Array>,
  input: Float32Array,
  shape: readonly [number, number, number, number],
  target: Float32Array,
  BUFFER_USAGE_STORAGE_COPY_DST: number,
  BUFFER_USAGE_UNIFORM_COPY_DST: number,
): Promise<Float32Array> => {
  const inputGram = await runGramMatrix(
    gpuDevice,
    runUnaryShader,
    input,
    shape,
    BUFFER_USAGE_UNIFORM_COPY_DST,
  );
  const targetGram = await runGramMatrix(
    gpuDevice,
    runUnaryShader,
    target,
    shape,
    BUFFER_USAGE_UNIFORM_COPY_DST,
  );
  const gramGrad = await runContentLossBackward(
    gpuDevice,
    runUnaryShader,
    inputGram,
    targetGram,
    BUFFER_USAGE_STORAGE_COPY_DST,
  );
  return runGramBackward(
    gpuDevice,
    runUnaryShader,
    input,
    shape,
    gramGrad,
    BUFFER_USAGE_STORAGE_COPY_DST,
    BUFFER_USAGE_UNIFORM_COPY_DST,
  );
};

export const runStyleLossBackwardFromTargetGram = async (
  gpuDevice: GPUDevice | null,
  runUnaryShader: (
    code: string,
    input: Float32Array,
    outCount: number,
    extraEntries?: GPUBindGroupEntry[],
  ) => Promise<Float32Array>,
  input: Float32Array,
  shape: readonly [number, number, number, number],
  targetGram: Float32Array,
  BUFFER_USAGE_STORAGE_COPY_DST: number,
  BUFFER_USAGE_UNIFORM_COPY_DST: number,
): Promise<Float32Array> => {
  const inputGram = await runGramMatrix(
    gpuDevice,
    runUnaryShader,
    input,
    shape,
    BUFFER_USAGE_UNIFORM_COPY_DST,
  );
  const gramGrad = await runContentLossBackward(
    gpuDevice,
    runUnaryShader,
    inputGram,
    targetGram,
    BUFFER_USAGE_STORAGE_COPY_DST,
  );
  return runGramBackward(
    gpuDevice,
    runUnaryShader,
    input,
    shape,
    gramGrad,
    BUFFER_USAGE_STORAGE_COPY_DST,
    BUFFER_USAGE_UNIFORM_COPY_DST,
  );
};
