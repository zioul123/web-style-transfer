import { runConv2dBackwardInputBuffer, runConv2dForwardBuffer, runConv2dReluForwardBuffer } from "../../../ops/convolution/conv2d.run";
import { runGramMatrixBuffer } from "../../../ops/gram/gram.run";
import { runContentLossBackwardBuffer } from "../../../ops/loss/contentLoss.run";
import { runMseBuffer, runMseBufferToScalarBuffer } from "../../../ops/loss/mse.run";
import { runStyleLossBackwardFromTargetGramBuffer } from "../../../ops/loss/styleLoss.run";
import { runNormalizeBackwardBuffer, runNormalizeForwardBuffer } from "../../../ops/normalization/normalize.run";
import { runMaxPool2dBackwardBuffer, runMaxPool2dForwardBuffer } from "../../../ops/pooling/maxpool.run";
import { runReluBackwardBuffer, runReluForwardBuffer } from "../../../ops/relu/relu.run";
import type { GpuBufferRef, OwnedGpuBuffer } from "../../../runtime/bufferKernels";
import { runBinaryOpToBuffer } from "../../../runtime/shaderRunner";
import type { FirstPoolRuntimeContext, TensorShape4D } from "./types";
import { runFusedUpdateClampBuffer, runScalarMulBuffer, runUnfusedUpdateClampBuffer } from "./updateKernels";

export type AcquireTempBuffer = (size: number, usage: number) => GPUBuffer;

type ReadbackMseArgs = {
  acquire: AcquireTempBuffer;
  a: GpuBufferRef;
  b: GpuBufferRef;
  count: number;
  reductionUsage: number;
  readUsage: number;
  mapModeRead: number;
};

type ScalarMseArgs = {
  acquire: AcquireTempBuffer;
  a: GpuBufferRef;
  b: GpuBufferRef;
  count: number;
  reductionUsage: number;
};

export type FirstPoolTrackedOps = ReturnType<typeof createFirstPoolTrackedOps>;

export const createFirstPoolTrackedOps = (
  device: GPUDevice,
  runtimeContext: FirstPoolRuntimeContext,
) => {
  const own = runtimeContext.trackOwned;

  const forward = {
    normalizeForward: async (input: GpuBufferRef, shape: TensorShape4D, mean: readonly number[], std: readonly number[]): Promise<GpuBufferRef> => own(await runNormalizeForwardBuffer(input, shape, mean, std)),
    conv2dForward: async (input: GpuBufferRef, inputShape: TensorShape4D, weights: Float32Array, weightShape: readonly [number, number, number, number], bias: readonly number[]): Promise<GpuBufferRef> => own(await runConv2dForwardBuffer(input, inputShape, weights, weightShape, bias)),
    conv2dReluForward: async (input: GpuBufferRef, inputShape: TensorShape4D, weights: Float32Array, weightShape: readonly [number, number, number, number], bias: readonly number[]): Promise<GpuBufferRef> => own(await runConv2dReluForwardBuffer(input, inputShape, weights, weightShape, bias)),
    reluForward: async (input: GpuBufferRef, count: number): Promise<GpuBufferRef> => own(await runReluForwardBuffer(input, count)),
    maxPoolForward: async (input: GpuBufferRef, shape: TensorShape4D): Promise<GpuBufferRef> => own(await runMaxPool2dForwardBuffer(input, shape)),
    gram: async (input: GpuBufferRef, shape: TensorShape4D): Promise<GpuBufferRef> => own(await runGramMatrixBuffer(input, shape)),
  };

  const loss = {
    mseScalarBuffer: async ({ acquire, a, b, count, reductionUsage }: ScalarMseArgs): Promise<GpuBufferRef> => own(await runMseBufferToScalarBuffer(device, acquire, a, b, count, reductionUsage)),
    mseScalarReadback: async ({ acquire, a, b, count, reductionUsage, readUsage, mapModeRead }: ReadbackMseArgs): Promise<number> => runMseBuffer(device, acquire, a, b, count, reductionUsage, readUsage, mapModeRead),
    scalarMul: (input: GpuBufferRef, count: number, scalar: number): GpuBufferRef => own(runScalarMulBuffer(device, input, count, scalar)),
    add: (a: GpuBufferRef, b: GpuBufferRef, count: number): GpuBufferRef => own({ buffer: runBinaryOpToBuffer(device, "add", a.buffer, b.buffer, count, "tensorTensor"), owned: true }),
  };

  const backward = {
    styleLossBackward: async (relu: GpuBufferRef, shape: TensorShape4D, targetGram: GpuBufferRef): Promise<GpuBufferRef> => own(await runStyleLossBackwardFromTargetGramBuffer(relu, shape, targetGram)),
    contentLossBackward: async (relu: GpuBufferRef, target: GpuBufferRef, count: number): Promise<GpuBufferRef> => own(await runContentLossBackwardBuffer(relu, target, count)),
    reluBackward: async (reluOut: GpuBufferRef, gradOut: GpuBufferRef, count: number): Promise<GpuBufferRef> => own(await runReluBackwardBuffer(reluOut, gradOut, count)),
    conv2dBackwardInput: async (gradOut: GpuBufferRef, inShape: TensorShape4D, weights: Float32Array, weightShape: readonly [number, number, number, number]): Promise<GpuBufferRef> => own(await runConv2dBackwardInputBuffer(gradOut, inShape, weights, weightShape)),
    maxPoolBackward: async (reluOut: GpuBufferRef, reluShape: TensorShape4D, gradPool: GpuBufferRef): Promise<GpuBufferRef> => own(await runMaxPool2dBackwardBuffer(reluOut, reluShape, gradPool)),
    normalizeBackward: async (gradNorm: GpuBufferRef, shape: TensorShape4D, std: readonly number[]): Promise<GpuBufferRef> => own(await runNormalizeBackwardBuffer(gradNorm, shape, std)),
  };

  const update = {
    updateClamp: (input: GpuBufferRef, grad: GpuBufferRef, count: number, learningRate: number, useFused: boolean): OwnedGpuBuffer => {
      if (useFused) return runFusedUpdateClampBuffer(device, input, grad, count, learningRate);
      const unfused = runUnfusedUpdateClampBuffer(device, input, grad, count, learningRate);
      own(...unfused.intermediates);
      return unfused.clamped;
    },
  };

  return { ...forward, ...loss, ...backward, ...update };
};
