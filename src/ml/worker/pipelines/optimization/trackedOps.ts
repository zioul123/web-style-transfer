import {
  runConv2dBackwardInputBuffer,
  runConv2dForwardBuffer,
  runConv2dReluForwardBuffer,
  type Conv2dBackwardStaticBuffers,
  type Conv2dForwardStaticBuffers,
} from "../../ops/convolution/conv2d.run";
import { runGramMatrixBuffer } from "../../ops/gram/gram.run";
import { runContentLossBackwardBuffer } from "../../ops/loss/contentLoss.run";
import { runMseBufferToScalarBuffer } from "../../ops/loss/mse.run";
import {
  runStyleLossTermsFromTargetGramBuffer,
  runStyleLossBackwardFusedFromGramDiffBuffer,
  runStyleLossBackwardFromInputGramBuffer,
  runStyleLossBackwardFromTargetGramBuffer,
} from "../../ops/loss/styleLoss.run";
import { runWeightedScalarSumBuffer } from "../../ops/loss/weightedScalarSum.run";
import {
  runNormalizeBackwardBuffer,
  runNormalizeForwardBuffer,
} from "../../ops/normalization/normalize.run";
import {
  runMaxPool2dBackwardBuffer,
  runMaxPool2dForwardBuffer,
} from "../../ops/pooling/maxpool.run";
import {
  runReluBackwardBuffer,
  runReluForwardBuffer,
} from "../../ops/relu/relu.run";
import {
  setKernelRuntimeOptions,
  type GpuBufferRef,
  type OwnedGpuBuffer,
} from "../../runtime/bufferKernels";
import { BUFFER_USAGE_STORAGE_COPY_SRC } from "../../runtime/gpuFlags";
import type {
  OptimizationRuntimeContext,
  OptimizationShape4D,
} from "../../runtime/optimizationContext";
import { runBinaryOpToBuffer } from "../../runtime/shaderRunner";
import {
  runFusedUpdateClampBuffer,
  runScalarMulBuffer,
  runUnfusedUpdateClampBuffer,
} from "./updateKernels";
import type {
  WorkerKernelFamily,
  WorkerKernelOptimizationFlags,
  WorkerKernelStats,
} from "../../../../types";

export type TensorShape4D = OptimizationShape4D;

export type OptimizationTrackedOps = ReturnType<
  typeof createOptimizationTrackedOps
>;
export type OptimizationPersistentKernelResources = {
  getConvForwardStaticBuffers: (
    weight: Float32Array,
    bias: Float32Array,
  ) => Conv2dForwardStaticBuffers;
  getConvBackwardStaticBuffers: (
    weight: Float32Array,
    weightShape: TensorShape4D,
  ) => Conv2dBackwardStaticBuffers;
};

export const createOptimizationTrackedOps = (
  device: GPUDevice,
  runtimeContext: OptimizationRuntimeContext,
  kernelFlags?: WorkerKernelOptimizationFlags,
  collectKernelStats?: boolean,
  persistentKernelResources?: OptimizationPersistentKernelResources,
) => {
  setKernelRuntimeOptions({
    useCachedPipelines: kernelFlags?.useCachedPipelines ?? false,
  });

  const kernelStats: WorkerKernelStats = {};
  const recordKernelStat = (
    family: WorkerKernelFamily,
    elapsedMs: number,
  ): void => {
    const existing = kernelStats[family];
    if (existing === undefined) {
      kernelStats[family] = { calls: 1, elapsedMs };
    } else {
      existing.calls += 1;
      existing.elapsedMs += elapsedMs;
    }
  };

  const withKernelStat = async <T>(
    family: WorkerKernelFamily,
    run: () => Promise<T>,
  ): Promise<T> => {
    if (!collectKernelStats) return run();
    const start = performance.now();
    const result = await run();
    recordKernelStat(family, performance.now() - start);
    return result;
  };

  const withKernelStatSync = <T>(
    family: WorkerKernelFamily,
    run: () => T,
  ): T => {
    if (!collectKernelStats) return run();
    const start = performance.now();
    const result = run();
    recordKernelStat(family, performance.now() - start);
    return result;
  };

  const own = runtimeContext.trackOwned;
  const acquireMseBuffer = (size: number, usage: number): GPUBuffer => {
    const floatCount = Math.ceil(size / Float32Array.BYTES_PER_ELEMENT);
    return runtimeContext.acquireTemp(
      [1, 1, 1, floatCount],
      usage,
      "mse-reduction",
    );
  };

  const forward = {
    normalizeForward: async (
      input: GpuBufferRef,
      shape: TensorShape4D,
      mean: readonly number[],
      std: readonly number[],
    ): Promise<GpuBufferRef> =>
      withKernelStat("pointwiseUpdate", async () =>
        own(
          await runNormalizeForwardBuffer(
            input,
            shape,
            mean,
            std,
            kernelFlags?.useVec4Pointwise === true,
          ),
        ),
      ),
    conv2dForward: async (
      input: GpuBufferRef,
      inputShape: TensorShape4D,
      weights: Float32Array,
      weightShape: TensorShape4D,
      bias: ArrayLike<number>,
    ): Promise<GpuBufferRef> =>
      withKernelStat("convForward", async () =>
        own(
          await runConv2dForwardBuffer(
            input,
            inputShape,
            weights,
            weightShape,
            bias,
            kernelFlags?.usePersistentWeightBuffers === true &&
              bias instanceof Float32Array
              ? persistentKernelResources?.getConvForwardStaticBuffers(
                  weights,
                  bias,
                )
              : undefined,
            kernelFlags?.convForwardKernel ?? "scalar",
          ),
        ),
      ),
    conv2dReluForward: async (
      input: GpuBufferRef,
      inputShape: TensorShape4D,
      weights: Float32Array,
      weightShape: TensorShape4D,
      bias: ArrayLike<number>,
    ): Promise<GpuBufferRef> =>
      withKernelStat("convForward", async () =>
        own(
          await runConv2dReluForwardBuffer(
            input,
            inputShape,
            weights,
            weightShape,
            bias,
            kernelFlags?.usePersistentWeightBuffers === true &&
              bias instanceof Float32Array
              ? persistentKernelResources?.getConvForwardStaticBuffers(
                  weights,
                  bias,
                )
              : undefined,
            kernelFlags?.convForwardKernel ?? "scalar",
          ),
        ),
      ),
    reluForward: async (
      input: GpuBufferRef,
      count: number,
    ): Promise<GpuBufferRef> =>
      withKernelStat("pointwiseUpdate", async () =>
        own(
          await runReluForwardBuffer(
            input,
            count,
            kernelFlags?.useVec4Pointwise === true,
          ),
        ),
      ),
    maxPoolForward: async (
      input: GpuBufferRef,
      shape: TensorShape4D,
    ): Promise<GpuBufferRef> =>
      own(await runMaxPool2dForwardBuffer(input, shape)),
    gram: async (
      input: GpuBufferRef,
      shape: TensorShape4D,
    ): Promise<GpuBufferRef> =>
      withKernelStat("gramStyle", async () =>
        own(
          await runGramMatrixBuffer(
            input,
            shape,
            kernelFlags?.gramKernel ?? "scalar",
          ),
        ),
      ),
  };

  const loss = {
    mseScalarBuffer: async (
      a: GpuBufferRef,
      b: GpuBufferRef,
      count: number,
    ): Promise<GpuBufferRef> =>
      withKernelStat("gramStyle", async () =>
        own(
          await runMseBufferToScalarBuffer(
            device,
            acquireMseBuffer,
            a,
            b,
            count,
            BUFFER_USAGE_STORAGE_COPY_SRC,
          ),
        ),
      ),
    scalarMul: (
      input: GpuBufferRef,
      count: number,
      scalar: number,
    ): GpuBufferRef =>
      withKernelStatSync("pointwiseUpdate", () =>
        own(
          runScalarMulBuffer(
            device,
            input,
            count,
            scalar,
            kernelFlags?.useVec4Pointwise === true,
          ),
        ),
      ),
    add: (a: GpuBufferRef, b: GpuBufferRef, count: number): GpuBufferRef =>
      withKernelStatSync("pointwiseUpdate", () =>
        own({
          buffer: runBinaryOpToBuffer(
            device,
            "add",
            a.buffer,
            b.buffer,
            count,
            "tensorTensor",
            kernelFlags?.useVec4Pointwise === true,
          ),
          owned: true,
        }),
      ),
    weightedScalarSum: (
      scalars: readonly GpuBufferRef[],
      weights: readonly number[],
    ): GpuBufferRef => own(runWeightedScalarSumBuffer(scalars, weights)),
    styleLossTerms: async (
      relu: GpuBufferRef,
      shape: TensorShape4D,
      targetGram: GpuBufferRef,
    ): Promise<{ inputGram: GpuBufferRef; styleSse: GpuBufferRef }> => {
      const terms = await withKernelStat("gramStyle", async () =>
        runStyleLossTermsFromTargetGramBuffer(
          acquireMseBuffer,
          relu,
          shape,
          targetGram,
          kernelFlags?.gramKernel ?? "scalar",
        ),
      );
      return { inputGram: own(terms.inputGram), styleSse: own(terms.styleSse) };
    },
  };

  const backward = {
    styleLossBackward: async (
      relu: GpuBufferRef,
      shape: TensorShape4D,
      targetGram: GpuBufferRef,
    ): Promise<GpuBufferRef> =>
      withKernelStat("gramStyle", async () =>
        own(
          await runStyleLossBackwardFromTargetGramBuffer(
            relu,
            shape,
            targetGram,
            kernelFlags?.gramKernel ?? "scalar",
          ),
        ),
      ),
    styleLossBackwardFromInputGram: async (
      relu: GpuBufferRef,
      shape: TensorShape4D,
      inputGram: GpuBufferRef,
      targetGram: GpuBufferRef,
    ): Promise<GpuBufferRef> =>
      withKernelStat("gramStyle", async () =>
        own(
          kernelFlags?.styleBackward === "fused-from-gram-diff"
            ? await runStyleLossBackwardFusedFromGramDiffBuffer(
                relu,
                shape,
                inputGram,
                targetGram,
              )
            : await runStyleLossBackwardFromInputGramBuffer(
                relu,
                shape,
                inputGram,
                targetGram,
              ),
        ),
      ),
    contentLossBackward: async (
      relu: GpuBufferRef,
      target: GpuBufferRef,
      count: number,
    ): Promise<GpuBufferRef> =>
      withKernelStat("pointwiseUpdate", async () =>
        own(
          await runContentLossBackwardBuffer(
            relu,
            target,
            count,
            kernelFlags?.useVec4Pointwise === true,
          ),
        ),
      ),
    reluBackward: async (
      reluOut: GpuBufferRef,
      gradOut: GpuBufferRef,
      count: number,
    ): Promise<GpuBufferRef> =>
      withKernelStat("pointwiseUpdate", async () =>
        own(
          await runReluBackwardBuffer(
            reluOut,
            gradOut,
            count,
            kernelFlags?.useVec4Pointwise === true,
          ),
        ),
      ),
    conv2dBackwardInput: async (
      gradOut: GpuBufferRef,
      inShape: TensorShape4D,
      weights: Float32Array,
      weightShape: TensorShape4D,
    ): Promise<GpuBufferRef> =>
      withKernelStat("convBackwardInput", async () =>
        own(
          await runConv2dBackwardInputBuffer(
            gradOut,
            inShape,
            weights,
            weightShape,
            kernelFlags?.usePersistentWeightBuffers === true
              ? persistentKernelResources?.getConvBackwardStaticBuffers(
                  weights,
                  weightShape,
                )
              : undefined,
            kernelFlags?.convBackwardInputKernel ?? "scalar",
          ),
        ),
      ),
    maxPoolBackward: async (
      reluOut: GpuBufferRef,
      reluShape: TensorShape4D,
      gradPool: GpuBufferRef,
    ): Promise<GpuBufferRef> =>
      withKernelStat("poolBackward", async () =>
        own(
          await runMaxPool2dBackwardBuffer(
            reluOut,
            reluShape,
            gradPool,
            kernelFlags?.usePoolBackwardScatter === true,
          ),
        ),
      ),
    normalizeBackward: async (
      gradNorm: GpuBufferRef,
      shape: TensorShape4D,
      std: readonly number[],
    ): Promise<GpuBufferRef> =>
      withKernelStat("pointwiseUpdate", async () =>
        own(
          await runNormalizeBackwardBuffer(
            gradNorm,
            shape,
            std,
            kernelFlags?.useVec4Pointwise === true,
          ),
        ),
      ),
  };

  const update = {
    updateClamp: (
      input: GpuBufferRef,
      grad: GpuBufferRef,
      count: number,
      learningRate: number,
      useFused: boolean,
    ): OwnedGpuBuffer => {
      return withKernelStatSync("pointwiseUpdate", () => {
        if (useFused) {
          return runFusedUpdateClampBuffer(
            device,
            input,
            grad,
            count,
            learningRate,
            kernelFlags?.useVec4Pointwise === true,
          );
        }
        const unfused = runUnfusedUpdateClampBuffer(
          device,
          input,
          grad,
          count,
          learningRate,
          kernelFlags?.useVec4Pointwise === true,
        );
        own(...unfused.intermediates);
        return unfused.clamped;
      });
    },
  };

  return {
    ...forward,
    ...loss,
    ...backward,
    ...update,
    getKernelStats: () => kernelStats,
  };
};
