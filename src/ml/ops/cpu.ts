import type { TensorData, TensorShape } from "../../types/worker-protocol/core";
import { assertSameLength } from "../tensor";

const indexNchw = (
  shape: TensorShape,
  n: number,
  c: number,
  h: number,
  w: number,
): number => ((n * shape[1] + c) * shape[2] + h) * shape[3] + w;

export const cpuAdd = (a: TensorData, b: TensorData): TensorData => {
  assertSameLength(a, b);
  const out: Float32Array = new Float32Array(a.values.length);
  for (let i = 0; i < out.length; i += 1) out[i] = a.values[i] + b.values[i];
  return { shape: a.shape, values: out };
};
export const cpuSub = (a: TensorData, b: TensorData): TensorData => {
  assertSameLength(a, b);
  const out = new Float32Array(a.values.length);
  for (let i = 0; i < out.length; i += 1) out[i] = a.values[i] - b.values[i];
  return { shape: a.shape, values: out };
};
export const cpuMul = (a: TensorData, b: TensorData): TensorData => {
  assertSameLength(a, b);
  const out = new Float32Array(a.values.length);
  for (let i = 0; i < out.length; i += 1) out[i] = a.values[i] * b.values[i];
  return { shape: a.shape, values: out };
};
export const cpuDiv = (a: TensorData, b: TensorData): TensorData => {
  assertSameLength(a, b);
  const out = new Float32Array(a.values.length);
  for (let i = 0; i < out.length; i += 1) out[i] = a.values[i] / b.values[i];
  return { shape: a.shape, values: out };
};
export const cpuClamp = (
  a: TensorData,
  min: number,
  max: number,
): TensorData => {
  const out = new Float32Array(a.values.length);
  for (let i = 0; i < out.length; i += 1)
    out[i] = Math.min(max, Math.max(min, a.values[i]));
  return { shape: a.shape, values: out };
};
export const cpuMse = (a: TensorData, b: TensorData): number => {
  assertSameLength(a, b);
  let sum = 0;
  for (let i = 0; i < a.values.length; i += 1) {
    const d = a.values[i] - b.values[i];
    sum += d * d;
  }
  return sum / a.values.length;
};

export const cpuReluForward = (input: TensorData): TensorData => {
  const out: Float32Array = new Float32Array(input.values.length);
  for (let i = 0; i < input.values.length; i += 1)
    out[i] = Math.max(0, input.values[i]);
  return { shape: input.shape, values: out };
};

export const cpuMaxPool2dForward = (input: TensorData): TensorData => {
  const [batch, channels, height, width] = input.shape;
  const outShape: TensorShape = [
    batch,
    channels,
    Math.floor(height / 2),
    Math.floor(width / 2),
  ];
  const out: Float32Array = new Float32Array(
    batch * channels * outShape[2] * outShape[3],
  );
  for (let n = 0; n < batch; n += 1)
    for (let c = 0; c < channels; c += 1)
      for (let oh = 0; oh < outShape[2]; oh += 1)
        for (let ow = 0; ow < outShape[3]; ow += 1) {
          let m = -Infinity;
          for (let kh = 0; kh < 2; kh += 1)
            for (let kw = 0; kw < 2; kw += 1) {
              const v =
                input.values[
                  indexNchw(input.shape, n, c, oh * 2 + kh, ow * 2 + kw)
                ];
              if (v > m) m = v;
            }
          out[indexNchw(outShape, n, c, oh, ow)] = m;
        }
  return { shape: outShape, values: out };
};

export const cpuNormalizeForward = (
  input: TensorData,
  mean: readonly number[],
  std: readonly number[],
): TensorData => {
  const [batch, channels, height, width] = input.shape;
  if (mean.length !== channels || std.length !== channels)
    throw new Error("Mean/std lengths must match input channels.");
  const out: Float32Array = new Float32Array(input.values.length);
  for (let n = 0; n < batch; n += 1)
    for (let c = 0; c < channels; c += 1)
      for (let h = 0; h < height; h += 1)
        for (let w = 0; w < width; w += 1) {
          const idx = indexNchw(input.shape, n, c, h, w);
          out[idx] = (input.values[idx] - mean[c]) / std[c];
        }
  return { shape: input.shape, values: out };
};

export const cpuConv2dForward = (
  input: TensorData,
  weight: TensorData,
  bias: readonly number[],
): TensorData => {
  const [batch, inChannels, height, width] = input.shape;
  const [outChannels, weightInChannels, kernelHeight, kernelWidth] =
    weight.shape;
  if (
    batch !== 1 ||
    kernelHeight !== 3 ||
    kernelWidth !== 3 ||
    inChannels !== weightInChannels ||
    bias.length !== outChannels
  )
    throw new Error("conv2d-forward only supports VGG-compatible params.");
  const outShape: TensorShape = [batch, outChannels, height, width];
  const out: Float32Array = new Float32Array(
    batch * outChannels * height * width,
  );
  for (let n = 0; n < batch; n += 1)
    for (let oc = 0; oc < outChannels; oc += 1)
      for (let oh = 0; oh < height; oh += 1)
        for (let ow = 0; ow < width; ow += 1) {
          let sum = bias[oc];
          for (let ic = 0; ic < inChannels; ic += 1)
            for (let kh = 0; kh < 3; kh += 1)
              for (let kw = 0; kw < 3; kw += 1) {
                const ih = oh + kh - 1;
                const iw = ow + kw - 1;
                if (ih < 0 || ih >= height || iw < 0 || iw >= width) continue;
                sum +=
                  input.values[indexNchw(input.shape, n, ic, ih, iw)] *
                  weight.values[indexNchw(weight.shape, oc, ic, kh, kw)];
              }
          out[indexNchw(outShape, n, oc, oh, ow)] = sum;
        }
  return { shape: outShape, values: out };
};

export const cpuReshapeChwFlatten = (input: TensorData): TensorData => {
  const [batch, channels, height, width] = input.shape;
  if (batch !== 1) throw new Error("reshape-chw-flatten expects batch size 1.");
  return {
    shape: [1, 1, channels, height * width],
    values: new Float32Array(input.values),
  };
};

export const cpuGramMatrix = (input: TensorData): TensorData => {
  const [batch, channels, height, width] = input.shape;
  if (batch !== 1) throw new Error("gram-matrix expects batch size 1.");
  const spatial: number = height * width;
  const out: Float32Array = new Float32Array(channels * channels);
  const norm: number = batch * channels * height * width;
  for (let row = 0; row < channels; row += 1)
    for (let col = 0; col < channels; col += 1) {
      let sum = 0;
      for (let i = 0; i < spatial; i += 1)
        sum +=
          input.values[row * spatial + i] * input.values[col * spatial + i];
      out[row * channels + col] = sum / norm;
    }
  return { shape: [1, 1, channels, channels], values: out };
};

export const cpuContentLoss = (
  input: TensorData,
  target: TensorData,
): number => {
  if (
    input.values.length !== target.values.length ||
    input.shape.some((v, i) => v !== target.shape[i])
  )
    return 0;
  return cpuMse(input, target);
};

export const cpuStyleLoss = (input: TensorData, target: TensorData): number => {
  const inputGram = cpuGramMatrix(input);
  const targetGram = cpuGramMatrix(target);
  return cpuMse(inputGram, targetGram);
};

export const cpuReluBackward = (
  input: TensorData,
  gradOut: TensorData,
): TensorData => {
  assertSameLength(input, gradOut);
  const out: Float32Array = new Float32Array(input.values.length);
  for (let i = 0; i < out.length; i += 1)
    out[i] = input.values[i] > 0 ? gradOut.values[i] : 0;
  return { shape: input.shape, values: out };
};

export const cpuMaxPool2dBackward = (
  input: TensorData,
  gradOut: TensorData,
): TensorData => {
  const [batch, channels, height, width] = input.shape;
  const outHeight: number = Math.floor(height / 2);
  const outWidth: number = Math.floor(width / 2);
  const expectedGradShape: TensorShape = [batch, channels, outHeight, outWidth];
  if (gradOut.shape.some((value, index) => value !== expectedGradShape[index]))
    throw new Error("maxpool2d-backward gradOut shape mismatch.");
  const gradIn: Float32Array = new Float32Array(input.values.length);
  for (let n = 0; n < batch; n += 1)
    for (let c = 0; c < channels; c += 1)
      for (let oh = 0; oh < outHeight; oh += 1)
        for (let ow = 0; ow < outWidth; ow += 1) {
          let maxH = oh * 2;
          let maxW = ow * 2;
          let maxVal = input.values[indexNchw(input.shape, n, c, maxH, maxW)];
          for (let kh = 0; kh < 2; kh += 1)
            for (let kw = 0; kw < 2; kw += 1) {
              const ih = oh * 2 + kh;
              const iw = ow * 2 + kw;
              const v = input.values[indexNchw(input.shape, n, c, ih, iw)];
              if (v > maxVal) {
                maxVal = v;
                maxH = ih;
                maxW = iw;
              }
            }
          gradIn[indexNchw(input.shape, n, c, maxH, maxW)] +=
            gradOut.values[indexNchw(gradOut.shape, n, c, oh, ow)];
        }
  return { shape: input.shape, values: gradIn };
};

export const cpuNormalizeBackward = (
  gradOut: TensorData,
  std: readonly number[],
): TensorData => {
  const [batch, channels, height, width] = gradOut.shape;
  if (std.length !== channels)
    throw new Error("Std length must match input channels.");
  const gradIn: Float32Array = new Float32Array(gradOut.values.length);
  for (let n = 0; n < batch; n += 1)
    for (let c = 0; c < channels; c += 1)
      for (let h = 0; h < height; h += 1)
        for (let w = 0; w < width; w += 1) {
          const idx = indexNchw(gradOut.shape, n, c, h, w);
          gradIn[idx] = gradOut.values[idx] / std[c];
        }
  return { shape: gradOut.shape, values: gradIn };
};

export const cpuConv2dBackwardInput = (
  inputShape: TensorShape,
  gradOut: TensorData,
  weight: TensorData,
): TensorData => {
  const [batch, inChannels, height, width] = inputShape;
  const [outChannels, weightInChannels, kernelHeight, kernelWidth] =
    weight.shape;
  if (
    batch !== 1 ||
    kernelHeight !== 3 ||
    kernelWidth !== 3 ||
    inChannels !== weightInChannels
  )
    throw new Error(
      "conv2d-backward-input only supports VGG-compatible params.",
    );
  const expectedGradShape: TensorShape = [batch, outChannels, height, width];
  if (gradOut.shape.some((value, index) => value !== expectedGradShape[index]))
    throw new Error("conv2d-backward-input gradOut shape mismatch.");
  const gradIn: Float32Array = new Float32Array(
    batch * inChannels * height * width,
  );
  for (let n = 0; n < batch; n += 1)
    for (let ic = 0; ic < inChannels; ic += 1)
      for (let ih = 0; ih < height; ih += 1)
        for (let iw = 0; iw < width; iw += 1) {
          let sum = 0;
          for (let oc = 0; oc < outChannels; oc += 1)
            for (let kh = 0; kh < 3; kh += 1)
              for (let kw = 0; kw < 3; kw += 1) {
                const oh = ih - kh + 1;
                const ow = iw - kw + 1;
                if (oh < 0 || oh >= height || ow < 0 || ow >= width) continue;
                sum +=
                  gradOut.values[indexNchw(gradOut.shape, n, oc, oh, ow)] *
                  weight.values[indexNchw(weight.shape, oc, ic, kh, kw)];
              }
          gradIn[indexNchw(inputShape, n, ic, ih, iw)] = sum;
        }
  return { shape: inputShape, values: gradIn };
};
