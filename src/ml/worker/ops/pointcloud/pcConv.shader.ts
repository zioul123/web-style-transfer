export const makePointCloudConvForwardShader = (count: number): string => `
struct PointCloudConvUniforms {
  pointCount: u32,
  inputChannels: u32,
  sampleCount: u32,
  neighborCount: u32,
  outputChannels: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
}
@group(0) @binding(0) var<storage, read> inputFeatures: array<f32>;
@group(0) @binding(1) var<storage, read> knnIndices: array<u32>;
@group(0) @binding(2) var<storage, read> knnWeights: array<f32>;
@group(0) @binding(3) var<storage, read> weightValues: array<f32>;
@group(0) @binding(4) var<storage, read> biasValues: array<f32>;
@group(0) @binding(5) var<storage, read> kernelIndexMap: array<u32>;
@group(0) @binding(6) var<uniform> uniforms: PointCloudConvUniforms;
@group(0) @binding(7) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= ${count}u) { return; }
  let outputChannel = i % uniforms.outputChannels;
  let sample = i / uniforms.outputChannels;
  var sum = biasValues[outputChannel];
  for (var kernel: u32 = 0u; kernel < 9u; kernel = kernel + 1u) {
    let mappedKernel = kernelIndexMap[kernel];
    let knnBase = (sample * 9u + kernel) * uniforms.neighborCount;
    for (var inputChannel: u32 = 0u; inputChannel < uniforms.inputChannels; inputChannel = inputChannel + 1u) {
      var weightedFeature = 0.0;
      for (var neighbor: u32 = 0u; neighbor < uniforms.neighborCount; neighbor = neighbor + 1u) {
        let knnIndex = knnBase + neighbor;
        let point = knnIndices[knnIndex];
        weightedFeature = weightedFeature + inputFeatures[point * uniforms.inputChannels + inputChannel] * knnWeights[knnIndex];
      }
      let weightIndex = (outputChannel * uniforms.inputChannels + inputChannel) * 9u + mappedKernel;
      sum = sum + weightedFeature * weightValues[weightIndex];
    }
  }
  out[i] = sum;
}`;

export const makePointCloudConvBackwardFeaturesShader = (
  count: number,
): string => `
struct PointCloudConvUniforms {
  pointCount: u32,
  inputChannels: u32,
  sampleCount: u32,
  neighborCount: u32,
  outputChannels: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
}
@group(0) @binding(0) var<storage, read> gradOutValues: array<f32>;
@group(0) @binding(1) var<storage, read> knnIndices: array<u32>;
@group(0) @binding(2) var<storage, read> knnWeights: array<f32>;
@group(0) @binding(3) var<storage, read> weightValues: array<f32>;
@group(0) @binding(4) var<storage, read> kernelIndexMap: array<u32>;
@group(0) @binding(5) var<uniform> uniforms: PointCloudConvUniforms;
@group(0) @binding(6) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= ${count}u) { return; }
  let point = i / uniforms.inputChannels;
  let inputChannel = i % uniforms.inputChannels;
  var sum = 0.0;
  for (var sample: u32 = 0u; sample < uniforms.sampleCount; sample = sample + 1u) {
    for (var kernel: u32 = 0u; kernel < 9u; kernel = kernel + 1u) {
      let mappedKernel = kernelIndexMap[kernel];
      let knnBase = (sample * 9u + kernel) * uniforms.neighborCount;
      for (var neighbor: u32 = 0u; neighbor < uniforms.neighborCount; neighbor = neighbor + 1u) {
        let knnIndex = knnBase + neighbor;
        if (knnIndices[knnIndex] != point) { continue; }
        let interpolationWeight = knnWeights[knnIndex];
        for (var outputChannel: u32 = 0u; outputChannel < uniforms.outputChannels; outputChannel = outputChannel + 1u) {
          let gradOutIndex = sample * uniforms.outputChannels + outputChannel;
          let weightIndex = (outputChannel * uniforms.inputChannels + inputChannel) * 9u + mappedKernel;
          sum = sum + gradOutValues[gradOutIndex] * weightValues[weightIndex] * interpolationWeight;
        }
      }
    }
  }
  out[i] = sum;
}`;

const pointCloudConvChunkUniforms = `
struct PointCloudConvChunkUniforms {
  pointCount: u32,
  inputChannels: u32,
  sampleCount: u32,
  neighborCount: u32,
  outputChannels: u32,
  batchStart: u32,
  batchSampleCount: u32,
  outputStart: u32,
}`;

export const makePointCloudConvInterpolateShader = (): string => `
${pointCloudConvChunkUniforms}
@group(0) @binding(0) var<storage, read> inputFeatures: array<f32>;
@group(0) @binding(1) var<storage, read> knnIndices: array<u32>;
@group(0) @binding(2) var<storage, read> knnWeights: array<f32>;
@group(0) @binding(3) var<uniform> uniforms: PointCloudConvChunkUniforms;
@group(0) @binding(4) var<storage, read_write> interpolated: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  let count = uniforms.batchSampleCount * 9u * uniforms.inputChannels;
  if (i >= count) { return; }
  let inputChannel = i % uniforms.inputChannels;
  let sampleKernel = i / uniforms.inputChannels;
  let localSample = sampleKernel / 9u;
  let kernel = sampleKernel - localSample * 9u;
  let sample = uniforms.batchStart + localSample;
  let knnBase = (sample * 9u + kernel) * uniforms.neighborCount;
  var sum = 0.0;
  for (var neighbor: u32 = 0u; neighbor < uniforms.neighborCount; neighbor = neighbor + 1u) {
    let knnIndex = knnBase + neighbor;
    let point = knnIndices[knnIndex];
    sum = sum + inputFeatures[point * uniforms.inputChannels + inputChannel] * knnWeights[knnIndex];
  }
  interpolated[i] = sum;
}`;

export const makePointCloudConvProjectShader = (): string => `
${pointCloudConvChunkUniforms}
@group(0) @binding(0) var<storage, read> interpolated: array<f32>;
@group(0) @binding(1) var<storage, read> weightValues: array<f32>;
@group(0) @binding(2) var<storage, read> biasValues: array<f32>;
@group(0) @binding(3) var<storage, read> kernelIndexMap: array<u32>;
@group(0) @binding(4) var<uniform> uniforms: PointCloudConvChunkUniforms;
@group(0) @binding(5) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  let count = uniforms.batchSampleCount * uniforms.outputChannels;
  if (i >= count) { return; }
  let outputChannel = i % uniforms.outputChannels;
  let localSample = i / uniforms.outputChannels;
  var sum = biasValues[outputChannel];
  for (var kernel: u32 = 0u; kernel < 9u; kernel = kernel + 1u) {
    let mappedKernel = kernelIndexMap[kernel];
    for (var inputChannel: u32 = 0u; inputChannel < uniforms.inputChannels; inputChannel = inputChannel + 1u) {
      let interpolatedIndex = (localSample * 9u + kernel) * uniforms.inputChannels + inputChannel;
      let weightIndex = (outputChannel * uniforms.inputChannels + inputChannel) * 9u + mappedKernel;
      sum = sum + interpolated[interpolatedIndex] * weightValues[weightIndex];
    }
  }
  out[(uniforms.batchStart + localSample) * uniforms.outputChannels + outputChannel] = sum;
}`;

export const makePointCloudConvBackwardProjectShader = (): string => `
${pointCloudConvChunkUniforms}
@group(0) @binding(0) var<storage, read> gradOutValues: array<f32>;
@group(0) @binding(1) var<storage, read> weightValues: array<f32>;
@group(0) @binding(2) var<storage, read> kernelIndexMap: array<u32>;
@group(0) @binding(3) var<uniform> uniforms: PointCloudConvChunkUniforms;
@group(0) @binding(4) var<storage, read_write> projected: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  let count = uniforms.batchSampleCount * 9u * uniforms.inputChannels;
  if (i >= count) { return; }
  let inputChannel = i % uniforms.inputChannels;
  let sampleKernel = i / uniforms.inputChannels;
  let localSample = sampleKernel / 9u;
  let kernel = sampleKernel - localSample * 9u;
  let sample = uniforms.batchStart + localSample;
  let mappedKernel = kernelIndexMap[kernel];
  var sum = 0.0;
  for (var outputChannel: u32 = 0u; outputChannel < uniforms.outputChannels; outputChannel = outputChannel + 1u) {
    let gradOutIndex = sample * uniforms.outputChannels + outputChannel;
    let weightIndex = (outputChannel * uniforms.inputChannels + inputChannel) * 9u + mappedKernel;
    sum = sum + gradOutValues[gradOutIndex] * weightValues[weightIndex];
  }
  projected[i] = sum;
}`;

export const makePointCloudConvReverseGatherShader = (): string => `
${pointCloudConvChunkUniforms}
@group(0) @binding(0) var<storage, read> projected: array<f32>;
@group(0) @binding(1) var<storage, read> reverseOffsets: array<u32>;
@group(0) @binding(2) var<storage, read> reverseSampleKernel: array<u32>;
@group(0) @binding(3) var<storage, read> reverseWeights: array<f32>;
@group(0) @binding(4) var<uniform> uniforms: PointCloudConvChunkUniforms;
@group(0) @binding(5) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = uniforms.outputStart + gid.x;
  let count = uniforms.pointCount * uniforms.inputChannels;
  if (i >= count) { return; }
  let point = i / uniforms.inputChannels;
  let inputChannel = i % uniforms.inputChannels;
  let batchEnd = uniforms.batchStart + uniforms.batchSampleCount;
  var sum = out[i];
  let entryEnd = reverseOffsets[point + 1u];
  for (var entry = reverseOffsets[point]; entry < entryEnd; entry = entry + 1u) {
    let globalSampleKernel = reverseSampleKernel[entry];
    let sample = globalSampleKernel / 9u;
    if (sample < uniforms.batchStart || sample >= batchEnd) { continue; }
    let kernel = globalSampleKernel - sample * 9u;
    let localSample = sample - uniforms.batchStart;
    let projectedIndex = (localSample * 9u + kernel) * uniforms.inputChannels + inputChannel;
    sum = sum + projected[projectedIndex] * reverseWeights[entry];
  }
  out[i] = sum;
}`;

export const makePointCloudConvZeroRangeShader = (): string => `
struct PointCloudConvRangeUniforms {
  start: u32,
  count: u32,
  _pad0: u32,
  _pad1: u32,
}
@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@group(0) @binding(1) var<uniform> uniforms: PointCloudConvRangeUniforms;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let localIndex = gid.x;
  if (localIndex >= uniforms.count) { return; }
  out[uniforms.start + localIndex] = 0.0;
}`;

export const makePointCloudConvZeroShader = (count: number): string => `
@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= ${count}u) { return; }
  out[i] = 0.0;
}`;
