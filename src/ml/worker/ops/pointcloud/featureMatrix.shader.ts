export const makePointwiseExpForwardShader = (count: number): string => `
@group(0) @binding(0) var<storage, read> inputValues: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= ${count}u) { return; }
  out[i] = exp(inputValues[i]);
}`;

export const makePointwiseExpBackwardShader = (count: number): string => `
@group(0) @binding(0) var<storage, read> inputValues: array<f32>;
@group(0) @binding(1) var<storage, read> gradOutValues: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= ${count}u) { return; }
  out[i] = gradOutValues[i] * exp(inputValues[i]);
}`;

export const makePointFeatureNormalizeForwardShader = (
  count: number,
): string => `
struct FeatureMatrixUniforms {
  pointCount: u32,
  channelCount: u32,
  _pad0: u32,
  _pad1: u32,
}
@group(0) @binding(0) var<storage, read> inputValues: array<f32>;
@group(0) @binding(1) var<storage, read> meanValues: array<f32>;
@group(0) @binding(2) var<storage, read> stdValues: array<f32>;
@group(0) @binding(3) var<uniform> uniforms: FeatureMatrixUniforms;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= ${count}u) { return; }
  let channel = i % uniforms.channelCount;
  out[i] = (inputValues[i] - meanValues[channel]) / stdValues[channel];
}`;

export const makePointFeatureNormalizeBackwardShader = (
  count: number,
): string => `
struct FeatureMatrixUniforms {
  pointCount: u32,
  channelCount: u32,
  _pad0: u32,
  _pad1: u32,
}
@group(0) @binding(0) var<storage, read> gradOutValues: array<f32>;
@group(0) @binding(1) var<storage, read> stdValues: array<f32>;
@group(0) @binding(2) var<uniform> uniforms: FeatureMatrixUniforms;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= ${count}u) { return; }
  let channel = i % uniforms.channelCount;
  out[i] = gradOutValues[i] / stdValues[channel];
}`;
