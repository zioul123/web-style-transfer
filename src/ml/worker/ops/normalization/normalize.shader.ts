export const makeNormalizeShader = (count: number): string => `
struct NormalizeUniforms {
  channels: u32,
  height: u32,
  width: u32,
}
@group(0) @binding(0) var<storage, read> inputValues: array<f32>;
@group(0) @binding(1) var<storage, read> meanValues: array<f32>;
@group(0) @binding(2) var<storage, read> stdValues: array<f32>;
@group(0) @binding(3) var<uniform> uniforms: NormalizeUniforms;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= ${count}u) { return; }
  let spatial = uniforms.height * uniforms.width;
  let c = (i / spatial) % uniforms.channels;
  out[i] = (inputValues[i] - meanValues[c]) / stdValues[c];
}`;

export const makeNormalizeBackwardShader = (count: number): string => `
struct NormalizeBackwardUniforms { channels: u32, height: u32, width: u32, _pad: u32 }
@group(0) @binding(0) var<storage, read> gradOutValues: array<f32>;
@group(0) @binding(1) var<storage, read> stdValues: array<f32>;
@group(0) @binding(2) var<uniform> uniforms: NormalizeBackwardUniforms;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= ${count}u) { return; }
  let spatial = uniforms.height * uniforms.width;
  let c = (i / spatial) % uniforms.channels;
  out[i] = gradOutValues[i] / stdValues[c];
}`;
