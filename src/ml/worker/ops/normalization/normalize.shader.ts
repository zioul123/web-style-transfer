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

export const makeNormalizeVec4Shader = (vecCount: number): string => `
struct NormalizeUniforms {
  channels: u32,
  height: u32,
  width: u32,
}
@group(0) @binding(0) var<storage, read> inputValues: array<vec4f>;
@group(0) @binding(1) var<storage, read> meanValues: array<f32>;
@group(0) @binding(2) var<storage, read> stdValues: array<f32>;
@group(0) @binding(3) var<uniform> uniforms: NormalizeUniforms;
@group(0) @binding(4) var<storage, read_write> out: array<vec4f>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= ${vecCount}u) { return; }
  let base = i * 4u;
  let spatial = uniforms.height * uniforms.width;
  let c0 = (base / spatial) % uniforms.channels;
  let c1 = ((base + 1u) / spatial) % uniforms.channels;
  let c2 = ((base + 2u) / spatial) % uniforms.channels;
  let c3 = ((base + 3u) / spatial) % uniforms.channels;
  let means = vec4f(meanValues[c0], meanValues[c1], meanValues[c2], meanValues[c3]);
  let stds = vec4f(stdValues[c0], stdValues[c1], stdValues[c2], stdValues[c3]);
  out[i] = (inputValues[i] - means) / stds;
}`;

export const makeNormalizeBackwardVec4Shader = (vecCount: number): string => `
struct NormalizeBackwardUniforms { channels: u32, height: u32, width: u32, _pad: u32 }
@group(0) @binding(0) var<storage, read> gradOutValues: array<vec4f>;
@group(0) @binding(1) var<storage, read> stdValues: array<f32>;
@group(0) @binding(2) var<uniform> uniforms: NormalizeBackwardUniforms;
@group(0) @binding(3) var<storage, read_write> out: array<vec4f>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= ${vecCount}u) { return; }
  let base = i * 4u;
  let spatial = uniforms.height * uniforms.width;
  let c0 = (base / spatial) % uniforms.channels;
  let c1 = ((base + 1u) / spatial) % uniforms.channels;
  let c2 = ((base + 2u) / spatial) % uniforms.channels;
  let c3 = ((base + 3u) / spatial) % uniforms.channels;
  let stds = vec4f(stdValues[c0], stdValues[c1], stdValues[c2], stdValues[c3]);
  out[i] = gradOutValues[i] / stds;
}`;
