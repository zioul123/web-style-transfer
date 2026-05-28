export const makeReluShader = (count: number): string => `
@group(0) @binding(0) var<storage, read> inputValues: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= ${count}u) { return; }
  out[i] = max(0.0, inputValues[i]);
}`;

export const makeReluBackwardShader = (count: number): string => `
@group(0) @binding(0) var<storage, read> inputValues: array<f32>;
@group(0) @binding(1) var<storage, read> gradOutValues: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= ${count}u) { return; }
  out[i] = select(0.0, gradOutValues[i], inputValues[i] > 0.0);
}`;

export const makeReluVec4Shader = (vecCount: number): string => `
@group(0) @binding(0) var<storage, read> inputValues: array<vec4f>;
@group(0) @binding(1) var<storage, read_write> out: array<vec4f>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= ${vecCount}u) { return; }
  out[i] = max(vec4f(0.0), inputValues[i]);
}`;

export const makeReluBackwardVec4Shader = (vecCount: number): string => `
@group(0) @binding(0) var<storage, read> inputValues: array<vec4f>;
@group(0) @binding(1) var<storage, read> gradOutValues: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> out: array<vec4f>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= ${vecCount}u) { return; }
  out[i] = select(vec4f(0.0), gradOutValues[i], inputValues[i] > vec4f(0.0));
}`;
