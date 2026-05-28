export const makeGramBackwardShader = (count: number): string => `
struct GramBackwardUniforms {
  channels: u32,
  spatial: u32,
  norm: u32,
  _pad: u32,
}
@group(0) @binding(0) var<storage, read> inputValues: array<f32>;
@group(0) @binding(1) var<storage, read> gradOutValues: array<f32>;
@group(0) @binding(2) var<uniform> uniforms: GramBackwardUniforms;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= ${count}u) { return; }
  let a = i / uniforms.spatial;
  let p = i % uniforms.spatial;
  var sum: f32 = 0.0;
  for (var b: u32 = 0u; b < uniforms.channels; b = b + 1u) {
    let gradAB = gradOutValues[a * uniforms.channels + b];
    let gradBA = gradOutValues[b * uniforms.channels + a];
    let inputBP = inputValues[b * uniforms.spatial + p];
    sum = sum + ((gradAB + gradBA) * inputBP);
  }
  out[i] = sum / f32(uniforms.norm);
}`;

export const makeGramMatrixShader = (count: number): string => `
struct GramUniforms {
  channels: u32,
  spatial: u32,
}
@group(0) @binding(0) var<storage, read> inputValues: array<f32>;
@group(0) @binding(1) var<uniform> uniforms: GramUniforms;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= ${count}u) { return; }
  let row = i / uniforms.channels;
  let col = i % uniforms.channels;
  var sum = 0.0;
  let rowBase = row * uniforms.spatial;
  let colBase = col * uniforms.spatial;
  for (var s: u32 = 0u; s < uniforms.spatial; s = s + 1u) {
    sum = sum + inputValues[rowBase + s] * inputValues[colBase + s];
  }
  let norm = f32(uniforms.channels * uniforms.spatial);
  out[i] = sum / norm;
}`;

export const makeSymmetricGramMatrixShader = (
  pairCount: number,
): string => `
struct GramUniforms {
  channels: u32,
  spatial: u32,
}
@group(0) @binding(0) var<storage, read> inputValues: array<f32>;
@group(0) @binding(1) var<uniform> uniforms: GramUniforms;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

fn triangularNumber(n: u32) -> u32 {
  return (n * (n + 1u)) / 2u;
}

fn pairFromIndex(index: u32, channels: u32) -> vec2<u32> {
  var row: u32 = 0u;
  loop {
    let next = triangularNumber(row + 1u);
    if (index < next || row + 1u >= channels) {
      break;
    }
    row = row + 1u;
  }
  let rowStart = triangularNumber(row);
  let col = index - rowStart;
  return vec2<u32>(row, col);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let pairIndex = gid.x;
  if (pairIndex >= ${pairCount}u) { return; }
  let pair = pairFromIndex(pairIndex, uniforms.channels);
  let row = pair.x;
  let col = pair.y;
  var sum = 0.0;
  let rowBase = row * uniforms.spatial;
  let colBase = col * uniforms.spatial;
  for (var s: u32 = 0u; s < uniforms.spatial; s = s + 1u) {
    sum = sum + inputValues[rowBase + s] * inputValues[colBase + s];
  }
  let norm = f32(uniforms.channels * uniforms.spatial);
  let gramValue = sum / norm;
  let rowMajor = row * uniforms.channels + col;
  out[rowMajor] = gramValue;
  if (row != col) {
    let colMajor = col * uniforms.channels + row;
    out[colMajor] = gramValue;
  }
}`;
