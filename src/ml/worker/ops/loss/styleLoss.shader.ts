export const makeStyleLossTermsShader = (gramCount: number): string => `
struct StyleLossUniforms {
  channels: u32,
  spatial: u32,
}
@group(0) @binding(0) var<storage, read> inputValues: array<f32>;
@group(0) @binding(1) var<storage, read> targetGram: array<f32>;
@group(0) @binding(2) var<uniform> uniforms: StyleLossUniforms;
@group(0) @binding(3) var<storage, read_write> outGram: array<f32>;
@group(0) @binding(4) var<storage, read_write> outSqDiff: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= ${gramCount}u) { return; }
  let row = i / uniforms.channels;
  let col = i % uniforms.channels;
  var sum = 0.0;
  let rowBase = row * uniforms.spatial;
  let colBase = col * uniforms.spatial;
  for (var s: u32 = 0u; s < uniforms.spatial; s = s + 1u) {
    sum = sum + inputValues[rowBase + s] * inputValues[colBase + s];
  }
  let norm = f32(uniforms.channels * uniforms.spatial);
  let gramValue = sum / norm;
  outGram[i] = gramValue;
  let d = gramValue - targetGram[i];
  outSqDiff[i] = d * d;
}`;
