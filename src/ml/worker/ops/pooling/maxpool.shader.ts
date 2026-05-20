export const makeMaxPool2dBackwardShader = (count: number): string => `
struct MaxPoolBackwardUniforms {
  channels: u32,
  inHeight: u32,
  inWidth: u32,
  outHeight: u32,
  outWidth: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
}
@group(0) @binding(0) var<storage, read> inputValues: array<f32>;
@group(0) @binding(1) var<storage, read> gradOutValues: array<f32>;
@group(0) @binding(2) var<uniform> uniforms: MaxPoolBackwardUniforms;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= ${count}u) { return; }
  let hw = uniforms.inHeight * uniforms.inWidth;
  let c = (i / hw) % uniforms.channels;
  let inOffset = i % hw;
  let ih = inOffset / uniforms.inWidth;
  let iw = inOffset % uniforms.inWidth;
  let oh = ih / 2u;
  let ow = iw / 2u;
  if (oh >= uniforms.outHeight || ow >= uniforms.outWidth) { out[i] = 0.0; return; }
  let outIdx = c * uniforms.outHeight * uniforms.outWidth + oh * uniforms.outWidth + ow;
  var maxH = oh * 2u;
  var maxW = ow * 2u;
  var maxVal = inputValues[c * hw + maxH * uniforms.inWidth + maxW];
  for (var kh: u32 = 0u; kh < 2u; kh = kh + 1u) {
    for (var kw: u32 = 0u; kw < 2u; kw = kw + 1u) {
      let candH = oh * 2u + kh;
      let candW = ow * 2u + kw;
      let v = inputValues[c * hw + candH * uniforms.inWidth + candW];
      if (v > maxVal) { maxVal = v; maxH = candH; maxW = candW; }
    }
  }
  out[i] = select(0.0, gradOutValues[outIdx], ih == maxH && iw == maxW);
}`;

export const makeMaxPool2dShader = (count: number): string => `
struct MaxPoolUniforms {
  channels: u32,
  inHeight: u32,
  inWidth: u32,
  outHeight: u32,
  outWidth: u32,
}
@group(0) @binding(0) var<storage, read> inputValues: array<f32>;
@group(0) @binding(1) var<uniform> uniforms: MaxPoolUniforms;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= ${count}u) { return; }
  let channelPlane = uniforms.outHeight * uniforms.outWidth;
  let c = (i / channelPlane) % uniforms.channels;
  let outOffset = i % channelPlane;
  let oh = outOffset / uniforms.outWidth;
  let ow = outOffset % uniforms.outWidth;
  let ih = oh * 2u;
  let iw = ow * 2u;
  let inputPlane = uniforms.inHeight * uniforms.inWidth;
  let base = c * inputPlane;
  let idx00 = base + ih * uniforms.inWidth + iw;
  let idx01 = idx00 + 1u;
  let idx10 = idx00 + uniforms.inWidth;
  let idx11 = idx10 + 1u;
  let m0 = max(inputValues[idx00], inputValues[idx01]);
  let m1 = max(inputValues[idx10], inputValues[idx11]);
  out[i] = max(m0, m1);
}`;
