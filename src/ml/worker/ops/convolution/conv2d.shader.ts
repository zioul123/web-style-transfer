export const makeConv2dBackwardInputShader = (count: number): string => `
struct Conv2dBackwardInputUniforms { inChannels: u32, outChannels: u32, height: u32, width: u32 }
@group(0) @binding(0) var<storage, read> gradOutValues: array<f32>;
@group(0) @binding(1) var<storage, read> weightValues: array<f32>;
@group(0) @binding(2) var<uniform> uniforms: Conv2dBackwardInputUniforms;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= ${count}u) { return; }
  let hw = uniforms.height * uniforms.width;
  let ic = (i / hw) % uniforms.inChannels;
  let inOffset = i % hw;
  let ih = inOffset / uniforms.width;
  let iw = inOffset % uniforms.width;
  var sum = 0.0;
  for (var oc: u32 = 0u; oc < uniforms.outChannels; oc = oc + 1u) {
    for (var kh: u32 = 0u; kh < 3u; kh = kh + 1u) {
      for (var kw: u32 = 0u; kw < 3u; kw = kw + 1u) {
        let ohSigned = i32(ih) - i32(kh) + 1;
        let owSigned = i32(iw) - i32(kw) + 1;
        if (ohSigned < 0 || ohSigned >= i32(uniforms.height) || owSigned < 0 || owSigned >= i32(uniforms.width)) { continue; }
        let oh = u32(ohSigned);
        let ow = u32(owSigned);
        let gradOutIndex = oc * hw + oh * uniforms.width + ow;
        let weightIndex = ((oc * uniforms.inChannels + ic) * 9u) + kh * 3u + kw;
        sum = sum + gradOutValues[gradOutIndex] * weightValues[weightIndex];
      }
    }
  }
  out[i] = sum;
}`

export const makeConv2dReluShader = (count: number): string => `
struct Conv2dReluUniforms { inChannels: u32, outChannels: u32, height: u32, width: u32 }
@group(0) @binding(0) var<storage, read> inputValues: array<f32>;
@group(0) @binding(1) var<storage, read> weightValues: array<f32>;
@group(0) @binding(2) var<storage, read> biasValues: array<f32>;
@group(0) @binding(3) var<uniform> uniforms: Conv2dReluUniforms;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= ${count}u) { return; }
  let hw = uniforms.height * uniforms.width;
  let oc = (i / hw) % uniforms.outChannels;
  let outOffset = i % hw;
  let oh = outOffset / uniforms.width;
  let ow = outOffset % uniforms.width;
  var sum = biasValues[oc];
  for (var ic: u32 = 0u; ic < uniforms.inChannels; ic = ic + 1u) {
    let inputBase = ic * hw;
    let weightBase = (oc * uniforms.inChannels + ic) * 9u;
    for (var kh: u32 = 0u; kh < 3u; kh = kh + 1u) {
      let khSigned = i32(kh) - 1;
      let ihSigned = i32(oh) + khSigned;
      if (ihSigned < 0 || ihSigned >= i32(uniforms.height)) { continue; }
      for (var kw: u32 = 0u; kw < 3u; kw = kw + 1u) {
        let kwSigned = i32(kw) - 1;
        let iwSigned = i32(ow) + kwSigned;
        if (iwSigned < 0 || iwSigned >= i32(uniforms.width)) { continue; }
        let inputIndex = inputBase + u32(ihSigned) * uniforms.width + u32(iwSigned);
        let weightIndex = weightBase + kh * 3u + kw;
        sum = sum + inputValues[inputIndex] * weightValues[weightIndex];
      }
    }
  }
  out[i] = max(0.0, sum);
}`

export const makeConv2dShader = (count: number): string => `
struct Conv2dUniforms { inChannels: u32, outChannels: u32, height: u32, width: u32 }
@group(0) @binding(0) var<storage, read> inputValues: array<f32>;
@group(0) @binding(1) var<storage, read> weightValues: array<f32>;
@group(0) @binding(2) var<storage, read> biasValues: array<f32>;
@group(0) @binding(3) var<uniform> uniforms: Conv2dUniforms;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= ${count}u) { return; }
  let hw = uniforms.height * uniforms.width;
  let oc = (i / hw) % uniforms.outChannels;
  let outOffset = i % hw;
  let oh = outOffset / uniforms.width;
  let ow = outOffset % uniforms.width;
  var sum = biasValues[oc];
  for (var ic: u32 = 0u; ic < uniforms.inChannels; ic = ic + 1u) {
    let inputBase = ic * hw;
    let weightBase = (oc * uniforms.inChannels + ic) * 9u;
    for (var kh: u32 = 0u; kh < 3u; kh = kh + 1u) {
      let khSigned = i32(kh) - 1;
      let ihSigned = i32(oh) + khSigned;
      if (ihSigned < 0 || ihSigned >= i32(uniforms.height)) { continue; }
      for (var kw: u32 = 0u; kw < 3u; kw = kw + 1u) {
        let kwSigned = i32(kw) - 1;
        let iwSigned = i32(ow) + kwSigned;
        if (iwSigned < 0 || iwSigned >= i32(uniforms.width)) { continue; }
        let ih = u32(ihSigned);
        let iw = u32(iwSigned);
        let inputIndex = inputBase + ih * uniforms.width + iw;
        let weightIndex = weightBase + kh * 3u + kw;
        sum = sum + inputValues[inputIndex] * weightValues[weightIndex];
      }
    }
  }
  out[i] = sum;
}`
