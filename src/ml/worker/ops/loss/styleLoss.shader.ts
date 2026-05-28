export const makeStyleLossTermsShader = (gramCount: number): string => `
struct StyleLossUniforms {
  channels: u32,
  spatial: u32,
}
@group(0) @binding(0) var<storage, read> inputValues: array<f32>;
@group(0) @binding(1) var<storage, read> targetGram: array<f32>;
@group(0) @binding(2) var<uniform> uniforms: StyleLossUniforms;
@group(0) @binding(3) var<storage, read_write> outGram: array<f32>;
@group(0) @binding(4) var<storage, read_write> outPartialSqDiffSums: array<f32>;
var<workgroup> sqDiffCache: array<f32, 64>;
@compute @workgroup_size(64)
fn main(
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(workgroup_id) wid: vec3<u32>,
) {
  let i = gid.x;
  let localIndex = lid.x;
  var sqDiff = 0.0;
  if (i < ${gramCount}u) {
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
    sqDiff = d * d;
  }
  sqDiffCache[localIndex] = sqDiff;
  workgroupBarrier();

  var step: u32 = 32u;
  loop {
    if (localIndex < step) {
      sqDiffCache[localIndex] = sqDiffCache[localIndex] + sqDiffCache[localIndex + step];
    }
    workgroupBarrier();
    if (step == 1u) {
      break;
    }
    step = step / 2u;
  }

  if (localIndex == 0u) {
    outPartialSqDiffSums[wid.x] = sqDiffCache[0];
  }
}`;

export const makeStyleLossTermsSymmetricShader = (
  channels: number,
  pairCount: number,
): string => `
struct StyleLossUniforms {
  channels: u32,
  spatial: u32,
}
@group(0) @binding(0) var<storage, read> inputValues: array<f32>;
@group(0) @binding(1) var<storage, read> targetGram: array<f32>;
@group(0) @binding(2) var<uniform> uniforms: StyleLossUniforms;
@group(0) @binding(3) var<storage, read_write> outGram: array<f32>;
@group(0) @binding(4) var<storage, read_write> outPartialSqDiffSums: array<f32>;
var<workgroup> sqDiffCache: array<f32, 64>;

fn triangularNumber(n: u32) -> u32 {
  return (n * (n + 1u)) / 2u;
}

fn pairFromIndex(index: u32, maxChannels: u32) -> vec2<u32> {
  var row: u32 = 0u;
  loop {
    let next = triangularNumber(row + 1u);
    if (index < next || row + 1u >= maxChannels) {
      break;
    }
    row = row + 1u;
  }
  let rowStart = triangularNumber(row);
  let col = index - rowStart;
  return vec2<u32>(row, col);
}

@compute @workgroup_size(64)
fn main(
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(workgroup_id) wid: vec3<u32>,
) {
  let pairIndex = gid.x;
  let localIndex = lid.x;
  var sqDiff = 0.0;
  if (pairIndex < ${pairCount}u) {
    let pair = pairFromIndex(pairIndex, ${channels}u);
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
    outGram[rowMajor] = gramValue;
    let diff = gramValue - targetGram[rowMajor];
    sqDiff = diff * diff;
    if (row != col) {
      let colMajor = col * uniforms.channels + row;
      outGram[colMajor] = gramValue;
      let mirrorDiff = gramValue - targetGram[colMajor];
      sqDiff = sqDiff + (mirrorDiff * mirrorDiff);
    }
  }
  sqDiffCache[localIndex] = sqDiff;
  workgroupBarrier();

  var step: u32 = 32u;
  loop {
    if (localIndex < step) {
      sqDiffCache[localIndex] = sqDiffCache[localIndex] + sqDiffCache[localIndex + step];
    }
    workgroupBarrier();
    if (step == 1u) {
      break;
    }
    step = step / 2u;
  }

  if (localIndex == 0u) {
    outPartialSqDiffSums[wid.x] = sqDiffCache[0];
  }
}`;
