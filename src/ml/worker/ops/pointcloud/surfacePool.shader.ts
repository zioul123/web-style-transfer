export const makeSurfacePoolForwardShader = (count: number): string => `
struct SurfacePoolUniforms {
  inputPointCount: u32,
  outputPointCount: u32,
  channelCount: u32,
  validInputCount: u32,
}
@group(0) @binding(0) var<storage, read> inputFeatures: array<f32>;
@group(0) @binding(1) var<storage, read> mappingValues: array<i32>;
@group(0) @binding(2) var<uniform> uniforms: SurfacePoolUniforms;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= ${count}u) { return; }
  let outputPoint = i / uniforms.channelCount;
  let channel = i % uniforms.channelCount;
  var hasMappedInput = false;
  var maxValue = -3.4028234663852886e38;
  var mean = 0.0;
  for (var point: u32 = 0u; point < uniforms.inputPointCount; point = point + 1u) {
    let mapped = mappingValues[point];
    if (mapped < 0) { continue; }
    let value = inputFeatures[point * uniforms.channelCount + channel];
    mean = mean + value;
    if (u32(mapped) == outputPoint) {
      hasMappedInput = true;
      maxValue = max(maxValue, value);
    }
  }
  if (hasMappedInput) {
    out[i] = maxValue;
  } else {
    out[i] = mean / f32(uniforms.validInputCount);
  }
}`;

export const makeSurfacePoolBackwardShader = (count: number): string => `
struct SurfacePoolUniforms {
  inputPointCount: u32,
  outputPointCount: u32,
  channelCount: u32,
  validInputCount: u32,
}
@group(0) @binding(0) var<storage, read> inputFeatures: array<f32>;
@group(0) @binding(1) var<storage, read> gradOutValues: array<f32>;
@group(0) @binding(2) var<storage, read> mappingValues: array<i32>;
@group(0) @binding(3) var<uniform> uniforms: SurfacePoolUniforms;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= ${count}u) { return; }
  let point = i / uniforms.channelCount;
  let channel = i % uniforms.channelCount;
  let mapped = mappingValues[point];
  if (mapped < 0) {
    out[i] = 0.0;
    return;
  }

  var orphanGrad = 0.0;
  for (var outputPoint: u32 = 0u; outputPoint < uniforms.outputPointCount; outputPoint = outputPoint + 1u) {
    var hasOutput = false;
    for (var candidate: u32 = 0u; candidate < uniforms.inputPointCount; candidate = candidate + 1u) {
      if (mappingValues[candidate] == i32(outputPoint)) {
        hasOutput = true;
      }
    }
    if (!hasOutput) {
      orphanGrad = orphanGrad + gradOutValues[outputPoint * uniforms.channelCount + channel];
    }
  }

  var maxValue = -3.4028234663852886e38;
  var tieCount = 0u;
  for (var candidate: u32 = 0u; candidate < uniforms.inputPointCount; candidate = candidate + 1u) {
    if (mappingValues[candidate] != mapped) { continue; }
    let value = inputFeatures[candidate * uniforms.channelCount + channel];
    if (value > maxValue) {
      maxValue = value;
      tieCount = 1u;
    } else if (value == maxValue) {
      tieCount = tieCount + 1u;
    }
  }

  var grad = orphanGrad / f32(uniforms.validInputCount);
  let inputValue = inputFeatures[i];
  if (inputValue == maxValue) {
    grad = grad + gradOutValues[u32(mapped) * uniforms.channelCount + channel] / f32(tieCount);
  }
  out[i] = grad;
}`;
