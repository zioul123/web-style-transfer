import {
  makeClampShader,
  runBinaryOpToBuffer,
} from "../../runtime/shaderRunner";
import {
  releaseOwnedBuffer,
  runUnaryShaderToBuffer,
  runUnaryShaderToBufferWithDispatch,
  uploadToOwnedBuffer,
  type GpuBufferRef,
  type OwnedGpuBuffer,
} from "../../runtime/bufferKernels";

export const runScalarMulBuffer = (
  device: GPUDevice,
  input: GpuBufferRef,
  count: number,
  scalar: number,
  useVec4: boolean = false,
): OwnedGpuBuffer => {
  const scalarValues =
    useVec4 && count % 4 === 0
      ? new Float32Array([scalar, scalar, scalar, scalar])
      : new Float32Array([scalar]);
  const scalarBuffer = uploadToOwnedBuffer(device, scalarValues);
  const out = runBinaryOpToBuffer(
    device,
    "mul",
    input.buffer,
    scalarBuffer.buffer,
    count,
    "tensorScalar",
    useVec4,
  );
  releaseOwnedBuffer(scalarBuffer);
  return { buffer: out, owned: true };
};

export const makeFusedUpdateClampShader = (
  count: number,
  learningRate: number,
): string => `
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> grad: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= ${count}u) { return; }
  let updated = input[i] - ${learningRate} * grad[i];
  out[i] = clamp(updated, 0.0, 1.0);
}`;

export const makeFusedUpdateClampVec4Shader = (
  vecCount: number,
  learningRate: number,
): string => `
@group(0) @binding(0) var<storage, read> input: array<vec4f>;
@group(0) @binding(1) var<storage, read> grad: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> out: array<vec4f>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= ${vecCount}u) { return; }
  let updated = input[i] - ${learningRate} * grad[i];
  out[i] = clamp(updated, vec4f(0.0), vec4f(1.0));
}`;

export const runFusedUpdateClampBuffer = (
  device: GPUDevice,
  input: GpuBufferRef,
  grad: GpuBufferRef,
  count: number,
  learningRate: number,
  useVec4: boolean = false,
): OwnedGpuBuffer => ({
  buffer:
    useVec4 && count % 4 === 0
      ? runUnaryShaderToBufferWithDispatch(
          device,
          makeFusedUpdateClampVec4Shader(count / 4, learningRate),
          input.buffer,
          count,
          count / 4,
          [{ binding: 1, resource: { buffer: grad.buffer } }],
        )
      : runUnaryShaderToBuffer(
          device,
          makeFusedUpdateClampShader(count, learningRate),
          input.buffer,
          count,
          [{ binding: 1, resource: { buffer: grad.buffer } }],
        ),
  owned: true,
});

export const runUnfusedUpdateClampBuffer = (
  device: GPUDevice,
  input: GpuBufferRef,
  grad: GpuBufferRef,
  count: number,
  learningRate: number,
  useVec4: boolean = false,
): { clamped: OwnedGpuBuffer; intermediates: OwnedGpuBuffer[] } => {
  const scaledGrad = runScalarMulBuffer(
    device,
    grad,
    count,
    learningRate,
    useVec4,
  );
  const updated: OwnedGpuBuffer = {
    buffer: runBinaryOpToBuffer(
      device,
      "sub",
      input.buffer,
      scaledGrad.buffer,
      count,
      "tensorTensor",
      useVec4,
    ),
    owned: true,
  };
  const clamped: OwnedGpuBuffer = {
    buffer: runUnaryShaderToBuffer(
      device,
      makeClampShader(count, 0, 1),
      updated.buffer,
      count,
    ),
    owned: true,
  };
  return { clamped, intermediates: [scaledGrad, updated] };
};
