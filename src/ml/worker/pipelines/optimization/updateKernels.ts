import { makeClampShader, runBinaryOpToBuffer } from "../../runtime/shaderRunner";
import { releaseOwnedBuffer, runUnaryShaderToBuffer, uploadToOwnedBuffer, type GpuBufferRef, type OwnedGpuBuffer } from "../../runtime/bufferKernels";

export const runScalarMulBuffer = (
  device: GPUDevice,
  input: GpuBufferRef,
  count: number,
  scalar: number,
): OwnedGpuBuffer => {
  const scalarBuffer = uploadToOwnedBuffer(device, new Float32Array([scalar]));
  const out = runBinaryOpToBuffer(device, "mul", input.buffer, scalarBuffer.buffer, count, "tensorScalar");
  releaseOwnedBuffer(scalarBuffer);
  return { buffer: out, owned: true };
};

export const makeFusedUpdateClampShader = (count: number, learningRate: number): string => `
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

export const runFusedUpdateClampBuffer = (
  device: GPUDevice,
  input: GpuBufferRef,
  grad: GpuBufferRef,
  count: number,
  learningRate: number,
): OwnedGpuBuffer => ({
  buffer: runUnaryShaderToBuffer(device, makeFusedUpdateClampShader(count, learningRate), input.buffer, count, [
    { binding: 1, resource: { buffer: grad.buffer } },
  ]),
  owned: true,
});

export const runUnfusedUpdateClampBuffer = (
  device: GPUDevice,
  input: GpuBufferRef,
  grad: GpuBufferRef,
  count: number,
  learningRate: number,
): { clamped: OwnedGpuBuffer; intermediates: OwnedGpuBuffer[] } => {
  const scaledGrad = runScalarMulBuffer(device, grad, count, learningRate);
  const updated: OwnedGpuBuffer = {
    buffer: runBinaryOpToBuffer(device, "sub", input.buffer, scaledGrad.buffer, count, "tensorTensor"),
    owned: true,
  };
  const clamped: OwnedGpuBuffer = {
    buffer: runUnaryShaderToBuffer(device, makeClampShader(count, 0, 1), updated.buffer, count),
    owned: true,
  };
  return { clamped, intermediates: [scaledGrad, updated] };
};
