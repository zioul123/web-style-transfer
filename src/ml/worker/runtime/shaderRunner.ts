const BINARY_SHADER_SNIPPETS: Record<'add' | 'sub' | 'mul' | 'div', { tensorTensor: string; tensorScalar: string; scalarTensor: string }> = {
  add: { tensorTensor: 'out[i] = a[i] + b[i];', tensorScalar: 'out[i] = a[i] + b[0];', scalarTensor: 'out[i] = b[0] + a[i];' },
  sub: { tensorTensor: 'out[i] = a[i] - b[i];', tensorScalar: 'out[i] = a[i] - b[0];', scalarTensor: 'out[i] = b[0] - a[i];' },
  mul: { tensorTensor: 'out[i] = a[i] * b[i];', tensorScalar: 'out[i] = a[i] * b[0];', scalarTensor: 'out[i] = b[0] * a[i];' },
  div: { tensorTensor: 'out[i] = a[i] / b[i];', tensorScalar: 'out[i] = a[i] / b[0];', scalarTensor: 'out[i] = b[0] / a[i];' },
}

export const makeBinaryOpShader = (op: 'add' | 'sub' | 'mul' | 'div', count: number, mode: 'tensorTensor' | 'tensorScalar' | 'scalarTensor'): string => `
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= ${count}u) { return; }
  ${BINARY_SHADER_SNIPPETS[op][mode]}
}`

export const makeClampShader = (count: number, min: number, max: number): string => `
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= ${count}u) { return; }
  out[i] = clamp(a[i], ${min}, ${max});
}`
