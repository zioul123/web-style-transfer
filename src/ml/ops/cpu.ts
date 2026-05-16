import { Tensor } from '../core/tensor';

function binaryOp(a: Tensor, b: Tensor, op: (x: number, y: number) => number): Tensor {
  if (a.data.length !== b.data.length) throw new Error('Shape mismatch');
  const out = new Float32Array(a.data.length);
  for (let i = 0; i < out.length; i++) out[i] = op(a.data[i], b.data[i]);
  return new Tensor(out, a.shape);
}

export const cpuOps = {
  add: (a: Tensor, b: Tensor) => binaryOp(a, b, (x, y) => x + y),
  sub: (a: Tensor, b: Tensor) => binaryOp(a, b, (x, y) => x - y),
  mul: (a: Tensor, b: Tensor) => binaryOp(a, b, (x, y) => x * y),
  div: (a: Tensor, b: Tensor) => binaryOp(a, b, (x, y) => x / y),
  clamp: (a: Tensor, min = 0, max = 1) => {
    const out = new Float32Array(a.data.length);
    for (let i = 0; i < out.length; i++) out[i] = Math.min(max, Math.max(min, a.data[i]));
    return new Tensor(out, a.shape);
  },
  mse: (a: Tensor, b: Tensor) => {
    if (a.data.length !== b.data.length) throw new Error('Shape mismatch');
    let acc = 0;
    for (let i = 0; i < a.data.length; i++) {
      const d = a.data[i] - b.data[i];
      acc += d * d;
    }
    return acc / a.data.length;
  }
};
