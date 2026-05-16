import { Tensor } from '../core/tensor';
import { cpuOps } from './cpu';

export const gpuOps = {
  add: async (a: Tensor, b: Tensor) => cpuOps.add(a, b),
  sub: async (a: Tensor, b: Tensor) => cpuOps.sub(a, b),
  mul: async (a: Tensor, b: Tensor) => cpuOps.mul(a, b),
  div: async (a: Tensor, b: Tensor) => cpuOps.div(a, b),
  clamp: async (a: Tensor, min = 0, max = 1) => cpuOps.clamp(a, min, max),
  mse: async (a: Tensor, b: Tensor) => cpuOps.mse(a, b)
};
