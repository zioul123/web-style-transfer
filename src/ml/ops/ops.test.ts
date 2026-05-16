import { describe, expect, it } from 'vitest';
import { Tensor } from '../core/tensor';
import { cpuOps } from './cpu';
import { gpuOps } from './gpu';

const shape: [number, number, number, number] = [1, 1, 4, 4];
const a = new Tensor([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16], shape);
const b = new Tensor([16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1], shape);

describe('Phase 1 ops parity', () => {
  it('matches add/sub/mul/div/clamp outputs', async () => {
    const add = await gpuOps.add(a, b);
    expect(Array.from(add.data)).toEqual(Array.from(cpuOps.add(a, b).data));

    const sub = await gpuOps.sub(a, b);
    expect(Array.from(sub.data)).toEqual(Array.from(cpuOps.sub(a, b).data));

    const mul = await gpuOps.mul(a, b);
    expect(Array.from(mul.data)).toEqual(Array.from(cpuOps.mul(a, b).data));

    const div = await gpuOps.div(a, b);
    expect(Array.from(div.data)).toEqual(Array.from(cpuOps.div(a, b).data));

    const clamp = await gpuOps.clamp(new Tensor([-1, 0.5, 2, 1, ...Array(12).fill(0)] as number[], shape));
    expect(Array.from(clamp.data.slice(0, 4))).toEqual([0, 0.5, 1, 1]);
  });

  it('matches mse scalar', async () => {
    const cpu = cpuOps.mse(a, b);
    const gpu = await gpuOps.mse(a, b);
    expect(gpu).toBeCloseTo(cpu, 7);
  });
});
