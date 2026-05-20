import { expect, test } from "@playwright/test";
import {
  cpuAdd,
  cpuClamp,
  cpuDiv,
  cpuMse,
  cpuMul,
  cpuSub,
  createTensor,
} from "../src/ml";
import {
  isWorkerTensorScalarOpResponse,
  isWorkerTensorVectorOpResponse,
  type WorkerRequest,
  type WorkerResponse,
  type WorkerTensor,
  type WorkerTensorOperand,
} from "../src/types";
import { gotoStableApp } from "./helpers/appPage";

const shape = [1, 1, 2, 2] as const;
const tensorValues = [1, 2, 3, 4];
const tensor: WorkerTensor = { shape, values: tensorValues };
const tensorB: WorkerTensor = { shape, values: [5, 6, 7, 8] };

const makeWorkerMseTensor = (
  length: number,
): { shape: readonly [number, number, number, number]; values: number[] } => ({
  shape: [1, 1, 1, length] as const,
  values: Array.from(
    { length },
    (_unused: unknown, index: number): number => ((index % 11) - 5) / 3,
  ),
});

const expectVector = (response: WorkerResponse): number[] => {
  expect(response.type).toBe("tensor-op-result");
  if (response.type !== "tensor-op-result")
    throw new Error("Expected tensor-op-result.");
  expect(isWorkerTensorVectorOpResponse(response)).toBeTruthy();
  if (!isWorkerTensorVectorOpResponse(response))
    throw new Error("Expected vector response.");
  return response.values;
};

const expectError = (response: WorkerResponse): string => {
  expect(response.type).toBe("tensor-op-result");
  if (response.type !== "tensor-op-result")
    throw new Error("Expected tensor-op-result.");
  expect(response.ok).toBeFalsy();
  if (response.ok) throw new Error("Expected error response.");
  return response.message;
};

test("phase 1 exhaustive scalar/tensor op coverage", async ({ page }) => {
  await gotoStableApp(page);

  const result = await page.evaluate(
    async ({ inputTensor, inputTensorB }) => {
      const worker = new Worker(
        new URL("/src/styleTransfer.worker.ts", window.location.origin),
        { type: "module" },
      );

      const ask = (payload: WorkerRequest): Promise<WorkerResponse> =>
        new Promise((resolve) => {
          const handler = (event: MessageEvent<WorkerResponse>): void => {
            if (event.data.id === payload.id) {
              worker.removeEventListener("message", handler);
              resolve(event.data);
            }
          };
          worker.addEventListener("message", handler);
          worker.postMessage(payload);
        });

      const tensorOperand: WorkerTensorOperand = {
        kind: "tensor",
        tensor: inputTensor,
      };
      const tensorOperandB: WorkerTensorOperand = {
        kind: "tensor",
        tensor: inputTensorB,
      };
      const scalarTwo: WorkerTensorOperand = { kind: "scalar", scalar: 2 };
      const scalarTen: WorkerTensorOperand = { kind: "scalar", scalar: 10 };

      const init = await ask({ type: "init-webgpu", id: `${Date.now()}-init` });

      const combos: Record<string, WorkerResponse> = {};
      for (const op of ["add", "sub", "mul", "div"] as const) {
        combos[`${op}-tt`] = await ask({
          type: "tensor-op",
          id: `${Date.now()}-${op}-tt`,
          op,
          a: tensorOperand,
          b: tensorOperandB,
        });
        combos[`${op}-ts`] = await ask({
          type: "tensor-op",
          id: `${Date.now()}-${op}-ts`,
          op,
          a: tensorOperand,
          b: scalarTwo,
        });
        combos[`${op}-st`] = await ask({
          type: "tensor-op",
          id: `${Date.now()}-${op}-st`,
          op,
          a: scalarTen,
          b: tensorOperand,
        });
        combos[`${op}-ss`] = await ask({
          type: "tensor-op",
          id: `${Date.now()}-${op}-ss`,
          op,
          a: scalarTen,
          b: scalarTwo,
        });
      }

      const mse_tt = await ask({
        type: "tensor-op",
        id: `${Date.now()}-mse-tt`,
        op: "mse",
        a: tensorOperand,
        b: tensorOperandB,
      });
      const mse_ts = await ask({
        type: "tensor-op",
        id: `${Date.now()}-mse-ts`,
        op: "mse",
        a: tensorOperand,
        b: scalarTwo,
      });
      const mse_st = await ask({
        type: "tensor-op",
        id: `${Date.now()}-mse-st`,
        op: "mse",
        a: scalarTwo,
        b: tensorOperand,
      });
      const mse_ss = await ask({
        type: "tensor-op",
        id: `${Date.now()}-mse-ss`,
        op: "mse",
        a: scalarTen,
        b: scalarTwo,
      });

      const clampTensor = await ask({
        type: "tensor-op",
        id: `${Date.now()}-clamp-t`,
        op: "clamp",
        a: {
          kind: "tensor",
          tensor: { shape: inputTensor.shape, values: [-1, 0.5, 1.5, 0.25] },
        },
        clampMin: 0,
        clampMax: 1,
      });
      const clampScalar = await ask({
        type: "tensor-op",
        id: `${Date.now()}-clamp-s`,
        op: "clamp",
        a: { kind: "scalar", scalar: 2.5 },
        clampMin: 0,
        clampMax: 1,
      });

      worker.terminate();
      return {
        init,
        combos,
        mse_tt,
        mse_ts,
        mse_st,
        mse_ss,
        clampTensor,
        clampScalar,
      };
    },
    { inputTensor: tensor, inputTensorB: tensorB },
  );

  expect(result.init.type).toBe("webgpu-init-result");
  if (result.init.type !== "webgpu-init-result")
    throw new Error("Expected webgpu-init-result");
  expect(result.init.ok).toBeTruthy();

  const opFns = { add: cpuAdd, sub: cpuSub, mul: cpuMul, div: cpuDiv };
  for (const op of ["add", "sub", "mul", "div"] as const) {
    const tt = expectVector(result.combos[`${op}-tt`]);
    const ts = expectVector(result.combos[`${op}-ts`]);
    const st = expectVector(result.combos[`${op}-st`]);
    const ss = expectError(result.combos[`${op}-ss`]);

    expect(tt).toEqual(
      Array.from(
        opFns[op](
          createTensor(shape, tensorValues),
          createTensor(shape, tensorB.values),
        ).values,
      ),
    );
    expect(ts).toEqual(
      Array.from(
        opFns[op](
          createTensor(shape, tensorValues),
          createTensor(shape, [2, 2, 2, 2]),
        ).values,
      ),
    );
    expect(st).toEqual(
      Array.from(
        opFns[op](
          createTensor(shape, [10, 10, 10, 10]),
          createTensor(shape, tensorValues),
        ).values,
      ),
    );
    expect(ss).toContain("At least one operand must be a tensor");
  }

  expect(result.mse_tt.type).toBe("tensor-op-result");
  if (result.mse_tt.type !== "tensor-op-result")
    throw new Error("Expected tensor-op-result");
  expect(isWorkerTensorScalarOpResponse(result.mse_tt)).toBeTruthy();
  if (!isWorkerTensorScalarOpResponse(result.mse_tt))
    throw new Error("Expected scalar MSE response");
  expect(result.mse_tt.scalar).toBeCloseTo(
    cpuMse(
      createTensor(shape, tensorValues),
      createTensor(shape, tensorB.values),
    ),
  );

  expect(expectError(result.mse_ts)).toContain(
    "MSE requires both operands to be tensors",
  );
  expect(expectError(result.mse_st)).toContain(
    "MSE requires both operands to be tensors",
  );
  expect(expectError(result.mse_ss)).toContain(
    "MSE requires both operands to be tensors",
  );

  expect(expectVector(result.clampTensor)).toEqual(
    Array.from(
      cpuClamp(createTensor(shape, [-1, 0.5, 1.5, 0.25]), 0, 1).values,
    ),
  );
  expect(expectVector(result.clampScalar)).toEqual([1]);
});

test("phase 1 mse GPU reduction parity for >64 lengths", async ({ page }) => {
  await gotoStableApp(page);

  const mseLengths: readonly number[] = [
    64,
    64 * 5,
    64 * 5 + 1,
    64 * 5 + 32,
    64 * 5 + 63,
  ];

  const phase1MseResult: {
    init: WorkerResponse;
    responses: Array<{ length: number; response: WorkerResponse }>;
  } = await page.evaluate(async (lengths: readonly number[]) => {
    const worker = new Worker(
      new URL("/src/styleTransfer.worker.ts", window.location.origin),
      { type: "module" },
    );
    const ask = (payload: WorkerRequest): Promise<WorkerResponse> =>
      new Promise((resolve) => {
        const handler = (event: MessageEvent<WorkerResponse>): void => {
          if (event.data.id === payload.id) {
            worker.removeEventListener("message", handler);
            resolve(event.data);
          }
        };
        worker.addEventListener("message", handler);
        worker.postMessage(payload);
      });

    const init = await ask({
      type: "init-webgpu",
      id: `${Date.now()}-mse-init`,
    });
    const responses: Array<{ length: number; response: WorkerResponse }> = [];
    for (const length of lengths) {
      const aValues: number[] = Array.from(
        { length },
        (_unused: unknown, index: number): number => ((index % 11) - 5) / 3,
      );
      const bValues: number[] = Array.from(
        { length },
        (_unused: unknown, index: number): number => ((index % 7) - 3) / 2,
      );
      const response = await ask({
        type: "tensor-op",
        id: `${Date.now()}-mse-${length}`,
        op: "mse",
        a: {
          kind: "tensor",
          tensor: { shape: [1, 1, 1, length] as const, values: aValues },
        },
        b: {
          kind: "tensor",
          tensor: { shape: [1, 1, 1, length] as const, values: bValues },
        },
      });
      responses.push({ length, response });
    }

    worker.terminate();
    return { init, responses };
  }, mseLengths);

  expect(phase1MseResult.init.type).toBe("webgpu-init-result");
  if (phase1MseResult.init.type !== "webgpu-init-result")
    throw new Error("Expected webgpu-init-result");
  expect(phase1MseResult.init.ok).toBeTruthy();

  for (const result of phase1MseResult.responses) {
    expect(result.response.type).toBe("tensor-op-result");
    if (result.response.type !== "tensor-op-result")
      throw new Error(`Expected tensor-op-result for length ${result.length}.`);
    expect(isWorkerTensorScalarOpResponse(result.response)).toBeTruthy();
    if (!isWorkerTensorScalarOpResponse(result.response))
      throw new Error(
        `Expected scalar MSE response for length ${result.length}.`,
      );

    const aTensor = makeWorkerMseTensor(result.length);
    const bTensor = {
      shape: [1, 1, 1, result.length] as const,
      values: Array.from(
        { length: result.length },
        (_unused: unknown, index: number): number => ((index % 7) - 3) / 2,
      ),
    };
    const expectedMse: number = cpuMse(
      createTensor(aTensor.shape, aTensor.values),
      createTensor(bTensor.shape, bTensor.values),
    );
    expect(result.response.scalar).toBeCloseTo(expectedMse, 6);
  }
});
