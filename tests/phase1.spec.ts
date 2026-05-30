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
import type {
  WorkerResponse,
  WorkerTensor,
  WorkerTensorOperand,
} from "../src/types";
import { gotoStableApp } from "./helpers/appPage";
import { expectScalarCloseTo } from "./helpers/tensorAssertions";
import {
  createStyleTransferWorkerClient,
  expectWebGpuInitOk,
  expectWorkerTensorErrorResponse,
  expectWorkerTensorScalarResponse,
  expectWorkerTensorVectorResponse,
} from "./helpers/workerClient";

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

test("phase 1 exhaustive scalar/tensor op coverage", async ({ page }) => {
  await gotoStableApp(page);
  const workerClient = await createStyleTransferWorkerClient(page);

  try {
    const tensorOperand: WorkerTensorOperand = { kind: "tensor", tensor };
    const tensorOperandB: WorkerTensorOperand = {
      kind: "tensor",
      tensor: tensorB,
    };
    const scalarTwo: WorkerTensorOperand = { kind: "scalar", scalar: 2 };
    const scalarTen: WorkerTensorOperand = { kind: "scalar", scalar: 10 };

    const init = await workerClient.initWebGpu();
    const combos: Record<string, WorkerResponse> = {};
    for (const op of ["add", "sub", "mul", "div"] as const) {
      combos[`${op}-tt`] = await workerClient.ask({
        type: "tensor-op",
        id: `phase1-${op}-tt`,
        op,
        a: tensorOperand,
        b: tensorOperandB,
      });
      combos[`${op}-ts`] = await workerClient.ask({
        type: "tensor-op",
        id: `phase1-${op}-ts`,
        op,
        a: tensorOperand,
        b: scalarTwo,
      });
      combos[`${op}-st`] = await workerClient.ask({
        type: "tensor-op",
        id: `phase1-${op}-st`,
        op,
        a: scalarTen,
        b: tensorOperand,
      });
      combos[`${op}-ss`] = await workerClient.ask({
        type: "tensor-op",
        id: `phase1-${op}-ss`,
        op,
        a: scalarTen,
        b: scalarTwo,
      });
    }

    const mseTt = await workerClient.ask({
      type: "tensor-op",
      id: "phase1-mse-tt",
      op: "mse",
      a: tensorOperand,
      b: tensorOperandB,
    });
    const mseTs = await workerClient.ask({
      type: "tensor-op",
      id: "phase1-mse-ts",
      op: "mse",
      a: tensorOperand,
      b: scalarTwo,
    });
    const mseSt = await workerClient.ask({
      type: "tensor-op",
      id: "phase1-mse-st",
      op: "mse",
      a: scalarTwo,
      b: tensorOperand,
    });
    const mseSs = await workerClient.ask({
      type: "tensor-op",
      id: "phase1-mse-ss",
      op: "mse",
      a: scalarTen,
      b: scalarTwo,
    });
    const clampTensor = await workerClient.ask({
      type: "tensor-op",
      id: "phase1-clamp-t",
      op: "clamp",
      a: {
        kind: "tensor",
        tensor: { shape: tensor.shape, values: [-1, 0.5, 1.5, 0.25] },
      },
      clampMin: 0,
      clampMax: 1,
    });
    const clampScalar = await workerClient.ask({
      type: "tensor-op",
      id: "phase1-clamp-s",
      op: "clamp",
      a: { kind: "scalar", scalar: 2.5 },
      clampMin: 0,
      clampMax: 1,
    });

    expectWebGpuInitOk(init);

    const opFns = { add: cpuAdd, sub: cpuSub, mul: cpuMul, div: cpuDiv };
    for (const op of ["add", "sub", "mul", "div"] as const) {
      const tt = expectWorkerTensorVectorResponse(combos[`${op}-tt`]).values;
      const ts = expectWorkerTensorVectorResponse(combos[`${op}-ts`]).values;
      const st = expectWorkerTensorVectorResponse(combos[`${op}-st`]).values;
      const ss = expectWorkerTensorErrorResponse(combos[`${op}-ss`]).message;

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

    const mseTtResponse = expectWorkerTensorScalarResponse(mseTt);
    expect(mseTtResponse.scalar).toBeCloseTo(
      cpuMse(
        createTensor(shape, tensorValues),
        createTensor(shape, tensorB.values),
      ),
    );

    expect(expectWorkerTensorErrorResponse(mseTs).message).toContain(
      "MSE requires both operands to be tensors",
    );
    expect(expectWorkerTensorErrorResponse(mseSt).message).toContain(
      "MSE requires both operands to be tensors",
    );
    expect(expectWorkerTensorErrorResponse(mseSs).message).toContain(
      "MSE requires both operands to be tensors",
    );

    expect(expectWorkerTensorVectorResponse(clampTensor).values).toEqual(
      Array.from(
        cpuClamp(createTensor(shape, [-1, 0.5, 1.5, 0.25]), 0, 1).values,
      ),
    );
    expect(expectWorkerTensorVectorResponse(clampScalar).values).toEqual([1]);
  } finally {
    await workerClient.dispose();
  }
});

test("phase 1 mse GPU reduction parity for >64 lengths", async ({ page }) => {
  await gotoStableApp(page);
  const workerClient = await createStyleTransferWorkerClient(page);

  try {
    const mseLengths: readonly number[] = [
      64,
      64 * 5,
      64 * 5 + 1,
      64 * 5 + 32,
      64 * 5 + 63,
    ];
    const init = await workerClient.initWebGpu();
    expectWebGpuInitOk(init);

    for (const length of mseLengths) {
      const aTensor = makeWorkerMseTensor(length);
      const bTensor = {
        shape: [1, 1, 1, length] as const,
        values: Array.from(
          { length },
          (_unused: unknown, index: number): number => ((index % 7) - 3) / 2,
        ),
      };
      const response = await workerClient.ask({
        type: "tensor-op",
        id: `phase1-mse-${length}`,
        op: "mse",
        a: { kind: "tensor", tensor: aTensor },
        b: { kind: "tensor", tensor: bTensor },
      });

      const scalarResponse = expectWorkerTensorScalarResponse(response);
      const expectedMse: number = cpuMse(
        createTensor(aTensor.shape, aTensor.values),
        createTensor(bTensor.shape, bTensor.values),
      );
      expectScalarCloseTo(scalarResponse.scalar, expectedMse, 6);
    }
  } finally {
    await workerClient.dispose();
  }
});

test("worker protocol emits typed responses for success and failure paths", async ({
  page,
}) => {
  await gotoStableApp(page);
  const workerClient = await createStyleTransferWorkerClient(page);

  try {
    const pong = await workerClient.ask({ type: "ping", id: "phase1-ping" });
    const tensorRoundtrip = await workerClient.ask({
      type: "tensor-roundtrip",
      id: "phase1-tensor-roundtrip",
      tensor: { shape: [1, 1, 1, 2], values: [2, 4] },
    });
    const unknown = await workerClient.ask({
      type: "not-a-real-message",
      id: "phase1-unknown",
    });

    expect(pong.type).toBe("pong");

    expect(tensorRoundtrip.type).toBe("tensor-roundtrip-result");
    if (tensorRoundtrip.type !== "tensor-roundtrip-result") {
      throw new Error("Expected tensor-roundtrip-result response.");
    }
    expect(tensorRoundtrip.ok).toBeTruthy();

    expect(unknown.type).toBe("error");
    if (unknown.type !== "error") {
      throw new Error("Expected worker error response.");
    }
    expect(typeof unknown.message).toBe("string");
    expect(unknown.message.length).toBeGreaterThan(0);
  } finally {
    await workerClient.dispose();
  }
});
