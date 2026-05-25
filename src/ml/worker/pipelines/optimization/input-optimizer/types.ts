import type { WorkerRequest } from "../../../../../types";

export type InputOptimizerMode = Extract<
  WorkerRequest,
  { type: "run-style-transfer" }
>["optimizer"];

export type InputOptimizerConfig = {
  optimizer: InputOptimizerMode;
  count: number;
  learningRate: number;
  adamBeta1: number;
  adamBeta2: number;
  adamEpsilon: number;
  lbfgsMemory: number;
  lbfgsEpsilon: number;
};

export type InputOptimizerVectorOps<TVector> = {
  clone: (input: TVector) => Promise<TVector>;
  zeros: () => Promise<TVector>;
  sub: (a: TVector, b: TVector) => Promise<TVector>;
  scale: (input: TVector, scalar: number) => Promise<TVector>;
  addScaled: (
    input: TVector,
    direction: TVector,
    scalar: number,
  ) => Promise<TVector>;
  addScaledByDot: (
    input: TVector,
    direction: TVector,
    dotLeft: TVector,
    dotRight: TVector,
    dotScale: number,
    dotBias: number,
  ) => Promise<TVector>;
  addScaledByDotAndScalarBuffer: (
    input: TVector,
    direction: TVector,
    dotLeft: TVector,
    dotRight: TVector,
    dotScale: number,
    scalarBuffer: TVector,
    scalarScale: number,
  ) => Promise<TVector>;
  dotToScalarBuffer: (a: TVector, b: TVector) => Promise<TVector>;
  addScaledByScalarBuffer: (
    input: TVector,
    direction: TVector,
    scalarBuffer: TVector,
    scalarScale: number,
  ) => Promise<TVector>;
  readScalarBuffer: (scalarBuffer: TVector) => Promise<number>;
  dot: (a: TVector, b: TVector) => Promise<number>;
  dotPairWithRight: (
    leftA: TVector,
    leftB: TVector,
    right: TVector,
  ) => Promise<readonly [number, number]>;
  absSum: (input: TVector) => Promise<number>;
  updateClamp: (
    input: TVector,
    direction: TVector,
    learningRate: number,
  ) => Promise<TVector>;
  adamUpdateClamp: (
    input: TVector,
    grad: TVector,
    m: TVector,
    v: TVector,
    step: number,
    config: InputOptimizerConfig,
  ) => Promise<TVector>;
  dispose: (input: TVector) => void;
};

export type InputOptimizer<TVector> = {
  step: (input: TVector, grad: TVector, step: number) => Promise<TVector>;
  dispose: () => void;
};
