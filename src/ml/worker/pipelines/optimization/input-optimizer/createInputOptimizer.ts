import type {
  InputOptimizer,
  InputOptimizerConfig,
  InputOptimizerVectorOps,
} from "./types";

type LbfgsHistoryEntry<TVector> = {
  s: TVector;
  y: TVector;
  rho: number;
};

export const createInputOptimizer = <TVector>(
  config: InputOptimizerConfig,
  ops: InputOptimizerVectorOps<TVector>,
): InputOptimizer<TVector> => {
  let adamM: TVector | null = null;
  let adamV: TVector | null = null;
  let previousInput: TVector | null = null;
  let previousGrad: TVector | null = null;
  const lbfgsHistory: LbfgsHistoryEntry<TVector>[] = [];

  const disposeHistoryEntry = (entry: LbfgsHistoryEntry<TVector>): void => {
    ops.dispose(entry.s);
    ops.dispose(entry.y);
  };

  const runSgd = async (input: TVector, grad: TVector): Promise<TVector> =>
    ops.updateClamp(input, grad, config.learningRate);

  const runAdam = async (
    input: TVector,
    grad: TVector,
    step: number,
  ): Promise<TVector> => {
    if (adamM === null) adamM = await ops.zeros();
    if (adamV === null) adamV = await ops.zeros();
    return ops.adamUpdateClamp(input, grad, adamM, adamV, step, config);
  };

  const addLbfgsHistory = async (
    input: TVector,
    grad: TVector,
  ): Promise<void> => {
    if (previousInput === null || previousGrad === null) return;
    const s = await ops.sub(input, previousInput);
    const y = await ops.sub(grad, previousGrad);
    const sy = await ops.dot(s, y);
    if (sy > config.lbfgsEpsilon) {
      lbfgsHistory.push({ s, y, rho: 1 / sy });
      if (lbfgsHistory.length > config.lbfgsMemory) {
        const removed = lbfgsHistory.shift();
        if (removed !== undefined) disposeHistoryEntry(removed);
      }
      return;
    }
    ops.dispose(s);
    ops.dispose(y);
  };

  const makeLbfgsDirection = async (grad: TVector): Promise<TVector> => {
    let q = await ops.clone(grad);
    const alphas = new Float32Array(lbfgsHistory.length);
    for (let i = lbfgsHistory.length - 1; i >= 0; i -= 1) {
      const { s, y, rho } = lbfgsHistory[i];
      const alpha = rho * (await ops.dot(s, q));
      alphas[i] = alpha;
      const nextQ = await ops.addScaled(q, y, -alpha);
      ops.dispose(q);
      q = nextQ;
    }

    if (lbfgsHistory.length > 0) {
      const last = lbfgsHistory[lbfgsHistory.length - 1];
      const ys = await ops.dot(last.y, last.s);
      const yy = await ops.dot(last.y, last.y);
      if (yy > config.lbfgsEpsilon) {
        const scaled = await ops.scale(q, ys / yy);
        ops.dispose(q);
        q = scaled;
      }
    }

    for (let i = 0; i < lbfgsHistory.length; i += 1) {
      const { s, y, rho } = lbfgsHistory[i];
      const beta = rho * (await ops.dot(y, q));
      const nextQ = await ops.addScaled(q, s, alphas[i] - beta);
      ops.dispose(q);
      q = nextQ;
    }

    return q;
  };

  const runLbfgs = async (input: TVector, grad: TVector): Promise<TVector> => {
    await addLbfgsHistory(input, grad);
    const direction = await makeLbfgsDirection(grad);
    const nextInput = await ops.updateClamp(
      input,
      direction,
      config.learningRate,
    );
    ops.dispose(direction);

    const nextPreviousInput = await ops.clone(input);
    const nextPreviousGrad = await ops.clone(grad);
    if (previousInput !== null) ops.dispose(previousInput);
    if (previousGrad !== null) ops.dispose(previousGrad);
    previousInput = nextPreviousInput;
    previousGrad = nextPreviousGrad;
    return nextInput;
  };

  const step = async (
    input: TVector,
    grad: TVector,
    stepIndex: number,
  ): Promise<TVector> => {
    if (config.optimizer === "sgd") return runSgd(input, grad);
    if (config.optimizer === "adam") return runAdam(input, grad, stepIndex);
    return runLbfgs(input, grad);
  };

  const dispose = (): void => {
    if (adamM !== null) ops.dispose(adamM);
    if (adamV !== null) ops.dispose(adamV);
    if (previousInput !== null) ops.dispose(previousInput);
    if (previousGrad !== null) ops.dispose(previousGrad);
    for (const entry of lbfgsHistory) disposeHistoryEntry(entry);
    adamM = null;
    adamV = null;
    previousInput = null;
    previousGrad = null;
    lbfgsHistory.length = 0;
  };

  return { step, dispose };
};
