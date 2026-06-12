import type {
  InputOptimizer,
  InputOptimizerConfig,
  InputOptimizerVectorOps,
} from "./types";

type LbfgsHistoryEntry<TVector> = {
  oldDir: TVector;
  oldStep: TVector;
  rho: number;
};

export const createInputOptimizer = <TVector>(
  config: InputOptimizerConfig,
  ops: InputOptimizerVectorOps<TVector>,
): InputOptimizer<TVector> => {
  let adamM: TVector | null = null;
  let adamV: TVector | null = null;
  let previousGrad: TVector | null = null;
  let previousStep: TVector | null = null;
  let lbfgsIteration = 0;
  let lbfgsHDiag = 1;
  const lbfgsHistory: LbfgsHistoryEntry<TVector>[] = [];

  const disposeHistoryEntry = (entry: LbfgsHistoryEntry<TVector>): void => {
    ops.dispose(entry.oldDir);
    ops.dispose(entry.oldStep);
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

  const addLbfgsHistory = async (grad: TVector): Promise<void> => {
    if (previousStep === null || previousGrad === null) return;
    const oldStep = previousStep;
    let oldDir: TVector | null = null;
    try {
      oldDir = await ops.sub(grad, previousGrad);
      const [ys, yy] = await ops.dotPairWithRight(oldStep, oldDir, oldDir);
      previousStep = null;
      if (ys > 1e-10) {
        lbfgsHDiag = ys / yy;
        lbfgsHistory.push({ oldDir, oldStep, rho: 1 / ys });
        oldDir = null;
        if (lbfgsHistory.length > config.lbfgsMemory) {
          const removed = lbfgsHistory.shift();
          if (removed !== undefined) disposeHistoryEntry(removed);
        }
        return;
      }
      ops.dispose(oldStep);
    } finally {
      if (oldDir !== null) ops.dispose(oldDir);
    }
  };

  const makeLbfgsDirection = async (grad: TVector): Promise<TVector> => {
    // Two-loop L-BFGS recursion. We keep alpha_i as GPU scalar buffers to avoid
    // scalar readbacks between loops; the second loop consumes those buffers.
    let q = await ops.scale(grad, -1);
    const alphaBuffers: TVector[] = [];
    const rhos = new Float32Array(lbfgsHistory.length);
    for (let i = lbfgsHistory.length - 1; i >= 0; i -= 1) {
      const { oldDir, oldStep, rho } = lbfgsHistory[i];
      const alphaBuffer = await ops.dotToScalarBuffer(oldStep, q);
      alphaBuffers[i] = alphaBuffer;
      rhos[i] = rho;
      // q <- q - (rho_i * alphaRaw_i) * y_i, with alphaRaw_i = dot(s_i, q)
      const nextQ = await ops.addScaledByScalarBuffer(
        q,
        oldDir,
        alphaBuffer,
        -rho,
      );
      ops.dispose(q);
      q = nextQ;
    }

    const scaled = await ops.scale(q, lbfgsHDiag);
    ops.dispose(q);
    q = scaled;

    for (let i = 0; i < lbfgsHistory.length; i += 1) {
      const { oldDir, oldStep, rho } = lbfgsHistory[i];
      // q <- q + (rho_i * alphaRaw_i - rho_i * dot(y_i, q)) * s_i
      const nextQ = await ops.addScaledByDotAndScalarBuffer(
        q,
        oldStep,
        oldDir,
        q,
        -rho,
        alphaBuffers[i],
        rhos[i],
      );
      ops.dispose(alphaBuffers[i]);
      ops.dispose(q);
      q = nextQ;
    }

    return q;
  };

  const runLbfgs = async (input: TVector, grad: TVector): Promise<TVector> => {
    lbfgsIteration += 1;
    if (lbfgsIteration === 1) {
      lbfgsHDiag = 1;
    } else {
      await addLbfgsHistory(grad);
    }
    const direction = await makeLbfgsDirection(grad);
    let nextInput: TVector;
    let nextPreviousStep: TVector;
    if (lbfgsIteration === 1) {
      const stepSizeBuffer = await ops.lbfgsInitialStepSizeBuffer(
        grad,
        config.learningRate,
      );
      [nextInput, nextPreviousStep] =
        await ops.updateClampAndScaleByScalarBuffer(
          input,
          direction,
          stepSizeBuffer,
          -1,
          1,
        );
      ops.dispose(stepSizeBuffer);
    } else {
      [nextInput, nextPreviousStep] = await ops.updateClampAndScale(
        input,
        direction,
        -config.learningRate,
        config.learningRate,
      );
    }

    const nextPreviousGrad = await ops.clone(grad);
    if (previousGrad !== null) ops.dispose(previousGrad);
    if (previousStep !== null) ops.dispose(previousStep);
    previousGrad = nextPreviousGrad;
    previousStep = nextPreviousStep;
    ops.dispose(direction);
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
    if (previousGrad !== null) ops.dispose(previousGrad);
    if (previousStep !== null) ops.dispose(previousStep);
    for (const entry of lbfgsHistory) disposeHistoryEntry(entry);
    adamM = null;
    adamV = null;
    previousGrad = null;
    previousStep = null;
    lbfgsIteration = 0;
    lbfgsHDiag = 1;
    lbfgsHistory.length = 0;
  };

  return { step, dispose };
};
