import type { TensorShape, WorkerRequest, WorkerResponse, WorkerTensor } from '../types'

type AskWorker = (payload: WorkerRequest) => Promise<WorkerResponse>

type VggConvWeights = { shape: [number, number, number, number]; values: number[]; bias: number[] }

type FirstPoolGraphWeights = { conv1_1: VggConvWeights; conv1_2: VggConvWeights; conv2_1: VggConvWeights }

type FirstPoolOptimizationConfig = {
  inputShape: TensorShape
  mean: [number, number, number]
  std: [number, number, number]
  contentImageValues: number[]
  styleImageValues: number[]
  initialInputValues: number[]
  weights: FirstPoolGraphWeights
  contentWeight: number
  styleWeightConv1_1: number
  styleWeightConv2_1: number
  learningRate: number
  steps: number
}

const getVector = (response: WorkerResponse, label: string): number[] => {
  if (response.type !== 'tensor-op-result' || !response.ok || !('values' in response)) throw new Error(`${label} expected tensor values`)
  return response.values
}
const getScalar = (response: WorkerResponse, label: string): number => {
  if (response.type !== 'tensor-op-result' || !response.ok || !('scalar' in response)) throw new Error(`${label} expected scalar`)
  return response.scalar
}

const convForward = async (ask: AskWorker, id: string, input: WorkerTensor, weights: VggConvWeights): Promise<WorkerTensor> => ({
  shape: [1, weights.shape[0], input.shape[2], input.shape[3]],
  values: getVector(await ask({ type: 'tensor-op', id, op: 'conv2d-forward', input, weight: { shape: weights.shape, values: weights.values }, bias: weights.bias }), id),
})

export const runFirstPoolOptimizerLoop = async (ask: AskWorker, config: FirstPoolOptimizationConfig): Promise<{ losses: number[]; finalValues: number[] }> => {
  const normalizedContent: WorkerTensor = { shape: config.inputShape, values: getVector(await ask({ type: 'tensor-op', id: 'p5-content-norm', op: 'normalize-forward', input: { shape: config.inputShape, values: config.contentImageValues }, mean: config.mean, std: config.std }), 'content-norm') }
  const normalizedStyle: WorkerTensor = { shape: config.inputShape, values: getVector(await ask({ type: 'tensor-op', id: 'p5-style-norm', op: 'normalize-forward', input: { shape: config.inputShape, values: config.styleImageValues }, mean: config.mean, std: config.std }), 'style-norm') }

  const styleConv1 = await convForward(ask, 'p5-style-conv1_1', normalizedStyle, config.weights.conv1_1)
  const styleConv1Relu: WorkerTensor = { shape: styleConv1.shape, values: getVector(await ask({ type: 'tensor-op', id: 'p5-style-relu1_1', op: 'relu-forward', input: styleConv1 }), 'style-relu1') }
  const styleConv2 = await convForward(ask, 'p5-style-conv1_2', styleConv1Relu, config.weights.conv1_2)
  const styleConv2Relu: WorkerTensor = { shape: styleConv2.shape, values: getVector(await ask({ type: 'tensor-op', id: 'p5-style-relu1_2', op: 'relu-forward', input: styleConv2 }), 'style-relu2') }
  const stylePool: WorkerTensor = { shape: [1, 64, 8, 8], values: getVector(await ask({ type: 'tensor-op', id: 'p5-style-pool1', op: 'maxpool2d-forward', input: styleConv2Relu }), 'style-pool1') }
  const styleConv3 = await convForward(ask, 'p5-style-conv2_1', stylePool, config.weights.conv2_1)

  const contentConv1 = await convForward(ask, 'p5-content-conv1_1', normalizedContent, config.weights.conv1_1)
  const contentConv1Relu: WorkerTensor = { shape: contentConv1.shape, values: getVector(await ask({ type: 'tensor-op', id: 'p5-content-relu1_1', op: 'relu-forward', input: contentConv1 }), 'content-relu1') }
  const contentConv2 = await convForward(ask, 'p5-content-conv1_2', contentConv1Relu, config.weights.conv1_2)

  let inputValues: number[] = Array.from(config.initialInputValues)
  const losses: number[] = []

  for (let step = 0; step < config.steps; step += 1) {
    const normalizedInput: WorkerTensor = { shape: config.inputShape, values: getVector(await ask({ type: 'tensor-op', id: `p5-step-${step}-norm`, op: 'normalize-forward', input: { shape: config.inputShape, values: inputValues }, mean: config.mean, std: config.std }), 'input-norm') }
    const conv1 = await convForward(ask, `p5-step-${step}-conv1_1`, normalizedInput, config.weights.conv1_1)
    const relu1: WorkerTensor = { shape: conv1.shape, values: getVector(await ask({ type: 'tensor-op', id: `p5-step-${step}-relu1_1`, op: 'relu-forward', input: conv1 }), 'relu1') }
    const conv2 = await convForward(ask, `p5-step-${step}-conv1_2`, relu1, config.weights.conv1_2)
    const relu2: WorkerTensor = { shape: conv2.shape, values: getVector(await ask({ type: 'tensor-op', id: `p5-step-${step}-relu1_2`, op: 'relu-forward', input: conv2 }), 'relu2') }
    const pool1: WorkerTensor = { shape: [1, 64, 8, 8], values: getVector(await ask({ type: 'tensor-op', id: `p5-step-${step}-pool1`, op: 'maxpool2d-forward', input: relu2 }), 'pool1') }
    const conv3 = await convForward(ask, `p5-step-${step}-conv2_1`, pool1, config.weights.conv2_1)

    const styleLoss1 = getScalar(await ask({ type: 'tensor-op', id: `p5-step-${step}-style1`, op: 'style-loss', input: conv1, target: styleConv1, styleWeight: config.styleWeightConv1_1 }), 'style1')
    const styleLoss2 = getScalar(await ask({ type: 'tensor-op', id: `p5-step-${step}-style2`, op: 'style-loss', input: conv3, target: styleConv3, styleWeight: config.styleWeightConv2_1 }), 'style2')
    const contentLoss = getScalar(await ask({ type: 'tensor-op', id: `p5-step-${step}-content`, op: 'content-loss', input: conv2, target: contentConv2, contentWeight: config.contentWeight }), 'content')
    losses.push(styleLoss1 + styleLoss2 + contentLoss)

    const gradConv1 = getVector(await ask({ type: 'tensor-op', id: `p5-step-${step}-g-style1`, op: 'style-loss-backward', input: conv1, target: styleConv1, styleWeight: config.styleWeightConv1_1 }), 'g-style1')
    const gradConv3 = getVector(await ask({ type: 'tensor-op', id: `p5-step-${step}-g-style2`, op: 'style-loss-backward', input: conv3, target: styleConv3, styleWeight: config.styleWeightConv2_1 }), 'g-style2')
    const gradContent = getVector(await ask({ type: 'tensor-op', id: `p5-step-${step}-g-content`, op: 'content-loss-backward', input: conv2, target: contentConv2, contentWeight: config.contentWeight }), 'g-content')

    const gradPoolFromConv3 = getVector(await ask({ type: 'tensor-op', id: `p5-step-${step}-b-conv2_1`, op: 'conv2d-backward-input', inputShape: pool1.shape, gradOut: { shape: conv3.shape, values: gradConv3 }, weight: { shape: config.weights.conv2_1.shape, values: config.weights.conv2_1.values } }), 'b-conv2_1')
    const gradRelu2 = getVector(await ask({ type: 'tensor-op', id: `p5-step-${step}-b-pool1`, op: 'maxpool2d-backward', input: relu2, gradOut: { shape: pool1.shape, values: gradPoolFromConv3 } }), 'b-pool1')
    const gradConv2FromPool = getVector(await ask({ type: 'tensor-op', id: `p5-step-${step}-b-relu1_2`, op: 'relu-backward', input: conv2, gradOut: { shape: relu2.shape, values: gradRelu2 } }), 'b-relu1_2')
    const gradConv2Total = gradConv2FromPool.map((value, index) => value + gradContent[index])
    const gradRelu1FromConv2 = getVector(await ask({ type: 'tensor-op', id: `p5-step-${step}-b-conv1_2`, op: 'conv2d-backward-input', inputShape: relu1.shape, gradOut: { shape: conv2.shape, values: gradConv2Total }, weight: { shape: config.weights.conv1_2.shape, values: config.weights.conv1_2.values } }), 'b-conv1_2')
    const gradConv1FromConv2 = getVector(await ask({ type: 'tensor-op', id: `p5-step-${step}-b-relu1_1`, op: 'relu-backward', input: conv1, gradOut: { shape: relu1.shape, values: gradRelu1FromConv2 } }), 'b-relu1_1')
    const gradConv1Total = gradConv1FromConv2.map((value, index) => value + gradConv1[index])
    const gradNorm = getVector(await ask({ type: 'tensor-op', id: `p5-step-${step}-b-conv1_1`, op: 'conv2d-backward-input', inputShape: normalizedInput.shape, gradOut: { shape: conv1.shape, values: gradConv1Total }, weight: { shape: config.weights.conv1_1.shape, values: config.weights.conv1_1.values } }), 'b-conv1_1')
    const gradInput = getVector(await ask({ type: 'tensor-op', id: `p5-step-${step}-b-norm`, op: 'normalize-backward', input: { shape: config.inputShape, values: inputValues }, gradOut: { shape: config.inputShape, values: gradNorm }, std: config.std }), 'b-norm')
    inputValues = inputValues.map((value, index) => Math.max(0, Math.min(1, value - config.learningRate * gradInput[index])))
  }

  return { losses, finalValues: inputValues }
}
