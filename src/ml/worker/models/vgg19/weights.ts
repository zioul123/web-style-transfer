import type { TensorShape4D } from "../../runtime/tensorShapes";

export type Vgg19WeightsRecord = Record<
  string,
  number[] | [number, number, number, number]
>;

export type Vgg19ConvLayerCacheEntry = {
  shape: TensorShape4D;
  values: Float32Array;
  bias: Float32Array;
};

export type Vgg19ConvLayerCache = Record<
  number,
  Vgg19ConvLayerCacheEntry | undefined
>;

const MAX_VGG19_FEATURE_LAYER_INDEX = 29;

const isFiniteNumber = (value: unknown): value is number =>
  typeof value === "number" && Number.isFinite(value);

const isNumberArray = (value: unknown): value is number[] =>
  Array.isArray(value) && value.every(isFiniteNumber);

const isTensorShape4D = (value: unknown): value is TensorShape4D =>
  Array.isArray(value) &&
  value.length === 4 &&
  value.every(
    (dimension) => Number.isInteger(dimension) && dimension > 0,
  );

export const parseVgg19ConvLayerCache = (
  weights: Vgg19WeightsRecord,
): Vgg19ConvLayerCache => {
  const convLayerCache: Vgg19ConvLayerCache = {};
  for (
    let layerIndex = 0;
    layerIndex <= MAX_VGG19_FEATURE_LAYER_INDEX;
    layerIndex += 1
  ) {
    const weightShape = weights[`conv${layerIndex}.weightShape`];
    const weightValues = weights[`conv${layerIndex}.weightValues`];
    const biasValues = weights[`conv${layerIndex}.biasValues`];
    if (
      isTensorShape4D(weightShape) &&
      isNumberArray(weightValues) &&
      isNumberArray(biasValues)
    ) {
      convLayerCache[layerIndex] = {
        shape: weightShape,
        values: new Float32Array(weightValues),
        bias: new Float32Array(biasValues),
      };
    }
  }
  return convLayerCache;
};
