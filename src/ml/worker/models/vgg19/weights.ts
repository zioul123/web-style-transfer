import type { TensorShape4D } from "../../runtime/tensorShapes";

export type Vgg19WeightsRecord = Record<
  string,
  number[] | [number, number, number, number]
>;

type ManifestTensorEntry = {
  shape: number[];
  dtype: "float32";
  shard: string;
  offset: number;
  length: number;
};

type ManifestLayerEntry = {
  weight: ManifestTensorEntry;
  bias: ManifestTensorEntry;
};

export type Vgg19WeightsManifest = {
  modelId: string;
  version: string;
  format: "float32-le";
  layers: Record<string, ManifestLayerEntry>;
  shards: Array<{ name: string; byteLength: number }>;
  checksums: { algorithm: "sha256"; files: Record<string, string> };
  quantization: { scheme: "none"; dtype: "float32"; scale: null; zeroPoint: null };
};

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
  Array.isArray(value) && value.length === 4 && value.every((d) => Number.isInteger(d) && d > 0);

const expectedFloatByteLength = (shape: readonly number[]): number =>
  shape.reduce((a, b) => a * b, 1) * 4;

export const parseVgg19ConvLayerCache = (
  weights: Vgg19WeightsRecord,
): Vgg19ConvLayerCache => {
  const convLayerCache: Vgg19ConvLayerCache = {};
  for (let layerIndex = 0; layerIndex <= MAX_VGG19_FEATURE_LAYER_INDEX; layerIndex += 1) {
    const weightShape = weights[`conv${layerIndex}.weightShape`];
    const weightValues = weights[`conv${layerIndex}.weightValues`];
    const biasValues = weights[`conv${layerIndex}.biasValues`];
    if (isTensorShape4D(weightShape) && isNumberArray(weightValues) && isNumberArray(biasValues)) {
      convLayerCache[layerIndex] = {
        shape: weightShape,
        values: new Float32Array(weightValues),
        bias: new Float32Array(biasValues),
      };
    }
  }
  return convLayerCache;
};

const hex = (bytes: Uint8Array): string => Array.from(bytes).map((b) => b.toString(16).padStart(2, "0")).join("");

const sha256 = async (buffer: ArrayBuffer): Promise<string> => {
  const digest = await crypto.subtle.digest("SHA-256", buffer);
  return hex(new Uint8Array(digest));
};

export const parseVgg19ManifestBackedLayerCache = async (
  manifest: Vgg19WeightsManifest,
  shardBuffers: Record<string, ArrayBuffer>,
): Promise<Vgg19ConvLayerCache> => {
  if (manifest.format !== "float32-le") throw new Error(`Unsupported weights format: ${manifest.format}`);
  if (manifest.checksums.algorithm !== "sha256") throw new Error("Unsupported checksum algorithm.");

  for (const shardMeta of manifest.shards) {
    const buffer = shardBuffers[shardMeta.name];
    if (buffer === undefined) throw new Error(`Missing shard buffer: ${shardMeta.name}`);
    if (buffer.byteLength !== shardMeta.byteLength) {
      throw new Error(`Shard ${shardMeta.name} length mismatch: ${buffer.byteLength} vs ${shardMeta.byteLength}.`);
    }
    const expected = manifest.checksums.files[shardMeta.name];
    if (expected !== undefined) {
      const actual = await sha256(buffer);
      if (actual !== expected) throw new Error(`Checksum validation failed for shard ${shardMeta.name}.`);
    }
  }

  const cache: Vgg19ConvLayerCache = {};
  for (let layerIndex = 0; layerIndex <= MAX_VGG19_FEATURE_LAYER_INDEX; layerIndex += 1) {
    const layer = manifest.layers[`conv${layerIndex}`];
    if (layer === undefined) continue;
    const { weight, bias } = layer;
    const weightExpected = expectedFloatByteLength(weight.shape);
    const biasExpected = expectedFloatByteLength(bias.shape);
    if (weight.length !== weightExpected) throw new Error(`conv${layerIndex} weight byte length mismatch.`);
    if (bias.length !== biasExpected) throw new Error(`conv${layerIndex} bias byte length mismatch.`);
    const weightShard = shardBuffers[weight.shard];
    const biasShard = shardBuffers[bias.shard];
    if (weightShard === undefined || biasShard === undefined) throw new Error(`Missing shard for conv${layerIndex}.`);
    if (weight.offset + weight.length > weightShard.byteLength) throw new Error(`conv${layerIndex} weight range out of bounds.`);
    if (bias.offset + bias.length > biasShard.byteLength) throw new Error(`conv${layerIndex} bias range out of bounds.`);

    const weightValues = new Float32Array(weightShard.slice(weight.offset, weight.offset + weight.length));
    const biasValues = new Float32Array(biasShard.slice(bias.offset, bias.offset + bias.length));
    if (!isTensorShape4D(weight.shape)) throw new Error(`conv${layerIndex} weight shape invalid.`);
    if (bias.shape.length !== 1 || !Number.isInteger(bias.shape[0])) throw new Error(`conv${layerIndex} bias shape invalid.`);
    if (biasValues.length !== bias.shape[0]) throw new Error(`conv${layerIndex} bias shape/value mismatch.`);
    cache[layerIndex] = { shape: weight.shape, values: weightValues, bias: biasValues };
  }
  return cache;
};
