import type { TensorShape4D } from "../../runtime/tensorShapes";

export type Vgg19WeightsRecord = Record<
  string,
  number[] | [number, number, number, number]
>;

type ManifestTensorEntry = {
  shape: number[];
  dtype: "float32" | "float16" | "int8";
  shard: string;
  offset: number;
  length: number;
  quantization?: {
    scheme: "per-channel-symmetric";
    axis: 0;
    scale: number[];
    zeroPoint: number[];
  };
};

type ManifestLayerEntry = {
  weight: ManifestTensorEntry;
  bias: ManifestTensorEntry;
};

export type Vgg19WeightsManifest = {
  modelId: string;
  version: string;
  format: "fp32-le" | "fp16-le" | "int8-per-channel-le";
  layers: Record<string, ManifestLayerEntry>;
  shards: Array<{ name: string; byteLength: number }>;
  checksums: { algorithm: "sha256"; files: Record<string, string> };
  quantization: { scheme: "none" | "per-channel-symmetric"; dtype: string };
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

const tensorElemCount = (shape: readonly number[]): number => shape.reduce((a, b) => a * b, 1);
const expectedByteLength = (shape: readonly number[], dtype: ManifestTensorEntry["dtype"]): number =>
  tensorElemCount(shape) * (dtype === "float32" ? 4 : dtype === "float16" ? 2 : 1);
const decodeWeights = (entry: ManifestTensorEntry, shard: ArrayBuffer): Float32Array => {
  const bytes = shard.slice(entry.offset, entry.offset + entry.length);
  if (entry.dtype === "float32") return new Float32Array(bytes);
  if (entry.dtype === "float16") {
    const f16 = new Uint16Array(bytes);
    const out = new Float32Array(f16.length);
    for (let i = 0; i < f16.length; i += 1) {
      const u = f16[i];
      const s = (u & 0x8000) !== 0 ? -1 : 1;
      const e = (u >>> 10) & 0x1f;
      const f = u & 0x3ff;
      out[i] = e === 0 ? s * 2 ** (-14) * (f / 1024) : e === 31 ? (f === 0 ? s * Infinity : Number.NaN) : s * 2 ** (e - 15) * (1 + f / 1024);
    }
    return out;
  }
  const q = new Int8Array(bytes);
  if (entry.quantization === undefined) throw new Error("Missing quantization metadata for int8 tensor.");
  const channelCount = entry.shape[entry.quantization.axis];
  if (entry.quantization.scale.length !== channelCount) throw new Error("Invalid int8 quantization scale length.");
  const channelStride = tensorElemCount(entry.shape.slice(1));
  const out = new Float32Array(q.length);
  for (let channelIndex = 0; channelIndex < channelCount; channelIndex += 1) {
    const scale = entry.quantization.scale[channelIndex];
    const zero = entry.quantization.zeroPoint[channelIndex] ?? 0;
    const start = channelIndex * channelStride;
    const end = start + channelStride;
    for (let i = start; i < end; i += 1) out[i] = (q[i] - zero) * scale;
  }
  return out;
};

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
  if (!["fp32-le", "fp16-le", "int8-per-channel-le"].includes(manifest.format)) throw new Error(`Unsupported weights format: ${manifest.format}`);
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
    const weightExpected = expectedByteLength(weight.shape, weight.dtype);
    const biasExpected = expectedByteLength(bias.shape, bias.dtype);
    if (weight.length !== weightExpected) throw new Error(`conv${layerIndex} weight byte length mismatch.`);
    if (bias.length !== biasExpected) throw new Error(`conv${layerIndex} bias byte length mismatch.`);
    const weightShard = shardBuffers[weight.shard];
    const biasShard = shardBuffers[bias.shard];
    if (weightShard === undefined || biasShard === undefined) throw new Error(`Missing shard for conv${layerIndex}.`);
    if (weight.offset + weight.length > weightShard.byteLength) throw new Error(`conv${layerIndex} weight range out of bounds.`);
    if (bias.offset + bias.length > biasShard.byteLength) throw new Error(`conv${layerIndex} bias range out of bounds.`);

    const weightValues = decodeWeights(weight, weightShard);
    const biasValues = new Float32Array(biasShard.slice(bias.offset, bias.offset + bias.length));
    if (!isTensorShape4D(weight.shape)) throw new Error(`conv${layerIndex} weight shape invalid.`);
    if (bias.shape.length !== 1 || !Number.isInteger(bias.shape[0])) throw new Error(`conv${layerIndex} bias shape invalid.`);
    if (biasValues.length !== bias.shape[0]) throw new Error(`conv${layerIndex} bias shape/value mismatch.`);
    cache[layerIndex] = { shape: weight.shape, values: weightValues, bias: biasValues };
  }
  return cache;
};
