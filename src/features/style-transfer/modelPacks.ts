import {
  parseVgg19ManifestBackedLayerCache,
  type Vgg19WeightsManifest,
} from "../../ml/worker/models/vgg19/weights";
import type { FullWeights } from "./types/controller";
import { vgg19ModelUrl } from "../../shared/assetUrls";
import {
  clearModelCache,
  createModelCacheKey,
  readCachedModelPack,
  readModelCacheStatus,
  writeCachedModelPack,
  type ModelCacheStatus,
  type ModelCacheTier,
} from "./modelCache";

export const VGG_PACK_OPTIONS = [
  { name: "fp32", label: "FP32" },
  { name: "fp16", label: "FP16" },
  { name: "int8-per-channel", label: "INT8 per-channel" },
  { name: "int8log-per-channel", label: "INT8 log per-channel" },
  { name: "int4-experimental", label: "INT4 experimental" },
  { name: "int4log-experimental", label: "INT4 log experimental" },
] as const;

export type VggPackName = (typeof VGG_PACK_OPTIONS)[number]["name"];

export type VggPackLoadState = "downloading" | "cached" | "ready";

export type VggPackLoadResult = {
  weights: FullWeights;
  state: VggPackLoadState;
  cacheStatus: ModelCacheStatus;
};

export const DEFAULT_VGG_PACK: VggPackName = "int4log-experimental";
export const VGG_PACK_STORAGE_KEY = "style-transfer.vgg-pack";

const loadJson = async <T>(url: string): Promise<T> => {
  const response = await fetch(url);
  if (!response.ok)
    throw new Error(`Failed to load ${url}: ${response.status}`);
  return (await response.json()) as T;
};

const toFullWeights = async (
  manifest: Vgg19WeightsManifest,
  shardBuffers: Record<string, ArrayBuffer>,
): Promise<FullWeights> => {
  const cache = await parseVgg19ManifestBackedLayerCache(
    manifest,
    shardBuffers,
  );
  const weights: FullWeights = {};
  for (let layerIndex = 0; layerIndex <= 29; layerIndex += 1) {
    const layer = cache[layerIndex];
    if (layer === undefined) continue;
    weights[`conv${layerIndex}.weightShape`] = [...layer.shape];
    weights[`conv${layerIndex}.weightValues`] = layer.values;
    weights[`conv${layerIndex}.biasValues`] = layer.bias;
  }
  return weights;
};

const downloadShards = async (
  pack: VggPackName,
  manifest: Vgg19WeightsManifest,
): Promise<Record<string, ArrayBuffer>> => {
  const shardEntries = await Promise.all(
    manifest.shards.map(async (shard) => {
      const response = await fetch(vgg19ModelUrl(`${pack}/${shard.name}`));
      if (!response.ok) {
        throw new Error(`Missing weight shard: ${shard.name}`);
      }
      return [shard.name, await response.arrayBuffer()] as const;
    }),
  );
  return Object.fromEntries(shardEntries);
};

export const loadVgg19ManifestWeightsForPack = async (
  pack: VggPackName,
): Promise<VggPackLoadResult> => {
  const manifest = await loadJson<Vgg19WeightsManifest>(
    vgg19ModelUrl(`${pack}/manifest.json`),
  );
  const cacheKey = createModelCacheKey(
    manifest.modelId,
    manifest.version,
    pack as ModelCacheTier,
  );

  const cached = await readCachedModelPack(cacheKey);
  if (cached !== null) {
    try {
      const weights = await toFullWeights(cached.manifest, cached.shards);
      return {
        weights,
        state: "cached",
        cacheStatus: await readModelCacheStatus(),
      };
    } catch {
      // Cached payload failed integrity checks in parser; fall through to re-download.
    }
  }

  const shards = await downloadShards(pack, manifest);
  const weights = await toFullWeights(manifest, shards);
  const totalBytes = Object.values(shards).reduce(
    (acc, buffer) => acc + buffer.byteLength,
    0,
  );
  await writeCachedModelPack({
    cacheKey,
    modelId: manifest.modelId,
    version: manifest.version,
    tier: pack as ModelCacheTier,
    manifest,
    shards,
    totalBytes,
    updatedAtMs: Date.now(),
  });

  return {
    weights,
    state: "ready",
    cacheStatus: await readModelCacheStatus(),
  };
};

export const clearCachedVggModelPacks = async (): Promise<ModelCacheStatus> => {
  await clearModelCache();
  return await readModelCacheStatus();
};

export const getCachedVggModelPackStatus =
  async (): Promise<ModelCacheStatus> => await readModelCacheStatus();
