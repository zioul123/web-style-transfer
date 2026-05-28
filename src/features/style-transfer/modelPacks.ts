import {
  parseVgg19ManifestBackedLayerCache,
  type Vgg19WeightsManifest,
} from "../../ml/worker/models/vgg19/weights";
import type { FullWeights } from "./types/controller";

export const VGG_PACK_OPTIONS = [
  { name: "fp32", label: "FP32" },
  { name: "fp16", label: "FP16" },
  { name: "int8-per-channel", label: "INT8 per-channel" },
  { name: "int8log-per-channel", label: "INT8 log per-channel" },
  { name: "int4-experimental", label: "INT4 experimental" },
  { name: "int4log-experimental", label: "INT4 log experimental" },
] as const;

export type VggPackName = (typeof VGG_PACK_OPTIONS)[number]["name"];

export const DEFAULT_VGG_PACK: VggPackName = "int8-per-channel";
export const VGG_PACK_STORAGE_KEY = "style-transfer.vgg-pack";

const loadJson = async <T,>(url: string): Promise<T> => {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Failed to load ${url}: ${response.status}`);
  return (await response.json()) as T;
};

export const loadVgg19ManifestWeightsForPack = async (
  pack: VggPackName,
): Promise<FullWeights> => {
  const manifest = await loadJson<Vgg19WeightsManifest>(
    `/vgg19-models/${pack}/manifest.json`,
  );
  const shardEntries = await Promise.all(
    manifest.shards.map(async (shard) => {
      const response = await fetch(`/vgg19-models/${pack}/${shard.name}`);
      if (!response.ok) {
        throw new Error(`Missing weight shard: ${shard.name}`);
      }
      return [shard.name, await response.arrayBuffer()] as const;
    }),
  );
  const shardBuffers: Record<string, ArrayBuffer> = Object.fromEntries(shardEntries);
  const cache = await parseVgg19ManifestBackedLayerCache(manifest, shardBuffers);
  const weights: FullWeights = {};
  for (let layerIndex = 0; layerIndex <= 29; layerIndex += 1) {
    const layer = cache[layerIndex];
    if (layer === undefined) continue;
    weights[`conv${layerIndex}.weightShape`] = [...layer.shape];
    weights[`conv${layerIndex}.weightValues`] = Array.from(layer.values);
    weights[`conv${layerIndex}.biasValues`] = Array.from(layer.bias);
  }
  return weights;
};
