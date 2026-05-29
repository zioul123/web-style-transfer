import {
  parseVgg19ManifestBackedLayerCache,
  type Vgg19WeightsManifest,
  type Vgg19WeightsRecord,
} from "../../src/ml/worker/models/vgg19/weights";

export type Phase3FullPassFixture = {
  inputShape: [number, number, number, number];
  inputImageValues: number[];
  contentImageValues: number[];
  styleImageValues: number[];
  mean: [number, number, number];
  std: [number, number, number];
  styleLayerIndices: number[];
  contentLayerIndex: number;
  expectedStyleLossByLayer: Record<string, number>;
  expectedStyleLossTotal: number;
  expectedContentLoss: number;
  expectedTotalLoss: number;
  expectedGradients?: {
    content?: number[];
    styleByLayer?: Record<string, number[]>;
    total?: number[];
  };
};

export type FullPassModelSource =
  | "legacy-fp32-json"
  | "manifest-fp32"
  | "manifest-int8-per-channel";

export type FullPassArtifacts = {
  fixture: Phase3FullPassFixture;
  weights: Vgg19WeightsRecord;
  modelSource: FullPassModelSource;
  lossPrecision: number;
  firstStepGradientMeanTolerance: number;
  firstStepGradientMaxTolerance: number;
  weightedGradientMeanTolerance: number;
  weightedGradientMaxTolerance: number;
};

export type FullPassArtifactsResult =
  | { ok: true; artifacts: FullPassArtifacts }
  | { ok: false; reason: "missing-fixtures"; message: string };

const fullPassBaseUrl = "/vgg19-phase3-full-pass";
const modelBaseUrl = "/vgg19-models";

const loadJson = async <T>(url: string): Promise<T | null> => {
  const response = await fetch(url);
  if (!response.ok) return null;
  const text = await response.text();
  try {
    return JSON.parse(text) as T;
  } catch {
    return null;
  }
};

const layerCacheToWeightsRecord = (
  layerCache: Awaited<ReturnType<typeof parseVgg19ManifestBackedLayerCache>>,
): Vgg19WeightsRecord => {
  const weights: Vgg19WeightsRecord = {};
  for (let layerIndex = 0; layerIndex <= 29; layerIndex += 1) {
    const layer = layerCache[layerIndex];
    if (layer === undefined) continue;
    weights[`conv${layerIndex}.weightShape`] = [...layer.shape] as [
      number,
      number,
      number,
      number,
    ];
    weights[`conv${layerIndex}.weightValues`] = Array.from(layer.values);
    weights[`conv${layerIndex}.biasValues`] = Array.from(layer.bias);
  }
  return weights;
};

const loadLegacyWeights = async (): Promise<Vgg19WeightsRecord | null> =>
  loadJson<Vgg19WeightsRecord>(
    `${fullPassBaseUrl}/vgg19_conv0_to_conv28_weights.json`,
  );

const loadManifestWeights = async (
  pack: "fp32" | "int8-per-channel",
): Promise<Vgg19WeightsRecord | null> => {
  const manifest = await loadJson<Vgg19WeightsManifest>(
    `${modelBaseUrl}/${pack}/manifest.json`,
  );
  if (manifest === null) return null;
  const shardEntries = await Promise.all(
    manifest.shards.map(async (shard) => {
      const response = await fetch(`${modelBaseUrl}/${pack}/${shard.name}`);
      if (!response.ok) return null;
      return [shard.name, await response.arrayBuffer()] as const;
    }),
  );
  if (shardEntries.some((entry) => entry === null)) return null;
  const shardMap: Record<string, ArrayBuffer> = Object.fromEntries(
    shardEntries as Array<readonly [string, ArrayBuffer]>,
  );
  const layerCache = await parseVgg19ManifestBackedLayerCache(
    manifest,
    shardMap,
  );
  return layerCacheToWeightsRecord(layerCache);
};

const loadWeights = async (): Promise<{
  weights: Vgg19WeightsRecord;
  modelSource: FullPassModelSource;
} | null> => {
  const legacyWeights = await loadLegacyWeights();
  if (legacyWeights !== null) {
    return { weights: legacyWeights, modelSource: "legacy-fp32-json" };
  }

  const fp32Weights = await loadManifestWeights("fp32");
  if (fp32Weights !== null) {
    return { weights: fp32Weights, modelSource: "manifest-fp32" };
  }

  const int8Weights = await loadManifestWeights("int8-per-channel");
  if (int8Weights !== null) {
    return {
      weights: int8Weights,
      modelSource: "manifest-int8-per-channel",
    };
  }

  return null;
};

const tolerancesFor = (
  modelSource: FullPassModelSource,
): Omit<FullPassArtifacts, "fixture" | "weights" | "modelSource"> => {
  if (modelSource === "manifest-int8-per-channel") {
    return {
      lossPrecision: 1,
      firstStepGradientMeanTolerance: 0.04,
      firstStepGradientMaxTolerance: 0.25,
      weightedGradientMeanTolerance: 0.1,
      weightedGradientMaxTolerance: 0.65,
    };
  }

  return {
    lossPrecision: 4,
    firstStepGradientMeanTolerance: 0.02,
    firstStepGradientMaxTolerance: 0.2,
    weightedGradientMeanTolerance: 0.02,
    weightedGradientMaxTolerance: 0.6,
  };
};

export const loadFullPassArtifacts =
  async (): Promise<FullPassArtifactsResult> => {
    const fixture = await loadJson<Phase3FullPassFixture>(
      `${fullPassBaseUrl}/vgg19_phase3_full_pass_fixture.json`,
    );
    if (fixture === null) {
      return {
        ok: false,
        reason: "missing-fixtures",
        message:
          'Missing phase3 full-pass fixture. Run python-reference/export_vgg19_phase3_full_pass.py --packs "" --no-legacy-json first.',
      };
    }

    const weightsResult = await loadWeights();
    if (weightsResult === null) {
      return {
        ok: false,
        reason: "missing-fixtures",
        message:
          "Missing VGG19 weights. Generate legacy full-pass weights or provide a manifest-backed fp32/int8-per-channel model pack.",
      };
    }

    return {
      ok: true,
      artifacts: {
        fixture,
        weights: weightsResult.weights,
        modelSource: weightsResult.modelSource,
        ...tolerancesFor(weightsResult.modelSource),
      },
    };
  };
