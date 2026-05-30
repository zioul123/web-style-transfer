type TestInfoLike = {
  annotations: Array<{ type: string; description?: string }>;
};

export type FixtureAvailability =
  | { ok: true }
  | { ok: false; reason: string; message: string };

export type VggFirstPoolWeightsFixture = {
  conv1WeightShape: [number, number, number, number];
  conv1WeightValues: number[];
  conv1BiasValues: number[];
  conv2WeightShape: [number, number, number, number];
  conv2WeightValues: number[];
  conv2BiasValues: number[];
  conv3WeightShape: [number, number, number, number];
  conv3WeightValues: number[];
  conv3BiasValues: number[];
};

export type VggFirstPoolCaseFixture = {
  inputShape: [number, number, number, number];
  inputValues: number[];
  normalizedValues?: number[];
  mean: [number, number, number];
  std: [number, number, number];
  expectedShape?: [number, number, number, number];
  expectedValues?: number[];
};

export type VggFirstPoolArtifactsResult =
  | {
      ok: true;
      weights: VggFirstPoolWeightsFixture;
      caseData: VggFirstPoolCaseFixture;
    }
  | { ok: false; reason: "weights-missing" | "case-missing"; message: string };

export const firstPoolFixtureMissingMessage =
  "Missing VGG19 first-pool fixtures. Run python-reference/export_vgg19_first_pool.py first.";

export const fetchJsonOrNull = async <T>(url: string): Promise<T | null> => {
  const response = await fetch(url);
  if (!response.ok) return null;

  try {
    return (await response.json()) as T;
  } catch {
    return null;
  }
};

export const loadVggFirstPoolArtifacts =
  async (): Promise<VggFirstPoolArtifactsResult> => {
    const weights = await fetchJsonOrNull<VggFirstPoolWeightsFixture>(
      "/vgg19-first-pool/vgg19_first_pool_weights.json",
    );
    if (weights === null) {
      return {
        ok: false,
        reason: "weights-missing",
        message: firstPoolFixtureMissingMessage,
      };
    }

    const caseData = await fetchJsonOrNull<VggFirstPoolCaseFixture>(
      "/vgg19-first-pool/vgg19_first_pool_case_madeira16.json",
    );
    if (caseData === null) {
      return {
        ok: false,
        reason: "case-missing",
        message: firstPoolFixtureMissingMessage,
      };
    }

    return { ok: true, weights, caseData };
  };

export const skipIfMissingFixture = (
  skip: (condition: boolean, description?: string) => void,
  availability: FixtureAvailability,
): void => {
  skip(!availability.ok, availability.ok ? undefined : availability.message);
};

export const attachMissingFixtureAnnotation = (
  testInfo: TestInfoLike,
  availability: FixtureAvailability,
): void => {
  if (availability.ok) return;
  testInfo.annotations.push({
    type: availability.reason,
    description: availability.message,
  });
};

export type ModelPackManifest = {
  shards: Array<{ name: string }>;
};

export const checkModelPackAvailability = async (
  baseUrl: string,
  pack: string,
): Promise<FixtureAvailability> => {
  const manifest = await fetchJsonOrNull<ModelPackManifest>(
    `${baseUrl}/${pack}/manifest.json`,
  );
  if (manifest === null) {
    return {
      ok: false,
      reason: "missing-model-pack",
      message: `Missing ${pack} model-pack manifest at ${baseUrl}/${pack}/manifest.json.`,
    };
  }

  const missingShard = await findMissingShard(baseUrl, pack, manifest);
  if (missingShard !== null) {
    return {
      ok: false,
      reason: "missing-model-pack",
      message: `Missing ${pack} model-pack shard at ${baseUrl}/${pack}/${missingShard}.`,
    };
  }

  return { ok: true };
};

export const checkFp32ModelPackAvailability = async (
  baseUrl = "/vgg19-models",
): Promise<FixtureAvailability> => checkModelPackAvailability(baseUrl, "fp32");

const findMissingShard = async (
  baseUrl: string,
  pack: string,
  manifest: ModelPackManifest,
): Promise<string | null> => {
  for (const shard of manifest.shards) {
    const response = await fetch(`${baseUrl}/${pack}/${shard.name}`, {
      method: "HEAD",
    });
    if (!response.ok) return shard.name;
  }
  return null;
};

export const expectFixtureAvailable = (
  availability: FixtureAvailability,
): void => {
  if (!availability.ok) throw new Error(availability.message);
};
