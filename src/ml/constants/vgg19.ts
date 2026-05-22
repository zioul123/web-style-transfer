// Canonical tap indexing scheme for style-transfer requests:
// all indices are torch vgg19.features layer indices for post-ReLU activations.
// These correspond to relu1_1, relu2_1, relu3_1, relu4_1, relu5_1 (style)
// and relu4_2 (content).
export const VGG19_RELU_TAP_STYLE_LAYER_INDICES: readonly number[] = [
  1, 6, 11, 20, 29,
];
export const VGG19_RELU_TAP_CONTENT_LAYER_INDEX: number = 22;

// torch vgg19.features indices for ReLU/MaxPool up to relu5_1 range (0..29)
export const VGG19_RELU_LAYER_INDICES_UP_TO_CONV5_1: readonly number[] = [
  1, 3, 6, 8, 11, 13, 15, 17, 20, 22, 24, 26, 29,
];
export const VGG19_POOL_LAYER_INDICES_UP_TO_CONV5_1: readonly number[] = [
  4, 9, 18, 27,
];

export const assertValidVgg19ReluTapIndices = (
  styleLayerIndices: readonly number[],
  contentLayerIndex: number,
): void => {
  const allowed = new Set<number>(VGG19_RELU_LAYER_INDICES_UP_TO_CONV5_1);
  const styleSet = new Set<number>();
  for (const index of styleLayerIndices) {
    if (!allowed.has(index))
      throw new Error(
        `Invalid style tap ReLU layer index ${index}. Expected a VGG19 ReLU layer index from the fixed features schedule.`,
      );
    if (styleSet.has(index))
      throw new Error(`Duplicate style tap ReLU layer index ${index}.`);
    styleSet.add(index);
  }
  if (styleLayerIndices.length === 0)
    throw new Error(
      "styleLayerIndices must include at least one ReLU tap index.",
    );
  if (!allowed.has(contentLayerIndex))
    throw new Error(
      `Invalid content tap ReLU layer index ${contentLayerIndex}. Expected a VGG19 ReLU layer index from the fixed features schedule.`,
    );
};
