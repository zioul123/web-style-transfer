export const VGG19_STYLE_LAYER_INDICES: readonly number[] = [0, 5, 10, 19, 28];
export const VGG19_CONTENT_LAYER_INDEX: number = 21;

// torch vgg19.features indices for ReLU/MaxPool up to relu5_1 range (0..29)
export const VGG19_RELU_LAYER_INDICES_UP_TO_CONV5_1: readonly number[] = [
  1, 3, 6, 8, 11, 13, 15, 17, 20, 22, 24, 26, 29,
];
export const VGG19_POOL_LAYER_INDICES_UP_TO_CONV5_1: readonly number[] = [
  4, 9, 18, 27,
];
