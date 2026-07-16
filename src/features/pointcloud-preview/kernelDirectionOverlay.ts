import type {
  ConvolutionKernelDirectionIndex,
  ConvolutionKernelPathGroup,
  ConvolutionKernelPathPoint,
} from "./types";

export type KernelDirectionOverlayPositions = {
  readonly sourcePositions: Float32Array;
  readonly targetPositions: Float32Array;
  readonly linePositions: Float32Array;
};

const emptyOverlayPositions = (): KernelDirectionOverlayPositions => ({
  sourcePositions: new Float32Array(),
  targetPositions: new Float32Array(),
  linePositions: new Float32Array(),
});

const isFinitePoint = (
  point: ConvolutionKernelPathPoint | undefined,
): point is ConvolutionKernelPathPoint =>
  point !== undefined &&
  point.every((coordinate) => Number.isFinite(coordinate));

export const buildKernelDirectionOverlayPositions = (
  groups: readonly ConvolutionKernelPathGroup[],
  directionIndex: ConvolutionKernelDirectionIndex,
): KernelDirectionOverlayPositions => {
  if (groups.length === 0) {
    return emptyOverlayPositions();
  }

  const sourceCoordinates: number[] = [];
  const targetCoordinates: number[] = [];
  const lineCoordinates: number[] = [];

  groups.forEach((group) => {
    const source = group[0]?.[0];
    const selectedPath = group[directionIndex];
    const target = selectedPath?.[selectedPath.length - 1];
    if (!isFinitePoint(source) || !isFinitePoint(target)) {
      return;
    }

    sourceCoordinates.push(source[0], source[1], source[2]);
    targetCoordinates.push(target[0], target[1], target[2]);
    lineCoordinates.push(
      source[0],
      source[1],
      source[2],
      target[0],
      target[1],
      target[2],
    );
  });

  return {
    sourcePositions: new Float32Array(sourceCoordinates),
    targetPositions: new Float32Array(targetCoordinates),
    linePositions: new Float32Array(lineCoordinates),
  };
};
