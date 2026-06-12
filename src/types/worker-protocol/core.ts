export type TensorShape = readonly [number, number, number, number];

export type TensorData = {
  shape: TensorShape;
  values: Float32Array;
};

export type FeatureMatrixShape = {
  pointCount: number;
  channelCount: number;
};

export type FeatureMatrixData = FeatureMatrixShape & {
  values: Float32Array;
};

export type WorkerTensor = {
  shape: TensorShape;
  values: number[];
};

export type WorkerFeatureMatrix = FeatureMatrixShape & {
  values: number[];
};

export type PointCloudKnnData = {
  sampleCount: number;
  kernelPointCount: number;
  neighborCount: number;
  indices: Uint32Array;
  weights: Float32Array;
};

export type WorkerFloatArray = number[] | Float32Array;
export type WorkerUnsignedIndexArray = number[] | Uint32Array | Int32Array;
export type WorkerSignedIndexArray = number[] | Int32Array;

export type WorkerPointCloudKnn = {
  sampleCount: number;
  kernelPointCount: number;
  neighborCount: number;
  indices: WorkerUnsignedIndexArray;
  weights: WorkerFloatArray;
};

export type SurfacePoolData = {
  inputPointCount: number;
  outputPointCount: number;
  mapping: Int32Array;
};

export type WorkerSurfacePoolMap = {
  inputPointCount: number;
  outputPointCount: number;
  mapping: WorkerSignedIndexArray;
};

export type WorkerTensorOperand =
  | { kind: "tensor"; tensor: WorkerTensor }
  | { kind: "scalar"; scalar: number };

export type WorkerRoundtripResponse =
  | {
      type: "tensor-roundtrip-result";
      id: string;
      ok: true;
      tensor: WorkerTensor;
    }
  | { type: "tensor-roundtrip-result"; id: string; ok: false; message: string };

export type WorkerTensorScalarOpResponse = {
  type: "tensor-op-result";
  id: string;
  ok: true;
  scalar: number;
};

export type WorkerTensorVectorOpResponse = {
  type: "tensor-op-result";
  id: string;
  ok: true;
  values: number[];
};

export type WorkerTensorOpErrorResponse = {
  type: "tensor-op-result";
  id: string;
  ok: false;
  message: string;
};

export type WorkerTensorOpResponse =
  | WorkerTensorScalarOpResponse
  | WorkerTensorVectorOpResponse
  | WorkerTensorOpErrorResponse;

export const isWorkerTensorScalarOpResponse = (
  value: WorkerTensorOpResponse,
): value is WorkerTensorScalarOpResponse => value.ok && "scalar" in value;

export const isWorkerTensorVectorOpResponse = (
  value: WorkerTensorOpResponse,
): value is WorkerTensorVectorOpResponse => value.ok && "values" in value;
