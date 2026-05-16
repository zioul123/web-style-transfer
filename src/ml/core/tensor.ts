export type TensorShape = [number, number, number, number];

export class Tensor {
  readonly data: Float32Array;
  readonly shape: TensorShape;

  constructor(data: Float32Array | number[], shape: TensorShape) {
    const size = shape[0] * shape[1] * shape[2] * shape[3];
    const arr = data instanceof Float32Array ? data : new Float32Array(data);
    if (arr.length !== size) throw new Error(`Size mismatch: got ${arr.length}, expected ${size}`);
    this.data = arr;
    this.shape = shape;
  }

  static zeros(shape: TensorShape): Tensor {
    return new Tensor(new Float32Array(shape[0] * shape[1] * shape[2] * shape[3]), shape);
  }
}
