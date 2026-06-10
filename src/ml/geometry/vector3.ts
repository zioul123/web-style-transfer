export type Vec2 = readonly [number, number];
export type Vec3 = readonly [number, number, number];

export const dimensionsPerVec3 = 3;
export const dimensionsPerVec2 = 2;

export const vec3At = (values: ArrayLike<number>, index: number): Vec3 => {
  const baseIndex = index * dimensionsPerVec3;
  return [values[baseIndex], values[baseIndex + 1], values[baseIndex + 2]];
};

export const vec2At = (values: ArrayLike<number>, index: number): Vec2 => {
  const baseIndex = index * dimensionsPerVec2;
  return [values[baseIndex], values[baseIndex + 1]];
};

export const subtractVec3 = (left: Vec3, right: Vec3): Vec3 => [
  left[0] - right[0],
  left[1] - right[1],
  left[2] - right[2],
];

export const addVec3 = (left: Vec3, right: Vec3): Vec3 => [
  left[0] + right[0],
  left[1] + right[1],
  left[2] + right[2],
];

export const scaleVec3 = (value: Vec3, scale: number): Vec3 => [
  value[0] * scale,
  value[1] * scale,
  value[2] * scale,
];

export const dotVec3 = (left: Vec3, right: Vec3): number =>
  left[0] * right[0] + left[1] * right[1] + left[2] * right[2];

export const crossVec3 = (left: Vec3, right: Vec3): Vec3 => [
  left[1] * right[2] - left[2] * right[1],
  left[2] * right[0] - left[0] * right[2],
  left[0] * right[1] - left[1] * right[0],
];

export const squaredLengthVec3 = (value: Vec3): number => dotVec3(value, value);

export const lengthVec3 = (value: Vec3): number =>
  Math.sqrt(squaredLengthVec3(value));

const cleanZero = (value: number): number => (Object.is(value, -0) ? 0 : value);

export const normalizeVec3 = (value: Vec3): Vec3 => {
  const length = lengthVec3(value);
  if (length === 0 || !Number.isFinite(length)) {
    return [0, 0, 0];
  }
  const normalized = scaleVec3(value, 1 / length);
  return [
    cleanZero(normalized[0]),
    cleanZero(normalized[1]),
    cleanZero(normalized[2]),
  ];
};

export const distanceVec3 = (left: Vec3, right: Vec3): number =>
  lengthVec3(subtractVec3(left, right));

export const squaredDistanceVec3 = (left: Vec3, right: Vec3): number =>
  squaredLengthVec3(subtractVec3(left, right));
