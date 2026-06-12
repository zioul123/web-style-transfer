export class FingerprintBuilder {
  private hashA = 0x811c9dc5;
  private hashB = 0x9e3779b9;
  private readonly floatScratch = new Float32Array(1);
  private readonly wordScratch = new Uint32Array(this.floatScratch.buffer);

  addWord(value: number): void {
    const word = value >>> 0;
    const rotatedWord = (word << 13) | (word >>> 19);
    this.hashA = Math.imul(this.hashA ^ word, 0x01000193) >>> 0;
    this.hashB =
      (Math.imul(this.hashB ^ rotatedWord, 0x85ebca6b) + 0x27d4eb2f) >>> 0;
  }

  addFloat32(value: number): void {
    this.floatScratch[0] = value;
    this.addWord(this.wordScratch[0]);
  }

  addString(value: string): void {
    this.addWord(value.length);
    for (let index = 0; index < value.length; index += 1) {
      this.addWord(value.charCodeAt(index));
    }
  }

  digest(): string {
    return `${this.hashA.toString(16).padStart(8, "0")}${this.hashB
      .toString(16)
      .padStart(8, "0")}`;
  }
}

export const fingerprintFloat32 = (values: Float32Array): string => {
  const fingerprint = new FingerprintBuilder();
  addFloat32ToFingerprint(fingerprint, values);
  return fingerprint.digest();
};

export const addFloat32ToFingerprint = (
  fingerprint: FingerprintBuilder,
  values: Float32Array,
): void => {
  fingerprint.addWord(values.length);
  for (const value of values) fingerprint.addFloat32(value);
};

export const addUint32ToFingerprint = (
  fingerprint: FingerprintBuilder,
  values: Uint32Array,
): void => {
  fingerprint.addWord(values.length);
  for (const value of values) fingerprint.addWord(value);
};

export const addInt32ToFingerprint = (
  fingerprint: FingerprintBuilder,
  values: Int32Array,
): void => {
  fingerprint.addWord(values.length);
  for (const value of values) fingerprint.addWord(value);
};
