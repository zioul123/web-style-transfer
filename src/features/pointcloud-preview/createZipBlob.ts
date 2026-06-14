type ZipEntry = {
  readonly name: string;
  readonly data: Uint8Array;
};

const textEncoder = new TextEncoder();

const crc32Table = (() => {
  const table = new Uint32Array(256);
  for (let index = 0; index < table.length; index += 1) {
    let value = index;
    for (let bit = 0; bit < 8; bit += 1) {
      value = (value & 1) !== 0 ? 0xedb88320 ^ (value >>> 1) : value >>> 1;
    }
    table[index] = value >>> 0;
  }
  return table;
})();

const crc32 = (data: Uint8Array): number => {
  let checksum = 0xffffffff;
  for (const value of data) {
    checksum = (checksum >>> 8) ^ crc32Table[(checksum ^ value) & 0xff]!;
  }
  return (checksum ^ 0xffffffff) >>> 0;
};

const writeUint16 = (view: DataView, offset: number, value: number): void => {
  view.setUint16(offset, value, true);
};

const writeUint32 = (view: DataView, offset: number, value: number): void => {
  view.setUint32(offset, value, true);
};

const copyBytes = (
  target: Uint8Array,
  offset: number,
  source: Uint8Array,
): number => {
  target.set(source, offset);
  return offset + source.length;
};

export const createZipBlob = (entries: readonly ZipEntry[]): Blob => {
  const encodedEntries = entries.map((entry) => ({
    ...entry,
    nameBytes: textEncoder.encode(entry.name),
    checksum: crc32(entry.data),
  }));
  const localSize = encodedEntries.reduce(
    (size, entry) => size + 30 + entry.nameBytes.length + entry.data.length,
    0,
  );
  const centralSize = encodedEntries.reduce(
    (size, entry) => size + 46 + entry.nameBytes.length,
    0,
  );
  const output = new Uint8Array(localSize + centralSize + 22);
  const view = new DataView(output.buffer);
  const localOffsets: number[] = [];
  let offset = 0;

  for (const entry of encodedEntries) {
    localOffsets.push(offset);
    writeUint32(view, offset, 0x04034b50);
    writeUint16(view, offset + 4, 20);
    writeUint16(view, offset + 6, 0x0800);
    writeUint16(view, offset + 8, 0);
    writeUint16(view, offset + 10, 0);
    writeUint16(view, offset + 12, 0);
    writeUint32(view, offset + 14, entry.checksum);
    writeUint32(view, offset + 18, entry.data.length);
    writeUint32(view, offset + 22, entry.data.length);
    writeUint16(view, offset + 26, entry.nameBytes.length);
    writeUint16(view, offset + 28, 0);
    offset += 30;
    offset = copyBytes(output, offset, entry.nameBytes);
    offset = copyBytes(output, offset, entry.data);
  }

  const centralOffset = offset;
  encodedEntries.forEach((entry, index) => {
    writeUint32(view, offset, 0x02014b50);
    writeUint16(view, offset + 4, 20);
    writeUint16(view, offset + 6, 20);
    writeUint16(view, offset + 8, 0x0800);
    writeUint16(view, offset + 10, 0);
    writeUint16(view, offset + 12, 0);
    writeUint16(view, offset + 14, 0);
    writeUint32(view, offset + 16, entry.checksum);
    writeUint32(view, offset + 20, entry.data.length);
    writeUint32(view, offset + 24, entry.data.length);
    writeUint16(view, offset + 28, entry.nameBytes.length);
    writeUint16(view, offset + 30, 0);
    writeUint16(view, offset + 32, 0);
    writeUint16(view, offset + 34, 0);
    writeUint16(view, offset + 36, 0);
    writeUint32(view, offset + 38, 0);
    writeUint32(view, offset + 42, localOffsets[index]!);
    offset += 46;
    offset = copyBytes(output, offset, entry.nameBytes);
  });

  writeUint32(view, offset, 0x06054b50);
  writeUint16(view, offset + 4, 0);
  writeUint16(view, offset + 6, 0);
  writeUint16(view, offset + 8, encodedEntries.length);
  writeUint16(view, offset + 10, encodedEntries.length);
  writeUint32(view, offset + 12, offset - centralOffset);
  writeUint32(view, offset + 16, centralOffset);
  writeUint16(view, offset + 20, 0);

  return new Blob([output], { type: "application/zip" });
};
