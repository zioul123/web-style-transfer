import type { Vgg19WeightsManifest } from "../../ml/worker/models/vgg19/weights";

const DB_NAME = "style-transfer-model-cache";
const DB_VERSION = 1;
const STORE_NAME = "packs";

export type ModelCacheTier =
  | "fp32"
  | "fp16"
  | "int8-per-channel"
  | "int8log-per-channel"
  | "int4-experimental"
  | "int4log-experimental";

type CachedModelPackRecord = {
  cacheKey: string;
  modelId: string;
  version: string;
  tier: ModelCacheTier;
  manifest: Vgg19WeightsManifest;
  shards: Record<string, ArrayBuffer>;
  totalBytes: number;
  updatedAtMs: number;
};

export type ModelCacheStatus = {
  bytes: number;
  packs: number;
};

const hasIndexedDb = (): boolean => typeof indexedDB !== "undefined";

const openCacheDb = async (): Promise<IDBDatabase | null> => {
  if (!hasIndexedDb()) return null;
  return await new Promise<IDBDatabase>((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, DB_VERSION);
    request.onerror = () => reject(request.error);
    request.onupgradeneeded = () => {
      const db = request.result;
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        db.createObjectStore(STORE_NAME, { keyPath: "cacheKey" });
      }
    };
    request.onsuccess = () => resolve(request.result);
  });
};

const withStore = async <T,>(
  mode: IDBTransactionMode,
  run: (store: IDBObjectStore) => Promise<T>,
): Promise<T | null> => {
  const db = await openCacheDb();
  if (db === null) return null;
  try {
    const tx = db.transaction(STORE_NAME, mode);
    const store = tx.objectStore(STORE_NAME);
    const result = await run(store);
    await new Promise<void>((resolve, reject) => {
      tx.oncomplete = () => resolve();
      tx.onerror = () => reject(tx.error);
      tx.onabort = () => reject(tx.error);
    });
    return result;
  } finally {
    db.close();
  }
};

export const createModelCacheKey = (
  modelId: string,
  version: string,
  tier: ModelCacheTier,
): string => `${modelId}:${version}:${tier}`;

export const readCachedModelPack = async (
  cacheKey: string,
): Promise<CachedModelPackRecord | null> => {
  const record = await withStore("readonly", async (store) => {
    return await new Promise<CachedModelPackRecord | null>((resolve, reject) => {
      const request = store.get(cacheKey);
      request.onerror = () => reject(request.error);
      request.onsuccess = () => resolve((request.result as CachedModelPackRecord | undefined) ?? null);
    });
  });
  return record;
};

export const writeCachedModelPack = async (
  record: CachedModelPackRecord,
): Promise<void> => {
  await withStore("readwrite", async (store) => {
    await new Promise<void>((resolve, reject) => {
      const request = store.put(record);
      request.onerror = () => reject(request.error);
      request.onsuccess = () => resolve();
    });
  });
};

export const clearModelCache = async (): Promise<void> => {
  await withStore("readwrite", async (store) => {
    await new Promise<void>((resolve, reject) => {
      const request = store.clear();
      request.onerror = () => reject(request.error);
      request.onsuccess = () => resolve();
    });
  });
};

export const readModelCacheStatus = async (): Promise<ModelCacheStatus> => {
  const result = await withStore("readonly", async (store) => {
    return await new Promise<ModelCacheStatus>((resolve, reject) => {
      const request = store.getAll();
      request.onerror = () => reject(request.error);
      request.onsuccess = () => {
        const rows = (request.result as CachedModelPackRecord[]) ?? [];
        const bytes = rows.reduce((acc, row) => acc + row.totalBytes, 0);
        resolve({ bytes, packs: rows.length });
      };
    });
  });
  return result ?? { bytes: 0, packs: 0 };
};
