const ensureTrailingSlash = (value: string): string =>
  value.endsWith("/") ? value : `${value}/`;

const trimSlashes = (value: string): string => value.replace(/^\/+|\/+$/g, "");

const joinUrl = (base: string, path: string): string => {
  const normalizedPath = trimSlashes(path);
  if (normalizedPath.length === 0) return ensureTrailingSlash(base);
  return `${ensureTrailingSlash(base)}${normalizedPath}`;
};

export const assetUrl = (path: string): string =>
  joinUrl(import.meta.env.BASE_URL, path);

export const vgg19ModelUrl = (path: string): string => {
  const configuredBase = import.meta.env.VITE_VGG19_MODEL_BASE_URL;
  const base =
    configuredBase === undefined || configuredBase.trim().length === 0
      ? assetUrl("vgg19-models")
      : configuredBase.trim();
  return joinUrl(base, path);
};
