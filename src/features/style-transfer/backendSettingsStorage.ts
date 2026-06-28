export type StyleTransferBackendPreference = "auto" | "webgpu";

export type StyleTransferBackendSettings = {
  readonly backendPreference: StyleTransferBackendPreference;
  readonly backendUrl: string;
};

export const BACKEND_SETTINGS_STORAGE_KEY =
  "web-style-transfer.backend-settings.v1";

export const DEFAULT_BACKEND_URL = (() => {
  const configured = import.meta.env.VITE_STYLE_TRANSFER_BACKEND_URL;
  return configured === undefined || configured.trim().length === 0
    ? "http://127.0.0.1:8000"
    : configured.trim();
})();

export const DEFAULT_BACKEND_SETTINGS: StyleTransferBackendSettings = {
  backendPreference: "auto",
  backendUrl: DEFAULT_BACKEND_URL,
};

export type BackendSettingsParseResult =
  | { readonly ok: true; readonly settings: StyleTransferBackendSettings }
  | { readonly ok: false };

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value);

export const isBackendPreference = (
  value: string,
): value is StyleTransferBackendPreference =>
  value === "auto" || value === "webgpu";

const normalizeBackendUrl = (value: unknown): string => {
  if (typeof value !== "string") return DEFAULT_BACKEND_URL;
  const trimmed = value.trim();
  return trimmed.length === 0 ? DEFAULT_BACKEND_URL : trimmed;
};

export const parseBackendSettings = (
  value: unknown,
): BackendSettingsParseResult => {
  if (!isRecord(value)) return { ok: false };
  const rawPreference = value.backendPreference;
  return {
    ok: true,
    settings: {
      backendPreference:
        typeof rawPreference === "string" && isBackendPreference(rawPreference)
          ? rawPreference
          : DEFAULT_BACKEND_SETTINGS.backendPreference,
      backendUrl: normalizeBackendUrl(value.backendUrl),
    },
  };
};

export const readBackendSettings = (): StyleTransferBackendSettings => {
  const storedSettings = localStorage.getItem(BACKEND_SETTINGS_STORAGE_KEY);
  if (storedSettings === null) return DEFAULT_BACKEND_SETTINGS;
  try {
    const parsed = parseBackendSettings(JSON.parse(storedSettings));
    return parsed.ok ? parsed.settings : DEFAULT_BACKEND_SETTINGS;
  } catch {
    return DEFAULT_BACKEND_SETTINGS;
  }
};

export const writeBackendSettings = (
  settings: StyleTransferBackendSettings,
): void => {
  localStorage.setItem(
    BACKEND_SETTINGS_STORAGE_KEY,
    JSON.stringify({
      backendPreference: settings.backendPreference,
      backendUrl: normalizeBackendUrl(settings.backendUrl),
    }),
  );
};
