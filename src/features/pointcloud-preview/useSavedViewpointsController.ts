import { useCallback, useEffect, useState } from "react";
import type { SavedViewpoint } from "./pointCloudPreviewModels";
import type { PreviewCameraState } from "./types";

const savedViewpointsStorageKey = "pointcloud-preview-saved-viewpoints";

const isPreviewCameraState = (value: unknown): value is PreviewCameraState => {
  if (typeof value !== "object" || value === null) {
    return false;
  }
  const candidate = value as Record<string, unknown>;
  return (
    Array.isArray(candidate.position) &&
    candidate.position.length === 3 &&
    candidate.position.every((entry) => typeof entry === "number") &&
    Array.isArray(candidate.target) &&
    candidate.target.length === 3 &&
    candidate.target.every((entry) => typeof entry === "number")
  );
};

const parseSavedViewpointEntries = (
  entries: readonly unknown[],
  reindexIds: boolean,
): SavedViewpoint[] =>
  entries.flatMap((entry, index): SavedViewpoint[] => {
    if (typeof entry !== "object" || entry === null) {
      return [];
    }
    const candidate = entry as Record<string, unknown>;
    if (
      typeof candidate.label !== "string" ||
      typeof candidate.swapYZ !== "boolean" ||
      !isPreviewCameraState(candidate.camera)
    ) {
      return [];
    }
    const preservedId =
      typeof candidate.id === "number" ? candidate.id : index + 1;
    return [
      {
        id: reindexIds ? index + 1 : preservedId,
        label: candidate.label,
        camera: candidate.camera,
        swapYZ: candidate.swapYZ,
      },
    ];
  });

const readSavedViewpoints = (): readonly SavedViewpoint[] => {
  if (typeof window === "undefined") {
    return [];
  }

  try {
    const raw = window.localStorage.getItem(savedViewpointsStorageKey);
    if (raw === null) {
      return [];
    }
    const parsed = JSON.parse(raw) as unknown;
    if (Array.isArray(parsed)) {
      return parseSavedViewpointEntries(parsed, false);
    }
    if (typeof parsed !== "object" || parsed === null) {
      return [];
    }

    const legacyEntries = Object.values(parsed as Record<string, unknown>)
      .filter(Array.isArray)
      .flatMap((value) => value);
    return parseSavedViewpointEntries(legacyEntries, true);
  } catch {
    return [];
  }
};

const nextViewpointIdFor = (viewpoints: readonly SavedViewpoint[]): number =>
  viewpoints.reduce(
    (highestId, viewpoint) => Math.max(highestId, viewpoint.id),
    0,
  ) + 1;

type UseSavedViewpointsControllerOptions = {
  readonly currentCameraState: PreviewCameraState | null;
  readonly swapYZ: boolean;
};

export type SavedViewpointsController = {
  readonly savedViewpoints: readonly SavedViewpoint[];
  readonly saveViewpoint: () => void;
  readonly renameViewpoint: (id: number, label: string) => void;
  readonly updateViewpoint: (id: number) => void;
  readonly deleteViewpoint: (id: number) => void;
};

export const useSavedViewpointsController = ({
  currentCameraState,
  swapYZ,
}: UseSavedViewpointsControllerOptions): SavedViewpointsController => {
  const [initialViewpoints] = useState<readonly SavedViewpoint[]>(() =>
    readSavedViewpoints(),
  );
  const [savedViewpoints, setSavedViewpoints] =
    useState<readonly SavedViewpoint[]>(initialViewpoints);
  const [nextViewpointId, setNextViewpointId] = useState<number>(() =>
    nextViewpointIdFor(initialViewpoints),
  );

  useEffect(() => {
    window.localStorage.setItem(
      savedViewpointsStorageKey,
      JSON.stringify(savedViewpoints),
    );
  }, [savedViewpoints]);

  const saveViewpoint = useCallback((): void => {
    if (currentCameraState === null) {
      return;
    }
    const nextId = nextViewpointId;
    setSavedViewpoints((currentViews) => [
      {
        id: nextId,
        label: `View ${nextId}`,
        camera: currentCameraState,
        swapYZ,
      },
      ...currentViews,
    ]);
    setNextViewpointId(nextId + 1);
  }, [currentCameraState, nextViewpointId, swapYZ]);

  const renameViewpoint = useCallback((id: number, label: string): void => {
    setSavedViewpoints((currentViews) =>
      currentViews.map((viewpoint) =>
        viewpoint.id === id ? { ...viewpoint, label } : viewpoint,
      ),
    );
  }, []);

  const updateViewpoint = useCallback(
    (id: number): void => {
      if (currentCameraState === null) {
        return;
      }
      setSavedViewpoints((currentViews) =>
        currentViews.map((viewpoint) =>
          viewpoint.id === id
            ? {
                ...viewpoint,
                camera: currentCameraState,
                swapYZ,
              }
            : viewpoint,
        ),
      );
    },
    [currentCameraState, swapYZ],
  );

  const deleteViewpoint = useCallback((id: number): void => {
    setSavedViewpoints((currentViews) =>
      currentViews.filter((viewpoint) => viewpoint.id !== id),
    );
  }, []);

  return {
    savedViewpoints,
    saveViewpoint,
    renameViewpoint,
    updateViewpoint,
    deleteViewpoint,
  };
};
