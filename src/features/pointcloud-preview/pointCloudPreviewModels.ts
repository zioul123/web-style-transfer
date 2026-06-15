import type { PointCloudMeshData, PreviewCameraState } from "./types";

export type UploadedPointCloudFileStatus =
  | "idle"
  | "loading"
  | "ready"
  | "error";

export type UploadedPointCloudFile = {
  readonly id: number;
  readonly file: File;
  readonly label: string;
  readonly status: UploadedPointCloudFileStatus;
  readonly errorMessage: string | null;
};

export type LoadedAssetState =
  | {
      readonly status: "loading";
      readonly sourceLabel: string;
      readonly errorMessage: null;
      readonly data: null;
    }
  | {
      readonly status: "ready";
      readonly sourceLabel: string;
      readonly errorMessage: null;
      readonly data: PointCloudMeshData;
    }
  | {
      readonly status: "error";
      readonly sourceLabel: string;
      readonly errorMessage: string;
      readonly data: null;
    };

export type SavedViewpoint = {
  readonly id: number;
  readonly label: string;
  readonly camera: PreviewCameraState;
  readonly swapYZ: boolean;
};

export type BatchScreenshotProgress = {
  readonly completed: number;
  readonly total: number;
  readonly label: string;
};

export const uploadedFileStatusLabel = (
  status: UploadedPointCloudFileStatus,
): string => (status === "idle" ? "queued" : status);
