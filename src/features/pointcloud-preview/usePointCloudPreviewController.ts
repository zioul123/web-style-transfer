import {
  useCallback,
  useRef,
  useState,
  type Dispatch,
  type MutableRefObject,
  type SetStateAction,
} from "react";
import type {
  PointCloudHitSample,
  PointCloudPreviewCameraCommand,
  PointCloudPreviewViewAxis,
  PointCloudPreviewViewSettings,
  PreviewCameraState,
} from "./types";

export const defaultPointCloudPreviewPointSize = 0.035;

export const defaultPointCloudPreviewViewSettings: PointCloudPreviewViewSettings =
  {
    showMesh: true,
    showPoints: false,
    showWireframe: false,
    showGroundPlane: false,
    meshColorMode: "fragment-knn",
    pointSize: defaultPointCloudPreviewPointSize,
    backgroundColor: "white",
    disableGammaDecoding: false,
    brightness: 1,
    swapYZ: false,
  };

export type NextPointCloudPreviewCameraCommand =
  | {
      readonly type: "frame";
    }
  | {
      readonly type: "restore";
      readonly camera: PreviewCameraState;
    }
  | {
      readonly type: "snap-axis";
      readonly axis: PointCloudPreviewViewAxis;
    };

type UsePointCloudPreviewControllerOptions = {
  readonly initialViewSettings?: PointCloudPreviewViewSettings;
};

type ResetPreviewInteractionOptions = {
  readonly clearCameraState?: boolean;
};

export type PointCloudPreviewController = {
  readonly viewSettings: PointCloudPreviewViewSettings;
  readonly updateViewSettings: (
    patch: Partial<PointCloudPreviewViewSettings>,
  ) => void;
  readonly selectedHit: PointCloudHitSample | null;
  readonly setSelectedHit: Dispatch<SetStateAction<PointCloudHitSample | null>>;
  readonly framesPerSecond: number;
  readonly setFramesPerSecond: Dispatch<SetStateAction<number>>;
  readonly cameraCommand: PointCloudPreviewCameraCommand;
  readonly cameraState: PreviewCameraState | null;
  readonly cameraStateRef: MutableRefObject<PreviewCameraState | null>;
  readonly handleCameraStateChange: (state: PreviewCameraState) => void;
  readonly issueCameraCommand: (
    nextCommand: NextPointCloudPreviewCameraCommand,
  ) => number;
  readonly resetPreviewInteraction: (
    options?: ResetPreviewInteractionOptions,
  ) => void;
};

export const usePointCloudPreviewController = ({
  initialViewSettings,
}: UsePointCloudPreviewControllerOptions = {}): PointCloudPreviewController => {
  const cameraStateRef = useRef<PreviewCameraState | null>(null);
  const cameraCommandIdRef = useRef<number>(0);
  const [viewSettings, setViewSettings] =
    useState<PointCloudPreviewViewSettings>(
      () => initialViewSettings ?? defaultPointCloudPreviewViewSettings,
    );
  const [selectedHit, setSelectedHit] = useState<PointCloudHitSample | null>(
    null,
  );
  const [cameraCommand, setCameraCommand] =
    useState<PointCloudPreviewCameraCommand>({
      id: 0,
      type: "frame",
    });
  const [cameraState, setCameraState] = useState<PreviewCameraState | null>(
    null,
  );
  const [framesPerSecond, setFramesPerSecond] = useState<number>(0);

  const updateViewSettings = useCallback(
    (patch: Partial<PointCloudPreviewViewSettings>): void => {
      setViewSettings((currentSettings) => ({
        ...currentSettings,
        ...patch,
      }));
    },
    [],
  );

  const handleCameraStateChange = useCallback(
    (state: PreviewCameraState): void => {
      cameraStateRef.current = state;
      setCameraState(state);
    },
    [],
  );

  const issueCameraCommand = useCallback(
    (nextCommand: NextPointCloudPreviewCameraCommand): number => {
      const nextId = cameraCommandIdRef.current + 1;
      cameraCommandIdRef.current = nextId;
      setCameraCommand({
        ...nextCommand,
        id: nextId,
      });
      return nextId;
    },
    [],
  );

  const resetPreviewInteraction = useCallback(
    (options: ResetPreviewInteractionOptions = {}): void => {
      setSelectedHit(null);
      setFramesPerSecond(0);
      if (options.clearCameraState === true) {
        cameraStateRef.current = null;
        setCameraState(null);
      }
    },
    [],
  );

  return {
    viewSettings,
    updateViewSettings,
    selectedHit,
    setSelectedHit,
    framesPerSecond,
    setFramesPerSecond,
    cameraCommand,
    cameraState,
    cameraStateRef,
    handleCameraStateChange,
    issueCameraCommand,
    resetPreviewInteraction,
  };
};
