/// <reference lib="webworker" />

export type GpuDispatchRecord = {
  label: string;
  workgroups: readonly [number, number, number];
  count: number;
};

export type GpuDispatchCoverageSnapshot = {
  enabled: boolean;
  records: GpuDispatchRecord[];
};

type DispatchRecorderState = {
  enabled: boolean;
  patched: boolean;
  records: Map<string, GpuDispatchRecord>;
  patchedDevices: WeakSet<GPUDevice>;
};

type CoverageComputePassEncoder = GPUComputePassEncoder & {
  __styleTransferCurrentPipelineLabel?: string;
};

type DispatchRecorderGlobal = typeof globalThis & {
  __styleTransferDispatchRecorder?: DispatchRecorderState;
};

const getRecorderGlobal = (): DispatchRecorderGlobal => globalThis;

const makeRecordKey = (
  label: string,
  workgroups: readonly [number, number, number],
): string => `${label}|${workgroups[0]}x${workgroups[1]}x${workgroups[2]}`;

const getRecorderState = (): DispatchRecorderState => {
  const recorderGlobal = getRecorderGlobal();
  recorderGlobal.__styleTransferDispatchRecorder ??= {
    enabled: false,
    patched: false,
    records: new Map<string, GpuDispatchRecord>(),
    patchedDevices: new WeakSet<GPUDevice>(),
  };
  return recorderGlobal.__styleTransferDispatchRecorder;
};

const recordDispatch = (
  state: DispatchRecorderState,
  label: string,
  workgroups: readonly [number, number, number],
): void => {
  const key = makeRecordKey(label, workgroups);
  const existing = state.records.get(key);
  if (existing !== undefined) {
    state.records.set(key, { ...existing, count: existing.count + 1 });
    return;
  }
  state.records.set(key, { label, workgroups, count: 1 });
};

const patchGpuComputePassPrototype = (state: DispatchRecorderState): void => {
  if (state.patched) return;
  if (typeof GPUComputePassEncoder === "undefined") return;

  const originalSetPipeline = GPUComputePassEncoder.prototype.setPipeline;
  GPUComputePassEncoder.prototype.setPipeline = function setPipelineWithLabel(
    this: CoverageComputePassEncoder,
    pipeline: GPUComputePipeline,
  ): void {
    this.__styleTransferCurrentPipelineLabel =
      pipeline.label || "unlabeled-pipeline";
    originalSetPipeline.call(this, pipeline);
  };

  const originalDispatchWorkgroups =
    GPUComputePassEncoder.prototype.dispatchWorkgroups;
  GPUComputePassEncoder.prototype.dispatchWorkgroups =
    function dispatchWorkgroupsWithCoverage(
      this: CoverageComputePassEncoder,
      workgroupCountX: number,
      workgroupCountY: number = 1,
      workgroupCountZ: number = 1,
    ): void {
      const latestState = getRecorderState();
      if (latestState.enabled) {
        recordDispatch(
          latestState,
          this.__styleTransferCurrentPipelineLabel ?? "pipeline-not-set",
          [workgroupCountX, workgroupCountY, workgroupCountZ],
        );
      }
      originalDispatchWorkgroups.call(
        this,
        workgroupCountX,
        workgroupCountY,
        workgroupCountZ,
      );
    };

  state.patched = true;
};

const getPipelineLabelFromStack = (): string => {
  const stack = new Error().stack ?? "";
  const frame = stack
    .split("\n")
    .map((line) => line.trim())
    .find(
      (line) =>
        line.includes("/src/ml/worker/") &&
        !line.includes("dispatchRecorder.ts"),
    );
  if (frame === undefined) return "unlabeled-pipeline";
  const match = frame.match(/\/src\/ml\/worker\/([^:)]+)(?::(\d+))?/);
  if (match === null) return "unlabeled-pipeline";
  return match[2] === undefined ? match[1] : `${match[1]}:${match[2]}`;
};

export const patchGpuDeviceForDispatchRecording = (device: GPUDevice): void => {
  const state = getRecorderState();
  if (state.patchedDevices.has(device)) return;
  const originalCreateComputePipeline =
    device.createComputePipeline.bind(device);
  device.createComputePipeline = (
    descriptor: GPUComputePipelineDescriptor,
  ): GPUComputePipeline => {
    descriptor.label ||= getPipelineLabelFromStack();
    return originalCreateComputePipeline(descriptor);
  };
  state.patchedDevices.add(device);
};

export const enableGpuDispatchRecording = (): void => {
  const state = getRecorderState();
  state.enabled = true;
  patchGpuComputePassPrototype(state);
};

export const resetGpuDispatchRecording = (): void => {
  const state = getRecorderState();
  state.records.clear();
};

export const getGpuDispatchCoverageSnapshot =
  (): GpuDispatchCoverageSnapshot => {
    const state = getRecorderState();
    return {
      enabled: state.enabled,
      records: Array.from(state.records.values()).sort((a, b) =>
        a.label.localeCompare(b.label),
      ),
    };
  };

export const maybeEnableGpuDispatchRecordingFromLocation = (): void => {
  const scopeLocation = self.location;
  if (scopeLocation.search.includes("dispatchCoverage=1")) {
    enableGpuDispatchRecording();
  }
};
