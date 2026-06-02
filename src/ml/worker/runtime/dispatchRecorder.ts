/// <reference lib="webworker" />

type DispatchRecord = {
  type: "__gpu-dispatch-coverage";
  label: string;
  workgroups: readonly [number, number, number];
};

type CoverageComputePassEncoder = GPUComputePassEncoder & {
  __styleTransferPipelineLabel?: string;
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

const postDispatchRecord = (
  label: string,
  workgroups: readonly [number, number, number],
): void => {
  const record: DispatchRecord = {
    type: "__gpu-dispatch-coverage",
    label,
    workgroups,
  };
  self.postMessage(record);
};

export const installGpuDispatchRecorder = (): void => {
  if (typeof GPUComputePassEncoder === "undefined") return;

  const originalSetPipeline = GPUComputePassEncoder.prototype.setPipeline;
  GPUComputePassEncoder.prototype.setPipeline = function setPipelineWithLabel(
    this: CoverageComputePassEncoder,
    pipeline: GPUComputePipeline,
  ): void {
    this.__styleTransferPipelineLabel = pipeline.label || "unlabeled-pipeline";
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
      postDispatchRecord(
        this.__styleTransferPipelineLabel ?? "pipeline-not-set",
        [workgroupCountX, workgroupCountY, workgroupCountZ],
      );
      originalDispatchWorkgroups.call(
        this,
        workgroupCountX,
        workgroupCountY,
        workgroupCountZ,
      );
    };

  const originalRequestDevice = GPUAdapter.prototype.requestDevice;
  GPUAdapter.prototype.requestDevice = async function requestDeviceWithCoverage(
    this: GPUAdapter,
    descriptor?: GPUDeviceDescriptor,
  ): Promise<GPUDevice> {
    const device = await originalRequestDevice.call(this, descriptor);
    const originalCreateComputePipeline =
      device.createComputePipeline.bind(device);
    device.createComputePipeline = (
      pipelineDescriptor: GPUComputePipelineDescriptor,
    ): GPUComputePipeline => {
      pipelineDescriptor.label ||= getPipelineLabelFromStack();
      return originalCreateComputePipeline(pipelineDescriptor);
    };
    return device;
  };
};
