/// <reference lib="webworker" />

import { mountMessageRouter } from "./ml/worker/main-thread-protocol/messageRouter";
import { installGpuDispatchRecorder } from "./ml/worker/runtime/dispatchRecorder";

if (self.location.search.includes("dispatchCoverage=1")) {
  installGpuDispatchRecorder();
}

mountMessageRouter();
