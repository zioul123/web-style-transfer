/// <reference lib="webworker" />

import { mountMessageRouter } from "./ml/worker/main-thread-protocol/messageRouter";
import { maybeEnableGpuDispatchRecordingFromLocation } from "./ml/worker/runtime/dispatchRecorder";

maybeEnableGpuDispatchRecordingFromLocation();
mountMessageRouter();
