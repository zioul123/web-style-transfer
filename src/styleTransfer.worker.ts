/// <reference lib="webworker" />

import { mountMessageRouter } from "./ml/worker/main-thread-protocol/messageRouter";

mountMessageRouter();
