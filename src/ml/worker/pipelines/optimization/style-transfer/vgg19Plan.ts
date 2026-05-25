export type Vgg19PlanStep =
  | {
      kind: "conv-relu";
      convLayerIndex: number;
      reluLayerIndex: number;
    }
  | {
      kind: "pool";
      poolLayerIndex: number;
    };

export const VGG19_STYLE_TRANSFER_PLAN: readonly Vgg19PlanStep[] = [
  { kind: "conv-relu", convLayerIndex: 0, reluLayerIndex: 1 },
  { kind: "conv-relu", convLayerIndex: 2, reluLayerIndex: 3 },
  { kind: "pool", poolLayerIndex: 4 },
  { kind: "conv-relu", convLayerIndex: 5, reluLayerIndex: 6 },
  { kind: "conv-relu", convLayerIndex: 7, reluLayerIndex: 8 },
  { kind: "pool", poolLayerIndex: 9 },
  { kind: "conv-relu", convLayerIndex: 10, reluLayerIndex: 11 },
  { kind: "conv-relu", convLayerIndex: 12, reluLayerIndex: 13 },
  { kind: "conv-relu", convLayerIndex: 14, reluLayerIndex: 15 },
  { kind: "conv-relu", convLayerIndex: 16, reluLayerIndex: 17 },
  { kind: "pool", poolLayerIndex: 18 },
  { kind: "conv-relu", convLayerIndex: 19, reluLayerIndex: 20 },
  { kind: "conv-relu", convLayerIndex: 21, reluLayerIndex: 22 },
  { kind: "conv-relu", convLayerIndex: 23, reluLayerIndex: 24 },
  { kind: "conv-relu", convLayerIndex: 25, reluLayerIndex: 26 },
  { kind: "pool", poolLayerIndex: 27 },
  { kind: "conv-relu", convLayerIndex: 28, reluLayerIndex: 29 },
];
