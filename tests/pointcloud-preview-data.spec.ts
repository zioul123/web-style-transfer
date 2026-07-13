import { readFileSync } from "node:fs";
import { expect, test } from "@playwright/test";
import {
  buildPointCloudMeshData,
  parsePointCloudMeshText,
} from "../src/features/pointcloud-preview/loadPointCloudMesh";
import {
  ablationDimensionDefinitionByKey,
  buildAblationExperimentKey,
  buildAblationMatrixCellKey,
  parseAblationExperimentFilename,
  parseAblationExperimentFilenames,
  summarizeAblationDimensions,
} from "../src/features/pointcloud-preview/ablation/experimentFilenames";
import { buildAblationMatrix } from "../src/features/pointcloud-preview/ablation/ablationMatrix";
import { sampleInterpolatedColor } from "../src/features/pointcloud-preview/math/interpolation";
import {
  buildKdTree3d,
  findKNearestNeighbors,
} from "../src/ml/geometry/kdTree3d";
import type {
  ConvolutionKernelPathGroup,
  PointCloudMeshJson,
} from "../src/features/pointcloud-preview/types";

const tinyFixture = JSON.parse(
  readFileSync(
    new URL(
      "../python-reference/pointcloud-style-transfer/pc-and-mesh-tiny-example.json",
      import.meta.url,
    ),
    "utf8",
  ),
) as PointCloudMeshJson;

const buildKernelPathGroup = (
  anchor: readonly [number, number, number],
): ConvolutionKernelPathGroup =>
  Array.from({ length: 8 }, (_, pathIndex) => [
    anchor,
    [anchor[0] + (pathIndex + 1) * 0.1, anchor[1], anchor[2] + 0.25] as const,
  ]);

test("point-cloud preview parser converts the tiny fixture into typed arrays", () => {
  const data = buildPointCloudMeshData(tinyFixture);

  expect(data.meshVertexCount).toBe(4);
  expect(data.meshFaceCount).toBe(2);
  expect(data.pointCount).toBe(3);
  expect(data.meshPositions).toBeInstanceOf(Float32Array);
  expect(data.meshIndices).toBeInstanceOf(Uint32Array);
  expect(data.meshVertexColors).toHaveLength(12);
  expect(data.bounds.min).toEqual([-1, 0, -1]);
  expect(data.bounds.max).toEqual([1, 0, 1]);
  expect(data.spatialHash.sortedPointPositions).toHaveLength(
    data.pointPositions.length,
  );
  expect(data.spatialHash.sortedPointColors).toHaveLength(
    data.pointColors.length,
  );
  expect(data.spatialHash.cellOffsets).toHaveLength(
    data.spatialHash.cellCount + 1,
  );
  expect(data.spatialHash.cellOffsets[data.spatialHash.cellCount]).toBe(
    data.pointCount,
  );
  expect(data.sourceKind).toBe("point-cloud-mesh");
  expect(data.convolutionKernelLevels).toEqual([]);
});

test("point-cloud preview parser derives convolution kernel levels and anchors", () => {
  const data = buildPointCloudMeshData({
    ...tinyFixture,
    level_0_paths: [
      buildKernelPathGroup([0, 0, 0]),
      buildKernelPathGroup([1, 0, 0]),
    ],
    level_2_paths: [buildKernelPathGroup([0.5, 0, 0.5])],
  });

  expect(data.sourceKind).toBe("convolution-kernels");
  expect(data.convolutionKernelLevels.map((level) => level.levelIndex)).toEqual(
    [0, 2],
  );
  expect(data.convolutionKernelLevels[0]?.groupCount).toBe(2);
  expect(
    Array.from(data.convolutionKernelLevels[0]?.anchorPositions ?? []),
  ).toEqual([0, 0, 0, 1, 0, 0]);
  expect(data.convolutionKernelLevels[1]?.groupCount).toBe(1);
  expect(data.convolutionKernelLevels[1]?.groups[0]?.length).toBe(8);
});

test("point-cloud preview parser rejects malformed convolution kernel groups", () => {
  const invalidText = JSON.stringify({
    ...tinyFixture,
    level_0_paths: [[[[0, 0, 0]]]],
  });

  expect(() => parsePointCloudMeshText(invalidText)).toThrow(
    /must contain exactly 8 geodesic paths/i,
  );
});

test("point-cloud spatial hash preserves all point/color tuples after sorting", () => {
  const data = buildPointCloudMeshData({
    pc_xyz: [
      [2, 0, 0],
      [0, 0, 0],
      [1, 0, 0],
      [3, 0, 0],
    ],
    pc_rgb: [
      [0.2, 0.1, 0.4],
      [0.9, 0.8, 0.7],
      [0.3, 0.5, 0.7],
      [0.1, 0.9, 0.2],
    ],
    m_verts: [
      [0, 0, 0],
      [1, 0, 0],
      [1, 0, 1],
    ],
    m_faces: [[0, 1, 2]],
  });

  const originalTuples = Array.from(
    { length: data.pointCount },
    (_, index) =>
      `${data.pointPositions[index * 3]},${data.pointPositions[index * 3 + 1]},${data.pointPositions[index * 3 + 2]}|${data.pointColors[index * 3]},${data.pointColors[index * 3 + 1]},${data.pointColors[index * 3 + 2]}`,
  ).sort();
  const hashedTuples = Array.from(
    { length: data.pointCount },
    (_, index) =>
      `${data.spatialHash.sortedPointPositions[index * 3]},${data.spatialHash.sortedPointPositions[index * 3 + 1]},${data.spatialHash.sortedPointPositions[index * 3 + 2]}|${data.spatialHash.sortedPointColors[index * 3]},${data.spatialHash.sortedPointColors[index * 3 + 1]},${data.spatialHash.sortedPointColors[index * 3 + 2]}`,
  ).sort();

  expect(hashedTuples).toEqual(originalTuples);
});

test("point-cloud preview parser rejects mismatched point colour counts", () => {
  const invalidText = JSON.stringify({
    ...tinyFixture,
    pc_rgb: tinyFixture.pc_rgb.slice(0, 2),
  });

  expect(() => parsePointCloudMeshText(invalidText)).toThrow(
    /pc_xyz and pc_rgb must contain the same number of entries/i,
  );
});

test("point-cloud preview parser rejects out-of-range mesh faces", () => {
  const invalidText = JSON.stringify({
    ...tinyFixture,
    m_faces: [[0, 1, 9]],
  });

  expect(() => parsePointCloudMeshText(invalidText)).toThrow(
    /references vertex 9/i,
  );
});

test("3-NN interpolation returns the exact point colour on an exact hit", () => {
  const pointPositions = new Float32Array([0, 0, 0, 2, 0, 0, 0, 2, 0]);
  const pointColors = new Float32Array([1, 0, 0, 0, 1, 0, 0, 0, 1]);
  const kdTree = buildKdTree3d(pointPositions);

  const sample = sampleInterpolatedColor(
    kdTree,
    pointPositions,
    pointColors,
    [0, 0, 0],
  );

  expect(sample.color[0]).toBeCloseTo(1);
  expect(sample.color[1]).toBeCloseTo(0);
  expect(sample.color[2]).toBeCloseTo(0);
  expect(sample.neighbors[0].index).toBe(0);
});

test("3-NN interpolation follows inverse-squared distance weighting", () => {
  const pointPositions = new Float32Array([1, 0, 0, 2, 0, 0, 4, 0, 0]);
  const pointColors = new Float32Array([1, 0, 0, 0, 1, 0, 0, 0, 1]);
  const kdTree = buildKdTree3d(pointPositions);

  const sample = sampleInterpolatedColor(
    kdTree,
    pointPositions,
    pointColors,
    [0, 0, 0],
  );

  const weights = [1, 1 / 4, 1 / 16];
  const totalWeight = weights.reduce((sum, value) => sum + value, 0);
  expect(sample.color[0]).toBeCloseTo(weights[0] / totalWeight);
  expect(sample.color[1]).toBeCloseTo(weights[1] / totalWeight);
  expect(sample.color[2]).toBeCloseTo(weights[2] / totalWeight);
});

test("k-d tree nearest-3 query stays ordered on the tiny fixture", () => {
  const data = buildPointCloudMeshData(tinyFixture);

  const neighbors = findKNearestNeighbors(
    data.kdTree,
    data.pointPositions,
    [0.86, 0, -0.18],
    3,
  );

  expect(neighbors).toHaveLength(3);
  expect(neighbors[0].index).toBe(0);
  expect(neighbors[0].squaredDistance).toBeLessThanOrEqual(
    neighbors[1].squaredDistance,
  );
  expect(neighbors[1].squaredDistance).toBeLessThanOrEqual(
    neighbors[2].squaredDistance,
  );
});

test("point-cloud ablation parser reads name_expt filenames with output steps", () => {
  const filename =
    "1000000sw_0cw_0.001tv_L2_2c-spf_192x256s-img_SIMPLE_AXIS_RAW_COLORS_MAX_SPECTRAL_4knn_0.7rf_2gr_20000tris_1.5std-attn_300steps_step320.json";

  const result = parseAblationExperimentFilename(filename);

  expect(result.ok).toBe(true);
  if (!result.ok) {
    throw new Error(result.failure.reason);
  }
  expect(result.file.config).toEqual({
    styleWeight: 1000000,
    contentWeight: 0,
    tvWeight: 0.001,
    tvMode: "L2",
    contentSamplesPerFace: 2,
    styleResolution: "192x256",
    styleSamplesPerFace: null,
    frameMode: "SIMPLE_AXIS",
    colorMode: "RAW_COLORS",
    poolMode: "MAX",
    distanceMeasure: "SPECTRAL",
    knn: 4,
    radiusFactor: 0.7,
    geodesicRatio: 2,
    maxTriangles: 20000,
    attenuationLabel: "1.5std",
    optimizationSteps: 300,
    outputStep: 320,
  });

  const maxTrianglesSummary = summarizeAblationDimensions([result.file]).find(
    (summary) => summary.definition.key === "maxTriangles",
  );
  expect(maxTrianglesSummary?.values).toEqual([
    {
      value: 20000,
      label: "20000",
      key: "number:20000",
      count: 1,
    },
  ]);
});

test("point-cloud ablation parser reads list-valued style weights", () => {
  const filename =
    "1.0-0.5-0.25sw_3cw_0.5tv_L1_2c-spf_192x256s-img_SIMPLE_AXIS_RAW_COLORS_MAX_EUCLIDEAN_12knn_1.25rf_3gr_2.0std-attn_60steps_step987.json";

  const result = parseAblationExperimentFilename(filename);

  expect(result.ok).toBe(true);
  if (!result.ok) {
    throw new Error(result.failure.reason);
  }
  expect(result.file.config.styleWeight).toEqual([1, 0.5, 0.25]);

  const summaries = summarizeAblationDimensions([result.file]);
  const styleWeightSummary = summaries.find(
    (summary) => summary.definition.key === "styleWeight",
  );
  expect(styleWeightSummary?.values).toEqual([
    {
      value: "1-0.5-0.25",
      label: "1-0.5-0.25",
      key: "string:1-0.5-0.25",
      count: 1,
    },
  ]);
});

test("point-cloud ablation parser reads mapped labels and style samples per face", () => {
  const filename =
    "7sw_3cw_0.5tv_L1_2.5c-spf_4.5s-spf_PCP_LOGIT_AVG_SPECTRAL_12knn_1.25rf_3gr_2.0std-attn_60steps_step987.json";

  const result = parseAblationExperimentFilename(filename);

  expect(result.ok).toBe(true);
  if (!result.ok) {
    throw new Error(result.failure.reason);
  }
  expect(result.file.config.frameMode).toBe("PCP");
  expect(result.file.config.colorMode).toBe("LOGIT");
  expect(result.file.config.poolMode).toBe("AVG");
  expect(result.file.config.distanceMeasure).toBe("SPECTRAL");
  expect(result.file.config.tvMode).toBe("L1");
  expect(result.file.config.contentSamplesPerFace).toBe(2.5);
  expect(result.file.config.styleResolution).toBeNull();
  expect(result.file.config.styleSamplesPerFace).toBe(4.5);
  expect(result.file.config.outputStep).toBe(987);
});

test("point-cloud ablation parser accepts absent style source and legacy max-pool filenames", () => {
  const filename =
    "1sw_2cw_0tv_L2_8c-spf_SPL-S_RAW_COLORS_EUCLIDEAN_4knn_0.7rf_2gr_3.0std-attn_120steps.json";

  const result = parseAblationExperimentFilename(filename);

  expect(result.ok).toBe(true);
  if (!result.ok) {
    throw new Error(result.failure.reason);
  }
  expect(result.file.config.frameMode).toBe("SPL-S");
  expect(result.file.config.poolMode).toBe("MAX");
  expect(result.file.config.styleResolution).toBeNull();
  expect(result.file.config.styleSamplesPerFace).toBeNull();
  expect(result.file.config.maxTriangles).toBeNull();
  expect(result.file.config.outputStep).toBeNull();

  const bstResult = parseAblationExperimentFilename(
    "1sw_2cw_0tv_L1_8c-spf_BST-S_RAW_COLORS_MAX_EUCLIDEAN_4knn_0.7rf_2gr_3.0std-attn_120steps_step60.json",
  );
  expect(bstResult.ok).toBe(true);
  if (!bstResult.ok) {
    throw new Error(bstResult.failure.reason);
  }
  expect(bstResult.file.config.frameMode).toBe("BST-S");
  expect(bstResult.file.config.outputStep).toBe(60);
});

test("point-cloud ablation parser normalizes legacy biharmonic distance labels", () => {
  const result = parseAblationExperimentFilename(
    "1sw_2cw_0tv_L1_8c-spf_SIMPLE_AXIS_RAW_COLORS_MAX_BIHARMONIC_4knn_0.7rf_2gr_3.0std-attn_120steps_step60.json",
  );

  expect(result.ok).toBe(true);
  if (!result.ok) {
    throw new Error(result.failure.reason);
  }
  expect(result.file.config.distanceMeasure).toBe("SPECTRAL");
  expect(result.file.config.poolMode).toBe("MAX");
});

test("point-cloud ablation batch parser reports unparsed filenames", () => {
  const result = parseAblationExperimentFilenames([
    "1sw_2cw_0tv_L2_8c-spf_SIMPLE_AXIS_RAW_COLORS_EUCLIDEAN_4knn_0.7rf_2gr_3.0std-attn_120steps.json",
    "notes.txt",
  ]);

  expect(result.files).toHaveLength(1);
  expect(result.failures).toEqual([
    {
      filename: "notes.txt",
      reason:
        "Filename does not match the expected point-cloud ablation experiment pattern.",
    },
  ]);
});

test("point-cloud ablation summaries sort values deterministically", () => {
  const filenames = [
    "1sw_2cw_10tv_L2_16c-spf_640x480s-img_SIMPLE_AXIS_RAW_COLORS_EUCLIDEAN_4knn_0.7rf_2gr_3.0std-attn_120steps_step60.json",
    "1sw_2cw_1tv_L1_2c-spf_192x256s-img_BST-S_RAW_COLORS_MAX_SPECTRAL_4knn_0.7rf_2gr_1.5std-attn_120steps_step320.json",
    "1sw_2cw_0.1tv_L2_4c-spf_320x240s-img_PCP_LOGIT_AVG_SPECTRAL_4knn_0.7rf_2gr_2.0std-attn_120steps_step160.json",
    "1sw_2cw_0.001tv_L1_8c-spf_4s-spf_SPL-S_RAW_COLORS_AVG_EUCLIDEAN_4knn_0.7rf_2gr_1.0std-attn_120steps_step120.json",
  ];
  const parsed = parseAblationExperimentFilenames(filenames);
  expect(parsed.failures).toEqual([]);

  const summaries = summarizeAblationDimensions(parsed.files);
  const byKey = new Map(
    summaries.map((summary) => [summary.definition.key, summary]),
  );

  expect(
    byKey.get("contentSamplesPerFace")?.values.map((value) => value.value),
  ).toEqual([2, 4, 8, 16]);
  expect(byKey.get("tvMode")?.values.map((value) => value.value)).toEqual([
    "L1",
    "L2",
  ]);
  expect(byKey.get("frameMode")?.values.map((value) => value.value)).toEqual([
    "PCP",
    "SPL-S",
    "BST-S",
    "SIMPLE_AXIS",
  ]);
  expect(byKey.get("poolMode")?.values.map((value) => value.value)).toEqual([
    "MAX",
    "AVG",
  ]);
  expect(
    byKey.get("styleResolution")?.values.map((value) => value.value),
  ).toEqual(["192x256", "320x240", "640x480"]);
  expect(byKey.get("maxTriangles")).toBeUndefined();
  expect(byKey.get("outputStep")?.values.map((value) => value.label)).toEqual([
    "step60",
    "step120",
    "step160",
    "step320",
  ]);
});

test("point-cloud ablation keys are stable for matrix and experiment lookup", () => {
  const parsed = parseAblationExperimentFilename(
    "1sw_2cw_0tv_L2_8c-spf_SIMPLE_AXIS_RAW_COLORS_EUCLIDEAN_4knn_0.7rf_2gr_3.0std-attn_120steps_step60.json",
  );
  expect(parsed.ok).toBe(true);
  if (!parsed.ok) {
    throw new Error(parsed.failure.reason);
  }

  const xDefinition = ablationDimensionDefinitionByKey.get(
    "contentSamplesPerFace",
  );
  const yDefinition = ablationDimensionDefinitionByKey.get("distanceMeasure");
  expect(xDefinition).toBeDefined();
  expect(yDefinition).toBeDefined();
  if (xDefinition === undefined || yDefinition === undefined) {
    throw new Error("Missing ablation dimension definitions.");
  }

  expect(
    buildAblationMatrixCellKey(xDefinition, 8, yDefinition, "EUCLIDEAN"),
  ).toBe("contentSamplesPerFace=number:8|distanceMeasure=string:EUCLIDEAN");
  expect(buildAblationExperimentKey(parsed.file)).toContain(
    "outputStep=number:60",
  );
});

test("point-cloud ablation matrix prunes empty axes after fixed filters", () => {
  const filenames = [
    "1sw_2cw_0tv_L2_2c-spf_SIMPLE_AXIS_RAW_COLORS_EUCLIDEAN_4knn_0.7rf_2gr_3.0std-attn_120steps_step60.json",
    "1sw_2cw_0tv_L2_4c-spf_SIMPLE_AXIS_RAW_COLORS_SPECTRAL_4knn_0.7rf_2gr_3.0std-attn_120steps_step60.json",
    "1sw_2cw_0tv_L2_8c-spf_SIMPLE_AXIS_RAW_COLORS_SPECTRAL_4knn_0.7rf_2gr_3.0std-attn_120steps_step120.json",
  ];
  const parsed = parseAblationExperimentFilenames(filenames);
  const summaries = summarizeAblationDimensions(parsed.files);
  const matrix = buildAblationMatrix(parsed.files, summaries, {
    xAxis: "contentSamplesPerFace",
    yAxis: "distanceMeasure",
    fixedValues: new Map([["outputStep", [60]]]),
  });

  expect(matrix).not.toBeNull();
  expect(matrix?.xSummary.values.map((value) => value.value)).toEqual([2, 4]);
  expect(matrix?.rows.map((row) => row.yValue)).toEqual([
    "EUCLIDEAN",
    "SPECTRAL",
  ]);
  expect(
    matrix?.rows.map((row) => row.cells.map((cell) => cell.status)),
  ).toEqual([
    ["available", "missing"],
    ["missing", "available"],
  ]);
});
