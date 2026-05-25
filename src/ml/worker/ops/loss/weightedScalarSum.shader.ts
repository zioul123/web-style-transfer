export const makeWeightedScalarSumShader = (weights: readonly number[]): string => {
  if (weights.length === 0) {
    throw new Error("weighted scalar sum requires at least one weight.");
  }
  const bindingDecls: string[] = [];
  const sumTerms: string[] = [];
  for (let i = 0; i < weights.length; i += 1) {
    bindingDecls.push(`@group(0) @binding(${i}) var<storage, read> s${i}: array<f32>;`);
    sumTerms.push(`(s${i}[0] * ${weights[i]})`);
  }
  const outBindingIndex = weights.length;
  return `
${bindingDecls.join("\n")}
@group(0) @binding(${outBindingIndex}) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x != 0u) { return; }
  out[0] = ${sumTerms.join(" + ")};
}`;
};
