# Context

Context of the project should be obtained by reading `README.md` and `docs/webgpu-style-transfer-plan.md` prior to tackling your task.

# Testing

When you need to run Playwright e2e tests, your environment lacks certain dependencies (which manifests as missing shared library `libatk-1.0.so.0`), so you must use the following to install Playwright:

```bash
npx playwright install chromium
npx playwright install-deps chromium
```

When running some e2e tests, certain tests that require fixtures will be skipped by default. Prior to running them, you will need to generate the fixtures by running:

```bash
python python-reference/export_vgg19_phase3_full_pass.py
```

Do not commit these fixtures if they were not already commited, as they are very big.

# Style

Use explicit types when declaring state, e.g. instead of `const [myState, setMyState] = useState('abcde.');`, use `const [myState, setMyState] = useState<string>('abcde.');`.

When declaring types, use narrower types instead of general ones where possible, e.g. instead of `{ ok: boolean; result?: number; errorMsg?: string }`, use `{ ok: true; result: number} | { ok: false; errorMsg: string }`. Instead of `{ op: 'add' | 'clamp'; value1: number; value2?: number; clampMin?: number; clampMax?: number}`, use `{ op: 'add'; value1: number; value2: number } | { op: 'clamp'; value1: number }`.

Avoid using type casts where possible, e.g. if we have `funcCall(a): { ok: boolean; values: number[] } | { ok: boolean; scalar: number }`, then instead of `const b = funcCall(a) as { ok: boolean; values?: number[] }`, prefer that using a type guard to prevent tricky bugs down the line, e.g. `const b = funcCall(a); if (!isScalarResult(b)) { throw new Error("..."); }`.

# Finishing a task

Before concluding a task, make sure to run `npm run build`, and ensure the build is successful.

When concluding, remember to update the attached github PR description if applicable.
