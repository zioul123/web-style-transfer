# Context Routing

Start with `docs/code-map.md`. Load other guidance only when the task matches.

| Task signal                                                  | Read                                                |
| ------------------------------------------------------------ | --------------------------------------------------- |
| Any non-trivial repository change                            | `docs/code-map.md`                                  |
| Cross-layer ownership or data-flow change                    | `docs/architecture.md`                              |
| Scope, approval, dependency, migration, or security question | `docs/change-policy.md`                             |
| Final review                                                 | `docs/review-rubric.md`                             |
| Documentation coverage audit for an existing diff            | invoke `$repo-doc-audit`                            |
| Phase status or roadmap change                               | `docs/webgpu-style-transfer-plan.md`                |
| User-facing setup, commands, or behavior                     | relevant `README.md` section                        |
| Detailed current module flow                                 | relevant section of `docs/architecture-overview.md` |
| Model-pack format or hosting                                 | `public/vgg19-models/README.md`                     |
| PyTorch fixtures or exporters                                | invoke `$python-reference`                          |

Follow imports and nearby tests from the smallest matching row in
`docs/code-map.md`. Do not preload generated fixtures, model shards, broad
source directories, or every documentation file.

Update the same documents when their described behavior, ownership, commands,
formats, or status changes.
