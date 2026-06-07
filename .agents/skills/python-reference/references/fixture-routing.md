# Fixture Routing

| Need                              | Start with                                            | Usually inspect                                                                           |
| --------------------------------- | ----------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| Original style-transfer reference | `python-reference/style-transfer.py`                  | `README.md`                                                                               |
| First-pool forward fixture        | `python-reference/export_vgg19_first_pool.py`         | `public/vgg19-first-pool/README.md`, closest parity spec                                  |
| Full VGG/phase-3 fixture          | `python-reference/export_vgg19_phase3_full_pass.py`   | `python-reference/vgg19-phase3-full-pass-README.md`, `tests/helpers/fullPassArtifacts.ts` |
| Manual backward fixture           | `python-reference/export_phase4_backprop_fixtures.py` | `public/phase4-backprop/`, backward parity specs                                          |
| LBFGS fixture                     | `python-reference/export_lbfgs_fixtures.py`           | `public/lbfgs/`, LBFGS specs                                                              |
| Quantized model packs             | `python-reference/evaluate_vgg19_quantization.py`     | `public/vgg19-models/README.md`, model-pack loader and acceptance tests                   |

Use the committed compact fixtures for routine checks. The full phase-3 export
may require a large pretrained VGG19 download and can create large untracked
outputs, so run it only when exact full-pass evidence is necessary.

After activating `.venv`, invoke only the selected script. Do not run all
exporters as a default validation step.
