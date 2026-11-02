# Recorded results and evidence

The CSV files transcribe selected original notebook outputs. They do not represent a new training run. Original notebooks remain unchanged in the author's workspace; published notebooks clear outputs and prepend one provenance cell.

| Result | Original notebook | Zero-based original cell / location |
|---|---|---|
| 360M comparison table | `DVLM_PA2_1.ipynb` | Cell 33, `Evaluation metrics`; evaluator uses 50 prompts |
| 0/128 GSM8K pass@1 | `DVLM_PA2_1.ipynb` | Cell 33, GSM8K evaluation |
| RLVR 0.875–1.000 degenerate-group logs | `DVLM_PA2_1.ipynb` | Cell 31, RLVR training output |
| 71.37% reward-model accuracy | `DVLM.ipynb` | Cell 10, `Final Test Preference Accuracy` |
| SFT PPL 7.14 → 5.62 | `DVLM.ipynb` | Cell 13, validation-perplexity trace |
| Continuous Phase 2 vs Phase 3 accuracy | `PartA.ipynb` | Cell 120, comparison table |
| Continuous modality gaps | `PartA.ipynb` | Cell 159, analysis; computation in cell 138 |
| Discrete 99.8% VQA | `PartB.ipynb` | Cell 101, 500-pair evaluation; summary in cell 111 |
| Discrete generation limitations | `PartB.ipynb` | Cells 104–108 and summary in cell 111 |

The alignment preference-accuracy column repeated 0.692308 across methods in the original output. It is not presented as a separate comparative success claim here. The nonzero SFT self-comparison KL is retained and flagged rather than silently corrected.

No confidence intervals, multi-seed aggregates, or generalisation claims have been added. VLM results from different datasets are kept separate.
