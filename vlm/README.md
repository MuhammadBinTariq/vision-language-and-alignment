# Continuous connectors and discrete visual tokens

## Goal

How can visual information become usable to a language model, and what happens to its existing language behaviour during adaptation? I explored two systems: a continuous bridge from CLIP patch features, and a discrete VQ-VAE vocabulary supporting question answering and image generation.

The visual datasets differ. These experiments inspect mechanisms and limitations; they do not establish a general ranking of the architectures.

## Procedure

### A. Continuous connector

A frozen CLIP ViT-B/32 supplies **49 patch tokens of dimension 768**. An MLP maps them into the **960-dimensional** embedding space of `SmolLM2-360M-Instruct`.

1. Select 10,000 CIFAR-10 training images and 2,000 test images, stratified by class.
2. Generate class-derived captions and five VQA templates per image: 50,000 training and 10,000 validation pairs. These synthetic labels do not evaluate open-ended image understanding.
3. Train the connector with the LM and vision encoder frozen. Inspect output norms and caption generation; the notebook records subsequent connector rescaling.
4. Train connector and LoRA adapters with `L_VQA + λ L_LM`, replaying pure-text Alpaca examples separately from visual examples.
5. Continue with VQA-only tuning and compare exact-match accuracy, replay perplexity, and modality gap. Run compact replay-weight, norm-regularisation, and LoRA-rank ablations.

### B. Discrete VQ-VAE

A VQ-VAE compresses synthetic images into a **4×4 grid** of indices from a **256-entry codebook**. These become token IDs in the language-model vocabulary.

1. Generate six synthetic visual classes and train the VQ-VAE. Compare gradient and EMA codebook updates, reconstruction MSE, utilisation, and dead codes.
2. Freeze the visual tokenizer. Add 256 visual tokens and two delimiters to the 49,152-token vocabulary.
3. Learn overlay embeddings/output weights and LoRA adapters with `L_VQA + γ L_IMG + λ L_LM`.
4. Use task-specific logit masks, then inspect VQA, decoded images, replay perplexity, and first-token distributions under different generation prompts.

The recorded discrete run switched to **bf16** after fp16 produced NaN logits. The notebook includes pilots and recovery cells; review them before unattended execution.

## Recorded outcomes

| Experiment | Evaluation | Observed result |
|---|---|---|
| Continuous, VQA + replay | 10,000 validation VQA pairs | **97.16%** exact match |
| Continuous, subsequent VQA-only phase | Same 10,000 pairs | **96.73%** exact match |
| Continuous modality gap | Fixed 200-example subset | **1.0916 → 1.0668** from Phase 2 to Phase 3 |
| Discrete, mixed-objective model | 500 synthetic validation VQA pairs | **99.8%** exact match |
| Discrete image generation | 12 decoded samples, plus temperature/prompt checks | Coarse textures without clear class-specific structure |

The continuous model's lower modality gap did **not** coincide with better accuracy. This separated a geometric diagnostic from the behaviour it was intended to explain.

The discrete model used visual tokens to answer questions, yet its first-token generation distributions for “spiral” and “triangle” were almost identical. This suggests weak prompt conditioning in the inspected run. The notebook discusses objective balance, training budget, and token ordering as possible contributors; these remain hypotheses rather than isolated causal findings.

Replay experiments measure `R = PPL_after / PPL_before` on an Alpaca subset. Since Alpaca also supplies replay training, **R < 1 demonstrates adaptation to that distribution, not preservation of general language ability**. In a discrete high-rank/no-replay stress run, R rose to 2.258 and quick VQA accuracy fell to 40%. Capacity and training conditions changed together, so this is not a clean one-factor comparison.

The 97.16% and 99.8% results are not directly comparable: images, questions, task complexity, and evaluation sample sizes differ. [Metric tables and evidence](../results/README.md).

## Code and notebooks

| Entry point | Role |
|---|---|
| [continuous_vlm.ipynb](notebooks/continuous_vlm.ipynb) | Continuous experiment, ablations, and analysis |
| [discrete_vlm.ipynb](notebooks/discrete_vlm.ipynb) | VQ-VAE, vocabulary expansion, mixed training, generation analysis |
| [data_exploration.ipynb](notebooks/data_exploration.ipynb) | Early data preparation and loading |
| [`scripts/`](scripts) and [`configs/`](configs) | Script-based implementation from the shared AI623 workspace |

Reported results come from notebooks supplied by Muhammad Bin Tariq. The script defaults and notebook experiments are related implementations, not an asserted exact match. Script sources are recorded in [PROVENANCE.md](../PROVENANCE.md) and [SOURCE_MANIFEST.json](../SOURCE_MANIFEST.json); this documentation does not attribute every shared script exclusively to one person.

## Run

Use a separate GPU environment from the alignment project. Install a matching PyTorch/torchvision pair, then the listed dependencies. The recorded notebook environment includes PyTorch 2.10; requirements are not an exact historical lockfile.

```bash
cd vision-language-and-alignment/vlm
python -m venv .venv
# Activate .venv for your shell.
python -m pip install -r requirements.txt
python -m pip install jupyter umap-learn
jupyter notebook notebooks/continuous_vlm.ipynb
# Or open notebooks/discrete_vlm.ipynb
```

For script-based shape/pipeline checks:

```bash
python scripts/run_all_part_a.py --quick_debug
python scripts/run_all_part_b.py --quick_debug
```

These still load models and data. Omit `--quick_debug` for full runs. Individual stages accept `--config` and `--output_dir`; some also accept `--resume` and `--max_steps`. Run from `vlm/` because orchestration uses relative paths.

Model/data downloads and substantial GPU compute are required. The recorded discrete mixed-training pilot used an A100 and about 8.6 GB peak allocated memory; requirements vary by batch and stage. Generated files go under `outputs/`. Checkpoints and caches are not bundled. Full training was not rerun during repository preparation.
