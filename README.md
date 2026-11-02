# Vision, language, and alignment

How does a language model learn to use a new training signal, and what does that adaptation preserve or change? This repository collects experiments in preference optimisation and in connecting visual representations to a small language model.

The work was developed in the context of AI623: Deep Vision Language Models at LUMS. It is a new, curated repository, not a fork. It contains implementation code, experimental notebooks, and documentation of the recorded outcomes; it does not present these course experiments as published research.

## Start here

| Project | Question | Code and results |
|---|---|---|
| LLM alignment | How do DPO, PPO, GRPO, and verifiable rewards differ in their training signals and failure modes? | [Alignment README](alignment/README.md) |
| Continuous and discrete VLMs | How can visual information enter a language model, and how does visual adaptation affect language behaviour? | [VLM README](vlm/README.md) |

The alignment experiments are **entirely Muhammad Bin Tariq's work**. The separate DVLM research project on **task-associated representation geometry** was co-developed by **Muhammad Bin Tariq and Raahim A. Samad**; that research project is not copied into this repository. Do not infer common authorship from the fact that these projects were stored in the same course workspace.

The VLM notebooks were supplied by Muhammad Bin Tariq. The accompanying script implementations were recovered from the shared AI623 repository with source paths and commit recorded in [SOURCE_MANIFEST.json](SOURCE_MANIFEST.json). [PROVENANCE.md](PROVENANCE.md) explains those sources and the publication edits. This repository does not make a blanket claim of sole authorship over every imported VLM script.

## What the experiments showed

- **Alignment:** in the recorded 360M pilot, GRPO achieved a 56% learned-reward-model win rate against SFT on 50 held-out prompts, compared with 50% for PPO and 48% for DPO. This is a small automatic evaluation, not evidence of a general ranking. The RLVR run reached 0/128 GSM8K pass@1 and frequently produced groups with identical rewards.
- **Continuous VLM:** the CLIP–MLP–SmolLM2 model reached 97.16% exact-match accuracy on 10,000 templated CIFAR-10 VQA pairs. Further VQA-only tuning reduced the measured modality gap but lowered accuracy to 96.73%: geometric proximity was not a substitute for task performance.
- **Discrete VLM:** the VQ-VAE token model reached 99.8% on 500 synthetic VQA pairs, while generated images lacked clear class-specific structure. Reading visual tokens and generating them conditioned on text remained different capabilities.

The two VLM percentages come from different datasets and evaluation sizes. They are **not** a controlled comparison of architecture quality. All numbers above come from supplied notebook records, not a fresh training run during repository preparation.

## Repository structure

```text
alignment/              # Preference learning and verifiable-reward implementation
  notebooks/            # 360M runner and historical 1.7B experiment
  configs/              # Model, optimisation, and data configuration
vlm/                    # Script-based continuous and discrete VLM pipelines
  notebooks/            # Recorded-experiment source notebooks, outputs cleared
  scripts/              # Individual phases and full pipeline runners
  configs/
results/                # Compact transcriptions of the recorded metric tables
PROVENANCE.md           # Authorship, source snapshot, and publication edits
SOURCE_MANIFEST.json    # Imported script paths and original Git blob hashes
NOTEBOOK_PROVENANCE.json
```

## Running the work

Use separate Python environments for `alignment/` and `vlm/`; their dependencies are listed inside each directory. The project READMEs give commands, model access requirements, and notebook entry points. GPU training, model downloads, and dataset downloads are required for full runs. Checkpoints are not bundled.

The notebooks preserve experimental code and explanatory notes, but outputs and runtime metadata have been cleared. The recorded metrics are retained as small tables with source cell references. Historical notebook state, updated library versions, and the distinction between notebook experiments and script defaults mean an exact numerical rerun is not guaranteed.

## Related collaborative work

[ATML — Advanced Topics in Machine Learning](https://github.com/Raahim58/ATML) is the shared repository for work on model biases, domain adaptation/generalisation, compression, and federated learning. Its existing contributor credits are retained at the source. See [the portfolio guide](docs/atml-guide.md) for an overview and direct links.

No new blanket licence is assigned to imported code, model weights, or course materials. Existing attribution and the original terms of each dependency and dataset continue to apply.
