# Authorship and source provenance

## Project boundaries

| Work | Attribution |
|---|---|
| Alignment: reward modelling, SFT, PPO, DPO, GRPO, RLVR, evaluation and related notebooks | Muhammad Bin Tariq, sole author of this work, as explicitly confirmed by him |
| DVLM research project leading to current task-associated representation geometry research | Muhammad Bin Tariq and Raahim A. Samad; separate from the code packaged here |
| ATML | Collaborative work; the original repository lists Muhammad Bin Tariq, Raahim A. Samad, and Muhammad Sabeer Asad as equal contributors |
| VLM notebooks in this repository | Supplied by Muhammad Bin Tariq from his course workspace |
| VLM script implementation | Recovered from the shared AI623 source snapshot below; retained source attribution, without asserting exclusive authorship of each file |

The name of an account hosting code is not itself an authorship statement. In particular, the alignment implementation was present in the shared course repository, but the alignment work belongs entirely to Muhammad Bin Tariq.

## Imported code

Source: [Raahim58/Deep-Vision-Language-Models-](https://github.com/Raahim58/Deep-Vision-Language-Models-/tree/91fe1c49e6ed94b9ca584d856c5aba6ea5c4da33), commit `91fe1c49e6ed94b9ca584d856c5aba6ea5c4da33`.

- `Programming Assignments/PA2_1/code/` → `alignment/`
- `Programming Assignments/PA3/pa3_vlm/` → `vlm/`

Only relevant Python source, configuration, requirements, and project metadata were selected. The directory READMEs were rewritten. The manifest records the original blob hashes; it is a source record, not a claim that edited documentation still matches those hashes. No Git history or fork relationship was imported into the new GitHub repository.

The original source repository and local source notebooks have not been modified. The separate `Project/` research folder, textbooks, lecture material, dataset archives, compiled files, checkpoints, and unrelated diffusion assignment are not included.

## Notebook preparation

The published notebooks are mapped to their originals in [NOTEBOOK_PROVENANCE.json](NOTEBOOK_PROVENANCE.json). One explanatory cell was prepended, so a zero-based original cell index is one lower than its index in the published notebook.

Publication edits:

1. Clear saved outputs, execution counts, widget state, and runtime metadata; preserve code and explanatory text.
2. Add a provenance note and stable cell IDs.
3. Point the alignment runner at this repository and its `alignment/` folder; leave training logic unchanged.
4. Retain compact result transcriptions separately rather than presenting stale output as a new execution.

The historical 1.7B notebook is an exploratory record with state-revival cells. The 360M runner is the preferred alignment entry point. `data_exploration.ipynb` is an early VLM setup notebook; the continuous and discrete notebooks contain the fuller experiments.

## Validation scope

Repository preparation checks notebook JSON, Python syntax, local documentation links, and credential-like text. Full model training and numerical reproduction are not part of that check. The supplied records remain the evidence for the documented results. Original notebook interpretations are retained as historical commentary; the READMEs distinguish measured observations from proposed explanations.
