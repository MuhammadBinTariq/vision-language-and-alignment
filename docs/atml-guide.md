# ATML: model assumptions, generalisation, and training constraints

**Shared repository:** [Raahim58/ATML](https://github.com/Raahim58/ATML). The existing repository credits Muhammad Bin Tariq, Raahim A. Samad, and Muhammad Sabeer Asad as equal contributors. This guide links to that work; it does not copy or relabel it as an individual project.

## Goal

Investigate how architectural assumptions, objectives, and training conditions shape learned representations and their behaviour beyond the training distribution.

## Procedure and code map

| Investigation | Procedure | Source |
|---|---|---|
| Model biases | CNN/ViT, VAE/GAN, and CLIP experiments; inspect corruption sensitivity, shape/texture cues, generation, and semantic transfer | [PA1](https://github.com/Raahim58/ATML/tree/main/Programming%20Assignments/PA1%20-%20Model%20Biases) |
| Domain adaptation and generalisation | Unsupervised adaptation, held-out-domain evaluation, and CLIP prompt-learning experiments | [PA2](https://github.com/Raahim58/ATML/tree/main/Programming%20Assignments/PA2%20-%20DA%20%26%20DG) |
| Model compression | Pruning, quantisation, and knowledge distillation | [PA3](https://github.com/Raahim58/ATML/tree/main/Programming%20Assignments/PA3%20-%20Model%20Compression) |
| Federated learning | FedSGD and FedAvg; examine heterogeneity and extensions including FedProx, SCAFFOLD, FedGH, and FedSAM | [PA4](https://github.com/Raahim58/ATML/tree/main/Programming%20Assignments/PA4%20-%20FL) |

## Outcome

The repository contains implementations, reports, and saved analyses across these four directions. The PA1 write-up reports different failure patterns across model families: texture dependence in CNNs, different responses to spatial disruption in ViTs, stable but blurry VAE reconstructions, sharper but less stable GAN samples, and semantic transfer from CLIP. These are observations from the course settings, not universal rankings of the architectures.

The broader value of the collection is methodological: change a modelling or training assumption, inspect the resulting behaviour, and consider where the evaluation no longer supports a stronger conclusion. Consult the linked reports for task-specific settings and quantitative results.

This guide is a portfolio overview, not a new experimental run or a replacement for the shared repository's README.
