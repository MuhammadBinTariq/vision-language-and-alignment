# LLM alignment: preferences, relative rewards, and sparse feedback

**Author: Muhammad Bin Tariq.** The alignment work, including PPO, DPO, GRPO, and RLVR, is entirely my own. It is distinct from the jointly developed DVLM representation-geometry research project. [Provenance](../PROVENANCE.md) explains why the implementation was recovered from a shared course repository.

## Goal

I wanted to understand alignment methods through the training signals they produce. How does a pairwise preference become an update to a language model? What changes when a critic is replaced by group-relative rewards? And what happens when a verifier gives the same reward to every response in a group?

I implemented an alignment pipeline and compared these methods in small-model experiments, focusing on their updates, reference-policy constraints, and reward-signal failure modes.

## Procedure

1. Parse `Anthropic/hh-rlhf`, `harmless-base`, into prompt/chosen/rejected triplets at the final assistant turn.
2. Warm-start the policy with supervised fine-tuning on response tokens, and train a scalar reward model from pairwise preferences.
3. Start policy optimisation from SFT. DPO uses preference pairs directly; PPO uses sampled responses, a value model, GAE, a clipped objective, and KL regularisation; GRPO uses grouped rollouts and within-group advantages.
4. Replace the learned reward with exact-answer verification for GSM8K RLVR. Inspect reward dispersion, degenerate groups, response lengths, gradients, and KL behaviour.
5. Compare generated responses with the SFT reference using the learned reward model and sampled-token KL. Run focused ablations through `run_ablations.py`.

The modular implementation uses explicit training loops rather than delegating the algorithms to TRL or Hugging Face Trainer.

## Two experiments, kept separate

| Record | Policy | Scope |
|---|---|---|
| [Historical notebook](notebooks/legacy_1p7b.ipynb) | `SmolLM2-1.7B-Instruct` | Initial implementation, full harmless-base parsing, reward modelling, SFT, and later alignment exploration |
| [Modular runner](notebooks/run_alignment.ipynb) | `SmolLM2-360M` | Smaller configurable DPO/PPO/GRPO/RLVR comparison; source of the table below |

The 360M run uses `Llama-3.2-1B-Instruct` reward/value backbones and rank-8 LoRA on `q_proj` and `v_proj`. The recorded runner overrides script defaults: 1,024 HH training examples, 64 HH evaluation examples in the runtime configuration, a 256-prompt pool, 512 GSM8K training examples, 128 GSM8K evaluation examples, 20 PPO/GRPO steps, and 30 RLVR steps. The evaluator itself caps learned-reward comparisons at **50 prompts**, despite the larger configured prompt count.

## Recorded outcomes

| Method | Reward-model win rate vs SFT | Mean sampled-token KL |
|---|---:|---:|
| SFT self-comparison | 50% | 0.007988 |
| DPO | 48% | 0.007779 |
| PPO | 50% | 0.009007 |
| GRPO | 56% | 0.006629 |
| RLVR | 46% | 0.005580 |

Source: the `Evaluation metrics` output in original `DVLM_PA2_1.ipynb`, cell 33. [CSV table](../results/alignment-360m.csv).

GRPO had the highest observed win rate in this pilot and lower sampled-token KL than PPO. I would not treat six percentage points on 50 automatically judged prompts as a general advantage: these are short stochastic runs without confidence intervals or repeated seeds. The evaluator counts strictly higher reward scores as wins; ties are not half-wins. The SFT row is a diagnostic self-comparison, not a separate treatment. Its nonzero reported KL needs investigation before treating this table as a calibrated benchmark.

RLVR produced **0/128 GSM8K pass@1**. Logged batches show **87.5–100% reward-degenerate groups**. Without within-group reward variation, a relative-advantage update supplies little task signal. This failure is part of the outcome.

The earlier **1.7B** notebook separately records **71.37% reward-model test preference accuracy** and an SFT validation-perplexity trace from **7.14 to 5.62**. That SFT evaluation samples a small number of test batches. These figures are not the 360M comparison and should not be combined into one run.

## Run

Use Python 3.12 in a GPU environment with an appropriate PyTorch build. The requirements are broad version ranges, not a frozen reproduction lockfile.

```bash
git clone https://github.com/MuhammadBinTariq/vision-language-and-alignment.git
cd vision-language-and-alignment/alignment
python -m venv .venv
# Activate .venv for your shell.
python -m pip install -r requirements.txt
python -m pip install jupyter
jupyter notebook notebooks/run_alignment.ipynb
```

Locally, set `REPO_MODE = "current"` in the first configuration cell. In Colab, leave `REPO_MODE = "clone"`; it clones this repository and enters `alignment/`. Provide `HF_TOKEN` through the environment or Colab secrets. The Llama reward/value models require Hugging Face access. Never store credentials in notebook source.

The runner is preferred because it carries SFT and reward-model checkpoint paths into later stages. Review `RUN_*` switches and `OVERRIDES` before execution; ablation and evaluation are enabled by default. Individual stages can also run from this directory:

```bash
python train_sft.py --config configs/default.yaml --config configs/sft.yaml
python train_rm.py --config configs/default.yaml --config configs/rm.yaml
# Set models.sft_checkpoint and models.rm_checkpoint to generated paths
# in your YAML configuration before policy optimisation.
python train_rl.py --method dpo --config configs/default.yaml --config configs/dpo.yaml
python train_rl.py --method ppo --config configs/default.yaml --config configs/ppo.yaml
python train_rl.py --method grpo --config configs/default.yaml --config configs/grpo.yaml
python train_rl.py --method rlvr --config configs/default.yaml --config configs/rlvr.yaml
```

Generated checkpoints, metrics, and samples go under `runs/`, ignored by Git. Model weights, datasets, and checkpoints are not bundled. The historical 1.7B notebook contains exploratory/state-revival cells and is not the default clean runner.

## Reading the implementation

- [`data/parsing.py`](data/parsing.py): prompt/response boundaries.
- [`model/logprobs.py`](model/logprobs.py): response-token log probabilities.
- [`alignment/dpo.py`](alignment/dpo.py), [`alignment/ppo.py`](alignment/ppo.py), [`alignment/grpo.py`](alignment/grpo.py): objectives and updates.
- [`train_rl.py`](train_rl.py): rollout/training orchestration.
- [`eval.py`](eval.py): automatic comparisons and evaluation caps.

## Evidence and limits

Results are transcribed from supplied notebook outputs, not a new GPU run. Outputs have been cleared for publication; the code and historical commentary are retained. Source cell references are in [results/README.md](../results/README.md). Proposed explanations in notebook prose are not independently established causal findings.
