# HW1 Experiment Journal — DDPM on CelebA-64

Lab notebook for the KID-improvement investigation. Baseline target from the
handout (Q4c): KID < 0.005 on 1k samples. All KID values are torch-fidelity
`kernel_inception_distance_mean` against `data/celeba/train/images` (63,715
reference images), 1000 generated samples, EMA weights, unless noted.

## Baseline (trained 2026-09-17, run `logs/ddpm_20260917_205522/`)

- 79.30M-param U-Net (base 128, mult [1,2,2,4], res blocks 2, attn @16/8),
  batch 64, lr 1e-4, 100k iterations, linear β 1e-4→0.02, T=1000. ~14h on one L40.
- Loss: clean collapse to a stable ~0.015 plateau (healthy; untrained ≈ 1.0).
- Samples: visually good, diverse faces. `samples_0100000.png`.
- **Baseline KID: 0.0451 and 0.0472** (two independent sampling runs → run-to-run
  σ ≈ 0.002; per-run subset std ≈ 0.0007 understates between-run variance).

### Bug fixed before this run
First launch crashed at iteration 0: DDPM schedule buffers stayed on CPU while
timestep indices were CUDA. Cause: `BaseMethod.to()` overrides `nn.Module.to()`
and moves only `self.model`; additionally, train.py/sample.py never call
`method.to()`. Fix: `DDPM.to()` bypasses to `nn.Module.to` (moves buffers) and
`__init__` self-places at construction. Lesson: CPU dry runs structurally
cannot catch device-placement bugs.

## Evaluation-pipeline validation (2026-09-23, `scripts/kid_debug.sh`)

Motivated by baseline 0.045 vs target 0.005 despite good-looking samples.

| check | result | conclusion |
|---|---|---|
| real-vs-real control (two 31.8k halves of the reference set) | KID = −1.4e-5 ± 8.8e-5 | pipeline unbiased; matched distributions read ≈ 0 |
| EMA sanity (checkpoint shadow vs raw weights) | step 100000, 0/336 identical, mean rel. diff 7.7% | EMA trained and applied correctly |
| 1000-step rerun (fresh seed) | 0.0472 (vs 0.0451) | baseline stable |
| 300-step rerun | 0.0427 (vs first-run 0.0761) | the 0.076 was sampling noise; variance across sample sets grows at low step counts |

**Conclusion: the ~0.045 floor is a property of the model, not the measurement.**

## Q7 step-count ablation (first pass, `results/kid_*.txt`)

1000: 0.0451 / 900: 0.0441 / 700: 0.0464 / 500: 0.0413 / 300: 0.0761 (fluke;
rerun 0.0427) / 100: 0.0405. Flat within noise 100→1000; qualitative residual
speckle clearly visible at 100 steps (KID barely punishes it — Inception
features are texture/semantics-weighted). Sampler is strided *ancestral DDPM*
(evenly spaced timestep subsequence, effective coefficients; NOT DDIM), oracle-
verified exact at multiple strides. Note: a genuine step-count effect of a few
×10⁻³ would be invisible under a 0.045 floor — trend may sharpen if the floor drops.

## Improvement experiments (submitted 2026-09-26, results 2026-09-27)

| experiment | config | result | verdict |
|---|---|---|---|
| lr ×10 (1e-3, from scratch) | `ddpm_lr1e3.yaml` | **KID 0.64** after full 100k | model destroyed; undershoot hypothesis dead |
| posterior-variance sampling (σ²=β̃ instead of β) on baseline ckpt | `sample.py --variance posterior` | **KID 0.0528** (> baseline band 0.043–0.047) | sampling-noise hypothesis dead; slightly worse — keep σ²=β for all reporting |
| capacity+batch: 192ch (178M) × batch 128 | `ddpm_b128c192.yaml` | **OOM at 28s** on L40 | see below; resubmit at batch 96 |

β̃ implementation notes: β̃_t = β_t(1−ᾱ_{t−1})/(1−ᾱ_t); β̃/β = 0.45 at t=1,
0.86 at t=10, →1 at high t (difference concentrated in final detail-forming
steps — which made it the natural suspect for the floor; falsified anyway).
Oracle test passes in both variance modes (mean path identical by construction).

### OOM analysis (exp-b128c192)
Died in the first forward pass at ~21.65 GB used with 22.7 GB reported free on
a 44.4 GB L40 — allocations of every size failing: allocator **fragmentation**
signature (interleaved cuDNN workspace probing + large activation tensors), not
raw capacity. Activation demand was ~3× baseline (2× batch × 1.5× width).
Mitigations adopted: batch 128→96 (−25% activations) AND
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (PyTorch's own suggested fix).

## Decisions log

- **2026-09-26**: combined batch and capacity into one run (user decision:
  "improve first, attribute later"). `ddpm_b128.yaml` kept unused as the
  disambiguator if the combined run wins.
- **2026-09-27**: batch 96 for the capacity run (see OOM). New experiment:
  **resume baseline at lr/10** (`ddpm_resume1e5.yaml`: lr 1e-5, iterations
  →200k, `checkpoint.resume` set) — classic post-plateau lr decay; deliberately
  confounds "longer training" with "smaller steps". Required two train.py
  fixes, both CPU-validated: (1) resume restored the checkpoint's optimizer
  param_groups including its old lr, silently clobbering the config's value —
  now re-applied from config after load; (2) `checkpoint.resume` in the config
  was written by `--resume` but never read — now honored as fallback, so
  `train_eval.sh` needs no arg-forwarding.
- Verification handle: `train/learning_rate` is logged to wandb every 100
  steps — the resumed run's first points must read 1e-5.

## Open items

- In flight next: `exp-b96c192` (capacity+batch) and `exp-resume1e5` (decay).
- **Reality-check the 0.005 target** on Piazza/with classmates — three sensible
  interventions floor at ~0.043 with a validated pipeline; knowing the class
  distribution decides whether to keep spending GPU-days.
- Not yet tested: lr 2–3e-4 (middle ground), attribution split of batch vs
  capacity, Q6 extra credit (alternative parametrization, full retrain).
