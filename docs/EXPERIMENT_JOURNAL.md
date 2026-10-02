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

### OOM saga (exp-b128c192 → b96 → b64): three attempts, a real lesson

**Attempt 1 (batch 128, 2026-09-27):** died 28s in, during the *first forward
pass*, at ~21.65 GB used with **22.7 GB reported free** on a 44.4 GB L40 —
allocations of every size failing. Activation demand was ~3× baseline
(2× batch × 1.5× width). Initial read: caching-allocator fragmentation
(interleaved cuDNN workspace probing + large activation tensors); adopted
PyTorch's own suggested fix, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`,
plus batch 128→96.

**Attempt 2 (batch 96 + expandable_segments, 2026-09-28):** died *instantly* at
`model.to(device)` — uploading just the ~713 MB of weights to an empty GPU —
with the driver reporting `CUDA_ERROR_OUT_OF_MEMORY from cuMemAddressReserve`.
That API is the virtual-address-space reservation expandable_segments is built
on: **the fix itself was the failure.** Our stack is a CUDA 13.0 torch runtime
in forward-compatibility mode on a much older driver, and the driver-side VMM
support expandable_segments needs isn't there. Corollary: attempt 1's
"can't allocate with 22 GB free" at suspiciously-half-the-card now reads less
like fragmentation and more like a **~21 GB effective mapping ceiling of the
same runtime/driver mismatch**.

**Attempt 3 (batch 64, no allocator env var — submitted 2026-09-28):** design
under a ~21 GB working assumption: batch 64 × 192ch ≈ 13–15 GB total (safe);
batch 96 would have been 20–21 GB (borderline). This turns the run into a
**pure capacity test** — 1.5× width (178M) at the baseline batch — which is
more interpretable than the original combined design anyway. Survival through
minute one clears both prior failure points (weight upload, first forward).

**Lesson for Q8/future runs:** the `2.14.0+cu130` wheel on this cluster's
12.x-era drivers works for compute but has degraded memory semantics: ~half the
card's VRAM effectively mappable, and VMM-based allocator features broken.
The durable fix is an env built against the driver (e.g. `./setup-uv.sh
cuda126` with a pinned index) — worth doing before any run that needs >20 GB.

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

## 2026-09-28: two "failures" that were both infrastructure, not science

**exp-resume1e5: training SUCCEEDED.** 200k/200k steps in 10h08 at 2.78 it/s
(run dir `logs/ddpm_20260927_222014/`); wandb confirms `train/learning_rate
= 1e-5` for the whole resumed phase (the override worked) and start step
100000. Training loss after +100k at decayed lr: **0.0150 — identical to the
100k plateau** (decay bought no loss; KID verdict pending — loss and KID have
diverged before in this project). The job's FAILED status was a
**race in train_eval.sh**: its "newest logs/ddpm_2* dir" heuristic grabbed the
*concurrently running* b64 job's dir (created later, no final checkpoint yet)
→ error → exit 1. Fixed with a job-start marker file + finished-run filter.
Eval to run separately via new `scripts/eval_ckpt.sh`.

**exp-b64c192 (attempt 3): GPU cleared, HOST RAM killed it.** The GPU side
finally worked — nonfatal cuDNN workspace warnings under the ~24 GB ceiling,
then 2h05 of clean training at ~1.34 it/s (100k would be ~21 h). Died at step
9999 — the **first checkpoint save** — with `DefaultCPUAllocator: can't
allocate memory (errno 12)`: CPU RAM, not CUDA. torch.save stages every GPU
tensor through host memory (model 0.7 GB + EMA 0.7 GB + Adam 1.4 GB for the
178M model), on top of 8 dataloader workers and wandb, and the job's 32 GB
cgroup ran dry on a final 10 MB. Fixes for attempt 4: `num_workers` 8→4 and
`sbatch --mem=64G`. Cumulative OOM lesson now spans three distinct memory
systems: GPU mapping ceiling (attempt 1), driver VMM (attempt 2), host cgroup
(attempt 3) — same experiment, three different walls.

## 2026-09-29: final experiment verdicts

| experiment | KID | verdict |
|---|---|---|
| baseline (79M, 100k, lr 1e-4) | 0.0451 / 0.0472 | reference band 0.043–0.047 |
| capacity attempt 4 (178M, batch 64, 100k) | **0.0433 ± 0.0007** | NULL — inside the band; capacity is not the binding constraint |
| lr-decay resume (79M, +100k @ 1e-5, 200k total) | **0.0411 ± 0.0007** | small real gain (~2–3× run-to-run σ); train loss unchanged at 0.0150 — loss and KID decouple. **Best model**: `logs/ddpm_20260927_222014/checkpoints/ddpm_final.pt` |

Five hypotheses tested in total (lr ×10, lr decay+longer, posterior variance,
capacity, and implicitly batch via the aborted combined run); everything floors
at ~0.041–0.047 against a pipeline validated to ≈0 bias. Conclusion: within
this architecture/data/compute regime the model sits at KID ≈ 0.04; the
handout's 0.005 was not reached by any single reasonable intervention, and the
gap is unlikely to close without something categorically different (much longer
training, different regime) — or different context on the target itself
(Piazza check still outstanding).

## 2026-10-01/02: witness analysis — the closing diagnosis

Built a per-sample "witness" score (w(x) = mean k(x, real) − mean k(x, gen),
same inception-v3-compat features + cubic kernel as torch-fidelity's KID) and
ranked every generated sample; worst/best 5×5 grids per run in `results/<run>/`.

Findings:
- **lr1e3 (control): worst 25 = pure noise.** Tool validated — it surfaces
  catastrophe when catastrophe exists.
- **baseline & resume1e5: worst 25 = ordinary, respectable faces.** NO
  degenerate tail; the earlier "~2% garbage" fear does not exist at 1000 steps.
- **best 25 = frontal, smooth, evenly-lit "classic CelebA" portraits**; worst
  = off-axis poses, harsh lighting, dark/busy backgrounds.

**Diagnosis: the ~0.041 KID floor is a GLOBAL distribution shift** — every
sample mildly under-represents the real data's diversity tails (pose,
lighting, background variety) — not a tail of broken samples. This explains
all five null/weak ablation results at once: nothing we turned addresses
"uniformly slightly too generic," which is a training-regime property.
Closes the investigation; written up as the Q4c/Q8 conclusion.

## Decision (2026-09-29): freeze and write

- Adopt the 200k lr-decay model as THE reported model (KID 0.0411).
- Re-run the full Q7 step ablation once on this model for an internally
  consistent table (scripts/eval.sh auto-picks the newest final checkpoint,
  which is now this one).
- Then: writeup (Q4c with the validation story, Q7 table + samples, Q5, Q8),
  no further training experiments.
- Optional leftover idea (unpursued): quantify the ~2% dark/speckled sample
  tail on the final model; diagnostic only.
- **Reality-check the 0.005 target** on Piazza/with classmates — three sensible
  interventions floor at ~0.043 with a validated pipeline; knowing the class
  distribution decides whether to keep spending GPU-days.
- Not yet tested: lr 2–3e-4 (middle ground), attribution split of batch vs
  capacity, Q6 extra credit (alternative parametrization, full retrain).
