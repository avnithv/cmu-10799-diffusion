# HW1 Experiment Journal — DDPM on CelebA-64

Lab notebook, newest first. Target (handout Q4c): KID < 0.005, 1k samples.
All KID = torch-fidelity mean vs `data/celeba/train/images` (63,715 imgs),
1000 generated samples, EMA weights, σ²=β sampling unless noted.

**Status: best model KID 0.0411** (`logs/ddpm_20260927_222014/checkpoints/ddpm_final.pt`,
200k-step lr-decay resume). Border artifact found 10-02 and quantified: **~86%
of KID is the 4px border zone** (crop test 0.0451 → 0.0062); replicate-padding
fine-tune is the live fix experiment.

## 2026-10-02 — black-border artifact (user-spotted, verified)

- Observation: worst witness tiles carry black frames; worse score ⇒ more border.
- Numbers (outer 3px frame vs interior, per tile):
  - generated worst-25: frame/interior **0.52**, black frames **9–11/25**
  - generated best-25: 0.78–0.92, 3–4/25
  - real (n=1500): frame/interior **0.99**, black-frame rate **6.3%** — and those
    are dark-background *photos*, not letterboxes. **Model invents frames.**
- Mechanism: zero-padded convs mark border pixels → learned border mean is dark
  (CelebA edges = background) → 1000 sampling steps compound the nudge into
  hard frames at ≫ real rate. Inception features highly sensitive to frames.
- **Crop test result (baseline set): uncropped 0.0451 → cropped 0.00617.**
  The 4px border zone carries **~86% of KID**; interior quality is essentially
  at the 0.005 target. Explains all five null ablations (none touch padding).
  Caveat: cropped KID is diagnostic only — the reportable number stays full-image.
- Fix: `padding_mode` option (default `zeros` = bit-identical old behavior)
  threaded through all six padded convs. Verified: same param count /
  state_dict keys; zeros checkpoints load into replicate models.
  - `configs/ddpm_replicate.yaml` — scratch retrain, verified identical to the
    Sep-17 baseline recipe except `padding_mode: replicate` (one-variable
    experiment, ~14h). Fine-tune variant considered and dropped — scratch is
    the cleaner comparison.

## 2026-10-01/02 — witness analysis (per-sample KID attribution)

- Tool: w(x) = mean k(x, real) − mean k(x, gen), same inception-v3-compat
  features + cubic kernel as KID. Worst/best 5×5 grids in `results/<run>/`.
- lr1e3 control: worst-25 = pure noise ✓ (tool flags true catastrophe).
- baseline + resume1e5: worst-25 = **ordinary decent faces**. No degenerate tail.
- best-25 = frontal/smooth/evenly-lit; worst-25 = off-pose/harsh-light/dark-bg.
- Diagnosis: floor = **global distribution shift**, not a bad-sample tail —
  refined on 10-02 to the border artifact above as a concrete component.

## 2026-09-29 — final experiment verdicts; freeze

| experiment | KID | verdict |
|---|---|---|
| baseline (79M, b64, lr 1e-4, 100k) | 0.0451 / 0.0472 | reference band |
| lr ×10 (1e-3, scratch) | 0.64 | model destroyed |
| posterior-variance (β̃) sampling | 0.0528 | worse; keep σ²=β |
| capacity 2.25× (178M, b64) | 0.0433 | null |
| resume +100k @ lr/10 | **0.0411** | small real win (~2–3σ); best model |

- lr-decay improved KID with train loss unchanged (0.0150 → 0.0150): loss and
  sample quality decouple.
- Decision: freeze experiments; re-run Q7 on best model; write.

## 2026-09-28 — two infra failures, no science lost

- resume1e5: trained 200k/200k OK (10h08, 2.78 it/s; wandb lr = 1e-5 ✓). Job
  FAILED only from a run-dir race in train_eval.sh (grabbed concurrent
  sibling's dir). Fixed: start-marker + finished-run filter.
- b64c192: GPU fine (2h05 @ 1.34 it/s) but **host-RAM OOM at first checkpoint**
  (torch.save stages ~2.9GB via CPU + 8 workers under 32G cgroup). Fix:
  workers 4, `--mem=64G` → completed 09-29.

## 2026-09-27 — first verdicts + capacity-run OOM saga (attempts 1–2)

- lr1e3 → KID 0.64. posterior-variance → 0.0528. Both hypotheses dead.
- capacity attempt 1 (b128): OOM in first forward at 21.65GB used with
  **22.7GB "free"** (L40 44.4GB) — ~21GB effective mapping ceiling of
  cu130-runtime-on-12.x-driver.
- attempt 2 (b96 + `expandable_segments`): instant OOM at model upload via
  `cuMemAddressReserve` — **the env var itself is incompatible** (driver lacks
  VMM support). NEVER use expandable_segments in this env.

## 2026-09-26 — improvement experiments designed/submitted

- Combined b128+192ch into one run (user: "improve first, attribute later");
  `ddpm_b128.yaml` kept as disambiguator. Plus lr×10 and β̃-sampling (job 0).
- train.py resume fixes (CPU-validated): config lr re-applied after optimizer
  restore (checkpoint lr silently clobbered it); `checkpoint.resume` config
  field actually honored.

## 2026-09-23 — eval pipeline validated (believe the numbers)

| check | result |
|---|---|
| real-vs-real control (31.8k vs 31.8k) | KID **−1.4e-5 ± 8.8e-5** ≈ 0 |
| EMA sanity | step 100000, 0/336 identical, 7.7% rel. diff — healthy |
| 1000-step rerun | 0.0472 (vs 0.0451) → run-to-run σ ≈ 0.002 |
| 300-step rerun | 0.0427 (first run's 0.0761 was sampling noise) |

Conclusion: the ~0.045 floor is the model's, not the pipeline's.

## 2026-09-22 — Q7 step-count ablation (baseline model)

- 1000: 0.0451 · 900: 0.0441 · 700: 0.0464 · 500: 0.0413 · 300: 0.0761(→0.0427
  rerun) · 100: 0.0405. Flat within noise.
- Qualitative: visible speckle at 100 steps that KID barely punishes.
- Sampler = strided *ancestral DDPM* (evenly spaced subsequence, effective
  coefficients; NOT DDIM), oracle-verified exact.
- Note: step effects of a few ×10⁻³ are invisible under a 0.045 floor.

## 2026-09-17/18 — baseline trained

- 79.30M U-Net (base 128, mult [1,2,2,4], blocks 2, attn @16/8), b64, lr 1e-4,
  100k iters, β 1e-4→0.02 linear, T=1000. ~14h, one L40.
- Loss: clean plateau ≈ 0.015 (untrained ≈ 1.0). Samples: good, diverse faces.
- Pre-run bug: schedule buffers stranded on CPU — `BaseMethod.to()` overrides
  `nn.Module.to()` and moves only the model, and train.py never calls
  `method.to()`. Fixed in DDPM (`nn.Module.to` bypass + self-place in init).
  Lesson: CPU dry runs can't catch device-placement bugs.

## Environment lessons (cluster: cu130 torch on 12.x drivers)

- ~21GB effective VRAM ceiling on 44GB L40s; plan memory to that.
- `expandable_segments` crashes at init (no driver VMM) — never use.
- sm_61 (1080Ti) and sm_70 (V100/TitanV) unusable with this build; sm_75+ OK;
  L40 (sm_89) covered via sm_86 cubins.
- Host RAM: budget checkpoint serialization (~16B/param) + workers under the
  slurm `--mem` cgroup.
- pyarrow 14+ removed `PyExtensionType` → upgrade `datasets` (09-11 fix).
