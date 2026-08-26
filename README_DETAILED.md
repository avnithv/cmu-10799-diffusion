# CMU 10-799 Diffusion & Flow Matching — Detailed Codebase Guide

A file-by-file walkthrough of the starter code, a checklist of every gap you must fill in, and a roadmap of what the later homeworks will build on top of this repo. The original instructor README is in [README.md](README.md).

**Links:** [Course site](https://kellyyutonghe.github.io/10799S26/) · [Homework page](https://kellyyutonghe.github.io/10799S26/homework/) · [Schedule](https://kellyyutonghe.github.io/10799S26/schedule/) · [Gradescope](https://www.gradescope.com/courses/1207241)

---

## 1. How the pieces fit together

```
config YAML ──► train.py ──► create_model_from_config()  ──► UNet          (src/models/unet.py)
                   │     ──► DDPM.from_config()           ──► DDPM          (src/methods/ddpm.py)
                   │     ──► create_dataloader_from_config ──► CelebADataset (src/data/celeba.py)
                   │     ──► EMA                                             (src/utils/ema.py)
                   │
                   ├─ every step:      loss, metrics = method.compute_loss(x_0)   ← you write
                   ├─ every N steps:   generate_samples() → method.sample()       ← you write
                   │                   save_samples() → PNG grid in logs/…/samples ← you write
                   └─ every M steps:   checkpoint .pt (model + optimizer + EMA + config)

checkpoint .pt ──► sample.py ──► method.sample() ──► PNGs ──► scripts/evaluate_torch_fidelity.sh ──► FID / KID / IS
```

The design contract: **`train.py` / `sample.py` never know the math.** They only call `method.compute_loss()` and `method.sample()` on an object that subclasses `BaseMethod`. That's what lets later homeworks slot in `FlowMatching`, guided sampling, distillation, etc. without rewriting the trainer.

---

## 2. Every file, what it does, and whether you touch it

Legend: ✅ complete, use as-is · ✏️ has TODOs you must fill · ⚙️ config/infra you may edit

### Root

| File | Status | What it does |
|---|---|---|
| `train.py` | ✏️ | Main training loop. Handles: config loading, seeding, single-GPU / multi-GPU DDP (auto-detected from `torchrun` env vars, gated by `infrastructure.num_gpus`), mixed precision (`torch.amp` autocast + `GradScaler`), gradient clipping, EMA updates, W&B logging, periodic sampling, periodic checkpointing, `--resume`, and `--overfit-single-batch` (debug mode: trains on one batch forever — loss should go to ~0). **Note:** checkpoints are written to `logging.dir/<method>_<timestamp>/checkpoints/`, not `checkpoint.dir` (that key is unused). |
| `sample.py` | ✏️ | Loads a `.pt` checkpoint (which embeds its config), rebuilds the UNet + DDPM + EMA, applies EMA weights, and generates `--num_samples` images either as individual PNGs (`--output_dir`) or a single grid (`--grid`). Flags: `--num_steps`, `--batch_size`, `--seed`, `--no_ema`. |
| `download_dataset.py` | ✅ | Pulls `electronickale/cmu-10799-celeba64-subset` from HuggingFace and writes `images/` + `attributes.csv` per split to `--output_dir`. Needed only if you use `from_hub: false`. |
| `modal_app.py` | ✅ | Modal cloud definitions: container image, a persistent `/data` volume, `train_1gpu`…`train_8gpu` functions (L40S), `sample`, `download_dataset`, `evaluate_torch_fidelity`. CLI via `modal run modal_app.py --action {download,train,sample,evaluate}`. Not needed if you use Colab. |
| `setup.sh` / `setup-uv.sh` | ✅ | Create a venv named `.venv-{cpu,cuda1xx,rocm}` after auto-detecting `nvidia-smi` / `/opt/rocm`, then install the matching `environments/requirements-*.txt`. |
| `pyproject.toml` | ✅ | Package metadata; makes `src/` importable; ruff/black config. |
| `.gitignore` | ✅ | Ignores venvs, `data/`, `logs/`, `checkpoints/`, `*.pt`, `wandb/`, `.claude/`. |

### `src/models/` — the neural network

| File | Status | What it does |
|---|---|---|
| `blocks.py` | ✅ | All U-Net building blocks, fully implemented: `SinusoidalPositionalEmbedding` (t → sin/cos features), `TimestepEmbedding` (sinusoidal → MLP → `time_embed_dim`), `GroupNorm32` (fp32-safe GroupNorm), `ResBlock` (two convs + time-conditioning, optional FiLM "scale-shift" norm, handles channel changes), `AttentionBlock` (multi-head self-attention over spatial positions), `Downsample` (stride-2 conv), `Upsample` (nearest ×2 + conv). |
| `unet.py` | ✏️ | `UNet` class: constructor stores hyperparameters but **builds no layers**; `forward()` raises `NotImplementedError`. `create_model_from_config()` maps `config['model']` → constructor kwargs. Has a `__main__` smoke test (`python -m src.models.unet`). |

### `src/methods/` — the algorithm

| File | Status | What it does |
|---|---|---|
| `base.py` | ✅ | `BaseMethod(nn.Module, ABC)`: holds `self.model` and `self.device`; abstract `compute_loss(x_0) → (loss, metrics_dict)` and `sample(batch_size, image_shape) → tensor`; provides `train_mode()/eval_mode()/to()/parameters()/state_dict()`. Every method (DDPM now, flow matching later) subclasses this. |
| `ddpm.py` | ✏️ | `DDPM(BaseMethod)`: constructor takes `num_timesteps, beta_start, beta_end` and nothing else; `forward_process`, `compute_loss`, `reverse_process`, `sample` all raise `NotImplementedError`. `from_config()` reads `config['ddpm']`. Comments suggest `register_buffer` for the β/α schedules and a broadcast helper for `(B,) → (B,1,1,1)`. |

### `src/data/` — the dataset

| File | Status | What it does |
|---|---|---|
| `celeba.py` | ✏️ | `CelebADataset`: loads 64×64 CelebA either from the HF hub (`from_hub=True`; uses an Arrow cache at `root` if `root/dataset_dict.json` exists, otherwise downloads) or from a local `images/` folder. `__getitem__` returns a single image (no labels — unconditional generation). `_build_transforms()` is **empty** so it currently returns raw PIL images. Also `create_dataloader()`, `create_dataloader_from_config()`, and display helpers `normalize` ([0,1]→[-1,1]), `unnormalize`, `make_grid`, `save_image`. |

### `src/utils/`

| File | Status | What it does |
|---|---|---|
| `ema.py` | ✅ | `EMA(model, decay, warmup_steps)`: keeps a shadow copy of weights; `update()` after each optimizer step; `apply_shadow()` / `restore()` swap EMA weights in and out for sampling; `state_dict()` for checkpointing. |
| `logging_utils.py` | ✅ | `setup_logger()` (console + file) and `log_section()` banner printer. |

### `configs/`

| File | Status | Purpose |
|---|---|---|
| `ddpm_modal.yaml` | ✏️ | Paths for Modal (`/data/celeba`, `from_hub: true`). All `model`, `training`, `ddpm`, `sampling` values are blank. |
| `ddpm_babel.yaml` | ✏️ | Same, for CMU's Babel SLURM cluster (`from_hub: false`, local root). |
| `ddpm_colab.yaml` | ✏️ | Added for this fork: `data.root: /content/data/celeba`, logs → Google Drive, 1 GPU, AMP on. Same blanks to fill. |

The config sections and who consumes them:
`data` → `celeba.py` · `model` → `unet.py` · `training` → `train.py` (optimizer, EMA, schedule, logging cadence) · `ddpm` → `ddpm.py` · `sampling` → `sample()` call · `infrastructure` → device/DDP/AMP · `logging` → output dir + W&B.

### `scripts/`

| File | Purpose |
|---|---|
| `train.sh` | SLURM `sbatch` template (Babel: 4×L40S). Wraps `torchrun … train.py`. |
| `evaluate_torch_fidelity.sh` | Runs `sample.py` to dump N PNGs, then runs `fidelity --fid/--kid/--isc` against the real dataset folder. Flags: `--checkpoint --method --dataset-path --metrics --num-samples --num-steps`. |
| `evaluate_modal_torch_fidelity.sh` | Same, but dispatches to Modal. |
| `list_checkpoints.sh` | Lists `.pt` files on the Modal volume. |

### `notebooks/`

| File | Status | Purpose |
|---|---|---|
| `01_1d_playground.ipynb` | ✏️ | CPU-only sandbox: provided 1-D mixture-of-Gaussians datasets + trajectory plotter. TODO cells: a tiny time-conditioned MLP, a DDPM implementation (can import from `src/methods/ddpm.py`), a training loop, and sample/trajectory visualisation. Best place to debug the math before touching images. |
| `02_dataset_exploration.ipynb` | ✏️ | Builds a dataloader; you add visualisation (e.g. a grid of samples). |
| `03_sampling_visualizations.ipynb` | ✏️ | Loads a checkpoint, prints parameter count, then "Your DDPM sampling code". Its imports already reference `FlowMatching` — a preview of HW2. Instructor's reference: their model reached **FID ≈ 8 / KID ≈ 0.0035** after ~5 h on 4×L40S, and similar on 1×L40S. |
| `colab_train.ipynb` | ✅ | Added for this fork: clone → pip → mount Drive → dataset → train / resume / evaluate on Colab. |

### `docs/` and `environments/`

`docs/SETUP.md` (per-platform install), `docs/QUICKSTART-MODAL.md`, `docs/DIRECTORY-STRUCTURE.md`. `environments/requirements.txt` is the shared dependency list; `requirements-{cpu,cuda118,cuda121,cuda126,cuda129,rocm}.txt` just pin the PyTorch wheel index.

---

## 3. Gaps you must fill (HW1)

Every `TODO` / `NotImplementedError` in the repo, in the order they block each other.

### Step 0 — Data (unblocks everything)
- [ ] **`src/data/celeba.py::_build_transforms`** — at minimum `ToTensor()` + `Normalize([0.5]*3, [0.5]*3)` so `__getitem__` returns a `(3,64,64)` tensor in `[-1, 1]` as its docstring promises. Optional `RandomHorizontalFlip` when `self.augment and self.split == "train"`. Images are already 64×64.
- [ ] **`notebooks/02_dataset_exploration.ipynb`** — visualise a grid of dataset samples.

### Step 1 — Model
- [ ] **`src/models/unet.py::UNet.__init__`** — build: input conv → `TimestepEmbedding` → encoder levels (per level: `num_res_blocks` × `ResBlock`, `AttentionBlock` where the current resolution ∈ `attention_resolutions`, then `Downsample`) → middle (`ResBlock`, `AttentionBlock`, `ResBlock`) → decoder mirroring the encoder with skip concatenation and `Upsample` → `GroupNorm32` + SiLU + output conv. Track channels per level using `base_channels * channel_mult[i]`.
- [ ] **`UNet.forward(x, t)`** — embed `t`, run encoder while stacking skips, middle, decoder popping skips, output head. Verify with `python -m src.models.unet` (expects `(4,3,64,64)` in → `(4,3,64,64)` out).

### Step 2 — DDPM algorithm (`src/methods/ddpm.py`)
- [ ] **`__init__`** — build the β schedule (linear from `beta_start` to `beta_end` over `num_timesteps`), α, ᾱ (cumprod), and any derived quantities (√ᾱ, √(1−ᾱ), posterior variance…). Register them with `register_buffer` so they move with `.to(device)` and save with the checkpoint.
- [ ] **`forward_process(x_0, t, noise)`** — q(x_t | x_0) = √ᾱ_t·x_0 + √(1−ᾱ_t)·ε. Choose your own signature (the stub has none).
- [ ] **`compute_loss(x_0)`** — sample `t ~ U{0..T−1}` and `ε ~ N(0,I)`, noise `x_0`, predict ε with `self.model(x_t, t)`, return MSE and a metrics dict (e.g. `{'loss': …, 'mse': …}`; `train.py` expects a `'loss'` key for the progress bar).
- [ ] **`reverse_process(x_t, t)`** — one ancestral step: predict ε, compute the posterior mean, add σ_t·z (no noise at t = 0).
- [ ] **`sample(batch_size, image_shape, num_steps=…)`** — start from `N(0, I)`, loop t = T−1 … 0 calling `reverse_process`, return images in `[-1, 1]` (clamp is optional). `sample.py` passes `num_steps=` so accept it even if you only support the full T for now.
- [ ] **`state_dict` / `from_config`** — add any extra hyperparameters you introduced.

### Step 3 — Training and sampling glue
- [ ] **`train.py::generate_samples`** — replace `samples = None` with `method.sample(num_samples, image_shape, **sampling_kwargs)`; pass `config['sampling']['num_steps']` through.
- [ ] **`train.py::save_samples`** and **`sample.py::save_samples`** — `unnormalize` then `save_image(samples, save_path, nrow=int(sqrt(num_samples)))` (both helpers live in `src.data`). `sample.py` calls this with `num_samples=1` for individual files, so handle a batch of size 1 too.
- [ ] **`sample.py` line ~160** — extra kwargs to `method.sample()` if you added any.

### Step 4 — Config
- [ ] Fill every blank in **`configs/ddpm_colab.yaml`** (and/or `_modal`/`_babel`). A sane starting point for 64×64 CelebA:
  `base_channels: 128`, `channel_mult: [1,2,2,4]`, `num_res_blocks: 2`, `attention_resolutions: [16]`, `num_heads: 4`, `dropout: 0.1`, `use_scale_shift_norm: true`;
  `batch_size: 128`, `learning_rate: 2e-4`, `weight_decay: 0.0`, `betas: [0.9, 0.999]`, `ema_decay: 0.9999`, `ema_start: 5000`, `gradient_clip_norm: 1.0`, `num_iterations: 100000`, `log_every: 100`, `sample_every: 5000`, `save_every: 5000`, `num_samples: 64`;
  `num_timesteps: 1000`, `beta_start: 1e-4`, `beta_end: 0.02`; `sampling.num_steps: 1000`.

### Step 5 — Sanity checks before a long run
1. `python -m src.models.unet` — shapes.
2. `notebooks/01_1d_playground.ipynb` — DDPM on 1-D Gaussians (seconds on CPU; catches sign/indexing bugs in the schedule).
3. `python train.py --method ddpm --config configs/ddpm_colab.yaml --overfit-single-batch` — loss must collapse and samples must reproduce the batch.
4. Full run; evaluate with `scripts/evaluate_torch_fidelity.sh`; inspect in `03_sampling_visualizations.ipynb`.

---

## 4. What later homeworks build on this

The handouts themselves are Overleaf PDFs (not machine-readable), so below is what the homework page, syllabus, and lecture schedule state, plus how it maps onto this codebase. Confirm specifics against each handout when released.

| HW | Due (2026) | Weight | Focus |
|---|---|---|---|
| 1 | Jan 24 | 15 % | Set up codebase, implement DDPM |
| 2 | Feb 5 | 15 % | Implement flow matching; choose a path (Fidelity / Controllability / Speed) |
| 3 | Feb 15 | 20 % | Implement baseline methods for your path |
| 4 | Feb 27 | 20 % | Improve on your baselines ("beat your own baselines and compete with classmates") |
| Poster | Feb 25 submit / Feb 26 present | 15 % | Showcase the full project |

### HW2 — Flow matching
- Add **`src/methods/flow_matching.py`** with a `FlowMatching(BaseMethod)` class — the codebase already anticipates it: `scripts/train.sh` documents `sbatch scripts/train.sh flow_matching`, notebook 03 imports `from src.methods import DDPM, FlowMatching`, and the base class's `compute_loss`/`sample` interface is method-agnostic. Register it in `src/methods/__init__.py` and in the `if args.method == 'ddpm'` dispatch in `train.py` / `sample.py`.
- Same UNet, but `t` becomes continuous in `[0,1]` (the `unet.py` smoke test already feeds `torch.rand` timesteps — design your time embedding to accept floats). Loss = regress a velocity field on the linear interpolant; sampling = ODE integration (Euler / Heun) over `num_steps`.
- New config `configs/flow_matching_*.yaml` with a `flow_matching:` section instead of `ddpm:`.
- Notebook 01 is explicitly titled "Diffusion & **Flow Matching**" — extend it with the FM variant for a 1-D sanity check.
- Lecture 4 (score-based models) and 5 (flow matching) are the theory; lecture 6 (design space / solvers) informs the sampler.

### HW3 — Baselines for your chosen path
Everything here plugs into `sample()` (sampler-side) or `compute_loss()` + the model/config (training-side). Lecture coverage in parentheses.

- **Speed** (lectures 6, 10): fewer-step samplers — DDIM, DPM-Solver / higher-order ODE solvers — implemented as alternative `sample()` strategies selected by `sampling.sampler` (the config already has this key as a "placeholder where you can add more options"). Then distillation / consistency models / flow maps, which need a teacher–student training loop: a second model, a new method class, and changes to `train.py`'s loop. Evaluate FID/KID vs. number of function evaluations via `--num_steps`.
- **Controllability** (lecture 7): conditional generation. The dataset ships `attributes.csv` (CelebA attributes) but `__getitem__` deliberately drops labels — you'll return `(image, label)`, add a class/attribute embedding to the UNet (added to the time embedding), implement **classifier-free guidance** (label dropout in `compute_loss`, guided ε in `reverse_process`), and possibly training-free editing / inpainting via modified sampling.
- **Fidelity** (lecture 9): image-quality improvements — larger/better-tuned UNets, EDM-style preconditioning and noise schedules (lecture 6), latent diffusion (train on autoencoder latents; would add an encoder/decoder around the dataloader and sampler), or a DiT transformer backbone as a drop-in for `UNet` behind `create_model_from_config`.

### HW4 — Beat your baselines
Open-ended improvements on the same path; you're encouraged to use pre-trained models and open-source repos. Metrics stay FID/KID/IS via `evaluate_torch_fidelity.sh`, so keep sampling behind the `sample.py` interface so evaluation keeps working. Lectures 12–13 (discrete / masked diffusion, discrete flow matching) are also available as directions.

### Practical implications for how you write HW1
1. Keep **all** math inside the method class; never special-case DDPM in `train.py`.
2. Make the time embedding accept both integer timesteps and floats in `[0,1]`.
3. Make `sample()` accept `num_steps` and a `sampler` name now, even if only `"ddpm"` works — HW2/3 add options without changing call sites.
4. Keep the UNet constructor purely config-driven so a conditional variant or a different backbone can be swapped in via YAML.
5. Log FID/KID for every run from HW1 onward — HW4 is judged on beating your own numbers.
