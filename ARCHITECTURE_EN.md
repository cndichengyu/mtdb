> English translation of [ARCHITECTURE.md](ARCHITECTURE.md).

# MTDB Architecture and Implementation Notes

**English** | [中文](ARCHITECTURE.md)

> MTDB = **M**ulti-**T**imescale **D**ynamical **B**ase (multi-timescale dynamical base).
> This document explains how the conceptual architecture maps onto the code in this repository:
> the formulas, implementation locations, interfaces, training protocol, complexity, and
> limitations of each module. It is aimed at engineers who want to continue training or
> modify the architecture on this foundation.

---

## 0. Positioning and Boundaries

MTDB is the bottom layer of a hybrid AGI system: the **perception–intuition–memory substrate**. It:

- carries temporal information with sparse, multi-timescale, phase-synchronized dynamical states;
- carries stable patterns with working memory / attractor memory;
- uses prediction error (surprise) to drive memory writing, attention, and local plasticity;
- uses sleep replay to perform consolidation, pruning, and interface recalibration;
- outputs compressed representations `z(t)` plus surprise/confidence/memory index to upper layers, rather than tokens.

It is **not**:

- a model that can achieve AGI on its own;
- a substitute for language/logic/planning capabilities (these still belong to upper-layer world models, neuro-symbolic systems, and agent frameworks);
- a "faster version" of the Transformer — it addresses a different class of problems: continual learning, long-horizon dynamics,
  low-power sparse computation, embodied grounding, stable memory.

This repository is a **runnable research prototype**: all training can be completed on a laptop
CPU, and the experimental scale is given by the three presets in `configs/`. Its value lies in turning concepts into testable
code, metrics, and ablations, not in claiming that AGI has already been reached.

---

## 1. Design Principles

1. **Do not build intelligence by stacking parameter counts**: the core units are sparsely connected (fixed fan-in), the per-step cost is O(N·K),
   aimed at mapping onto 1–3 W class neuromorphic/digital hybrid hardware.
2. **Computation is carried in the temporal domain**: multi-timescale states, phase synchronization, and attractor dynamics are the
   carriers of memory and "intuition", rather than only doing feed-forward token prediction.
3. **Local learning first**: Hebb/STDP-style eligibility + prediction error + reward, three-factor modulation; the readout layer
   uses least squares/RLS, with no reliance on backpropagation anywhere.
4. **The bottom layer does not do language or logic**: it only outputs latent states, phase binding, surprise, confidence, and memory indices.
5. **Interface first**: the bottom-layer latent state must be consumable by upper-layer world models/symbolic systems; the interface
   protocol is explicitly defined in `mtdb/core/interface.py` and is slice-addressable.

---

## 2. Overall Data Flow

```text
                  s(t)  perceptual events / continuous features [F]
                  │
                  ▼
        ┌─────────────────────────────────────────────────────────────┐
        │ PhaseSyncRouter                                             │
        │ θ_i(t+1)=θ_i+2πΔt(f_i+K_c·sync term+input term+noise)       │
        └───────────────┬─────────────────────────────────────────────┘
                        │ θ(t), phase-cluster features Φ(t)
                        ▼
        ┌─────────────────────────────────────────────────────────────┐
        │ MultiTimescaleDynamics                                      │
        │ x(t+1)=α⊙x+(1-α)⊙tanh(Wφ(x)+Us+b+η)                         │
        │ α_i = exp(-Δt/τ_i), τ∈{10,100,1000}ms                       │
        └───────────────┬─────────────────────────────────────────────┘
                        │ x(t)
             ┌──────────┴────────────────────────┐
             ▼                                   ▼
  h(t)=P x(t) (compressed latent state)          self-prediction error
             │                                   │ e(t)=||h(t)-ĥ(t)|| normalized → surprise
             ▼                                   ▼
        ┌─────────────────────────────────────────────────────────────┐
        │ Memory: WorkingMemory / AttractorMemory                     │
        │ Read: content addressing + top-k prototype retrieval        │
        │ Write: triggered by surprise or novelty (cooldown)          │
        └───────────────┬─────────────────────────────────────────────┘
                        │
                        ▼
    z(t) = [h | Φ | M_wm | M_ltm | u | Δu | (surprise, sync, confidence)]
                        │
                        ▼
        Upper layers: world model / neuro-symbolic / agent / readout (ridge/RLS)
```

---

## 3. Modules and Code Mapping

| Conceptual layer (user's design) | Code location | Key classes/functions | Notes |
|---|---|---|---|
| A. Multi-timescale dynamics core | `mtdb/core/dynamics.py` | `MultiTimescaleDynamics` | sparse fan-in, three τ groups, spectral-radius initialization, homeostatic bias |
| B. Phase synchronization and routing | `mtdb/core/phase.py` | `PhaseSyncRouter` | Kuramoto phase, synchronized cluster features, cluster membership |
| C. Working memory / attractor memory | `mtdb/core/memory.py` | `WorkingMemory` `AttractorMemory` `SurpriseGate` | content-addressed slots, prototype attractors, surprise gating |
| D. Local plasticity and modulation | `mtdb/core/plasticity.py` + `dynamics.plastic_update` | `ThreeFactorPlasticity` `ErrorFeedback` | eligibility/STDP, error broadcast, Oja stabilization term, reward modulation |
| E. Sleep replay and consolidation | `mtdb/core/consolidation.py` | `ReplayBuffer` `SleepConsolidator` | replay by regime quota, prototype abstraction, pruning + renormalization, gated readout refit |
| Interface protocol | `mtdb/core/interface.py` + `system.py` | `InterfaceSpec` `InterfacePacket` `pack_interface` `MTDBSystem` | z(t) layout, field slices, top-down modulation |
| Backprop-free readout | `mtdb/training/readout.py` | `RidgeReadout` `RLSReadout` | closed-form least squares / Bayesian-initialized RLS |
| Three training stages | `mtdb/training/trainer.py` | `MTDBTrainer` | stage1/stage2/sleep/recall/ablation/stress |
| Synthetic tasks and metrics | `mtdb/tasks/synthetic.py`, `mtdb/eval/metrics.py` | `MultimodalWorld` `DelayedRecallWorld` | prediction, anomaly, delayed recall, forgetting stress test |

---

## 4. Mathematics and Implementation Details of Each Module

### 4.1 Multi-timescale dynamics core (A)

The continuous-time form given by the user:

```
τ_i dx_i/dt = -x_i + Σ_j W_ij φ(x_j) + Σ_k U_ik s_k(t) + η_i(t)
```

The code uses exponential Euler discretization (`dt_ms=10`):

```
α_i = exp(-Δt/τ_i)
x_i(t+1) = α_i x_i(t) + (1-α_i) · tanh( drive_i(t) )
drive_i  = Σ_{j∈pre(i)} W_ij φ(x_j(t)) + Σ_k U_ik s_k(t) + b_i(t) + η_i(t)
φ = tanh (default; the `activation` field is reserved for extending to spiking/saturating functions)
```

- **Sparse connectivity**: every unit has a fixed fan-in `in_degree` (e.g. 16/768≈2.1%),
  `W` is implemented as a `[N,K]` gather, per-step O(N·K), avoiding an N² dense matrix.
- **Spectral radius initialization**: `_rescale_to_spectral_radius` estimates the spectral radius of the sparse operator by power iteration
  and rescales it to `target_spectral_radius` (default 1.0), keeping the dynamics at the edge of stability.
- **Three time-constant groups**: `τ = (10, 100, 1000) ms`, ratio `(0.5, 0.3, 0.2)`.
  The fast group responds to instantaneous input, the medium/slow groups integrate context; this is the main workhorse of the delayed recall experiments.
- **Homeostatic balance and stability**: the homeostatic bias is updated as
  `b_i ← b_i - η_homeo(EMA(|x_i|) - target_rate)`, preventing long-term drift/saturation;
  `row_norm_cap` bounds the row norm after plasticity; the state range is kept bounded by tanh and decay.
- **Input**: `W_in` is a random fixed mapping (can be changed to plastic/spiking input in later experiments).
  The readout layer only sees a sparse random projection of `x`, `h(t)=P x(t)` (`projection.fan_in` fixed),
  to control the interface width.

Complexity: per step `O(N·K + H·K_h)`, where `H=h_dim` and `K_h` is the projection fan-in;
with 100k units and K=32, a step is about 3.2M multiply-adds, suitable for CPU/neuromorphic event-driven implementations.

### 4.2 Phase synchronization and routing (B)

One phase variable per unit (Kuramoto type):

```
dθ_i/dt = 2π f_i + K_c · mean_{j∈pre(i)} sin(θ_j - θ_i) + K_in (U_phase s)_i + noise
```

- the frequency `f_i` is given a base value per timescale group (8/6/4 Hz) + jitter;
- the input term makes units driven by the same sensory event undergo a phase reset, tending toward synchronization;
- synchronized clusters are obtained by phase binning: for cluster m,

```
r_m = | mean_{i∈m} exp(i θ_i) |          (coherence)
mx_m = mean_{i∈m} x_i
f_m = |m| / N
Φ(t) = [r_0..r_{C-1}, mx_0..mx_{C-1}, f_0..f_{C-1}]  (3C dimensions)
```

- the synchrony `sync = Σ_m r_m f_m` serves as an interface scalar and can be used by upper layers for event binding/routing decisions;
- `cluster_membership` is returned as an additional field of `InterfacePacket`, convenient for binding analysis.

**Current limitation**: the phase layer does not yet feed back into the dynamics gain (it only affects the interface and routing statistics); this is the key
extension point for the "phase-routing selection module" to be validated next.

### 4.3 Working memory and attractor memory (C)

**Working memory**: `S=32` (default) normalized content slots, content dimension = `h_dim`.

- read: `sim = slots·normalize(h)`, softmax-weighted sum → `M_wm`; returns best-hit and entropy;
- write: triggered only on high surprise or high novelty, with a cooldown `write_cooldown`;
- eviction: empty slots first; when full, replace the slot with the smallest usage (usage decays with reads/writes).

**Attractor long-term memory**: normalized prototypes `ξ^μ` with capacity `P=512` (default).

- read: `top-k` cosine similarity + softmax weighting (a modern-Hopfield-style retrieval approximation);
- write: novelty gate (write only if the maximum cosine with stored prototypes < `novelty_cos`), ring-buffer overwrite;
- retrieval evaluation: noisy prototype → top-k composition → cosine with the clean prototype, against the chance level of random prototypes.

**Surprise gate (SurpriseGate)**: normalizes the one-step self-prediction error by an exponential moving mean/variance,
`z_t = (e_t - μ)/σ`; it is treated as a surprise event only when `e_t > μ + kσ` and `z_t > floor`.
The write gate = `surprise OR novelty`, sharing a single write cooldown, to avoid memory being flooded by high-frequency noise.

Note: **memory writing is an inference behavior** (it can write even when `learn=False`), whereas "learning" is the update
at the weight/readout level. The evaluation protocol explains separately in §6 when writing is turned on/off.

### 4.4 Three-factor local plasticity (D)

For a sparse synapse `(i,j)`:

```
E_ij(t) = λ_E E_ij(t-1) + post_i(t) pre_j(t)              (eligibility/STDP trace)
ΔW_ij = lr_hebb · (post_i pre_j − post_i² W_ij)           (Oja-stabilized Hebb term)
       + lr_error · mod(t) · e_i(t) pre_j(t)               (prediction-error term)
       + lr_reward · r(t) · E_ij(t)                        (reward/task modulation)
W_ij  ← clip_per_step( W_ij + ΔW_ij ) − wd·W_ij
```

- `post=tanh(x_i)`, `pre=tanh(x_j)` come from the current step;
- error broadcast `e = B · error`, where `B` is a fixed random feedback matrix (feedback alignment style),
  requiring neither symmetric transpose weights nor BPTT;
- `mod(t) = 1 + gain·max(0, surprise)`, the write strength being modulated by the surprise signal;
- after each step a per-row norm cap is applied, preventing plasticity from causing the dynamics to diverge;
- the training log records `delta_abs_mean / hebb_abs_mean / error_abs_mean / ...`.

Complexity: O(N·K) per step, the same order as the dynamics; there is no need to store the activations of the whole temporal graph.

### 4.5 Sleep replay and consolidation (E)

The order of `SleepConsolidator.run(buffer, readout)` (important):

1. **Structural consolidation**: prune the weakest `prune_fraction` of synapses per row; renormalize each row to its original mean
   row norm (`normalize_rows`), and record the number of pruned synapses and the weight drift;
2. **Replay**: sample full sequence windows from `ReplayBuffer` according to the regime quotas, reset the dynamical state and
   run them again (without updating plasticity); write the top-surprise `h` of each window as a prototype into attractor
   memory and working memory;
3. **Readout consolidation**: refit by least squares on the `(z_t, s_{t+horizon})` samples over the replay windows,
   but **accept only when the replay error decreases** (`replay-mse gating`), preventing a single sleep from breaking the interface;
4. **Retrieval evaluation**: noisy retrieval against the attractor prototypes, reporting cosine vs chance cosine.

The replay buffer allocates quotas per regime (at most `buffer_size/2` per regime); otherwise the most recent online
task would evict all windows of the older task, and sleep would have no "old experience" to consolidate.

### 4.6 Backprop-free readout (interface training)

- **RidgeReadout**: standardizes features first, closed-form solution
  `W = (XᵀX + λnI)⁻¹ Xᵀ(Y-ȳ)`; this is the "offline sleep-style" interface fit, also used in stage-1.
- **RLSReadout**: recursive least squares, with `P` (inverse correlation matrix) and `θ` (weights) updated rank-one at each step,
  supporting a forgetting factor; used online in stage-2.
- **Bayesian initialization**: at the start of stage-2, the `(XᵀX+λnI)⁻¹` of the stage-1 ridge is used as the
  initial `P` of RLS, and the ridge weights as the initial `θ`. In this way online learning starts from a
  posterior that has "already digested the old data", significantly reducing severe early drift (code: `RidgeReadout.fit` saves `precision`,
  `RLSReadout.set_weights(..., precision=...)`).
- **Why not backprop**: the core assumption of the bottom layer is that "learning is local and can happen continuously at inference time".
  The readout layer is the only part that needs a supervised channel, and least squares/RLS suffices; no BPTT over the dynamics is needed.

---

## 5. Interface Protocol z(t)

The layout defined by `InterfaceSpec` (default configuration: `h=96`, phase `3C=24`, WM 32, LTM 32,
sensory channels 20, auxiliary 3 → 207 dimensions):

| Field | Dimensions | Meaning | Upper-layer use |
|---|---|---|---|
| `h` | 96 | slow/medium/fast mixed latent-state compression | world-model context input |
| `phase` | 3C | per-cluster coherence, mean state, fraction | event binding, routing, object-level representation |
| `wm` | 32 | working memory readout (compressed) | current situation/task context |
| `attractor` | 32 | long-term attractor prototype retrieval result (compressed) | long-term experience retrieval |
| `input` | F | current observation s(t) | world-model synchronization/conditioning |
| `delta` | F | s(t)-s(t-1) | instantaneous change/velocity information |
| `aux` | 3 | (surprise, sync, confidence) | trigger attention/memory writing/exploration/invoke symbolic reasoning |

- Top-down modulation (`UpperModulation`): the structures `reward`, `attention`, `query`, `target_error`
  are already defined; the current training script uses `target_error` to drive three-factor plasticity, with `reward` being the negative prediction
  error; `attention` is wired into `ThreeFactorPlasticity.step` as an optional per-unit mask.
- The memory index `memory_key` and `cluster_membership` are kept in `InterfacePacket` and are not forced
  into `z`, for upper layers that need symbolic-level tracking.
- The `include_sensory_channel` configuration can disable the two `input/delta` channels for ablation.

---

## 6. Training and Evaluation Protocol

### 6.1 Stage 1: unsupervised predictive-coding pretraining

- Data: the state stream `s(t)` of `MultimodalWorld`, target `y(t)=s(t+h)` (default h=5).
- Each epoch: run the whole training stream with `learn=True` (homeostasis, self-prediction RLS, and memory writing all enabled),
  collect interface features `z` and targets `y`, and refit in closed form with `RidgeReadout.fit`.
- Record the eval MSE of each epoch; this is the stage that "only learns prediction, not task labels".

### 6.2 Stage 2: online learning during the task

- Initialize `RLSReadout` from the stage-1 `RidgeReadout` weights and precision;
- On regime B, tick by tick: `z → ŷ`, the error is used for
  (a) the RLS readout update; (b) the error broadcast and reward modulation of `ThreeFactorPlasticity`;
- Record online MSE and the per-epoch A/B evaluation;
- Default `forget=0.9999` (conservative), plus a stress test with `forget=0.999` (aggressive forgetting).

### 6.3 Stage 3: sleep replay

See §4.5; default `replay_passes=2`, `prune_fraction=0.01`, full sequence windows.

### 6.4 Evaluation protocol (important)

- Prediction/anomaly evaluation: `learn=False`, **the self-predictor and the surprise statistics are frozen**; memory writing is **enabled**
  as an inference behavior (`allow_memory_write=True`), and the memory state is snapshotted/restored before and after evaluation, ensuring that evaluation does not
  contaminate the main model.
- Anomaly detection: `surprise` is the normalized one-step self-prediction error; a 3-step moving-average version and
  the "input change" baseline are also reported; rank-sum ROC-AUC is used.
- Delayed recall: an independent system performs two rounds of unsupervised adaptation on the recall stream, then fits a ridge readout; at evaluation time
  the same memory snapshot is restored for each sequence, the two conditions "writing on/frozen" are measured separately, and the top-1
  classification accuracy over the last 10 steps plus the per-timestep curves are reported.
- Forgetting stress test: restore the stage-1 base, learn online quickly on regime B with `forget=0.999`,
  quantify the forgetting of A, then run sleep replay and quantify the recovery of A/B.

---

## 7. Complexity, Parameters, and Scale

| Configuration | N | K | h_dim | Interface dimensions | Parameters (approx.) | Purpose |
|---|---|---|---|---|---|---|
| `debug` | 768 | 16 | 96 | 207 | 95k | quick laptop validation |
| `standard` | 2048 | 24 | 128 | 271 | ~300k | a single full experiment |
| `mvp` | 16384 | 32 | 256 | 435 | ~2M | larger-scale prototype (not run in CI) |

- Per-step core cost: `O(N·K + N·H_out)`; phase `O(N·K)`; readout RLS `O(D²)` (D = interface dimension,
  around 200 by default, affordable on CPU).
- The current implementation is "discrete tick + sparse gather", not yet true event-driven asynchronous computation;
  but structurally all operations depend only on local connectivity, so it can be mapped onto event-driven neuromorphic chips/analog slow dynamics.

---

## 8. Experimental Results

See [RESULTS.md](RESULTS.md). Summary (based on `configs/standard.yaml`, `demo_standard`):

- next-observation prediction MSE is better than persistence and linear-AR(k=4);
- anomaly detection AUC is higher than the input-change baseline;
- delayed recall (delay=40–48 steps) is significantly above chance, and the decay curve of memory with delay can be measured;
- online adaptation (conservative forgetting) improves both A and B; in the stress test aggressive forgetting causes clear forgetting of A,
  and after sleep replay both A and B recover significantly.

(Concrete numbers are governed by `results/demo_standard/results.json`, synchronized in README/RESULTS.)

---

## 9. Limitations and Risks (Honest List)

1. **Not AGI, nor a Transformer replacement**: this repository only validates a small part of the behavior of the bottom-layer base;
   it has no language, planning, or symbolic reasoning capability.
2. **Mostly synthetic tasks**: `MultimodalWorld` is a 10-dimensional toy world, and the conclusions cannot be directly extrapolated to real
   vision/audio/robotics data.
3. **Limited credit assignment for local learning**: the three-factor rule relies on fixed random feedback and scalar modulation; it can adapt online,
   but cannot carry the gradients of complex tasks; an upper layer is still needed for world-model/decision learning.
4. **Component ablations are not significant**: on short-horizon synthetic prediction, the difference made by removing phase/memory/slow time constants is small
   (see the ablation table in RESULTS); the value of the components needs to be validated with longer-horizon, more complex binding/recall tasks.
5. **Working memory shows no advantage in delayed recall**: the current recall task is mainly carried by slow dynamics,
   and freezing writing is even slightly better; the role of WM/attractors needs to be validated in multi-object binding, interference, and context-switching tasks.
6. **Sleep readout refitting is gated**: it is accepted only when the replay error decreases, avoiding "sleep damaging the model"; this also
   shows that there is still a gap between the replay distribution and the online distribution, which is a point for future improvement.
7. **RLS numerical stability**: float64, precision initialization, and the choice of forgetting factor all need tuning; currently a
   200-dimensional interface is acceptable, but larger scales will need blocking/regularization.
8. **No power/latency hardware measurements**: the 1–3 W mentioned in the text is a design target, not a measured result of this repository.
9. **The phase layer is not closed-loop**: phase currently mainly affects the interface features, without inversely modulating dynamics/routing gain.

---

## 10. Extension Roadmap

- **Closer to event-driven**: replace dense ticks with sparse spike thresholds and an asynchronous event queue;
  introduce `torch.sparse`/hand-written kernels or a neuromorphic backend.
- **Continuous time/multi-compartment**: replace the discrete dynamics with ODEs/spiking neurons, keeping multiple τ;
  add dendritic compartments and local voltage-dependent plasticity.
- **Closed-loop phase routing**: let synchronized clusters actually select "which modules/memory slots participate in the next step of computation",
  and add phase-coded order/interval readout.
- **Hierarchical MTDB**: the bottom base outputs to an upper-layer world model (differentiable or non-differentiable),
  with the interface `z(t)` connecting to Transformer/SSM/symbolic reasoners.
- **Real multimodal data**: event cameras (DVS), audio, touch/proprioception; start with self-supervised prediction.
- **Tasks and scale**: multi-object binding, occlusion, continual navigation, online adaptation/catastrophic forgetting benchmarks;
  scale to 10⁴–10⁵ units and perform power/latency evaluation.
- **Hardware mapping**: sparse fan-in + local three-factor rules are naturally suited to in-memory computing/neuromorphics;
  but quantization, noise robustness, and on-chip learning need to be validated.

---

## Appendix A: Reproduction Commands

```bash
# Environment
uv venv .venv --python 3.14
uv pip install --python .venv/bin/python -r requirements.txt

# Quick experiment (about 45s, CPU laptop)
.venv/bin/python scripts/train.py --config configs/debug.yaml

# Standard experiment (2048 units, a few minutes)
.venv/bin/python scripts/train.py --config configs/standard.yaml

# Unit tests
.venv/bin/python -m pytest tests -q

# Re-evaluate from a checkpoint
.venv/bin/python scripts/eval.py --checkpoint artifacts/demo_standard/checkpoint.pt
```

## Appendix B: Key Hyperparameters

| Parameter | Default | Role |
|---|---|---|
| `dynamics.tau_groups_ms` | (10,100,1000) | fast/medium/slow timescales |
| `dynamics.in_degree` | 16/24/32 | sparse fan-in |
| `dynamics.target_spectral_radius` | 1.0 | edge of dynamical stability |
| `phase.coupling_gain` | 0.4 | phase synchronization strength |
| `memory.surprise_k` | 2.0 | surprise threshold (σ) |
| `memory.wm_novelty_threshold` | 0.55 | WM content novelty threshold |
| `memory.attr_novelty_threshold` | 0.65 | LTM prototype novelty threshold |
| `plasticity.lr_error` | 4e-4 | learning rate of the three-factor error term |
| `readout.rls_forget` | 0.9999 | effective memory of the online readout |
| `consolidation.prune_fraction` | 0.01 | pruning fraction per row |
| `train.horizon` | 5 | prediction target step size |
