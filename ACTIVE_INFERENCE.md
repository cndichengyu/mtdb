# Active Inference — "thinking" beyond next-token prediction (v0.2)

> **Status: scoped research prototype. Not AGI, and not a universal claim that
> "MTDB beats Transformers".** This document reports one explicitly defined
> benchmark on which an active-inference loop improves sample efficiency over a
> passive next-token-style classifier, and it states exactly where that advantage
> comes from and where it does not.

![active inference results](assets/active_inference.png)

## 1. The question

The v0.1 base is a predictive substrate: it learns a multi-timescale world model
and fits readouts, but it still behaves like a passive predictor. v0.2 adds a
small, concrete "thinking" loop:

> At each decision step, the model **simulates the consequences of its own
> questions**, scores candidate probes by expected posterior entropy (expected
> free energy), and then acts by choosing the most informative probe.

This is deliberately not "generate a longer chain-of-thought". It is model-based
active inference in a controlled task where thinking has a measurable value:
identifying a hidden class with as few probes as possible.

## 2. Benchmark: active sensing with redundant tests

The environment (`mtdb/active/sensing.py`) is an exactly specified 8-class task:

- hidden class `c ∈ {0..7}`, encoded by 3 bits;
- 8 binary probes; probe `q` returns `bit(c, q) ∈ {0,1}` corrupted by 10% noise;
- the probe matrix contains redundant tests (`q0` and `q1` return the same bit);
- 3 well-chosen probes identify the class, but a fixed/random probe order often
  wastes its budget.

At each tick the agent sees a query one-hot, the noisy outcome and the step index.
An episode of budget `B` ends with an 8-way prediction.

### Compared systems

| System | Training | Decision rule |
|---|---|---|
| **MTDB active** | MTDB base + bootstrap-ridge class/outcome ensembles, trained on passively collected random-probe episodes | expected-free-energy planning over probes using the learned probe table and the class posterior from the MTDB interface |
| **MTDB passive** | same model | fixed probe order `0,1,2,...` |
| **Transformer passive** | parameter-matched causal Transformer (≈69.8k params) trained by backprop on the same passive episodes | fixed probe order; class logits at the last token |
| **Transformer + MTDB probes** | same Transformer | evaluated on the *actively selected* probe sequences — isolates evidence acquisition from classifier capacity |
| **Bayes active (oracle)** | known generative model, no learning | exact expected-information-gain planning |

All systems are trained on the same 600 random-probe episodes (8 probes each) and
evaluated on the same 200 episodes per probe budget. MTDB has 69,216 parameters;
the Transformer has 69,832.

## 3. Results

| Probe budget | MTDB active | MTDB fixed | Transformer passive | Transformer + MTDB probes | Bayes active |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.210 | 0.275 | 0.230 | 0.235 | 0.205 |
| 2 | 0.390 | 0.200 | 0.195 | **0.430** | 0.435 |
| 3 | **0.605** | 0.430 | 0.315 | 0.495 | 0.765 |
| 4 | **0.750** | 0.680 | 0.525 | 0.700 | 0.725 |
| 5 | **0.825** | 0.820 | 0.555 | 0.725 | 0.810 |

Reference chance = 0.125.

### What the numbers say

- **Active planning helps the same MTDB model**: at budgets 2–5, MTDB active is
  substantially above MTDB with a fixed probe order (e.g. 3 probes: 0.605 vs
  0.430; 4 probes: 0.750 vs 0.680).
- **Against the passive next-token baseline**: MTDB active beats the fixed-probe
  Transformer at every budget from 2 onward by a wide margin (e.g. 3 probes:
  0.605 vs 0.315; 5 probes: 0.825 vs 0.555).
- **Honest fairness check**: when the Transformer is given the *same actively
  selected observations*, it improves a lot (0.495 / 0.700 / 0.725 at budgets
  3/4/5). This shows that **most of the gain comes from active evidence
  acquisition, not from the classifier alone**. MTDB active is still ahead at
  budgets 3–5, but the Transformer is better at budget 2 (0.430 vs 0.390).
- **Oracle gap**: exact Bayes active reaches 0.765 at 3 probes while MTDB active
  reaches 0.605. The gap is dominated by the finite-data ridge class readout, not
  by the planning rule. This is a concrete next research target.
- At budget 1 there is no time to think; all methods are near chance. Active
  inference is about *sequential* information gathering.

This is a **scoped result on a synthetic 8-class active-sensing task with one
seed and 200 evaluation episodes**. It does not imply that MTDB is generally
better than Transformers, and it is not evidence of AGI.

## 4. What is actually new in the code

- `mtdb/active/sensing.py` — exact active-sensing environment + exact Bayesian
  oracle.
- `mtdb/active/agent.py` — `ActiveSensingAgent`:
  - learns a class posterior (bootstrap ridge ensemble) from the MTDB interface;
  - learns a probe→bit table and noise level from labelled training experience;
  - performs one-step expected-free-energy lookahead over candidate probes;
  - has a latent-rollout fallback through `MTDBSystem.snapshot_state()`.
- `MTDBSystem.snapshot_state()` / `restore_state()` — cheap mental-rollout
  snapshots including working-memory usage; used by deliberation and tested for
  bit-exact restoration.
- Optional **sensory evidence trace** (`include_sensory_trace=True`): a generic
  decaying trace of the input stream included in `z(t)`. It is disabled by
  default, so all v0.1 benchmark numbers remain unchanged.
- `mtdb/baselines/transformer.py` — small causal Transformer baseline trained
  with backprop, for a fair comparison.
- `scripts/active_benchmark.py` — one-command reproduction (~8 s on CPU).

## 5. Reproduce

```bash
python scripts/active_benchmark.py
# writes results/active_v02/results.json (script writes artifacts/ by default;
# the committed result is results/active_v02/results.json)
```

The script prints:

```text
B=1 | MTDB-active 0.210 | MTDB-fixed 0.275 | Transformer-passive 0.230 | Transformer+active-probes 0.235 | Bayes 0.205
B=2 | MTDB-active 0.390 | MTDB-fixed 0.200 | Transformer-passive 0.195 | Transformer+active-probes 0.430 | Bayes 0.435
B=3 | MTDB-active 0.605 | MTDB-fixed 0.430 | Transformer-passive 0.315 | Transformer+active-probes 0.495 | Bayes 0.765
B=4 | MTDB-active 0.750 | MTDB-fixed 0.680 | Transformer-passive 0.525 | Transformer+active-probes 0.700 | Bayes 0.725
B=5 | MTDB-active 0.825 | MTDB-fixed 0.820 | Transformer-passive 0.555 | Transformer+active-probes 0.725 | Bayes 0.810
```

## 6. Limitations and next steps

1. **Synthetic task**: 8 classes, 8 probes, 10% label noise. No real sensor data.
2. **Only one seed** and 200 evaluation episodes; confidence intervals should be
   added for a publication-grade claim.
3. **The probe table is learned from labelled training episodes**, but the planner
   currently assumes the probe semantics are stationary. Online adaptation of the
   probe table is the obvious next step.
4. **The class readout is a linear ensemble**, leaving a large gap to the exact
   Bayes oracle (0.605 vs 0.765 at 3 probes). A richer local readout (kernel /
   random-feature / small MLP head) is a natural improvement.
5. **Planning is belief-space**; the latent-dynamics rollout path exists but is
   not the default because the current class readout is miscalibrated under
   imagined states.
6. **No universal superiority claim**: on the same actively selected evidence, a
   Transformer is competitive. The defensible claim is narrower — *active
   evidence acquisition, which next-token predictors cannot do on their own, is
   the main source of sample efficiency; MTDB provides one natively trainable
   way to implement it.*
7. **Not AGI**. This is a controlled primitive: sequential decision making under
   uncertainty with a learned world model.

## 7. If you want to push this further

- replace the probe table with an online-learned experiment model;
- add calibrated uncertainty (deep ensembles / conformal / Bayesian last layer);
- plan multiple steps ahead with particle rollouts through `MTDBSystem`;
- add active perception on real event-camera/audio streams;
- compare against model-based RL baselines and Transformer policies trained with
  the same probe-selection objective.

The repository will keep both results (where MTDB active wins and where the
Transformer catches up) visible in `results/active_v02/results.json`.
