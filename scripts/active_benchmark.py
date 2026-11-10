#!/usr/bin/env python3
"""MTDB v0.3 active-inference benchmark.

Adds to the v0.2 benchmark:

* multi-step / particle planning (depth 1 vs depth 2+);
* temperature calibration + conformal prediction sets;
* online per-probe noise adaptation under a non-stationary (drifted) probe;
* a fair Transformer + active-policy baseline that uses the same planner;
* multiple seeds with 95% confidence intervals.

The task and protocol are deliberately small. Results are reported as scoped
evidence, not as a universal Transformer-beating or AGI claim.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mtdb.active import planning
from mtdb.active.agent import ActiveSensingAgent
from mtdb.active.sensing import ActiveSensingEnv, SensingEpisode
from mtdb.baselines.transformer import (
    CausalTransformerClassifier,
    train_transformer_classifier,
    transformer_param_count,
    transformer_predict,
)
from mtdb.config import ExperimentConfig
from mtdb.utils import get_logger, set_seed

T_CRIT_95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571}


def build_config(args) -> ExperimentConfig:
    cfg = ExperimentConfig.from_yaml(args.config)
    cfg.device = "cpu"
    cfg.include_sensory_trace = True
    cfg.include_sensory_channel = True
    cfg.dynamics.n_units = args.units
    cfg.dynamics.in_degree = args.in_degree
    cfg.dynamics.h_dim = args.h_dim
    cfg.dynamics.dt_ms = 100.0
    cfg.dynamics.noise_std = 0.0
    cfg.phase.phase_noise = 0.0
    cfg.memory.attr_capacity = 128
    cfg.memory.wm_slots = 16
    cfg.memory.sensory_trace_decay = 0.8
    cfg.train.ablations = False
    cfg.train.run_stage2 = False
    cfg.train.run_recall = False
    cfg.train.forgetting_stress = False
    cfg.train.plot = False
    return cfg


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def mtdb_accuracy(agent: ActiveSensingAgent, episodes: Sequence[SensingEpisode]) -> float:
    if not episodes:
        return 0.0
    return float(np.mean([agent.predict_episode(e) == e.label for e in episodes]))


def transformer_accuracy(
    model: CausalTransformerClassifier, episodes: Sequence[SensingEpisode]
) -> float:
    if not episodes:
        return 0.0
    preds = transformer_predict(model, episodes)
    return float(np.mean([int(p == e.label) for p, e in zip(preds, episodes)]))


def bayes_accuracy(env: ActiveSensingEnv, episodes: Sequence[SensingEpisode]) -> float:
    if not episodes:
        return 0.0
    return float(np.mean([env.bayes_predict(e) == e.label for e in episodes]))


def generate_passive(
    env: ActiveSensingEnv, policy: str, n: int, budget: int, seed: int
) -> List[SensingEpisode]:
    rng = np.random.default_rng(seed)
    if policy == "fixed":
        return [env.fixed_episode(rng, budget) for _ in range(n)]
    if policy == "random":
        return [env.random_episode(rng, budget) for _ in range(n)]
    if policy == "bayes":
        return [env.bayes_episode(rng, budget) for _ in range(n)]
    raise ValueError(policy)


def evaluate_mtdb_active(
    agent: ActiveSensingAgent,
    n: int,
    budget: int,
    seed: int,
    depth: int,
    n_particles: int,
    adaptive_noise: bool = False,
) -> List[SensingEpisode]:
    rng = np.random.default_rng(seed)
    eps = []
    for _ in range(n):
        eps.append(
            agent.plan_episode(
                rng, budget, depth=depth, n_particles=n_particles,
                adaptive_noise=adaptive_noise,
            )
        )
    return eps


@torch.no_grad()
def transformer_posterior(
    model: CausalTransformerClassifier,
    features: Sequence[np.ndarray],
    temperature: float = 1.0,
) -> np.ndarray:
    x = torch.tensor(np.stack(features), dtype=torch.float32).unsqueeze(0)
    logits = model(x)[0, -1] / max(temperature, 1e-3)
    return torch.softmax(logits, dim=-1).numpy()


def run_transformer_active(
    model: CausalTransformerClassifier,
    env: ActiveSensingEnv,
    code: np.ndarray,
    noise: np.ndarray,
    budget: int,
    depth: int,
    n_particles: int,
    rng: np.random.Generator,
    temperature: float = 1.0,
) -> SensingEpisode:
    label = int(rng.integers(0, env.n_classes))
    features: List[np.ndarray] = []
    queries: List[int] = []
    outcomes: List[float] = []
    planning_rng = np.random.default_rng(int(rng.integers(0, 2 ** 31 - 1)))
    for t in range(budget):
        post = transformer_posterior(model, features, temperature) if features else np.ones(env.n_classes) / env.n_classes
        q = planning.plan_query(
            post, code, noise, depth=depth, n_particles=n_particles, rng=planning_rng
        )
        o = env.observe(label, q, rng, step=t)
        features.append(env.feature(q, o, t, budget))
        queries.append(q)
        outcomes.append(o)
    return SensingEpisode(
        features=np.stack(features),
        label=label,
        queries=np.asarray(queries, dtype=np.int64),
        outcomes=np.asarray(outcomes, dtype=np.float32),
    )


def fit_transformer_temperature(
    model: CausalTransformerClassifier, episodes: Sequence[SensingEpisode]
) -> float:
    if not episodes:
        return 1.0
    T = min(e.features.shape[0] for e in episodes)
    X = torch.tensor(
        np.stack([e.features[:T] for e in episodes]), dtype=torch.float32
    )
    y = torch.tensor([e.label for e in episodes], dtype=torch.long)
    with torch.no_grad():
        logits = model(X)[:, -1]
    best_t, best_nll = 1.0, float("inf")
    for temperature in (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0):
        nll = float(
            torch.nn.functional.cross_entropy(logits / temperature, y)
        )
        if nll < best_nll:
            best_nll, best_t = nll, float(temperature)
    return best_t


def mean_ci(values: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    mean = float(arr.mean()) if n else 0.0
    std = float(arr.std(ddof=1)) if n > 1 else 0.0
    crit = T_CRIT_95.get(n - 1, 1.96)
    ci = crit * std / np.sqrt(n) if n > 1 else 0.0
    return {"mean": mean, "std": std, "ci95": float(ci), "n": n}


# ---------------------------------------------------------------------------
# one seed
# ---------------------------------------------------------------------------
def run_seed(seed: int, args) -> Dict[str, object]:
    t0 = time.time()
    cfg = build_config(args)
    env = ActiveSensingEnv()
    rng = np.random.default_rng(seed)
    episodes = env.random_batch(args.train_episodes, rng)
    split = max(1, int(len(episodes) * 0.8))
    fit_episodes = episodes[:split]
    calib_episodes = episodes[split:]

    agent = ActiveSensingAgent(cfg, env, seed=seed, n_ensemble=5)
    agent.fit(fit_episodes, passes=args.passes)

    transformer = train_transformer_classifier(
        fit_episodes,
        feature_dim=env.feature_dim,
        n_classes=env.n_classes,
        epochs=args.transformer_epochs,
        seed=seed,
    )
    calib_fixed = generate_passive(
        env, "fixed", len(calib_episodes), 5, seed + 6000
    )
    transformer_temp = fit_transformer_temperature(transformer, calib_fixed)

    # shared probe table (learned from labelled fit episodes, same for both models)
    labels = [ep.label for ep in fit_episodes for _ in range(len(ep.queries))]
    qs = [int(q) for ep in fit_episodes for q in ep.queries]
    outs = [float(o) for ep in fit_episodes for o in ep.outcomes]
    code, noise_est = planning.infer_code_and_noise(
        labels, qs, outs, env.n_classes, env.n_queries
    )
    # start planning from a common reliability prior; online drift adaptation
    # learns per-probe deviations later
    noise = np.full(env.n_queries, float(np.mean(noise_est)), dtype=np.float64)
    agent.code_hat = torch.as_tensor(code, dtype=torch.long)
    agent.noise_per_query = noise.copy()
    agent.noise_hat = float(noise.mean())

    # ---------------- calibration split --------------------------------
    calibration_eval = generate_passive(
        env, "fixed", args.eval_episodes, 5, seed + 7000
    )
    before = agent.calibration_report(calibration_eval)
    temperature = agent.calibrate(calib_fixed)
    conformal_q = agent.conformal_calibrate(calib_fixed, alpha=0.1)
    after = agent.calibration_report(calibration_eval)

    # ---------------- stationary benchmark -----------------------------
    rows = []
    for budget in args.budgets:
        s = seed + 1000 + budget
        fixed = generate_passive(env, "fixed", args.eval_episodes, budget, s)
        random = generate_passive(env, "random", args.eval_episodes, budget, s + 1)
        bayes = generate_passive(env, "bayes", args.eval_episodes, budget, s + 2)
        active_d1 = evaluate_mtdb_active(
            agent, args.eval_episodes, budget, s + 3, depth=1,
            n_particles=args.particles,
        )
        active_d2 = evaluate_mtdb_active(
            agent, args.eval_episodes, budget, s + 4, depth=2,
            n_particles=args.particles,
        )
        rng_trans = np.random.default_rng(s + 5)
        trans_active = [
            run_transformer_active(
                transformer, env, code, noise, budget, depth=2,
                n_particles=args.particles,
                rng=rng_trans,
                temperature=transformer_temp,
            )
            for _ in range(args.eval_episodes)
        ]
        rows.append(
            {
                "budget": int(budget),
                "mtdb_fixed": mtdb_accuracy(agent, fixed),
                "mtdb_random": mtdb_accuracy(agent, random),
                "mtdb_active_d1": mtdb_accuracy(agent, active_d1),
                "mtdb_active_d2": mtdb_accuracy(agent, active_d2),
                "transformer_passive": transformer_accuracy(transformer, fixed),
                "transformer_active_d2": transformer_accuracy(transformer, trans_active),
                "transformer_on_mtdb_active": transformer_accuracy(transformer, active_d2),
                "bayes_active": bayes_accuracy(env, bayes),
                "mtdb_active_d2_queries": float(np.mean([len(e.queries) for e in active_d2])),
            }
        )

    # ---------------- non-stationary drift -----------------------------
    env.configure_drift(
        switch_step=args.drift_switch,
        drift_queries=tuple(args.drift_queries),
        drift_noise=args.drift_noise,
    )
    initial_noise = noise.copy()
    drift_rows = []
    for budget in args.drift_budgets:
        s = seed + 5000 + budget
        # static planner: frozen reliability table
        agent.reset_noise(initial_noise)
        static_eps = evaluate_mtdb_active(
            agent, args.eval_episodes, budget, s, depth=1,
            n_particles=args.particles, adaptive_noise=False,
        )
        # adaptive planner: posterior-weighted online reliability updates
        agent.reset_noise(initial_noise)
        adaptive_eps = evaluate_mtdb_active(
            agent, args.eval_episodes, budget, s, depth=1,
            n_particles=args.particles, adaptive_noise=True,
        )
        rng_trans_drift = np.random.default_rng(s + 3)
        trans_drift = [
            run_transformer_active(
                transformer, env, code, initial_noise, budget, depth=1,
                n_particles=args.particles,
                rng=rng_trans_drift,
                temperature=transformer_temp,
            )
            for _ in range(args.eval_episodes)
        ]
        drift_rows.append(
            {
                "budget": int(budget),
                "mtdb_static_noise": mtdb_accuracy(agent, static_eps),
                "mtdb_adaptive_noise": mtdb_accuracy(agent, adaptive_eps),
                "transformer_active": transformer_accuracy(transformer, trans_drift),
                "noise_mean_after": float(np.mean(agent.noise_per_query)),
            }
        )
    env.clear_drift()

    return {
        "seed": int(seed),
        "params": {
            "mtdb": int(agent.system.n_parameters()),
            "transformer": int(transformer_param_count(transformer)),
        },
        "calibration": {
            "before": before,
            "after": after,
            "temperature": float(temperature),
            "conformal_q": float(conformal_q),
        },
        "stationary": rows,
        "drift": drift_rows,
        "time_s": time.time() - t0,
    }


# ---------------------------------------------------------------------------
# aggregation + plotting
# ---------------------------------------------------------------------------
def aggregate(per_seed: List[dict]) -> Dict[str, object]:
    out: Dict[str, object] = {"stationary": {}, "drift": {}}
    budgets = [r["budget"] for r in per_seed[0]["stationary"]]
    metrics = [k for k in per_seed[0]["stationary"][0] if k != "budget"]
    for budget in budgets:
        block = {}
        for metric in metrics:
            values = [
                r["stationary"][i][metric]
                for r in per_seed
                for i in range(len(r["stationary"]))
                if r["stationary"][i]["budget"] == budget
            ]
            block[metric] = mean_ci(values)
        out["stationary"][str(budget)] = block
    for row in per_seed[0]["drift"]:
        budget = row["budget"]
        block = {}
        for metric in row:
            if metric in ("budget",):
                continue
            values = [
                r["drift"][i][metric]
                for r in per_seed
                for i in range(len(r["drift"]))
                if r["drift"][i]["budget"] == budget
            ]
            block[metric] = mean_ci(values)
        out["drift"][str(budget)] = block
    return out


def plot_stationary(agg: Dict[str, object], out_path: Path) -> None:
    budgets = sorted(int(b) for b in agg["stationary"].keys())
    series = [
        ("bayes_active", "Bayes active (oracle)", "#111827", "o", "-"),
        ("mtdb_active_d2", "MTDB active, depth 2 (particles)", "#0f766e", "o", "-"),
        ("mtdb_active_d1", "MTDB active, depth 1 (EIG)", "#14b8a6", "s", "--"),
        ("mtdb_fixed", "MTDB fixed probes", "#5eead4", "s", "--"),
        ("transformer_active_d2", "Transformer + active depth 2", "#f59e0b", "^", "-"),
        ("transformer_passive", "Transformer passive", "#dc2626", "s", "--"),
    ]
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    for key, label, color, marker, ls in series:
        means = [agg["stationary"][str(b)][key]["mean"] for b in budgets]
        cis = [agg["stationary"][str(b)][key]["ci95"] for b in budgets]
        ax.errorbar(
            budgets, means, yerr=cis, marker=marker, color=color, lw=1.7,
            ls=ls, capsize=3, label=label,
        )
    ax.axhline(1.0 / 8.0, color="#64748b", ls=":", lw=0.9)
    ax.set_xlabel("probe budget")
    ax.set_ylabel("8-way accuracy")
    ax.set_ylim(0, 1.02)
    ax.set_title("Active inference: planning depth, calibration and fair baselines")
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_drift(agg: Dict[str, object], out_path: Path, switch_step: int = 3) -> None:
    budgets = sorted(int(b) for b in agg["drift"].keys())
    series = [
        ("mtdb_adaptive_noise", "MTDB active, online noise", "#0f766e", "o"),
        ("mtdb_static_noise", "MTDB active, frozen noise", "#94a3b8", "s"),
        ("transformer_active", "Transformer + active", "#f59e0b", "^"),
    ]
    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    for key, label, color, marker in series:
        means = [agg["drift"][str(b)][key]["mean"] for b in budgets]
        cis = [agg["drift"][str(b)][key]["ci95"] for b in budgets]
        ax.errorbar(budgets, means, yerr=cis, marker=marker, color=color,
                    lw=1.7, capsize=3, label=label)
    ax.set_xlabel(f"probe budget (drift starts after step {switch_step})")
    ax.set_ylabel("8-way accuracy")
    ax.set_ylim(0, 1.02)
    ax.set_title("Non-stationary probe reliability")
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="MTDB v0.3 active-inference benchmark")
    ap.add_argument("--config", default=str(ROOT / "configs" / "debug.yaml"))
    ap.add_argument("--out-dir", default=str(ROOT / "artifacts"))
    ap.add_argument("--run-name", default="active_v03")
    ap.add_argument("--units", type=int, default=768)
    ap.add_argument("--in-degree", type=int, default=16)
    ap.add_argument("--h-dim", type=int, default=96)
    ap.add_argument("--train-episodes", type=int, default=600)
    ap.add_argument("--eval-episodes", type=int, default=100)
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--budgets", type=int, nargs="+", default=[2, 3, 4, 5])
    ap.add_argument("--drift-budgets", type=int, nargs="+", default=[6, 8])
    ap.add_argument("--drift-switch", type=int, default=3)
    ap.add_argument("--drift-queries", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--drift-noise", type=float, default=0.4)
    ap.add_argument("--particles", type=int, default=12)
    ap.add_argument("--transformer-epochs", type=int, default=60)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.train_episodes = 80
        args.eval_episodes = 20
        args.passes = 1
        args.transformer_epochs = 15
        args.budgets = [2, 3]
        args.drift_budgets = [4]
        args.particles = 4
        args.seeds = [0]

    logger = get_logger("mtdb.active")
    t0 = time.time()
    per_seed = []
    for seed in args.seeds:
        set_seed(seed)
        logger.info("Running seed %d", seed)
        result = run_seed(seed, args)
        per_seed.append(result)
        logger.info(
            "seed %d | B=3 active-d2 %.3f active-d1 %.3f transformer %.3f | "
            "drift adaptive %.3f static %.3f",
            seed,
            next(r["mtdb_active_d2"] for r in result["stationary"] if r["budget"] == 3),
            next(r["mtdb_active_d1"] for r in result["stationary"] if r["budget"] == 3),
            next(r["transformer_passive"] for r in result["stationary"] if r["budget"] == 3),
            result["drift"][0]["mtdb_adaptive_noise"],
            result["drift"][0]["mtdb_static_noise"],
        )
    aggregate_rows = aggregate(per_seed)
    results = {
        "protocol": {
            "task": "8-class active sensing",
            "noise": 0.1,
            "train_episodes": args.train_episodes,
            "eval_episodes": args.eval_episodes,
            "seeds": args.seeds,
            "budgets": args.budgets,
            "depth2_particles": args.particles,
            "drift": {
                "switch_step": args.drift_switch,
                "queries": args.drift_queries,
                "noise": args.drift_noise,
                "budgets": args.drift_budgets,
            },
        },
        "params": per_seed[0]["params"],
        "per_seed": per_seed,
        "aggregate": aggregate_rows,
        "total_time_s": time.time() - t0,
        "note": (
            "Scoped synthetic benchmark. Active planning uses the same learned "
            "probe table for MTDB and the Transformer baseline; calibration is "
            "temperature scaling + split conformal. No universal superiority / "
            "AGI claim. See ACTIVE_INFERENCE.md."
        ),
    }
    run_dir = Path(args.out_dir) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "results.json").write_text(json.dumps(results, indent=2))
    plot_stationary(aggregate_rows, ROOT / "assets" / "active_inference.png")
    plot_drift(aggregate_rows, ROOT / "assets" / "active_drift.png", args.drift_switch)
    logger.info("Wrote %s and 2 figures", run_dir / "results.json")

    print("\n=== Active inference v0.3 (mean over %d seeds) ===" % len(args.seeds))
    for budget in args.budgets:
        b = aggregate_rows["stationary"][str(budget)]
        print(
            f"B={budget} | MTDB d2 {b['mtdb_active_d2']['mean']:.3f}±{b['mtdb_active_d2']['ci95']:.3f} "
            f"| d1 {b['mtdb_active_d1']['mean']:.3f} "
            f"| fixed {b['mtdb_fixed']['mean']:.3f} "
            f"| Transformer passive {b['transformer_passive']['mean']:.3f} "
            f"| Transformer active {b['transformer_active_d2']['mean']:.3f} "
            f"| Bayes {b['bayes_active']['mean']:.3f}"
        )
    print("\n=== Non-stationary drift (mean over seeds) ===")
    for budget in args.drift_budgets:
        b = aggregate_rows["drift"][str(budget)]
        print(
            f"B={budget} | adaptive noise {b['mtdb_adaptive_noise']['mean']:.3f} "
            f"| static noise {b['mtdb_static_noise']['mean']:.3f} "
            f"| Transformer active {b['transformer_active']['mean']:.3f}"
        )
    cal = per_seed[0]["calibration"]
    print(
        "\ncalibration seed0: ECE %.3f -> %.3f | NLL %.3f -> %.3f | T=%.2f | "
        "conformal coverage %.2f avg set %.2f"
        % (
            cal["before"].get("ece", float("nan")),
            cal["after"].get("ece", float("nan")),
            cal["before"].get("nll", float("nan")),
            cal["after"].get("nll", float("nan")),
            cal["temperature"],
            cal["after"].get("coverage", float("nan")),
            cal["after"].get("avg_set_size", float("nan")),
        )
    )


if __name__ == "__main__":
    main()
