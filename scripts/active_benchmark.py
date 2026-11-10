#!/usr/bin/env python3
"""Active-inference benchmark for MTDB v0.2.

The question is narrow and measurable: with the same passively collected training
episodes and the same 8-way active-sensing task, does an agent that **plans its
own probes by mental rollout** identify the hidden class with fewer probes than

  * the same MTDB base using a fixed/random probe policy, and
  * a parameter-matched causal Transformer trained on the same data?

The exact Bayesian active policy is reported as an oracle upper bound. The
Transformer is evaluated both on the passive probe sequences and on the
actively selected sequences, so the reader can separate "better evidence" from
"better classifier". No universal Transformer-beating claim is implied.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

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


def build_config(args) -> ExperimentConfig:
    cfg = ExperimentConfig.from_yaml(args.config)
    cfg.device = "cpu"
    cfg.include_sensory_trace = True
    cfg.include_sensory_channel = True
    cfg.dynamics.n_units = args.units
    cfg.dynamics.in_degree = args.in_degree
    cfg.dynamics.h_dim = args.h_dim
    cfg.dynamics.dt_ms = 100.0          # one probe = one 100 ms tick
    cfg.dynamics.noise_std = 0.0        # planning rollouts should be deterministic
    cfg.phase.phase_noise = 0.0
    cfg.memory.attr_capacity = 128
    cfg.memory.attr_read_dim = 32
    cfg.memory.wm_slots = 16
    cfg.memory.sensory_trace_decay = 0.8
    cfg.train.ablations = False
    cfg.train.run_stage2 = False
    cfg.train.run_recall = False
    cfg.train.forgetting_stress = False
    cfg.train.plot = False
    return cfg


def mtdb_accuracy(agent: ActiveSensingAgent, episodes: List[SensingEpisode]) -> float:
    if not episodes:
        return 0.0
    correct = 0
    for ep in episodes:
        correct += int(agent.predict_episode(ep) == ep.label)
    return correct / len(episodes)


def transformer_accuracy(
    model: CausalTransformerClassifier, episodes: List[SensingEpisode]
) -> float:
    if not episodes:
        return 0.0
    preds = transformer_predict(model, episodes)
    return float(np.mean([int(p == e.label) for p, e in zip(preds, episodes)]))


def bayes_accuracy(env: ActiveSensingEnv, episodes: List[SensingEpisode]) -> float:
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


def generate_active(
    agent: ActiveSensingAgent, n: int, budget: int, seed: int
) -> List[SensingEpisode]:
    rng = np.random.default_rng(seed)
    return [agent.plan_episode(rng, budget) for _ in range(n)]


def plot_results(rows: List[dict], out_path: Path, transformer_params: int, mtdb_params: int) -> None:
    budgets = [r["budget"] for r in rows]
    series = [
        ("bayes_active", "Bayes active (oracle)", "#111827", "o", "-"),
        ("mtdb_active", "MTDB active (EFE planning)", "#0f766e", "o", "-"),
        ("mtdb_fixed", "MTDB fixed probes", "#5eead4", "s", "--"),
        ("transformer_passive", "Transformer, fixed probes", "#dc2626", "s", "--"),
        ("transformer_active_queries", "Transformer, MTDB probes", "#f59e0b", "^", "-"),
        ("mtdb_random", "MTDB random probes", "#94a3b8", ".", ":"),
    ]
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    for key, label, color, marker, ls in series:
        ys = [r[key] for r in rows]
        ax.plot(budgets, ys, marker=marker, color=color, lw=1.7, ls=ls, label=label)
    chance = 1.0 / 8
    ax.axhline(chance, color="#64748b", ls=":", lw=0.9)
    ax.set_xlabel("probe budget (number of queries)")
    ax.set_ylabel("8-way accuracy")
    ax.set_ylim(0, 1.02)
    ax.set_title("Active inference vs passive next-token-style classifiers")
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    ax.text(
        0.01, 0.01,
        f"MTDB {mtdb_params:,} params · Transformer {transformer_params:,} params\n"
        "same training episodes; Transformer also sees the MTDB-selected probes",
        transform=ax.transAxes, fontsize=6.2, color="#475569", va="bottom",
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="MTDB active-inference benchmark")
    ap.add_argument("--config", default=str(ROOT / "configs" / "debug.yaml"))
    ap.add_argument("--out-dir", default=str(ROOT / "artifacts"))
    ap.add_argument("--run-name", default="active_v02")
    ap.add_argument("--units", type=int, default=768)
    ap.add_argument("--in-degree", type=int, default=16)
    ap.add_argument("--h-dim", type=int, default=96)
    ap.add_argument("--train-episodes", type=int, default=600)
    ap.add_argument("--eval-episodes", type=int, default=200)
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--budgets", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--transformer-epochs", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.train_episodes = 80
        args.eval_episodes = 24
        args.passes = 1
        args.transformer_epochs = 15
        args.budgets = [2, 3, 4]

    set_seed(args.seed)
    logger = get_logger("mtdb.active")
    t0 = time.time()
    cfg = build_config(args)
    env = ActiveSensingEnv()
    rng = np.random.default_rng(args.seed)
    train_episodes = env.random_batch(args.train_episodes, rng)

    logger.info("Training MTDB base + ensemble readouts on %d passive episodes",
                len(train_episodes))
    agent = ActiveSensingAgent(cfg, env, seed=args.seed, n_ensemble=5)
    agent.fit(train_episodes, passes=args.passes)

    logger.info("Training small causal Transformer baseline")
    transformer = train_transformer_classifier(
        train_episodes,
        feature_dim=env.feature_dim,
        n_classes=env.n_classes,
        epochs=args.transformer_epochs,
        seed=args.seed,
    )
    t_train = time.time() - t0
    logger.info("Training finished in %.1fs", t_train)

    rows: List[dict] = []
    for budget in args.budgets:
        seed = 1000 + budget
        fixed = generate_passive(env, "fixed", args.eval_episodes, budget, seed)
        random = generate_passive(env, "random", args.eval_episodes, budget, seed + 1)
        bayes = generate_passive(env, "bayes", args.eval_episodes, budget, seed + 2)
        active = generate_active(agent, args.eval_episodes, budget, seed + 3)
        row = {
            "budget": int(budget),
            "mtdb_fixed": mtdb_accuracy(agent, fixed),
            "mtdb_random": mtdb_accuracy(agent, random),
            "mtdb_active": mtdb_accuracy(agent, active),
            "transformer_passive": transformer_accuracy(transformer, fixed),
            "transformer_random": transformer_accuracy(transformer, random),
            "transformer_active_queries": transformer_accuracy(transformer, active),
            "bayes_active": bayes_accuracy(env, bayes),
            "bayes_fixed": bayes_accuracy(env, fixed),
            "mean_active_queries": float(np.mean([len(e.queries) for e in active])),
            "active_query_histogram": np.bincount(
                np.concatenate([e.queries for e in active]) if active else np.array([], dtype=int),
                minlength=env.n_queries,
            ).tolist(),
        }
        rows.append(row)
        logger.info(
            "budget %d | Bayes-active %.2f | MTDB-active %.2f | MTDB-fixed %.2f | "
            "Transformer-passive %.2f | Transformer+active-probes %.2f",
            budget, row["bayes_active"], row["mtdb_active"], row["mtdb_fixed"],
            row["transformer_passive"], row["transformer_active_queries"],
        )

    mtdb_params = int(agent.system.n_parameters())
    transformer_params = int(transformer_param_count(transformer))
    results = {
        "config": {
            "units": cfg.dynamics.n_units,
            "in_degree": cfg.dynamics.in_degree,
            "h_dim": cfg.dynamics.h_dim,
            "include_sensory_trace": cfg.include_sensory_trace,
            "train_episodes": args.train_episodes,
            "eval_episodes": args.eval_episodes,
            "passes": args.passes,
            "transformer_epochs": args.transformer_epochs,
            "noise": env.noise,
            "seed": args.seed,
        },
        "params": {"mtdb": mtdb_params, "transformer": transformer_params},
        "train_time_s": t_train,
        "rows": rows,
        "note": (
            "Synthetic 8-class active sensing. 'transformer_active_queries' is the "
            "Transformer classifier evaluated on the MTDB-selected probe sequences; "
            "it isolates evidence acquisition from classifier capacity. This is a "
            "scoped benchmark, not a universal Transformer-vs-MTDB claim."
        ),
    }
    run_dir = Path(args.out_dir) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "results.json").write_text(json.dumps(results, indent=2))
    plot_results(rows, ROOT / "assets" / "active_inference.png", transformer_params, mtdb_params)
    logger.info("Wrote %s and assets/active_inference.png", run_dir / "results.json")
    print("\n=== Active inference benchmark ===")
    for row in rows:
        print(
            f"B={row['budget']} | MTDB-active {row['mtdb_active']:.3f} "
            f"| MTDB-fixed {row['mtdb_fixed']:.3f} "
            f"| Transformer-passive {row['transformer_passive']:.3f} "
            f"| Transformer+active-probes {row['transformer_active_queries']:.3f} "
            f"| Bayes {row['bayes_active']:.3f}"
        )
    print(f"params: MTDB {mtdb_params:,} / Transformer {transformer_params:,}")


if __name__ == "__main__":
    main()
