#!/usr/bin/env python3
"""Evaluate a saved MTDB checkpoint on the synthetic prediction/anomaly tasks.

The final checkpoint contains the consolidated base + online RLS readout.
If ``checkpoint_stage1.pt`` is present next to it, the stage-1 ridge readout and
the post-stage-1 base are also evaluated (that is the clean anomaly-detection
model before online adaptation changes the self-predictor statistics).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from mtdb.config import ExperimentConfig
from mtdb.training.readout import RLSReadout
from mtdb.training.trainer import MTDBTrainer, _jsonable
from mtdb.utils import get_logger, set_seed


def _build_trainer(ckpt: dict, out: Path):
    cfg = ExperimentConfig.from_dict(ckpt["config"])
    trainer = MTDBTrainer(cfg, run_dir=out, logger=get_logger("mtdb.eval"))
    trainer.system.load_state_dict(ckpt["system"])
    trainer.readout.load_state_dict(ckpt["readout"])
    return trainer


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate an MTDB checkpoint")
    ap.add_argument("--checkpoint", default="artifacts/demo_standard/checkpoint.pt")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ckpt_path = Path(args.checkpoint)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    input_dim = int(ckpt.get("input_dim", 10))
    if input_dim != 10:
        raise SystemExit(
            "scripts/eval.py evaluates the shipped 10-D synthetic benchmark only. "
            f"This checkpoint uses input_dim={input_dim}; load your own data with "
            "mtdb.tasks.array.make_streams_from_arrays and evaluate the trainer directly."
        )
    set_seed(0)
    out_dir = Path(args.out) if args.out else ckpt_path.parent / "eval"
    trainer = _build_trainer(ckpt, out_dir)

    results = {
        "checkpoint": str(ckpt_path),
        "interface_dim": trainer.system.spec.total_dim,
        "n_parameters": trainer.system.n_parameters(),
        "final": {},
    }
    if "readout_online" in ckpt:
        rls = RLSReadout(
            trainer.system.spec.total_dim, trainer.input_dim,
            forget=trainer.cfg.readout.rls_forget,
            init_delta=trainer.cfg.readout.rls_init_delta,
        )
        rls.load_state_dict(ckpt["readout_online"])
        trainer.readout_online = rls
        results["final"]["readout"] = "rls"
        results["final"]["prediction_A"] = trainer._eval_prediction(
            trainer.eval_A, readout=rls
        )
        results["final"]["prediction_B"] = trainer._eval_prediction(
            trainer.eval_B, readout=rls
        )
    else:
        results["final"]["readout"] = "ridge"
        results["final"]["prediction_A"] = trainer._eval_prediction(trainer.eval_A)
        results["final"]["prediction_B"] = trainer._eval_prediction(trainer.eval_B)
    results["final"]["anomaly"] = {
        k: v for k, v in trainer._eval_anomaly().items() if k != "series"
    }

    stage1_path = ckpt_path.parent / "checkpoint_stage1.pt"
    if stage1_path.exists():
        s1 = torch.load(stage1_path, map_location="cpu", weights_only=False)
        trainer1 = _build_trainer(s1, out_dir / "stage1")
        results["stage1"] = {
            "prediction_A": trainer1._eval_prediction(trainer1.eval_A),
            "anomaly": {
                k: v for k, v in trainer1._eval_anomaly().items() if k != "series"
            },
        }

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "eval.json", "w", encoding="utf-8") as f:
        json.dump(_jsonable(results), f, ensure_ascii=False, indent=2)

    print("=== MTDB checkpoint evaluation ===")
    print(f"checkpoint: {ckpt_path}")
    print(f"interface: {results['interface_dim']} dims, params {results['n_parameters']:,}")
    fin = results["final"]
    print(f"final ({fin['readout']}) | A MSE {fin['prediction_A']['mse']:.5f} "
          f"| B MSE {fin['prediction_B']['mse']:.5f}")
    a = fin["anomaly"]
    print(f"final anomaly AUC {a['auc']:.3f} (smoothed {a['auc_smoothed']:.3f})")
    if "stage1" in results:
        p1 = results["stage1"]["prediction_A"]
        a1 = results["stage1"]["anomaly"]
        print(f"stage1 ridge | A MSE {p1['mse']:.5f} | anomaly AUC {a1['auc']:.3f} "
              f"(smoothed {a1['auc_smoothed']:.3f}, input-change {a1['auc_input_change']:.3f})")
    print(f"wrote {out_dir / 'eval.json'}")


if __name__ == "__main__":
    main()
