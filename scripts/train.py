#!/usr/bin/env python3
"""Train the MTDB base on the synthetic world or on your own feature streams.

Examples
--------
    # synthetic multimodal benchmark
    python scripts/train.py --config configs/debug.yaml
    python scripts/train.py --config configs/standard.yaml --run-name standard_v1

    # your own streams (.npz with at least keys 'train' and 'eval')
    python scripts/train.py --config configs/standard.yaml --data data/my_stream.npz

    # quick override
    python scripts/train.py --config configs/debug.yaml --set train.epochs=1
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mtdb.config import ExperimentConfig, apply_overrides
from mtdb.eval.plot import ReportPlotter
from mtdb.tasks.array import streams_from_npz
from mtdb.training.trainer import MTDBTrainer
from mtdb.utils import get_logger, set_seed


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="MTDB training base")
    p.add_argument("--config", default=str(ROOT / "configs" / "debug.yaml"))
    p.add_argument("--run-name", default=None)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                   help="dotted config overrides, e.g. dynamics.n_units=512")
    p.add_argument("--data", default=None,
                   help="optional .npz with train/eval feature streams; "
                        "see mtdb/tasks/array.py for the format")
    p.add_argument("--no-normalize", action="store_true",
                   help="for --data: do not z-score features using train statistics")
    p.add_argument("--no-plot", action="store_true")
    return p


def main() -> None:
    args = build_parser().parse_args()
    cfg = ExperimentConfig.from_yaml(args.config)
    if args.run_name:
        cfg.run_name = args.run_name
    if args.out_dir:
        cfg.out_dir = args.out_dir
    apply_overrides(cfg, args.set)
    streams = None
    meta = None
    if args.data:
        streams, meta = streams_from_npz(
            args.data,
            horizon=cfg.train.horizon,
            normalize=not args.no_normalize,
        )
        # sensible defaults for a first custom-data run
        cfg.train.run_recall = False
        if not any(item.startswith("train.ablations") for item in args.set):
            cfg.train.ablations = False
        if not meta["has_regime_b"]:
            cfg.train.forgetting_stress = False
        if not args.run_name:
            cfg.run_name = f"custom_{Path(args.data).stem}"
    set_seed(cfg.seed)
    logger = get_logger("mtdb.train")
    logger.info("MTDB run '%s' | config %s | device %s | task %s",
                cfg.run_name, args.config, cfg.device,
                "custom-stream" if streams else "synthetic-multimodal")
    if meta:
        logger.info(
            "Data: %d train / %d eval sequences, F=%d, normalize=%s, regime-B=%s",
            meta["n_train"], meta["n_eval"], meta["input_dim"],
            meta["normalize"], meta["has_regime_b"],
        )
    t0 = time.time()
    trainer = MTDBTrainer(cfg, streams=streams)
    results = trainer.run()
    if cfg.train.plot and not args.no_plot:
        plots = ReportPlotter(results, trainer.run_dir).make_all()
        logger.info("Wrote %d figures", len(plots))
    logger.info("Total wall time: %.1fs", time.time() - t0)
    stage1 = results.get("stage1", {})
    if stage1:
        pred = stage1["prediction"]
        base = stage1["baselines"]
        anom = stage1["anomaly"]
        print("\n=== MTDB experiment summary ===")
        print(f"run dir        : {trainer.run_dir}")
        print(f"interface dim  : {results['interface_dim']}  "
              f"params: {results['n_parameters']:,}")
        print(f"prediction MSE : {pred['mse']:.5f} (NMSE {pred['nmse']:.3f})")
        print(f"persistence    : {base['persistence']['mse']:.5f} "
              f"(NMSE {base['persistence']['nmse']:.3f})")
        print(f"linear AR      : {base['linear_ar']['mse']:.5f} "
              f"(NMSE {base['linear_ar']['nmse']:.3f})")
        if anom and "auc" in anom:
            print(f"anomaly AUC    : {anom['auc']:.3f} "
                  f"(input-change {anom['auc_input_change']:.3f})")
        else:
            print("anomaly AUC    : skipped (no eval_anomaly stream)")
    if "stage2" in results:
        s2 = results["stage2"]
        print(f"forgetting A   : {s2['forgetting_A']:+.5f} "
              f"(sleep gain {s2['sleep_gain_A']:+.5f})")
        print(f"adaptation B   : {s2['adaptation_gain_B']:+.5f}")
    if "stage2_stress" in results:
        st = results["stage2_stress"]
        print(f"stress forget A: {st['forgetting_A']:+.5f} "
              f"(sleep gain {st['sleep_gain_A']:+.5f})")
        print(f"stress forget B: {st['sleep_gain_B']:+.5f} recovered by sleep")
    if "recall" in results:
        rc = results["recall"]
        print(f"recall acc     : {rc['accuracy_with_memory_writes']:.3f} "
              f"(chance {rc['chance']:.3f}, delay {rc['delay_steps']})")
        dl = rc.get("accuracy_by_delay") or []
        if dl:
            print("recall vs delay: " + ", ".join(
                f"{r['delay']}->{r['accuracy']:.2f}" for r in dl))
    print(f"results json   : {trainer.run_dir / 'results.json'}")
    print("===============================\n")


if __name__ == "__main__":
    main()
