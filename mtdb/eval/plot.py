"""Matplotlib report figures for an MTDB experiment."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .metrics import roc_curve

PALETTE = {
    "mtdb": "#0f766e",
    "baseline": "#94a3b8",
    "ablation": "#f59e0b",
    "sleep": "#7c3aed",
    "online": "#dc2626",
    "memory": "#2563eb",
}


class ReportPlotter:
    def __init__(self, results: Dict[str, object], run_dir: str | Path):
        self.results = results
        self.run_dir = Path(run_dir)
        self.plots_dir = self.run_dir / "plots"
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        plt.rcParams.update(
            {
                "figure.dpi": 130,
                "savefig.dpi": 130,
                "font.size": 9,
                "axes.spines.top": False,
                "axes.spines.right": False,
                "axes.grid": True,
                "grid.alpha": 0.25,
            }
        )

    def make_all(self) -> List[Path]:
        paths: List[Path] = []
        for name, fn in [
            ("prediction_comparison.png", self.prediction_comparison),
            ("stage1_training_curve.png", self.stage1_curve),
            ("anomaly_detection.png", self.anomaly_detection),
            ("delayed_recall.png", self.delayed_recall),
            ("online_adaptation.png", self.online_adaptation),
        ]:
            try:
                p = fn()
                if p is not None:
                    paths.append(p)
            except Exception as exc:  # plotting must never break the run
                print(f"[plot] {name} skipped: {exc}")
        return paths

    # ------------------------------------------------------------- utilities
    def _fig(self, ncols: int = 1, nrows: int = 1, size=(7.0, 3.6), **kwargs):
        fig, axes = plt.subplots(nrows, ncols, figsize=size, **kwargs)
        self._axes = axes
        return fig, axes

    def _save(self, fig, name: str) -> Path:
        path = self.plots_dir / name
        fig.tight_layout()
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        return path

    # ------------------------------------------------------------ prediction
    def prediction_comparison(self) -> Optional[Path]:
        stage1 = self.results.get("stage1", {})
        if not stage1:
            return None
        bars = [
            ("Mean", stage1["baselines"]["mean"].get("nmse", np.nan), "baseline"),
            ("Persistence", stage1["baselines"]["persistence"].get("nmse", np.nan), "baseline"),
            ("Linear AR(k=4)", stage1["baselines"]["linear_ar"].get("nmse", np.nan), "baseline"),
            ("MTDB (full)", stage1["prediction"].get("nmse", np.nan), "mtdb"),
        ]
        ablations = self.results.get("ablations", [])
        if ablations:
            fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.5))
        else:
            fig, axes = plt.subplots(1, 1, figsize=(5.2, 3.5))
            axes = [axes]
        ax = axes[0]
        labels = [b[0] for b in bars]
        vals = [b[1] for b in bars]
        colors = [PALETTE[b[2]] for b in bars]
        bars_artist = ax.bar(labels, vals, color=colors)
        ax.set_ylabel("NMSE (lower is better)")
        ax.set_title(f"Next observation prediction @ horizon {self.results.get('horizon', 5)}")
        ax.tick_params(axis="x", rotation=20)
        for artist, v in zip(bars_artist, vals):
            if np.isfinite(v):
                ax.text(artist.get_x() + artist.get_width() / 2, v, f"{v:.3f}",
                        ha="center", va="bottom", fontsize=8)
        if len(axes) > 1 and ablations:
            ax2 = axes[1]
            names = [r["name"] for r in ablations]
            nm = [r.get("nmse", np.nan) for r in ablations]
            cols = [PALETTE["mtdb"] if n == "full" else PALETTE["ablation"] for n in names]
            artist = ax2.bar(names, nm, color=cols)
            ax2.set_ylabel("NMSE")
            ax2.set_title("Ablations")
            ax2.tick_params(axis="x", rotation=25)
            for a, v in zip(artist, nm):
                if np.isfinite(v):
                    ax2.text(a.get_x() + a.get_width() / 2, v, f"{v:.3f}",
                             ha="center", va="bottom", fontsize=8)
        return self._save(fig, "prediction_comparison.png")

    # --------------------------------------------------------------- stage 1
    def stage1_curve(self) -> Optional[Path]:
        history = self.results.get("stage1", {}).get("history", [])
        if len(history) < 1:
            return None
        fig, ax = plt.subplots(figsize=(4.6, 3.2))
        epochs = [h["epoch"] for h in history]
        mses = [h["eval_mse"] for h in history]
        ax.plot(epochs, mses, marker="o", color=PALETTE["mtdb"], label="MTDB readout")
        if len(mses) > 1:
            ax.set_yscale("log")
        ax.set_xlabel("unsupervised epoch")
        ax.set_ylabel("eval MSE")
        ax.set_title("Stage 1 predictive-coding pretraining")
        ax.legend(frameon=False)
        return self._save(fig, "stage1_training_curve.png")

    # ---------------------------------------------------------------- anomaly
    def anomaly_detection(self) -> Optional[Path]:
        anomaly = self.results.get("stage1", {}).get("anomaly")
        if not anomaly:
            return None
        fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.3))
        series = anomaly["series"]
        s = np.asarray(series["surprise"])
        lab = np.asarray(series["label"])
        ax = axes[0]
        ax.plot(s, color=PALETTE["mtdb"], lw=1.3, label="surprise (normalised)")
        ax.fill_between(range(len(lab)), 0, max(s.max(), 1e-6),
                        where=lab.astype(bool), color="#ef4444", alpha=0.22,
                        label="anomaly")
        ax.set_xlabel("time step")
        ax.set_ylabel("surprise")
        ax.set_title("Surprise response to injected corruptions")
        ax.legend(frameon=False, fontsize=7)
        ax2 = axes[1]
        key_s = "scores_smoothed" if "scores_smoothed" in anomaly else "scores"
        auc_s = anomaly.get("auc_smoothed", anomaly["auc"])
        for key, label, color in [
            (key_s, f"MTDB surprise (AUC {auc_s:.3f})", PALETTE["mtdb"]),
            ("change_scores", f"input change (AUC {anomaly['auc_input_change']:.3f})",
             PALETTE["baseline"]),
        ]:
            fpr, tpr = roc_curve(anomaly[key], anomaly["labels"])
            ax2.plot(fpr, tpr, color=color, label=label, lw=1.6)
        ax2.plot([0, 1], [0, 1], "--", color="#64748b", lw=0.8)
        ax2.set_xlabel("false positive rate")
        ax2.set_ylabel("true positive rate")
        ax2.set_title("Anomaly detection ROC")
        ax2.legend(frameon=False, fontsize=7, loc="lower right")
        return self._save(fig, "anomaly_detection.png")

    # ----------------------------------------------------------------- recall
    def delayed_recall(self) -> Optional[Path]:
        recall = self.results.get("recall")
        if not recall:
            return None
        curve_w = np.asarray(recall["accuracy_by_t_with_writes"])
        curve_wo = np.asarray(recall["accuracy_by_t_without_writes"])
        if curve_w.size == 0:
            return None
        t = np.arange(curve_w.size)
        by_delay = recall.get("accuracy_by_delay") or []
        ncols = 2 if by_delay else 1
        fig, axes = plt.subplots(1, ncols, figsize=(9.6 if ncols == 2 else 6.4, 3.4))
        if ncols == 1:
            axes = [axes]
        ax = axes[0]
        cue_len = self.results["config"]["task"]["recall_cue_len"]
        scored_start = min(t.size, cue_len + recall["delay_steps"])
        ax.axvspan(0, cue_len, color="#f59e0b", alpha=0.12, label="cue")
        ax.axvspan(scored_start, t.size, color="#2563eb", alpha=0.08,
                   label="scored region")
        ax.plot(t, curve_w, color=PALETTE["memory"], lw=1.6,
                label=f"memory writes on ({recall['accuracy_with_memory_writes']:.2f})")
        ax.plot(t, curve_wo, color=PALETTE["baseline"], lw=1.4, ls="--",
                label=f"writes frozen ({recall['accuracy_without_memory_writes']:.2f})")
        ax.axhline(recall["chance"], color="#64748b", ls=":", lw=1.0,
                   label=f"chance ({recall['chance']:.2f})")
        ax.set_xlabel("time step")
        ax.set_ylabel("accuracy")
        ax.set_ylim(0, 1.05)
        ax.set_title(f"Recall curve, delay={recall['delay_steps']} steps")
        ax.legend(frameon=False, fontsize=7, loc="upper left")
        if ncols == 2:
            ax2 = axes[1]
            ds = [r["delay"] for r in by_delay]
            accs = [r["accuracy"] for r in by_delay]
            ax2.plot(ds, accs, marker="o", color=PALETTE["memory"], lw=1.6)
            ax2.axhline(recall["chance"], color="#64748b", ls=":", lw=1.0,
                        label="chance")
            ax2.set_xlabel("delay (steps)")
            ax2.set_ylabel("accuracy")
            ax2.set_ylim(0, 1.05)
            ax2.set_title("Memory accuracy vs delay")
            ax2.legend(frameon=False, fontsize=7)
        return self._save(fig, "delayed_recall.png")

    # ---------------------------------------------------------------- online
    def online_adaptation(self) -> Optional[Path]:
        stage2 = self.results.get("stage2")
        if not stage2:
            return None
        fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.3))
        ax = axes[0]
        curve = stage2["online_curve"]
        epochs = [c["epoch"] for c in curve]
        ax.plot(epochs, [c["online_mse"] for c in curve], marker="o",
                color=PALETTE["online"], label="online prediction MSE")
        ax.set_yscale("log")
        ax.set_xlabel("online epoch (regime B)")
        ax.set_ylabel("MSE")
        ax.set_title("Online adaptation to perturbed regime")
        ax.legend(frameon=False, fontsize=7)

        ax2 = axes[1]
        keys = [("before", "before"), ("after_online", "after online"),
                ("after_sleep", "after sleep")]
        x = np.arange(len(keys))
        width = 0.34
        for j, (regime, color) in enumerate([("A", "#2563eb"), ("B", "#dc2626")]):
            vals = [stage2[k][regime]["mse"] for k, _ in keys]
            ax2.bar(x + (j - 0.5) * width, vals, width, color=color,
                    label=f"regime {regime}")
        ax2.set_xticks(x)
        ax2.set_xticklabels([k for _, k in keys])
        ax2.set_ylabel("eval MSE")
        ax2.set_title("Forgetting and sleep consolidation")
        ax2.legend(frameon=False, fontsize=7)
        return self._save(fig, "online_adaptation.png")
