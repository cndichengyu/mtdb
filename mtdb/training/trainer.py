"""Three-stage training pipeline for the MTDB base.

Stage 1  Unsupervised predictive-coding pretraining
         - drive the base with the sensory stream, learn only self-prediction
           and homeostatic set-points (local rules)
         - fit the upper read-out with ridge least squares (no backprop)
Stage 2  Online task learning
         - adapt the read-out with RLS and, optionally, recurrent weights with
           the three-factor local rule, while the surprise gate writes memory
Stage 3  Sleep replay / consolidation
         - replay old+new experience, build attractor prototypes, prune and
           renormalise synapses, jointly re-consolidate the read-out

Every metric is computed with ``learn=False`` so that evaluation is frozen and
reproducible (except episodic memory reads/writes, which are inference-time
operations and can be toggled per experiment).
"""

from __future__ import annotations

import copy
import json
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from ..baselines.simple import LinearAR, evaluate_direct, mean_predictor, persistence
from ..config import ExperimentConfig
from ..core.consolidation import ReplayBuffer, SleepConsolidator
from ..core.system import MTDBSystem
from ..eval.metrics import (
    auc_roc,
    masked_accuracy,
    mse,
    nmse,
)
from ..tasks.synthetic import (
    FEATURE_DIM,
    DelayedRecallWorld,
    MultimodalWorld,
    make_prediction_set,
    make_recall_set,
)
from ..utils import get_logger, make_run_dir, resolve_device
from .readout import RidgeReadout, RLSReadout


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, float):
        return None if (np.isnan(obj) or np.isinf(obj)) else obj
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if torch.is_tensor(obj):
        return obj.detach().cpu().tolist()
    return obj


class MTDBTrainer:
    def __init__(
        self,
        cfg: ExperimentConfig,
        run_dir: Optional[str | Path] = None,
        logger=None,
        streams: Optional[Dict[str, List[dict]]] = None,
    ):
        self.cfg = cfg
        self.device = resolve_device(cfg.device)
        self.logger = logger or get_logger()
        self.run_dir = Path(run_dir) if run_dir else make_run_dir(cfg.out_dir, cfg.run_name)
        self.run_dir.mkdir(parents=True, exist_ok=True)

        seed = int(cfg.seed)
        # --- training / evaluation streams --------------------------------
        if streams is None:
            self.task_name = "synthetic-multimodal"
            self.world_train = MultimodalWorld(
                horizon=cfg.train.horizon,
                anomaly_rate=0.0,
                seed=cfg.task.seed_train + seed,
            )
            self.train_A = make_prediction_set(
                self.world_train, cfg.train.n_train, cfg.train.seq_len, regime="A"
            )
            self.train_B = make_prediction_set(
                self.world_train, cfg.train.n_train, cfg.train.seq_len, regime="B"
            )
            self.eval_A = make_prediction_set(
                MultimodalWorld(
                    horizon=cfg.train.horizon,
                    anomaly_rate=0.0,
                    seed=cfg.task.seed_eval + seed,
                ),
                cfg.train.n_eval,
                cfg.train.seq_len,
                regime="A",
            )
            self.eval_B = make_prediction_set(
                MultimodalWorld(
                    horizon=cfg.train.horizon,
                    anomaly_rate=0.0,
                    seed=cfg.task.seed_eval + seed + 1,
                ),
                cfg.train.n_eval,
                cfg.train.seq_len,
                regime="B",
            )
            self.eval_anom = make_prediction_set(
                MultimodalWorld(
                    horizon=cfg.train.horizon,
                    anomaly_rate=cfg.task.anomaly_rate,
                    anomaly_window=cfg.task.anomaly_window,
                    seed=cfg.task.seed_eval + seed + 2,
                ),
                cfg.train.n_eval,
                cfg.train.seq_len,
                regime="A",
                anomalies=True,
            )
            self.input_dim = FEATURE_DIM
        else:
            required = {"train_A", "eval_A"}
            missing = required - set(streams.keys())
            if missing:
                raise KeyError(f"streams is missing required keys: {sorted(missing)}")
            self.task_name = "custom-stream"
            self.train_A = list(streams["train_A"])
            self.eval_A = list(streams["eval_A"])
            self.train_B = list(streams.get("train_B") or self.train_A)
            self.eval_B = list(streams.get("eval_B") or self.eval_A)
            self.eval_anom = list(streams.get("eval_anom") or [])
            if not self.train_A or not self.eval_A:
                raise ValueError("streams['train_A'] and streams['eval_A'] must be non-empty")
            self.input_dim = int(self.train_A[0]["x"].shape[-1])
            stream_shapes_ok = all(
                seq["x"].dim() == 2
                and seq["x"].shape[-1] == self.input_dim
                and seq["y"].shape[-1] == self.input_dim
                for seq in self.train_A + self.eval_A
            )
            if not stream_shapes_ok:
                raise ValueError("all sequences must have x/y of shape [T, input_dim]")
            self.logger.info(
                "Custom streams: %d train / %d eval sequences, input_dim=%d, "
                "regime-B=%s, anomaly-eval=%s",
                len(self.train_A), len(self.eval_A), self.input_dim,
                bool(streams.get("train_B") or streams.get("eval_B")),
                bool(self.eval_anom),
            )

        self.buffer = ReplayBuffer(cfg.consolidation)
        self.system = self._build_system(cfg, seed=seed)
        self.readout = RidgeReadout(
            self.system.spec.total_dim, self.input_dim, cfg.readout.ridge_lambda
        )
        self.readout_online: Optional[RLSReadout] = None
        self.results: Dict[str, object] = {
            "config": cfg.to_dict(),
            "device": str(self.device),
            "interface_dim": self.system.spec.total_dim,
            "n_parameters": self.system.n_parameters(),
            "n_units": self.system.dynamics.n_units,
            "input_dim": self.input_dim,
            "task": self.task_name,
            "horizon": cfg.train.horizon,
            "seq_len": cfg.train.seq_len,
        }

    # ------------------------------------------------------------------ setup
    def _build_system(self, cfg: ExperimentConfig, seed: int) -> MTDBSystem:
        return MTDBSystem(
            cfg,
            input_dim=self.input_dim,
            upper_dim=self.input_dim,
            seed=seed,
            device=self.device,
        )

    # --------------------------------------------------------------- helpers
    @staticmethod
    def _predict(readout, Z: torch.Tensor) -> torch.Tensor:
        if hasattr(readout, "predict_batch"):
            return readout.predict_batch(Z)
        return readout.predict(Z)

    def _collect_features(
        self,
        seqs: List[dict],
        system: Optional[MTDBSystem] = None,
        learn: bool = True,
        allow_memory_write: bool = True,
        buffer: Optional[ReplayBuffer] = None,
        regime: int = 0,
        stride: int = 48,
    ) -> Dict[str, object]:
        system = system or self.system
        X: List[torch.Tensor] = []
        Y: List[torch.Tensor] = []
        surprises: List[float] = []
        gate_count = 0
        window = deque(maxlen=self.cfg.consolidation.window_len)
        for seq in seqs:
            x, y = seq["x"], seq["y"]
            system.reset_state()
            window.clear()
            T = x.shape[0]
            for t in range(T):
                window.append(x[t])
                out = system.step(x[t], learn=learn, allow_memory_write=allow_memory_write)
                X.append(out.packet.z)
                Y.append(y[t])
                surprises.append(out.surprise)
                gate_count += int(out.gate)
                if (
                    buffer is not None
                    and len(window) == window.maxlen
                    and (t + 1) % stride == 0
                ):
                    buffer.add(torch.stack(list(window)), out.surprise, regime=regime)
        return {
            "X": torch.stack(X),
            "Y": torch.stack(Y),
            "surprises": np.asarray(surprises, dtype=np.float32),
            "gate_count": gate_count,
            "n_steps": len(X),
        }

    def _eval_prediction(
        self,
        seqs: List[dict],
        system: Optional[MTDBSystem] = None,
        readout=None,
        allow_memory_write: bool = True,
    ) -> Dict[str, object]:
        system = system or self.system
        readout = readout or self.readout
        memory_snapshot = self._snapshot_memory(system)
        per_seq: List[float] = []
        preds: List[torch.Tensor] = []
        targets: List[torch.Tensor] = []
        surprises: List[float] = []
        for seq in seqs:
            x = seq["x"]
            system.reset_state()
            zs = []
            for t in range(x.shape[0]):
                out = system.step(x[t], learn=False, allow_memory_write=allow_memory_write)
                zs.append(out.packet.z)
                surprises.append(out.surprise)
            Z = torch.stack(zs)
            P = self._predict(readout, Z)
            targets.append(seq["y"])
            preds.append(P)
            per_seq.append(mse(P, seq["y"]))
        self._restore_memory(system, memory_snapshot)
        P_all = torch.cat(preds, dim=0)
        Y_all = torch.cat(targets, dim=0)
        return {
            "mse": float(np.mean(per_seq)),
            "mse_std": float(np.std(per_seq)),
            "nmse": nmse(P_all, Y_all),
            "per_sequence": [float(v) for v in per_seq],
            "surprise_mean": float(np.mean(surprises)) if surprises else 0.0,
        }

    def _eval_anomaly(self, seqs: Optional[List[dict]] = None) -> Dict[str, object]:
        seqs = seqs or self.eval_anom
        if not seqs:
            return {}
        scores: List[float] = []
        change_scores: List[float] = []
        labels: List[bool] = []
        series = None
        prev = None
        for si, seq in enumerate(seqs):
            x = seq["x"]
            self.system.reset_state()
            prev = None
            s_seq, b_seq = [], []
            for t in range(x.shape[0]):
                out = self.system.step(x[t], learn=False, allow_memory_write=False)
                s_seq.append(out.surprise)
                b_seq.append(0.0 if prev is None else float((x[t] - prev).norm()))
                prev = x[t]
            scores.extend(s_seq)
            change_scores.extend(b_seq)
            labels.extend([bool(v) for v in seq["anomaly"].tolist()])
            if si == 0:
                series = {
                    "surprise": list(map(float, s_seq)),
                    "change": list(map(float, b_seq)),
                    "label": [int(v) for v in seq["anomaly"].tolist()],
                }
        labels_np = np.asarray(labels, dtype=bool)
        scores_np = np.asarray(scores)
        change_np = np.asarray(change_scores)
        # temporal smoothing: a surprise event is a short episode, not a single tick
        win = 3
        pad = win // 2
        scores_pad = np.pad(scores_np, (pad, pad), mode="edge")
        scores_smooth = np.asarray(
            [scores_pad[i : i + win].mean() for i in range(scores_np.size)]
        )
        change_pad = np.pad(change_np, (pad, pad), mode="edge")
        change_smooth = np.asarray(
            [change_pad[i : i + win].mean() for i in range(change_np.size)]
        )
        return {
            "auc": auc_roc(scores_np, labels_np),
            "auc_smoothed": auc_roc(scores_smooth, labels_np),
            "auc_input_change": auc_roc(change_np, labels_np),
            "auc_input_change_smoothed": auc_roc(change_smooth, labels_np),
            "mean_surprise_normal": float(scores_np[~labels_np].mean())
            if (~labels_np).any()
            else 0.0,
            "mean_surprise_anomaly": float(scores_np[labels_np].mean())
            if labels_np.any()
            else 0.0,
            "n_anomaly_steps": int(labels_np.sum()),
            "gate_mean": float(self.system.gate.mean),
            "gate_std": float(self.system.gate.std),
            "series": series,
            "scores": scores_np.tolist(),
            "scores_smoothed": scores_smooth.tolist(),
            "change_scores": change_np.tolist(),
            "labels": labels_np.astype(int).tolist(),
        }

    # ---------------------------------------------------------------- stage 1
    def stage1_pretrain(self) -> Dict[str, object]:
        cfg = self.cfg
        t0 = time.time()
        history = []
        self.logger.info("Stage 1: unsupervised predictive-coding pretraining "
                         "(N=%d, K=%d, seq_len=%d)", cfg.dynamics.n_units,
                         cfg.dynamics.in_degree, cfg.train.seq_len)
        self.system.plasticity_enabled = False
        for epoch in range(cfg.train.epochs):
            collected = self._collect_features(
                self.train_A,
                learn=True,
                allow_memory_write=True,
                buffer=self.buffer,
                regime=0,
                stride=cfg.train.replay_stride,
            )
            self.readout.fit(collected["X"], collected["Y"])
            ev = self._eval_prediction(self.eval_A)
            history.append(
                {
                    "epoch": epoch + 1,
                    "train_steps": int(collected["n_steps"]),
                    "eval_mse": ev["mse"],
                    "eval_nmse": ev["nmse"],
                    "surprise_mean": float(collected["surprises"].mean()),
                    "gate_count": int(collected["gate_count"]),
                }
            )
            self.logger.info(
                "  epoch %d/%d | eval MSE %.5f (NMSE %.3f) | gates %d | %.1fs",
                epoch + 1, cfg.train.epochs, ev["mse"], ev["nmse"],
                collected["gate_count"], time.time() - t0,
            )

        final = self._eval_prediction(self.eval_A)
        baselines = self._compute_baselines()
        anomaly = self._eval_anomaly()
        self.logger.info(
            "Stage 1 done | MTDB MSE %.5f | persistence %.5f | linear-AR %.5f",
            final["mse"], baselines["persistence"]["mse"], baselines["linear_ar"]["mse"],
        )
        if anomaly:
            self.logger.info(
                "  anomaly AUC %.3f (input-change %.3f)",
                anomaly["auc"], anomaly["auc_input_change"],
            )
        result = {
            "history": history,
            "prediction": final,
            "baselines": baselines,
            "anomaly": anomaly,
            "system_stats": self.system.stats(),
            "time_s": time.time() - t0,
        }
        self.results["stage1"] = result
        return result

    def _compute_baselines(self) -> Dict[str, object]:
        persistence_preds = [persistence(seq, self.cfg.train.horizon) for seq in self.eval_A]
        persistence_res = evaluate_direct(persistence_preds, self.eval_A)
        mean_vec = mean_predictor(self.train_A)
        mean_preds = [mean_vec.expand_as(seq["x"]) for seq in self.eval_A]
        mean_res = evaluate_direct(mean_preds, self.eval_A)
        ar = LinearAR(order=4, ridge_lambda=self.cfg.readout.ridge_lambda).fit(self.train_A)
        ar_res = ar.evaluate(self.eval_A)
        for res in (persistence_res, mean_res, ar_res):
            res["nmse"] = res["mse"] / self._target_variance()
        return {
            "persistence": persistence_res,
            "mean": mean_res,
            "linear_ar": ar_res,
        }

    def _target_variance(self) -> float:
        Y = torch.cat([seq["y"] for seq in self.eval_A], dim=0)
        return max(float(Y.var()), 1e-8)

    # ---------------------------------------------------------------- stage 2
    def stage2_online_adaptation(self) -> Dict[str, object]:
        cfg = self.cfg
        if not cfg.train.run_stage2 or cfg.train.online_epochs <= 0:
            return {}
        t0 = time.time()
        self.logger.info("Stage 2: online learning on regime B "
                         "(RLS readout + three-factor plasticity)")
        self.readout_online = RLSReadout(
            self.system.spec.total_dim,
            self.input_dim,
            forget=cfg.readout.rls_forget,
            init_delta=cfg.readout.rls_init_delta,
        )
        self.readout_online.set_weights(
            self.readout.w,
            self.readout.b,
            x_mean=self.readout.x_mean,
            x_std=self.readout.x_std,
            precision=self.readout.precision,
        )
        # snapshot the post-stage-1 base for the forgetting stress experiment
        self._stage1_system_state = copy.deepcopy(self.system.state_dict())
        before = self._eval_pair(self.readout_online)
        self.system.plasticity_enabled = bool(cfg.plasticity.enabled)

        curve = []
        pl_accum: Dict[str, List[float]] = {}
        for epoch in range(cfg.train.online_epochs):
            mses, pl_stats = self._online_pass(self.readout_online, self.train_B, regime=1)
            epoch_eval = self._eval_pair(self.readout_online)
            curve.append(
                {
                    "epoch": epoch + 1,
                    "online_mse": float(np.mean(mses)),
                    "online_mse_last_quarter": float(np.mean(mses[len(mses) // 4 * 3:])),
                    "eval_A_mse": epoch_eval["A"]["mse"],
                    "eval_B_mse": epoch_eval["B"]["mse"],
                }
            )
            for st in pl_stats:
                for k, v in st.items():
                    pl_accum.setdefault(k, []).append(float(v))
            self.logger.info(
                "  epoch %d/%d | online MSE %.5f | readout updates %d | %.1fs",
                epoch + 1, cfg.train.online_epochs, float(np.mean(mses)),
                self.readout_online.n_updates, time.time() - t0,
            )

        after_online = self._eval_pair(self.readout_online)
        sleep_stats = SleepConsolidator(
            cfg.consolidation, self.system, horizon=cfg.train.horizon
        ).run(
            self.buffer,
            readout=self.readout_online,
            ridge_lambda=cfg.readout.ridge_lambda,
        )
        after_sleep = self._eval_pair(self.readout_online)

        plasticity = {
            k: float(np.mean(v)) for k, v in pl_accum.items()
        }
        result = {
            "before": before,
            "after_online": after_online,
            "after_sleep": after_sleep,
            "adaptation_gain_B": before["B"]["mse"] - after_online["B"]["mse"],
            "forgetting_A": after_online["A"]["mse"] - before["A"]["mse"],
            "sleep_gain_A": after_online["A"]["mse"] - after_sleep["A"]["mse"],
            "sleep_gain_B": after_online["B"]["mse"] - after_sleep["B"]["mse"],
            "sleep": sleep_stats,
            "plasticity": plasticity,
            "readout_online": {
                "theta_norm": float(self.readout_online.theta.norm()),
                "P_mean_diag": float(self.readout_online.P.diagonal().mean()),
                "n_updates": int(self.readout_online.n_updates),
            },
            "online_curve": curve,
            "buffer": {
                "size": len(self.buffer),
                "per_regime": self.buffer.n_per_regime(),
                "total_added": self.buffer.total_added,
            },
            "system_stats": self.system.stats(),
            "time_s": time.time() - t0,
        }
        self.logger.info(
            "Stage 2 done | A: %.5f -> %.5f (sleep %.5f) | B: %.5f -> %.5f (sleep %.5f)",
            before["A"]["mse"], after_online["A"]["mse"], after_sleep["A"]["mse"],
            before["B"]["mse"], after_online["B"]["mse"], after_sleep["B"]["mse"],
        )
        self.results["stage2"] = result
        # snapshot the consolidated base (used as the artifact checkpoint)
        self._main_system_state = copy.deepcopy(self.system.state_dict())
        self._main_readout_online_state = copy.deepcopy(self.readout_online.state_dict())
        return result

    def _online_pass(self, readout: RLSReadout, seqs: List[dict], regime: int = 1):
        cfg = self.cfg
        stride = cfg.train.replay_stride
        window = deque(maxlen=cfg.consolidation.window_len)
        mses: List[float] = []
        pl_stats: List[Dict[str, float]] = []
        for seq in seqs:
            x, y = seq["x"], seq["y"]
            self.system.reset_state()
            window.clear()
            T = x.shape[0]
            for t in range(T):
                window.append(x[t])
                out = self.system.step(x[t], learn=True, allow_memory_write=True)
                z = out.packet.z
                yhat = readout.predict(z)
                err = y[t] - yhat
                readout.update(z, y[t])
                if self.system.plasticity_enabled:
                    stats = self.system.apply_error_feedback(
                        err, surprise=out.surprise, reward=-float(err.norm())
                    )
                    if stats:
                        pl_stats.append(stats)
                mses.append(float(err.pow(2).mean()))
                if len(window) == window.maxlen and (t + 1) % stride == 0:
                    self.buffer.add(torch.stack(list(window)), out.surprise, regime=regime)
        return mses, pl_stats

    def _eval_pair(self, readout) -> Dict[str, object]:
        return {
            "A": self._eval_prediction(self.eval_A, readout=readout),
            "B": self._eval_prediction(self.eval_B, readout=readout),
        }

    # ------------------------------------------------- forgetting stress test
    @torch.no_grad()
    def forgetting_stress(self) -> Dict[str, object]:
        """Deliberately induce catastrophic forgetting, then show sleep recovery.

        The main stage-2 learner uses a conservative effective memory
        (forget=0.9999) and improves both regimes. Here we restore the
        post-stage-1 base and re-run online learning with aggressive forgetting
        (forget=0.999). The read-out drifts away from regime A; the sleep replay
        pass then re-consolidates it on replayed A+B experience.
        """
        cfg = self.cfg
        if (
            not cfg.train.forgetting_stress
            or not hasattr(self, "_stage1_system_state")
            or self.readout_online is None
        ):
            return {}
        t0 = time.time()
        self.logger.info("Forgetting stress test (aggressive RLS forgetting)")
        self.system.load_state_dict(copy.deepcopy(self._stage1_system_state))
        self.system.plasticity_enabled = bool(cfg.plasticity.enabled)
        rls = RLSReadout(
            self.system.spec.total_dim,
            self.input_dim,
            forget=cfg.train.stress_rls_forget,
            init_delta=cfg.readout.rls_init_delta,
        )
        rls.set_weights(
            self.readout.w, self.readout.b,
            x_mean=self.readout.x_mean, x_std=self.readout.x_std,
            precision=self.readout.precision,
        )
        before = self._eval_pair(rls)
        curve = []
        for epoch in range(cfg.train.stress_online_epochs):
            mses, _ = self._online_pass(rls, self.train_B, regime=1)
            ev = self._eval_pair(rls)
            curve.append(
                {
                    "epoch": epoch + 1,
                    "online_mse": float(np.mean(mses)),
                    "eval_A_mse": ev["A"]["mse"],
                    "eval_B_mse": ev["B"]["mse"],
                }
            )
        after_online = self._eval_pair(rls)
        sleep = SleepConsolidator(
            cfg.consolidation, self.system, horizon=cfg.train.horizon
        ).run(self.buffer, readout=rls, ridge_lambda=cfg.readout.ridge_lambda)
        after_sleep = self._eval_pair(rls)
        result = {
            "stress_rls_forget": cfg.train.stress_rls_forget,
            "before": before,
            "after_online": after_online,
            "after_sleep": after_sleep,
            "forgetting_A": after_online["A"]["mse"] - before["A"]["mse"],
            "adaptation_B": before["B"]["mse"] - after_online["B"]["mse"],
            "sleep_gain_A": after_online["A"]["mse"] - after_sleep["A"]["mse"],
            "sleep_gain_B": after_online["B"]["mse"] - after_sleep["B"]["mse"],
            "sleep": sleep,
            "online_curve": curve,
            "time_s": time.time() - t0,
        }
        self.logger.info(
            "Stress done | A: %.5f -> %.5f -> %.5f | B: %.5f -> %.5f -> %.5f",
            before["A"]["mse"], after_online["A"]["mse"], after_sleep["A"]["mse"],
            before["B"]["mse"], after_online["B"]["mse"], after_sleep["B"]["mse"],
        )
        self.results["stage2_stress"] = result
        # restore the consolidated main model so the checkpoint is coherent
        self.system.load_state_dict(copy.deepcopy(self._main_system_state))
        self.readout_online.load_state_dict(copy.deepcopy(self._main_readout_online_state))
        return result

    # ---------------------------------------------------------------- recall
    def run_recall(self) -> Dict[str, object]:
        cfg = self.cfg
        if not cfg.train.run_recall:
            return {}
        if self.task_name != "synthetic-multimodal":
            self.logger.info(
                "Delayed-recall probe skipped for custom streams "
                "(configure task.prediction or use the synthetic preset)."
            )
            return {}
        t0 = time.time()
        self.logger.info("Long-horizon memory test: delayed cue recall")
        world = DelayedRecallWorld(
            n_classes=cfg.task.recall_classes,
            cue_len=cfg.task.recall_cue_len,
            delay=cfg.task.recall_delay,
            seed=cfg.task.seed_train + cfg.seed + 42,
        )
        train = make_recall_set(world, cfg.train.n_recall_train)
        evals = make_recall_set(world, cfg.train.n_recall_eval)

        system = MTDBSystem(
            cfg, input_dim=FEATURE_DIM, upper_dim=FEATURE_DIM,
            seed=cfg.seed + 777, device=self.device,
        )
        system.plasticity_enabled = False
        # unsupervised adaptation of self-prediction to the recall stream
        for _pass in range(2):
            for seq in train:
                x = seq["x"]
                system.reset_state()
                for t in range(x.shape[0]):
                    system.step(x[t], learn=True, allow_memory_write=True)

        collected = self._collect_features(
            train, system=system, learn=False, allow_memory_write=True,
            buffer=None, regime=0,
        )
        readout = RidgeReadout(system.spec.total_dim, FEATURE_DIM,
                               cfg.readout.ridge_lambda).fit(collected["X"], collected["Y"])
        acc_w, curve_w = self._eval_recall(evals, system, readout, writes=True)
        acc_wo, curve_wo = self._eval_recall(evals, system, readout, writes=False)

        # accuracy vs delay: same trained interface, longer/shorter delays
        by_delay = []
        for d in cfg.task.recall_eval_delays:
            if d == cfg.task.recall_delay:
                by_delay.append(
                    {
                        "delay": int(d),
                        "accuracy": acc_w,
                        "accuracy_writes_off": acc_wo,
                        "n_eval": cfg.train.n_recall_eval,
                    }
                )
                continue
            w = DelayedRecallWorld(
                n_classes=cfg.task.recall_classes,
                cue_len=cfg.task.recall_cue_len,
                delay=int(d),
                seed=cfg.task.seed_train + cfg.seed + 1000 + int(d),
            )
            seqs = make_recall_set(w, cfg.train.n_recall_eval)
            a_on, _ = self._eval_recall(seqs, system, readout, writes=True)
            a_off, _ = self._eval_recall(seqs, system, readout, writes=False)
            by_delay.append(
                {
                    "delay": int(d),
                    "accuracy": a_on,
                    "accuracy_writes_off": a_off,
                    "n_eval": cfg.train.n_recall_eval,
                }
            )
        result = {
            "accuracy_with_memory_writes": acc_w,
            "accuracy_without_memory_writes": acc_wo,
            "chance": 1.0 / cfg.task.recall_classes,
            "accuracy_by_t_with_writes": curve_w,
            "accuracy_by_t_without_writes": curve_wo,
            "delay_steps": cfg.task.recall_delay,
            "accuracy_by_delay": by_delay,
            "n_eval": cfg.train.n_recall_eval,
            "n_train": cfg.train.n_recall_train,
            "wm_filled": float(system.wm.n_filled if system.wm is not None else 0),
            "time_s": time.time() - t0,
        }
        self.logger.info(
            "Recall accuracy (delay=%d): %.3f (memory writes on) / %.3f (off) | chance %.3f",
            cfg.task.recall_delay, acc_w, acc_wo, result["chance"],
        )
        self.results["recall"] = result
        return result

    @staticmethod
    def _snapshot_memory(system: MTDBSystem) -> Dict[str, object]:
        snap: Dict[str, object] = {}
        if system.wm is not None:
            snap["wm"] = {k: v.clone() for k, v in system.wm.state_dict().items()}
        if system.attractor is not None:
            snap["attractor"] = {
                k: (v.clone() if torch.is_tensor(v) else v)
                for k, v in system.attractor.state_dict().items()
            }
        return snap

    @staticmethod
    def _restore_memory(system: MTDBSystem, snap: Dict[str, object]) -> None:
        if "wm" in snap and system.wm is not None:
            system.wm.load_state_dict(snap["wm"])
        if "attractor" in snap and system.attractor is not None:
            system.attractor.load_state_dict(snap["attractor"])

    def _eval_recall(
        self, seqs: List[dict], system: MTDBSystem, readout, writes: bool
    ):
        snap = self._snapshot_memory(system)
        per_t_correct = None
        accs = []
        for seq in seqs:
            self._restore_memory(system, snap)
            x, y = seq["x"], seq["y"]
            system.reset_state()
            zs = []
            for t in range(x.shape[0]):
                out = system.step(x[t], learn=False, allow_memory_write=writes)
                zs.append(out.packet.z)
            Z = torch.stack(zs)
            pred = self._predict(readout, Z)
            mask = seq["eval_mask"]
            acc = masked_accuracy(pred, y, mask)
            accs.append(acc)
            correct = (pred.argmax(dim=-1) == y.argmax(dim=-1)).float()
            if per_t_correct is None:
                per_t_correct = correct.numpy()
            else:
                per_t_correct += correct.numpy()
        curve = (per_t_correct / max(1, len(seqs))).tolist() if per_t_correct is not None else []
        return float(np.mean(accs)), curve

    # -------------------------------------------------------------- ablations
    def run_ablations(self) -> List[Dict[str, object]]:
        cfg = self.cfg
        if not cfg.train.ablations:
            return []
        self.logger.info("Ablation study")
        variants = [
            ("full", lambda c: None),
            ("no_phase", lambda c: setattr(c.phase, "enabled", False)),
            ("no_memory", lambda c: setattr(c.memory, "enabled", False)),
            (
                "single_tau",
                lambda c: (
                    setattr(c.dynamics, "tau_groups_ms", (100.0, 100.0, 100.0)),
                    setattr(c.dynamics, "group_probs", (1.0, 0.0, 0.0)),
                ),
            ),
            (
                "fast_only",
                lambda c: (
                    setattr(c.dynamics, "tau_groups_ms", (10.0, 10.0, 10.0)),
                    setattr(c.dynamics, "group_probs", (1.0, 0.0, 0.0)),
                ),
            ),
        ]
        rows: List[Dict[str, object]] = []
        n_train = min(cfg.train.ablation_n_train, len(self.train_A))
        for name, mutate in variants:
            t0 = time.time()
            vcfg = copy.deepcopy(cfg)
            vcfg.train.epochs = min(cfg.train.ablation_epochs, cfg.train.epochs)
            vcfg.train.ablations = False
            vcfg.train.run_stage2 = False
            vcfg.train.run_recall = False
            vcfg.consolidation.enabled = False
            mutate(vcfg)
            system = self._build_system(vcfg, seed=cfg.seed + 13)
            readout = RidgeReadout(system.spec.total_dim, self.input_dim,
                                   vcfg.readout.ridge_lambda)
            for _epoch in range(vcfg.train.epochs):
                collected = self._collect_features(
                    self.train_A[:n_train], system=system, learn=True,
                    allow_memory_write=True, buffer=None, regime=0,
                )
                readout.fit(collected["X"], collected["Y"])
            ev = self._eval_prediction(self.eval_A, system=system, readout=readout)
            row = {
                "name": name,
                "mse": ev["mse"],
                "nmse": ev["nmse"],
                "params": system.n_parameters(),
                "interface_dim": system.spec.total_dim,
                "time_s": time.time() - t0,
            }
            rows.append(row)
            self.logger.info(
                "  %-10s MSE %.5f (NMSE %.3f) | params %d | %.1fs",
                name, row["mse"], row["nmse"], row["params"], row["time_s"],
            )
        self.results["ablations"] = rows
        return rows

    # ---------------------------------------------------------------- artifact
    def save(self) -> Path:
        results_path = self.run_dir / "results.json"
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(_jsonable(self.results), f, ensure_ascii=False, indent=2)
        self.cfg.save(self.run_dir / "config_resolved.yaml")
        if self.cfg.train.save_model:
            ckpt = {
                "config": self.cfg.to_dict(),
                "input_dim": self.input_dim,
                "task": self.task_name,
                "system": self.system.state_dict(),
                "readout": self.readout.state_dict(),
                "results": _jsonable(self.results),
            }
            if self.readout_online is not None:
                ckpt["readout_online"] = self.readout_online.state_dict()
            torch.save(ckpt, self.run_dir / "checkpoint.pt")
        # a clean post-stage-1 checkpoint makes the unsupervised readout and
        # anomaly metrics reproducible independently of later online adaptation
        if self.cfg.train.save_model and hasattr(self, "_stage1_system_state"):
            stage1_ckpt = {
                "config": self.cfg.to_dict(),
                "input_dim": self.input_dim,
                "task": self.task_name,
                "system": self._stage1_system_state,
                "readout": self.readout.state_dict(),
                "results": _jsonable(self.results),
            }
            torch.save(stage1_ckpt, self.run_dir / "checkpoint_stage1.pt")
        self.logger.info("Artifacts written to %s", self.run_dir)
        return self.run_dir

    # ------------------------------------------------------------------- run
    def run(self) -> Dict[str, object]:
        self.stage1_pretrain()
        self.stage2_online_adaptation()
        self.forgetting_stress()
        self.run_recall()
        self.run_ablations()
        self.save()
        return self.results
