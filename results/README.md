# Recorded results

These JSON files are the raw outputs of the runs reported in the repository
README. They are committed so the numbers and plots are auditable without
rerunning training.

| folder | preset | command |
|---|---|---|
| `demo_standard/` | 2048 units, full pipeline | `python scripts/train.py --config configs/standard.yaml --run-name demo_standard` |
| `demo_debug/` | 768 units, quick validation | `python scripts/train.py --config configs/debug.yaml --run-name demo_debug` |

Each `results.json` contains:

- stage-1 prediction metrics, baselines and anomaly ROC inputs;
- stage-2 online-adaptation metrics and sleep-consolidation statistics;
- the aggressive-forgetting stress test;
- delayed-recall accuracy vs delay;
- ablation table.

Model checkpoints and per-run plot folders are **not** committed to keep the
repository small; run the commands above to regenerate them. The five figures
shown in the README live in [`../assets/`](../assets/).
