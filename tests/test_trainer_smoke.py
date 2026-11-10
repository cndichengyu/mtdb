import json

from mtdb.training.trainer import MTDBTrainer


def test_tiny_trainer_end_to_end(tiny_cfg, tmp_path):
    tiny_cfg.out_dir = str(tmp_path)
    tiny_cfg.run_name = "smoke"
    tiny_cfg.train.run_stage2 = True
    tiny_cfg.train.online_epochs = 1
    tiny_cfg.train.run_recall = True
    tiny_cfg.train.ablations = False
    tiny_cfg.train.plot = False
    tiny_cfg.train.save_model = False
    trainer = MTDBTrainer(tiny_cfg)
    results = trainer.run()
    assert "stage1" in results
    assert "stage2" in results
    assert "recall" in results
    assert results["stage1"]["prediction"]["mse"] > 0
    assert 0.0 <= results["recall"]["accuracy_with_memory_writes"] <= 1.0
    path = trainer.run_dir / "results.json"
    assert path.exists()
    data = json.loads(path.read_text())
    assert data["stage1"]["prediction"]["mse"] > 0
