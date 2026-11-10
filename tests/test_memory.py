import torch

from mtdb.core.memory import AttractorMemory, SurpriseGate, WorkingMemory
from mtdb.utils import set_seed


def test_working_memory_write_read(tiny_cfg):
    set_seed(0)
    wm = WorkingMemory(tiny_cfg.memory, dim=16)
    assert wm.n_filled == 0
    c = torch.randn(16)
    slot = wm.write(c)
    assert slot == 0 and wm.n_filled == 1
    read, hit, ent = wm.read(c)
    assert read.shape == (16,)
    assert hit > 0.99
    wm.write(torch.randn(16))
    assert wm.n_filled == 2


def test_attractor_retrieval(tiny_cfg):
    set_seed(0)
    mem = AttractorMemory(tiny_cfg.memory, dim=16)
    patterns = torch.randn(8, 16)
    for p in patterns:
        mem.write(p)
    assert mem.count == 8
    read, hit, idx = mem.read(patterns[3])
    assert torch.cosine_similarity(read, patterns[3], dim=0) > 0.99
    # near-duplicate is skipped
    before = mem.n_writes
    mem.write(patterns[3] * 1.001)
    assert mem.n_writes == before
    stats = mem.retrieval_accuracy(8, 0.2)
    assert stats["cosine"] > stats["chance_cosine"]


def test_surprise_gate(tiny_cfg):
    gate = SurpriseGate(tiny_cfg.memory, warmup=2)
    surprises = []
    for i in range(200):
        raw = 0.1 + 0.01 * (i % 5)
        s, flag = gate.observe(raw)
        surprises.append(s)
    spike, flag = gate.observe(1.0)
    assert spike > 2.0 and flag is True
    frozen, _ = gate.evaluate(1.0)
    assert frozen > 2.0
    assert gate.normalize(1.0) == frozen
