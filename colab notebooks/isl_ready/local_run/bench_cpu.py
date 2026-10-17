"""Time real training steps of the patched model on this CPU (2 threads, like a Colab CPU runtime).
Usage: python bench_cpu.py <local root with content/ProgressiveTransformersSLP and content/isl_code>"""
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

root = Path(sys.argv[1]) / "content"
sys.path[:0] = [str(root / "ProgressiveTransformersSLP"), str(root / "isl_code")]
from constants import PAD_TOKEN
from loss import RegLoss
from model import build_model
from vocabulary import Vocabulary
import isign_train as it

threads = int(sys.argv[2]) if len(sys.argv) > 2 else 2
torch.set_num_threads(threads)
cfg = yaml.safe_load((root / "ProgressiveTransformersSLP" / "Configs" / "Base.yaml").read_text())
cfg["model"]["trg_size"] = 225
cfg["training"]["use_cuda"] = False
vocab = Vocabulary(tokens=[f"w{i}" for i in range(30000)])        # an iSign-sized vocabulary
model = build_model(cfg, src_vocab=vocab, trg_vocab=Vocabulary(tokens=[]))
n_params = sum(p.numel() for p in model.parameters())
n_compute = sum(p.numel() for n, p in model.named_parameters() if "src_embed" not in n)
print(f"threads {threads} | parameters {n_params / 1e6:.1f} M (excluding the word-embedding table {n_compute / 1e6:.1f} M)")
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
crit = RegLoss(cfg)
rng = np.random.RandomState(0)
from types import SimpleNamespace
from batch import Batch


def fake_batch(B, T, S=14):
    src = torch.tensor(rng.randint(4, 30000, size=(B, S)), dtype=torch.long)
    trg = torch.tensor(rng.randn(B, T, 226), dtype=torch.float32)
    return SimpleNamespace(src=(src, torch.full((B,), S)), trg=trg, frame_lengths=torch.full((B,), T),
                           landmark_valid=torch.ones(B, T, 75, dtype=torch.bool), file_paths=["x"] * B)


for B, T in ((8, 200), (8, 100), (4, 200)):
    raw = fake_batch(B, T)
    batch = Batch(torch_batch=raw, pad_index=vocab.stoi[PAD_TOKEN], model=model)
    it.train_step(model, batch, crit, opt, 5.0)                 # warm-up
    t0 = time.time()
    n = 3
    for _ in range(n):
        it.train_step(model, batch, crit, opt, 5.0)
    dt = (time.time() - t0) / n
    print(f"batch {B} x {T} frames: {dt:.2f} s/step -> {B * T / dt:,.0f} frames/s")
