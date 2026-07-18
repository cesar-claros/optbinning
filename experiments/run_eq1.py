"""Experiment E-Q1 — quantizer feasibility (P8 note).

Small convolutional autoencoder on MNIST/CIFAR-10 with three discrete
bottlenecks at a matched code budget (default 8^4 = 4096 codes): vanilla
VQ-EMA, FSQ (fixed uniform levels), and the OT learned-knot scalar
quantizer. Decision rule from the P8 note: the OT arm is interesting if
it matches FSQ reconstruction while beating VQ-EMA on utilization and
seed stability -- then the guarantees (structural utilization floor,
interval code map, audit table) come free.

HPC:
    python experiments/run_eq1.py -m dataset=mnist,cifar10 \
        'seed=range(0,3)' device=cuda out=outputs/eq1
"""

# Cesar Claros <cesar.claros@outlook.com>
# Copyright (C) 2026

import logging
import sys
import time

from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import hydra                                            # noqa: E402
import torch                                            # noqa: E402
from omegaconf import DictConfig                        # noqa: E402
from torch import nn                                    # noqa: E402

from experiments.datasets import DATA_DIR               # noqa: E402
from experiments.common import save_results             # noqa: E402
from experiments.paperq.quantizers import (FSQ, OTQuantizer,  # noqa: E402
                                           VQEMA)

logger = logging.getLogger(__name__)


class ConvAE(nn.Module):
    """Small conv autoencoder; the quantizer acts on per-position
    latent vectors (batch * H/4 * W/4 tokens of dimension D)."""

    def __init__(self, in_channels, d_channels, quantizer):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(128, d_channels, 1))
        self.decoder = nn.Sequential(
            nn.Conv2d(d_channels, 128, 1), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(64, in_channels, 4, 2, 1), nn.Sigmoid())
        self.quantizer = quantizer

    def tokens(self, x):
        z = self.encoder(x)                              # (B, D, h, w)
        b, d, h, w = z.shape
        return z.permute(0, 2, 3, 1).reshape(-1, d), (b, d, h, w)

    def forward(self, x, **q_kwargs):
        flat, (b, d, h, w) = self.tokens(x)
        zq, q_loss = self.quantizer(flat, **q_kwargs)
        zq = zq.reshape(b, h, w, d).permute(0, 3, 1, 2)
        return self.decoder(zq), q_loss


def _loaders(name, batch_size):
    from torchvision import datasets as tvd
    from torchvision import transforms
    root = str(DATA_DIR / "torchvision")
    cls = {"mnist": tvd.MNIST, "cifar10": tvd.CIFAR10}[name]
    tf = transforms.ToTensor()
    train = cls(root, train=True, download=True, transform=tf)
    test = cls(root, train=False, download=True, transform=tf)
    return (torch.utils.data.DataLoader(train, batch_size, shuffle=True,
                                        num_workers=2, drop_last=True),
            torch.utils.data.DataLoader(test, batch_size))


def _build(arm, cfg):
    d, m = cfg.d_channels, cfg.n_levels
    if arm == "vq":
        return VQEMA(m ** d, d, decay=cfg.vq_decay, beta=cfg.vq_beta)
    if arm == "fsq":
        return FSQ(d, m)
    if arm == "ot":
        return OTQuantizer(d, m, sinkhorn_iters=cfg.sinkhorn_iters,
                           commit_beta=cfg.get("ot_commit", 0.25))
    raise ValueError("unknown arm: {}".format(arm))


def _train_eval(arm, cfg, device):
    torch.manual_seed(cfg.seed)                # identical AE init per arm
    in_ch = 1 if cfg.dataset == "mnist" else 3
    net = ConvAE(in_ch, cfg.d_channels, _build(arm, cfg)).to(device)
    train_loader, test_loader = _loaders(cfg.dataset, cfg.batch_size)
    optim = torch.optim.Adam(net.parameters(), lr=cfg.lr)

    start = time.perf_counter()
    net.train()
    for epoch in range(cfg.epochs):
        frac = epoch / max(cfg.epochs - 1, 1)
        eps = cfg.eps_start * (cfg.eps_end / cfg.eps_start) ** frac
        q_kwargs = {"eps": eps} if arm == "ot" else {}
        for x, _ in train_loader:
            x = x.to(device)
            recon, q_loss = net(x, **q_kwargs)
            loss = nn.functional.mse_loss(recon, x) + q_loss
            optim.zero_grad()
            loss.backward()
            optim.step()
    fit_time = time.perf_counter() - start

    net.eval()
    n_codes = net.quantizer.n_codes
    usage = torch.zeros(n_codes, device=device)
    se_hard, se_soft, n_pix = 0.0, 0.0, 0
    with torch.no_grad():
        for x, _ in test_loader:
            x = x.to(device)
            recon, _ = net(x)                    # eval path (hard codes)
            se_hard += float((recon - x).pow(2).sum())
            n_pix += x.numel()
            flat, _ = net.tokens(x)
            usage += torch.bincount(net.quantizer.codes(flat),
                                    minlength=n_codes).float()
            if arm == "ot":
                net.train()                      # soft path, no grad
                recon_s, _ = net(x, eps=cfg.eps_end)
                net.eval()
                se_soft += float((recon_s - x).pow(2).sum())

    mse = se_hard / n_pix
    p = (usage / usage.sum()).cpu().numpy()
    ent = -np.sum(p[p > 0] * np.log(p[p > 0]))
    row = dict(arm=arm, n_codes=n_codes, mse=mse,
               psnr=float(-10 * np.log10(mse)),
               mse_soft=se_soft / n_pix if arm == "ot" else np.nan,
               perplexity=float(np.exp(ent)),
               dead_frac=float((p == 0).mean()),
               fit_time=fit_time)
    audit = (net.quantizer.audit().cpu().numpy() if arm == "ot" else None)
    return row, audit


def run(cfg):
    device = torch.device(cfg.device)
    rows, audit_rows = [], []
    for arm in cfg.arms:
        try:
            row, audit = _train_eval(arm, cfg, device)
            logger.info("%s: mse=%.5f psnr=%.2f perplexity=%.0f/%d "
                        "dead=%.3f", arm, row["mse"], row["psnr"],
                        row["perplexity"], row["n_codes"],
                        row["dead_frac"])
        except Exception:
            logger.exception("arm %s failed", arm)
            row, audit = dict(arm=arm, mse=np.nan), None
        row.update(dataset=cfg.dataset, seed=cfg.seed)
        rows.append(row)
        if audit is not None:
            for ch in range(audit.shape[0]):
                for k in range(audit.shape[1]):
                    audit_rows.append(dict(
                        dataset=cfg.dataset, seed=cfg.seed, channel=ch,
                        edge=k, value=float(audit[ch, k])))

    out = Path(cfg.out)
    tag = "eq1_{}_{}".format(cfg.dataset, cfg.seed)
    paths = [save_results(rows, out / tag)]
    if audit_rows:
        paths.append(save_results(audit_rows, out / (tag + "_audit")))
    logger.info("EQ1: wrote %s", paths[0])
    return paths


@hydra.main(version_base=None, config_path="../conf", config_name="eq1")
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
