"""
ex2.py — Replication of Figure 2 (MNIST MLP + dropout) from:
  "Adam: A Method for Stochastic Optimization"
  Kingma & Ba, ICLR 2015

Architecture: 784 → 1000 (ReLU, Dropout) → 1000 (ReLU, Dropout) → 10
Regularization: 50% dropout only (no L2 weight decay)
Objective:      cross-entropy (NLL)
LR schedule:    alpha_t = base_lr / sqrt(epoch)
Baselines:      SGD+Nesterov, AdaGrad, RMSprop, SGD  (torch.optim)
Adam:           hand-rolled using only torch.autograd (same as ex1.py)

Usage:
    python ex2.py                    # reads hyperparameters.yaml next to this file
    python ex2.py --cfg my_hps.yaml  # override config path
"""

import argparse
import math
import os
import pathlib
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets.mnist import read_image_file, read_label_file
import yaml


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def load_cfg(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Hand-rolled Adam  (torch.autograd only — no torch.optim.Adam)
# Identical implementation to ex1.py for consistency
# ---------------------------------------------------------------------------

class AdamCustom:
    """
    Adam optimiser — Algorithm 1 of Kingma & Ba (2015).
    Uses only parameter .grad tensors; no torch.optim internals.
    """

    def __init__(self, params, lr: float, beta1=0.9, beta2=0.999, eps=1e-8):
        self.params = list(params)
        self.lr     = lr
        self.beta1  = beta1
        self.beta2  = beta2
        self.eps    = eps
        self.t      = 0
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]

    def set_lr(self, lr: float):
        self.lr = lr

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()

    @torch.no_grad()
    def step(self):
        self.t += 1
        beta1, beta2, eps = self.beta1, self.beta2, self.eps
        bc1 = 1.0 - beta1 ** self.t
        bc2 = 1.0 - beta2 ** self.t

        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad
            self.m[i] = beta1 * self.m[i] + (1.0 - beta1) * g
            self.v[i] = beta2 * self.v[i] + (1.0 - beta2) * g * g
            m_hat = self.m[i] / bc1
            v_hat = self.v[i] / bc2
            p.data -= self.lr * m_hat / (torch.sqrt(v_hat) + eps)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    """
    Two fully-connected hidden layers with ReLU + Dropout.
    Architecture matches paper: 784 → 1000 → 1000 → 10
    Dropout is applied AFTER each hidden activation (training only).
    """

    def __init__(self, input_dim: int, hidden_units: int,
                 num_classes: int, dropout_rate: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_units),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_units, num_classes),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def get_mnist_loader(data_path: str, batch_size: int, train: bool) -> DataLoader:
    data_root = pathlib.Path(data_path)

    if train:
        images_path = data_root / "train-images.idx3-ubyte"
        labels_path = data_root / "train-labels.idx1-ubyte"
    else:
        images_path = data_root / "t10k-images.idx3-ubyte"
        labels_path = data_root / "t10k-labels.idx1-ubyte"

    images = read_image_file(images_path)
    labels = read_label_file(labels_path)

    # Normalize to [0,1] and flatten to 784
    images = images.float().div(255.0).view(images.size(0), -1)

    ds = TensorDataset(images, labels)
    return DataLoader(ds, batch_size=batch_size, shuffle=train,
                      num_workers=0, pin_memory=False)


# ---------------------------------------------------------------------------
# Training loop — one epoch
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, l2_lambda: float,
                device: torch.device) -> float:
    """
    Returns mean cross-entropy training cost over the epoch.
    Dropout layers are active during model.train().
    l2_lambda is 0.0 for this experiment (dropout only).
    """
    model.train()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    n_batches  = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()

        logits = model(x)
        loss   = criterion(logits, y)

        # L2 penalty (kept at 0.0 for this experiment; included for symmetry)
        if l2_lambda > 0.0:
            l2_reg = sum(p.pow(2).sum() for p in model.parameters())
            loss   = loss + l2_lambda * l2_reg

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches  += 1

    return total_loss / n_batches


# ---------------------------------------------------------------------------
# Build baseline optimisers
# ---------------------------------------------------------------------------

def build_optimizer(name: str, model: nn.Module,
                    opt_cfg: dict, epoch_lr: float):
    """Construct a torch.optim baseline with the scheduled lr."""
    if name == "sgd_nesterov":
        return torch.optim.SGD(
            model.parameters(),
            lr=epoch_lr,
            momentum=opt_cfg["momentum"],
            nesterov=opt_cfg["nesterov"],
        )
    elif name == "adagrad":
        return torch.optim.Adagrad(
            model.parameters(),
            lr=epoch_lr,
            initial_accumulator_value=opt_cfg["initial_accumulator_value"],
        )
    elif name == "rmsprop":
        return torch.optim.RMSprop(
            model.parameters(),
            lr=epoch_lr,
            alpha=opt_cfg["alpha"],
            eps=opt_cfg["epsilon"],
        )
    elif name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=epoch_lr,
            momentum=opt_cfg["momentum"],
            nesterov=opt_cfg["nesterov"],
        )
    else:
        raise ValueError(f"Unknown optimiser: {name}")


# ---------------------------------------------------------------------------
# One (seed × optimiser) trial
# ---------------------------------------------------------------------------

def run_trial(opt_name: str, seed: int, cfg: dict,
              device: torch.device) -> list[float]:
    set_seed(seed)

    exp          = cfg["mlp_experiment"]
    opt_cfgs     = cfg["mlp_optimizers"]
    num_epochs   = exp["num_epochs"]
    batch_size   = exp["batch_size"]
    l2_lambda    = exp["l2_lambda"]
    input_dim    = exp["input_dim"]
    hidden_units = exp["hidden_units"]
    num_classes  = exp["num_classes"]
    dropout_rate = exp["dropout_rate"]
    base_lr      = opt_cfgs[opt_name]["base_lr"]

    loader = get_mnist_loader(exp["data_path"], batch_size, train=True)
    model  = MLP(input_dim, hidden_units, num_classes, dropout_rate).to(device)

    # Build the optimizer once — keep it alive for the whole trial so
    # internal accumulators (AdaGrad, RMSprop, Adam m/v) are never reset.
    ocfg = opt_cfgs[opt_name]
    if opt_name == "adam":
        optimizer = AdamCustom(
            params=model.parameters(),
            lr=base_lr,
            beta1=ocfg["beta1"],
            beta2=ocfg["beta2"],
            eps=ocfg["epsilon"],
        )
    else:
        optimizer = build_optimizer(opt_name, model, opt_cfgs[opt_name], base_lr)

    costs = []
    for epoch in range(1, num_epochs + 1):
        # lr schedule: alpha_t = base_lr / sqrt(epoch)
        epoch_lr = base_lr / math.sqrt(epoch)

        # Update lr without rebuilding (preserves accumulator state)
        if isinstance(optimizer, AdamCustom):
            optimizer.set_lr(epoch_lr)
        else:
            for group in optimizer.param_groups:
                group["lr"] = epoch_lr

        cost = train_epoch(model, loader, optimizer, l2_lambda, device)
        costs.append(cost)
        print(f"  [{opt_name:14s}] seed={seed}  epoch={epoch:3d}/{num_epochs}"
              f"  cost={cost:.4f}")

    return costs


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

DISPLAY_NAMES = {
    "adam":         "Adam",
    "sgd_nesterov": "SGD+Nesterov",
    "adagrad":      "AdaGrad",
    "rmsprop":      "RMSprop",
    "sgd":          "SGD",
}

COLOURS = {
    "adam":         "#e05c00",
    "sgd_nesterov": "#0077b6",
    "adagrad":      "#2a9d8f",
    "rmsprop":      "#8338ec",
    "sgd":          "#c9184a",
}

LINE_STYLES = {
    "adam":         "-",
    "sgd_nesterov": "--",
    "adagrad":      "-.",
    "rmsprop":      ":",
    "sgd":          (0, (3, 1, 1, 1)),
}


def make_figure(all_costs: dict, cfg: dict):
    pcfg       = cfg["mlp_plot"]
    num_epochs = cfg["mlp_experiment"]["num_epochs"]
    epochs     = np.arange(1, num_epochs + 1)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.set_facecolor("#fafafa")
    fig.patch.set_facecolor("#ffffff")

    opt_order = ["sgd", "adagrad", "rmsprop", "sgd_nesterov", "adam"]

    for opt_name in opt_order:
        data  = all_costs[opt_name]          # (n_seeds, n_epochs)
        mean  = data.mean(axis=0)
        std   = data.std(axis=0)
        colour = COLOURS[opt_name]
        ls     = LINE_STYLES[opt_name]

        ax.plot(epochs, mean, label=DISPLAY_NAMES[opt_name],
                color=colour, linestyle=ls, linewidth=2.0)
        ax.fill_between(epochs, mean - std, mean + std,
                        color=colour, alpha=pcfg["shade_alpha"])

    ax.set_xlabel("iterations over entire dataset (epochs)", fontsize=11)
    ax.set_ylabel("training cost (NLL)", fontsize=11)
    ax.set_title("MNIST MLP + Dropout (128 hidden, ReLU)", fontsize=13,
                 fontweight="bold")
    ax.set_xlim(1, num_epochs)
    ax.set_ylim(0.0, 2.0)
    ax.set_yticks(np.arange(0.0, 2.0 + 1e-9, 0.125))
    ax.legend(fontsize=9, framealpha=0.9)
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.6)

    plt.tight_layout()
    out = pcfg["output_path"]
    fig.savefig(out, dpi=pcfg["dpi"])
    print(f"\nFigure saved → {out}")
    plt.close(fig)


def make_figure_seed0(all_costs: dict, cfg: dict, output_path: str):
    """
    Plot only seed 0 results (no averaging).
    all_costs: { opt_name: np.ndarray of shape (n_seeds, n_epochs) }
    """
    num_epochs = cfg["mlp_experiment"]["num_epochs"]
    epochs = np.arange(1, num_epochs + 1)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.set_facecolor("#fafafa")
    fig.patch.set_facecolor("#ffffff")

    opt_order = ["sgd", "adagrad", "rmsprop", "sgd_nesterov", "adam"]

    for opt_name in opt_order:
        data = all_costs[opt_name]          # (n_seeds, n_epochs)
        seed0 = data[0]
        colour = COLOURS[opt_name]
        ls     = LINE_STYLES[opt_name]

        ax.plot(epochs, seed0, label=DISPLAY_NAMES[opt_name],
                color=colour, linestyle=ls, linewidth=2.0)

    ax.set_xlabel("iterations over entire dataset (epochs)", fontsize=11)
    ax.set_ylabel("training cost (NLL)", fontsize=11)
    ax.set_title("MNIST MLP + Dropout (Seed 0)", fontsize=13, fontweight="bold")
    ax.set_xlim(1, num_epochs)
    ax.set_ylim(0.0, 2.0)
    ax.set_yticks(np.arange(0.0, 2.0 + 1e-9, 0.125))
    ax.legend(fontsize=9, framealpha=0.9)
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.6)

    plt.tight_layout()
    fig.savefig(output_path, dpi=cfg["mlp_plot"]["dpi"])
    print(f"\nSeed-0 figure saved → {output_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Adam paper Figure 2 replication — MLP + dropout on MNIST"
    )
    parser.add_argument(
        "--cfg",
        default=str(pathlib.Path(__file__).parent / "hyperparameters.yaml"),
        help="Path to hyperparameters YAML (default: hyperparameters.yaml next to this file)",
    )
    args = parser.parse_args()

    cfg    = load_cfg(args.cfg)

    # Force outputs into ../results/ex2 (relative to this script)
    output_dir = (pathlib.Path(__file__).parent / ".." / "results" / "ex2").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg["mlp_plot"]["output_path"] = str(output_dir / "figure2_mnist_mlp.png")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    print(f"Config : {args.cfg}\n")

    opt_names = list(cfg["mlp_optimizers"].keys())
    seeds     = cfg["mlp_experiment"]["seeds"]

    all_costs: dict[str, list] = {name: [] for name in opt_names}

    for opt_name in opt_names:
        print(f"=== Optimiser: {DISPLAY_NAMES[opt_name]} ===")
        for seed in seeds:
            costs = run_trial(opt_name, seed, cfg, device)
            all_costs[opt_name].append(costs)
        print()

    all_costs_np = {
        name: np.array(c) for name, c in all_costs.items()
    }

    make_figure(all_costs_np, cfg)

    # Seed-0 only figure (no averaging)
    seed0_dir = (pathlib.Path(__file__).parent / ".." / "results" / "ex2" / "no_avg").resolve()
    seed0_dir.mkdir(parents=True, exist_ok=True)
    seed0_path = seed0_dir / "figure2_mnist_mlp_seed0.png"
    make_figure_seed0(all_costs_np, cfg, str(seed0_path))

    out_dir = pathlib.Path(cfg["mlp_plot"]["output_path"]).parent
    np_path = out_dir / "costs_raw_mlp.npz"
    np.savez(np_path, **all_costs_np)
    print(f"Raw costs saved → {np_path}")


if __name__ == "__main__":
    main()