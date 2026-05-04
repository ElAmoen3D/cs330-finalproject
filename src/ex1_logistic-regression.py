"""
ex1.py — Replication of Figure 1 (MNIST panel) from:
  "Adam: A Method for Stochastic Optimization"
  Kingma & Ba, ICLR 2015

Reproduces: L2-regularised multi-class logistic regression on MNIST,
  comparing Adam (hand-rolled), SGD+Nesterov, AdaGrad, RMSprop, SGD.
  Learning-rate schedule: alpha_t = base_lr / sqrt(epoch)  (1-indexed).
  Results are averaged over multiple seeds; shaded bands = ±1 std dev.

Usage:
    python ex1.py                   # uses hyperparameters.yaml next to this file
    python ex1.py --cfg my_hps.yaml # override config path
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
# Model
# ---------------------------------------------------------------------------

class LogisticRegression(nn.Module):
    """Single linear layer — logistic (softmax) regression."""

    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.linear(x)          # raw logits; loss uses log_softmax


# ---------------------------------------------------------------------------
# Hand-rolled Adam  (torch.autograd only — no torch.optim.Adam)
# ---------------------------------------------------------------------------

class AdamCustom:
    """
    Adam optimiser implemented from first principles.
    Matches Algorithm 1 of Kingma & Ba (2015).

    Parameters
    ----------
    params   : iterable of torch.Tensors (with requires_grad=True)
    lr       : base step-size alpha (may be overridden each step via set_lr)
    beta1    : first-moment decay  (default 0.9)
    beta2    : second-moment decay (default 0.999)
    epsilon  : numerical stability  (default 1e-8)
    """

    def __init__(self, params, lr: float, beta1=0.9, beta2=0.999, eps=1e-8):
        self.params = list(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0                          # step counter
        # First and second moment estimates, initialised to zero
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]

    def set_lr(self, lr: float):
        """Allow the epoch-level schedule to update the step-size."""
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
        # Bias-correction terms (Algorithm 1, line 8–9)
        bc1 = 1.0 - beta1 ** self.t
        bc2 = 1.0 - beta2 ** self.t

        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad

            # Biased first and second moment estimates (lines 5–6)
            self.m[i] = beta1 * self.m[i] + (1.0 - beta1) * g
            self.v[i] = beta2 * self.v[i] + (1.0 - beta2) * g * g

            # Bias-corrected estimates (lines 8–9)
            m_hat = self.m[i] / bc1
            v_hat = self.v[i] / bc2

            # Parameter update (line 10)
            p.data -= self.lr * m_hat / (torch.sqrt(v_hat) + eps)


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
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=train,
        num_workers=0,
        pin_memory=False,
    )


# ---------------------------------------------------------------------------
# Training loop (one epoch)
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, l2_lambda: float, device) -> float:
    """
    Returns the mean negative log-likelihood (training cost) over all batches
    in the epoch — this is what the paper plots on the y-axis.
    L2 regularisation is applied manually so we control it for all optimisers.
    """
    model.train()
    criterion = nn.CrossEntropyLoss()           # NLL + softmax
    total_loss = 0.0
    n_batches = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        # ---- zero gradients ------------------------------------------------
        if isinstance(optimizer, AdamCustom):
            optimizer.zero_grad()
        else:
            optimizer.zero_grad()

        # ---- forward -------------------------------------------------------
        logits = model(x)
        loss = criterion(logits, y)

        # ---- L2 penalty (weight decay) applied uniformly -------------------
        l2_reg = torch.tensor(0.0, device=device)
        for param in model.parameters():
            l2_reg = l2_reg + param.pow(2).sum()
        loss = loss + l2_lambda * l2_reg

        # ---- backward & step -----------------------------------------------
        loss.backward()

        if isinstance(optimizer, AdamCustom):
            optimizer.step()
        else:
            optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


# ---------------------------------------------------------------------------
# Build optimisers
# ---------------------------------------------------------------------------

def build_optimizer(name: str, model: nn.Module, cfg: dict, epoch_lr: float):
    """
    Construct the appropriate optimiser for `name`.
    epoch_lr is the ALREADY SCHEDULED learning rate for this epoch.
    For the custom Adam we update lr via set_lr() externally.
    """
    ocfg = cfg["optimizers"][name]

    if name == "adam":
        return AdamCustom(
            params=model.parameters(),
            lr=epoch_lr,
            beta1=ocfg["beta1"],
            beta2=ocfg["beta2"],
            eps=ocfg["epsilon"],
        )

    elif name == "sgd_nesterov":
        return torch.optim.SGD(
            model.parameters(),
            lr=epoch_lr,
            momentum=ocfg["momentum"],
            nesterov=ocfg["nesterov"],
        )

    elif name == "adagrad":
        return torch.optim.Adagrad(
            model.parameters(),
            lr=epoch_lr,
            initial_accumulator_value=ocfg["initial_accumulator_value"],
        )

    elif name == "rmsprop":
        return torch.optim.RMSprop(
            model.parameters(),
            lr=epoch_lr,
            alpha=ocfg["alpha"],
            eps=ocfg["epsilon"],
        )

    elif name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=epoch_lr,
            momentum=ocfg["momentum"],
            nesterov=ocfg["nesterov"],
        )

    else:
        raise ValueError(f"Unknown optimiser: {name}")


# ---------------------------------------------------------------------------
# Run one (seed × optimiser) trial
# ---------------------------------------------------------------------------

def run_trial(
    opt_name: str,
    seed: int,
    cfg: dict,
    device: torch.device,
) -> list[float]:
    """
    Train for num_epochs epochs and return the per-epoch training cost.
    All optimiser instances are kept alive across epochs so internal
    accumulators (AdaGrad sum-of-squares, RMSprop EMA, Adam m/v) are
    never reset.  The scheduled lr is injected each epoch via param_groups
    for torch.optim baselines, and via set_lr() for custom Adam.
    """
    set_seed(seed)

    exp = cfg["experiment"]
    num_epochs  = exp["num_epochs"]
    batch_size  = exp["batch_size"]
    l2_lambda   = exp["l2_lambda"]
    input_dim   = exp["input_dim"]
    num_classes = exp["num_classes"]

    loader = get_mnist_loader(exp["data_path"], batch_size, train=True)

    model = LogisticRegression(input_dim, num_classes).to(device)
    nn.init.zeros_(model.linear.weight)
    nn.init.zeros_(model.linear.bias)

    ocfg    = cfg["optimizers"][opt_name]
    base_lr = ocfg["base_lr"]

    # Build the optimizer once — keep it alive for the whole trial
    optimizer = build_optimizer(opt_name, model, cfg, base_lr)

    costs = []
    for epoch in range(1, num_epochs + 1):
        # lr schedule: alpha_t = base_lr / sqrt(epoch) (Adam only)
        if isinstance(optimizer, AdamCustom):
            epoch_lr = base_lr / math.sqrt(epoch)
            optimizer.set_lr(epoch_lr)
        else:
            for group in optimizer.param_groups:
                group["lr"] = base_lr

        cost = train_epoch(model, loader, optimizer, l2_lambda, device)
        costs.append(cost)
        print(f"  [{opt_name:14s}] seed={seed}  epoch={epoch:3d}/{num_epochs}  cost={cost:.4f}")

    return costs


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

DISPLAY_NAMES = {
    "adam":        "Adam",
    "sgd_nesterov":"SGD+Nesterov",
    "adagrad":     "AdaGrad",
    "rmsprop":     "RMSprop",
    "sgd":         "SGD",
}

# Colour palette chosen to roughly match the paper's figure
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
    """
    all_costs: { opt_name: np.ndarray of shape (n_seeds, n_epochs) }
    """
    pcfg = cfg["plot"]
    num_epochs = cfg["experiment"]["num_epochs"]
    epochs = np.arange(1, num_epochs + 1)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.set_facecolor("#fafafa")
    fig.patch.set_facecolor("#ffffff")

    opt_order = ["sgd", "adagrad", "rmsprop", "sgd_nesterov", "adam"]

    for opt_name in opt_order:
        data = all_costs[opt_name]          # shape (n_seeds, n_epochs)
        mean = data.mean(axis=0)
        std  = data.std(axis=0)

        colour = COLOURS[opt_name]
        ls     = LINE_STYLES[opt_name]
        label  = DISPLAY_NAMES[opt_name]

        ax.plot(epochs, mean, label=label, color=colour,
                linestyle=ls, linewidth=2.0)
        ax.fill_between(epochs,
                         mean - std,
                         mean + std,
                         color=colour,
                         alpha=pcfg["shade_alpha"])

    ax.set_xlabel("iterations over entire dataset (epochs)", fontsize=11)
    ax.set_ylabel("training cost (NLL)", fontsize=11)
    ax.set_title("MNIST Logistic Regression", fontsize=13, fontweight="bold")
    ax.set_xlim(1, num_epochs)
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
    num_epochs = cfg["experiment"]["num_epochs"]
    epochs = np.arange(1, num_epochs + 1)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.set_facecolor("#fafafa")
    fig.patch.set_facecolor("#ffffff")

    opt_order = ["sgd", "adagrad", "rmsprop", "sgd_nesterov", "adam"]

    for opt_name in opt_order:
        data = all_costs[opt_name]          # shape (n_seeds, n_epochs)
        seed0 = data[0]

        colour = COLOURS[opt_name]
        ls     = LINE_STYLES[opt_name]
        label  = DISPLAY_NAMES[opt_name]

        ax.plot(epochs, seed0, label=label, color=colour,
                linestyle=ls, linewidth=2.0)

    ax.set_xlabel("iterations over entire dataset (epochs)", fontsize=11)
    ax.set_ylabel("training cost (NLL)", fontsize=11)
    ax.set_title("MNIST Logistic Regression (Seed 0)", fontsize=13, fontweight="bold")
    ax.set_xlim(1, num_epochs)
    ax.legend(fontsize=9, framealpha=0.9)
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.6)

    plt.tight_layout()
    fig.savefig(output_path, dpi=cfg["plot"]["dpi"])
    print(f"\nSeed-0 figure saved → {output_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Adam paper Figure 1 replication")
    parser.add_argument(
        "--cfg",
        default=str(pathlib.Path(__file__).parent / "hyperparameters.yaml"),
        help="Path to hyperparameters YAML (default: hyperparameters.yaml next to this file)",
    )
    args = parser.parse_args()

    cfg = load_cfg(args.cfg)

    # Force outputs into ../results/ex1 (relative to this script)
    output_dir = (pathlib.Path(__file__).parent / ".." / "results" / "ex1").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg["plot"]["output_path"] = str(output_dir / "figure1_mnist_logistic.png")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Config: {args.cfg}\n")

    opt_names = list(cfg["optimizers"].keys())
    seeds     = cfg["experiment"]["seeds"]

    # Collect results: { opt_name: list of per-seed cost vectors }
    all_costs: dict[str, list] = {name: [] for name in opt_names}

    for opt_name in opt_names:
        print(f"=== Optimiser: {DISPLAY_NAMES[opt_name]} ===")
        for seed in seeds:
            costs = run_trial(opt_name, seed, cfg, device)
            all_costs[opt_name].append(costs)
        print()

    # Convert to arrays
    all_costs_np = {
        name: np.array(costs)            # (n_seeds, n_epochs)
        for name, costs in all_costs.items()
    }

    make_figure(all_costs_np, cfg)

    # Seed-0 only figure (no averaging)
    seed0_dir = (pathlib.Path(__file__).parent / ".." / "results" / "ex1" / "no_avg").resolve()
    seed0_dir.mkdir(parents=True, exist_ok=True)
    seed0_path = seed0_dir / "figure1_mnist_logistic_seed0.png"
    make_figure_seed0(all_costs_np, cfg, str(seed0_path))

    # Also save raw numbers for the report
    out_dir = pathlib.Path(cfg["plot"]["output_path"]).parent
    np_path = out_dir / "costs_raw.npz"
    np.savez(np_path, **all_costs_np)
    print(f"Raw costs saved → {np_path}")


if __name__ == "__main__":
    main()