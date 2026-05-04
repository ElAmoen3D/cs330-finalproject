"""
ex1_logistic-regression_numpy.py — NumPy-only replication of Figure 1 (MNIST panel)
from:
  "Adam: A Method for Stochastic Optimization" (Kingma & Ba, ICLR 2015)

Same functionality as ex1_logistic-regression.py, but implemented in NumPy
(no torch/autograd). Uses the locally downloaded MNIST IDX files.
Outputs are saved to ../results/ex1/numpy.
"""

import argparse
import math
import pathlib
import random
import struct

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def load_cfg(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# MNIST IDX loading (local files)
# ---------------------------------------------------------------------------

def _read_idx_images(path: pathlib.Path) -> np.ndarray:
    with path.open("rb") as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16)
        )
        if magic != 2051:
            raise ValueError(f"Unexpected magic number {magic} in {path}")
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(num, rows * cols)


def _read_idx_labels(path: pathlib.Path) -> np.ndarray:
    with path.open("rb") as f:
        magic, num = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"Unexpected magic number {magic} in {path}")
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.astype(np.int64)


def load_mnist(data_path: str, train: bool) -> tuple[np.ndarray, np.ndarray]:
    data_root = pathlib.Path(data_path)
    if train:
        images_path = data_root / "train-images.idx3-ubyte"
        labels_path = data_root / "train-labels.idx1-ubyte"
    else:
        images_path = data_root / "t10k-images.idx3-ubyte"
        labels_path = data_root / "t10k-labels.idx1-ubyte"

    images = _read_idx_images(images_path).astype(np.float32) / 255.0
    labels = _read_idx_labels(labels_path)
    return images, labels


# ---------------------------------------------------------------------------
# Optimizers (NumPy)
# ---------------------------------------------------------------------------

class OptimizerBase:
    def step(self, params, grads):
        raise NotImplementedError


class SGD(OptimizerBase):
    def __init__(self, lr: float, momentum: float = 0.0, nesterov: bool = False):
        self.lr = lr
        self.momentum = momentum
        self.nesterov = nesterov
        self.v = None

    def set_lr(self, lr: float):
        self.lr = lr

    def step(self, params, grads):
        if self.v is None:
            self.v = [np.zeros_like(p) for p in params]

        for i, (p, g) in enumerate(zip(params, grads)):
            self.v[i] = self.momentum * self.v[i] + g
            if self.nesterov and self.momentum > 0.0:
                update = g + self.momentum * self.v[i]
            else:
                update = self.v[i]
            params[i] -= self.lr * update


class AdaGrad(OptimizerBase):
    def __init__(self, lr: float, initial_accumulator_value: float = 0.0, eps: float = 1e-8):
        self.lr = lr
        self.eps = eps
        self.acc = None
        self.initial_accumulator_value = initial_accumulator_value

    def set_lr(self, lr: float):
        self.lr = lr

    def step(self, params, grads):
        if self.acc is None:
            self.acc = [np.full_like(p, self.initial_accumulator_value) for p in params]

        for i, (p, g) in enumerate(zip(params, grads)):
            self.acc[i] = self.acc[i] + g * g
            params[i] -= self.lr * g / (np.sqrt(self.acc[i]) + self.eps)


class RMSprop(OptimizerBase):
    def __init__(self, lr: float, alpha: float = 0.99, eps: float = 1e-8):
        self.lr = lr
        self.alpha = alpha
        self.eps = eps
        self.avg_sq = None

    def set_lr(self, lr: float):
        self.lr = lr

    def step(self, params, grads):
        if self.avg_sq is None:
            self.avg_sq = [np.zeros_like(p) for p in params]

        for i, (p, g) in enumerate(zip(params, grads)):
            self.avg_sq[i] = self.alpha * self.avg_sq[i] + (1.0 - self.alpha) * (g * g)
            params[i] -= self.lr * g / (np.sqrt(self.avg_sq[i]) + self.eps)


class Adam(OptimizerBase):
    def __init__(self, lr: float, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = None
        self.v = None
        self.t = 0

    def set_lr(self, lr: float):
        self.lr = lr

    def step(self, params, grads):
        if self.m is None:
            self.m = [np.zeros_like(p) for p in params]
            self.v = [np.zeros_like(p) for p in params]

        self.t += 1
        bc1 = 1.0 - self.beta1 ** self.t
        bc2 = 1.0 - self.beta2 ** self.t

        for i, (p, g) in enumerate(zip(params, grads)):
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * (g * g)
            m_hat = self.m[i] / bc1
            v_hat = self.v[i] / bc2
            params[i] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


# ---------------------------------------------------------------------------
# Model + loss (NumPy)
# ---------------------------------------------------------------------------

def softmax_logits(logits: np.ndarray) -> np.ndarray:
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / np.sum(exp, axis=1, keepdims=True)


def forward(X: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    return X @ W.T + b


def cross_entropy_loss(logits: np.ndarray, y: np.ndarray) -> float:
    probs = softmax_logits(logits)
    n = y.shape[0]
    return -np.log(probs[np.arange(n), y] + 1e-12).mean()


def compute_grads(X: np.ndarray, y: np.ndarray, W: np.ndarray, b: np.ndarray, l2_lambda: float):
    logits = forward(X, W, b)
    probs = softmax_logits(logits)

    n = X.shape[0]
    probs[np.arange(n), y] -= 1.0
    probs = probs / n

    dW = probs.T @ X
    db = probs.sum(axis=0)

    if l2_lambda > 0.0:
        dW += 2.0 * l2_lambda * W
        db += 2.0 * l2_lambda * b

    loss = cross_entropy_loss(logits, y)
    if l2_lambda > 0.0:
        loss += l2_lambda * (np.sum(W * W) + np.sum(b * b))

    return loss, dW, db


# ---------------------------------------------------------------------------
# Training loop (one epoch)
# ---------------------------------------------------------------------------

def train_epoch(X: np.ndarray, y: np.ndarray, batch_size: int,
                W: np.ndarray, b: np.ndarray, optimizer: OptimizerBase,
                l2_lambda: float) -> float:
    n = X.shape[0]
    indices = np.random.permutation(n)
    total_loss = 0.0
    n_batches = 0

    for start in range(0, n, batch_size):
        batch_idx = indices[start:start + batch_size]
        Xb = X[batch_idx]
        yb = y[batch_idx]

        loss, dW, db = compute_grads(Xb, yb, W, b, l2_lambda)
        optimizer.step([W, b], [dW, db])

        total_loss += loss
        n_batches += 1

    return total_loss / n_batches


# ---------------------------------------------------------------------------
# Build optimisers
# ---------------------------------------------------------------------------

def build_optimizer(name: str, cfg: dict, epoch_lr: float) -> OptimizerBase:
    ocfg = cfg["optimizers"][name]

    if name == "adam":
        return Adam(
            lr=epoch_lr,
            beta1=ocfg["beta1"],
            beta2=ocfg["beta2"],
            eps=ocfg["epsilon"],
        )
    if name == "sgd_nesterov":
        return SGD(
            lr=epoch_lr,
            momentum=ocfg["momentum"],
            nesterov=ocfg["nesterov"],
        )
    if name == "adagrad":
        return AdaGrad(
            lr=epoch_lr,
            initial_accumulator_value=ocfg["initial_accumulator_value"],
        )
    if name == "rmsprop":
        return RMSprop(
            lr=epoch_lr,
            alpha=ocfg["alpha"],
            eps=ocfg["epsilon"],
        )
    if name == "sgd":
        return SGD(
            lr=epoch_lr,
            momentum=ocfg["momentum"],
            nesterov=ocfg["nesterov"],
        )

    raise ValueError(f"Unknown optimiser: {name}")


# ---------------------------------------------------------------------------
# Run one (seed × optimiser) trial
# ---------------------------------------------------------------------------

def run_trial(opt_name: str, seed: int, cfg: dict) -> list[float]:
    set_seed(seed)

    exp = cfg["experiment"]
    num_epochs  = exp["num_epochs"]
    batch_size  = exp["batch_size"]
    l2_lambda   = exp["l2_lambda"]
    input_dim   = exp["input_dim"]
    num_classes = exp["num_classes"]

    X_train, y_train = load_mnist(exp["data_path"], train=True)

    W = np.zeros((num_classes, input_dim), dtype=np.float32)
    b = np.zeros((num_classes,), dtype=np.float32)

    ocfg = cfg["optimizers"][opt_name]
    base_lr = ocfg["base_lr"]

    optimizer = build_optimizer(opt_name, cfg, base_lr)

    costs = []
    for epoch in range(1, num_epochs + 1):
        epoch_lr = base_lr / math.sqrt(epoch)
        optimizer.set_lr(epoch_lr)

        cost = train_epoch(X_train, y_train, batch_size, W, b, optimizer, l2_lambda)
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


def make_figure(all_costs: dict, cfg: dict, output_path: str):
    pcfg = cfg["plot"]
    num_epochs = cfg["experiment"]["num_epochs"]
    epochs = np.arange(1, num_epochs + 1)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.set_facecolor("#fafafa")
    fig.patch.set_facecolor("#ffffff")

    opt_order = ["sgd", "adagrad", "rmsprop", "sgd_nesterov", "adam"]

    for opt_name in opt_order:
        data = all_costs[opt_name]
        mean = data.mean(axis=0)
        std  = data.std(axis=0)

        colour = COLOURS[opt_name]
        ls     = LINE_STYLES[opt_name]
        label  = DISPLAY_NAMES[opt_name]

        ax.plot(epochs, mean, label=label, color=colour,
                linestyle=ls, linewidth=2.0)
        ax.fill_between(epochs, mean - std, mean + std,
                         color=colour, alpha=pcfg["shade_alpha"])

    ax.set_xlabel("iterations over entire dataset (epochs)", fontsize=11)
    ax.set_ylabel("training cost (NLL)", fontsize=11)
    ax.set_title("MNIST Logistic Regression (NumPy)", fontsize=13, fontweight="bold")
    ax.set_xlim(1, num_epochs)
    ax.legend(fontsize=9, framealpha=0.9)
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.6)

    plt.tight_layout()
    fig.savefig(output_path, dpi=pcfg["dpi"])
    print(f"\nFigure saved → {output_path}")
    plt.close(fig)


def make_figure_seed0(all_costs: dict, cfg: dict, output_path: str):
    num_epochs = cfg["experiment"]["num_epochs"]
    epochs = np.arange(1, num_epochs + 1)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.set_facecolor("#fafafa")
    fig.patch.set_facecolor("#ffffff")

    opt_order = ["sgd", "adagrad", "rmsprop", "sgd_nesterov", "adam"]

    for opt_name in opt_order:
        data = all_costs[opt_name]
        seed0 = data[0]

        colour = COLOURS[opt_name]
        ls     = LINE_STYLES[opt_name]
        label  = DISPLAY_NAMES[opt_name]

        ax.plot(epochs, seed0, label=label, color=colour,
                linestyle=ls, linewidth=2.0)

    ax.set_xlabel("iterations over entire dataset (epochs)", fontsize=11)
    ax.set_ylabel("training cost (NLL)", fontsize=11)
    ax.set_title("MNIST Logistic Regression (Seed 0, NumPy)", fontsize=13, fontweight="bold")
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
    parser = argparse.ArgumentParser(description="Adam paper Figure 1 replication (NumPy)")
    parser.add_argument(
        "--cfg",
        default=str(pathlib.Path(__file__).parent / "hyperparameters.yaml"),
        help="Path to hyperparameters YAML (default: hyperparameters.yaml next to this file)",
    )
    args = parser.parse_args()

    cfg = load_cfg(args.cfg)

    # Force outputs into ../results/ex1/numpy (relative to this script)
    output_dir = (pathlib.Path(__file__).parent / ".." / "results" / "ex1" / "numpy").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    opt_names = list(cfg["optimizers"].keys())
    seeds     = cfg["experiment"]["seeds"]

    all_costs: dict[str, list] = {name: [] for name in opt_names}

    for opt_name in opt_names:
        print(f"=== Optimiser: {DISPLAY_NAMES[opt_name]} ===")
        for seed in seeds:
            costs = run_trial(opt_name, seed, cfg)
            all_costs[opt_name].append(costs)
        print()

    all_costs_np = {
        name: np.array(costs) for name, costs in all_costs.items()
    }

    fig_path = output_dir / "figure1_mnist_logistic_numpy.png"
    make_figure(all_costs_np, cfg, str(fig_path))

    seed0_dir = output_dir / "no_avg"
    seed0_dir.mkdir(parents=True, exist_ok=True)
    seed0_path = seed0_dir / "figure1_mnist_logistic_seed0_numpy.png"
    make_figure_seed0(all_costs_np, cfg, str(seed0_path))

    np_path = output_dir / "costs_raw_numpy.npz"
    np.savez(np_path, **all_costs_np)
    print(f"Raw costs saved → {np_path}")

    # Log differences vs PyTorch outputs (if present)
    pytorch_path = (pathlib.Path(__file__).parent / ".." / "results" / "ex1" / "costs_raw.npz").resolve()
    log_path = output_dir / "pytorch_vs_numpy_diff.log"
    if pytorch_path.exists():
        pt = np.load(pytorch_path)
        npn = np.load(np_path)
        keys = sorted(set(pt.files) & set(npn.files))
        lines = []
        lines.append(f"PyTorch file: {pytorch_path}")
        lines.append(f"NumPy file:   {np_path}")
        lines.append("")
        lines.append("Per-optimizer absolute differences:")

        for k in keys:
            a = pt[k]
            b = npn[k]
            if a.shape != b.shape:
                lines.append(f"{k}: shape mismatch {a.shape} vs {b.shape}")
                continue
            diff = a - b
            lines.append(
                f"{k}: max_abs={np.max(np.abs(diff)):.6g}, mean_abs={np.mean(np.abs(diff)):.6g}"
            )

        missing = set(pt.files) ^ set(npn.files)
        if missing:
            lines.append("")
            lines.append("Mismatched keys: " + ", ".join(sorted(missing)))

        log_path.write_text("\n".join(lines))
        print(f"Differences log saved → {log_path}")
    else:
        print(f"PyTorch outputs not found at {pytorch_path}; skipping diff log.")


if __name__ == "__main__":
    main()
