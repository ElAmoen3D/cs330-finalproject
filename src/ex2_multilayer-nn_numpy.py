"""
ex2_multilayer-nn_numpy.py — NumPy-only replication of Figure 2 (MNIST MLP + dropout)
from:
  "Adam: A Method for Stochastic Optimization" (Kingma & Ba, ICLR 2015)

Same functionality as ex2_multilayer-nn.py, but implemented in NumPy
(no torch/autograd). Uses the locally downloaded MNIST IDX files.
Outputs are saved to ../results/ex2/numpy.
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
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
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

def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def softmax_logits(logits: np.ndarray) -> np.ndarray:
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / np.sum(exp, axis=1, keepdims=True)


def cross_entropy_loss(logits: np.ndarray, y: np.ndarray) -> float:
    probs = softmax_logits(logits)
    n = y.shape[0]
    return -np.log(probs[np.arange(n), y] + 1e-12).mean()


def dropout_mask(shape, drop_prob: float) -> np.ndarray:
    keep_prob = 1.0 - drop_prob
    mask = (np.random.rand(*shape) < keep_prob).astype(np.float32) / keep_prob
    return mask


def forward_backward(X: np.ndarray, y: np.ndarray,
                     W1: np.ndarray, b1: np.ndarray,
                     W2: np.ndarray, b2: np.ndarray,
                     W3: np.ndarray, b3: np.ndarray,
                     dropout_rate: float) -> tuple[float, list]:
    # Forward
    z1 = X @ W1.T + b1
    a1 = relu(z1)
    m1 = dropout_mask(a1.shape, dropout_rate)
    a1d = a1 * m1

    z2 = a1d @ W2.T + b2
    a2 = relu(z2)
    m2 = dropout_mask(a2.shape, dropout_rate)
    a2d = a2 * m2

    logits = a2d @ W3.T + b3
    loss = cross_entropy_loss(logits, y)

    # Backward
    probs = softmax_logits(logits)
    n = X.shape[0]
    probs[np.arange(n), y] -= 1.0
    dlogits = probs / n

    dW3 = dlogits.T @ a2d
    db3 = dlogits.sum(axis=0)

    da2d = dlogits @ W3
    da2 = da2d * m2
    dz2 = da2 * (z2 > 0.0)

    dW2 = dz2.T @ a1d
    db2 = dz2.sum(axis=0)

    da1d = dz2 @ W2
    da1 = da1d * m1
    dz1 = da1 * (z1 > 0.0)

    dW1 = dz1.T @ X
    db1 = dz1.sum(axis=0)

    grads = [dW1, db1, dW2, db2, dW3, db3]
    return loss, grads


# ---------------------------------------------------------------------------
# Training loop (one epoch)
# ---------------------------------------------------------------------------

def train_epoch(X: np.ndarray, y: np.ndarray, batch_size: int,
                params: list, optimizer: OptimizerBase,
                dropout_rate: float, l2_lambda: float) -> float:
    n = X.shape[0]
    indices = np.random.permutation(n)
    total_loss = 0.0
    n_batches = 0

    W1, b1, W2, b2, W3, b3 = params

    for start in range(0, n, batch_size):
        batch_idx = indices[start:start + batch_size]
        Xb = X[batch_idx]
        yb = y[batch_idx]

        loss, grads = forward_backward(Xb, yb, W1, b1, W2, b2, W3, b3, dropout_rate)

        if l2_lambda > 0.0:
            loss += l2_lambda * (
                np.sum(W1 * W1) + np.sum(W2 * W2) + np.sum(W3 * W3)
                + np.sum(b1 * b1) + np.sum(b2 * b2) + np.sum(b3 * b3)
            )
            grads[0] += 2.0 * l2_lambda * W1
            grads[1] += 2.0 * l2_lambda * b1
            grads[2] += 2.0 * l2_lambda * W2
            grads[3] += 2.0 * l2_lambda * b2
            grads[4] += 2.0 * l2_lambda * W3
            grads[5] += 2.0 * l2_lambda * b3

        optimizer.step(params, grads)

        total_loss += loss
        n_batches += 1

    return total_loss / n_batches


# ---------------------------------------------------------------------------
# Build optimisers
# ---------------------------------------------------------------------------

def build_optimizer(name: str, cfg: dict, epoch_lr: float) -> OptimizerBase:
    ocfg = cfg["mlp_optimizers"][name]

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

    X_train, y_train = load_mnist(exp["data_path"], train=True)

    # He initialization for ReLU layers
    W1 = (np.random.randn(hidden_units, input_dim).astype(np.float32)
          * np.sqrt(2.0 / input_dim))
    b1 = np.zeros((hidden_units,), dtype=np.float32)
    W2 = (np.random.randn(hidden_units, hidden_units).astype(np.float32)
          * np.sqrt(2.0 / hidden_units))
    b2 = np.zeros((hidden_units,), dtype=np.float32)
    W3 = (np.random.randn(num_classes, hidden_units).astype(np.float32)
          * np.sqrt(2.0 / hidden_units))
    b3 = np.zeros((num_classes,), dtype=np.float32)

    params = [W1, b1, W2, b2, W3, b3]

    optimizer = build_optimizer(opt_name, cfg, base_lr)

    costs = []
    for epoch in range(1, num_epochs + 1):
        epoch_lr = base_lr / math.sqrt(epoch)
        optimizer.set_lr(epoch_lr)

        cost = train_epoch(X_train, y_train, batch_size, params, optimizer, dropout_rate, l2_lambda)
        costs.append(cost)
        print(f"  [{opt_name:14s}] seed={seed}  epoch={epoch:3d}/{num_epochs}  cost={cost:.4f}")

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


def make_figure(all_costs: dict, cfg: dict, output_path: str):
    pcfg       = cfg["mlp_plot"]
    num_epochs = cfg["mlp_experiment"]["num_epochs"]
    epochs     = np.arange(1, num_epochs + 1)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.set_facecolor("#fafafa")
    fig.patch.set_facecolor("#ffffff")

    opt_order = ["sgd", "adagrad", "rmsprop", "sgd_nesterov", "adam"]

    for opt_name in opt_order:
        data  = all_costs[opt_name]
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
    ax.set_title("MNIST MLP + Dropout (NumPy)", fontsize=13, fontweight="bold")
    ax.set_xlim(1, num_epochs)
    ax.set_ylim(0.0, 2.0)
    ax.set_yticks(np.arange(0.0, 2.0 + 1e-9, 0.125))
    ax.legend(fontsize=9, framealpha=0.9)
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.6)

    plt.tight_layout()
    fig.savefig(output_path, dpi=pcfg["dpi"])
    print(f"\nFigure saved → {output_path}")
    plt.close(fig)


def make_figure_seed0(all_costs: dict, cfg: dict, output_path: str):
    num_epochs = cfg["mlp_experiment"]["num_epochs"]
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

        ax.plot(epochs, seed0, label=DISPLAY_NAMES[opt_name],
                color=colour, linestyle=ls, linewidth=2.0)

    ax.set_xlabel("iterations over entire dataset (epochs)", fontsize=11)
    ax.set_ylabel("training cost (NLL)", fontsize=11)
    ax.set_title("MNIST MLP + Dropout (Seed 0, NumPy)", fontsize=13, fontweight="bold")
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
        description="Adam paper Figure 2 replication — MLP + dropout on MNIST (NumPy)"
    )
    parser.add_argument(
        "--cfg",
        default=str(pathlib.Path(__file__).parent / "hyperparameters.yaml"),
        help="Path to hyperparameters YAML (default: hyperparameters.yaml next to this file)",
    )
    args = parser.parse_args()

    cfg = load_cfg(args.cfg)

    # Force outputs into ../results/ex2/numpy (relative to this script)
    output_dir = (pathlib.Path(__file__).parent / ".." / "results" / "ex2" / "numpy").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    opt_names = list(cfg["mlp_optimizers"].keys())
    seeds     = cfg["mlp_experiment"]["seeds"]

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

    fig_path = output_dir / "figure2_mnist_mlp_numpy.png"
    make_figure(all_costs_np, cfg, str(fig_path))

    seed0_dir = output_dir / "no_avg"
    seed0_dir.mkdir(parents=True, exist_ok=True)
    seed0_path = seed0_dir / "figure2_mnist_mlp_seed0_numpy.png"
    make_figure_seed0(all_costs_np, cfg, str(seed0_path))

    np_path = output_dir / "costs_raw_mlp_numpy.npz"
    np.savez(np_path, **all_costs_np)
    print(f"Raw costs saved → {np_path}")

    # Log differences vs PyTorch outputs (if present)
    pytorch_path = (pathlib.Path(__file__).parent / ".." / "results" / "ex2" / "costs_raw_mlp.npz").resolve()
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
