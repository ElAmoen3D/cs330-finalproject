#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN=${PYTHON_BIN:-python}

echo "Using Python: $PYTHON_BIN"

# Install dependencies
$PYTHON_BIN -m pip install --upgrade pip
$PYTHON_BIN -m pip install -r "$ROOT_DIR/src/requirements.txt"

echo "EXPERIMENT 1: Logistic Regression"
# Run experiments (PyTorch)
$PYTHON_BIN "$ROOT_DIR/src/ex1_logistic-regression.py"

echo "EXPERIMENT 2: Multilayer Neural Network"
$PYTHON_BIN "$ROOT_DIR/src/ex2_multilayer-nn.py"

# Run experiments (NumPy)
echo "EXPERIMENT 1: Logistic Regression (NumPy)"
$PYTHON_BIN "$ROOT_DIR/src/ex1_logistic-regression_numpy.py"

echo "EXPERIMENT 2: Multilayer Neural Network (NumPy)"
$PYTHON_BIN "$ROOT_DIR/src/ex2_multilayer-nn_numpy.py"

echo "All graphs generated under results/"
