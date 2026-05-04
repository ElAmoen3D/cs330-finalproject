# CS-330 Final Project - Gabriel Gonzalez
This repository contains all necessary code and documentation to replicate Diederik P. Kingma and Jimmy Lei Ba's publication "Adam: A Method for Stochastic Optimization." The repository also include a lab notebook and CMD script for ease of use.

## Repository Layout
- `src` contains all source code and supporting material, like hyperparameters and dependencies
- `docs` contains all necessary documentation for the repository as well as the development process
- `results` contains all program output
- `data` contains all datasets used. "DATA_INDEX.md" explains why each is there and what its used for

## Deliverables:
- Replication script: `run.sh` in this directory - run with necessary permissions
- All result graphs in results
- Lab notebook in `./docs`
- All code in `./src`

## Compiling and running
- All dependencies can be installed manually with `pip install ./src/requirements.txt`
- Run the script `run.sh` as superuser for full execution.

## Requirements:
- Python 3.12
- Libraries in `src/requirements.txt`