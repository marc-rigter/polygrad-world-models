# World Models via Policy-Guided Trajectory Diffusion (PolyGRAD)

<img src="https://github.com/marc-rigter/polygrad-world-models/blob/main/polygrad-world-models.gif" width="50%" height="50%"/>

Official code to reproduce the experiments for the paper [World Models via Policy-Guided Trajectory Diffusion](https://arxiv.org/abs/2312.08533).  PolyGRAD diffuses an initially random trajectory of states and actions into an on-policy trajectory, and uses the synthetic data for imagined on-policy RL training.

## Installation
1. Install [MuJoCo 2.1.0](https://github.com/deepmind/mujoco/releases) to `~/.mujoco/mujoco210`.
2. Create a conda environment and install the package.
```
cd polygrad-world-models
conda env create -f environment.yaml
conda activate polygrad-wm
pip install -e .
```

## Usage
To run online RL experiments:

```
python scripts/online_rl.py --config config.online_rl.hopper
```

The scripts/train_world_model.py can be used to reload a dataset and train a world model only.

## Citing this work

```
@article{rigter2023world,
  title={World Models via Policy-Guided Trajectory Diffusion},
  author={Rigter, Marc and Yamada, Jun and Posner, Ingmar},
  journal={arXiv preprint arXiv:2312.08533},
  year={2023}
}
```

## Acknowledgements
Our implementation utilises some of the code from [Diffuser](https://github.com/jannerm/diffuser).
