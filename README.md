# World Models via Policy-Guided Trajectory Diffusion (PolyGRAD)

<img src="https://github.com/marc-rigter/polygrad-world-models/blob/main/polygrad-world-models.gif" width="50%" height="50%"/>

Official code to reproduce the experiments for the paper [World Models via Policy-Guided Trajectory Diffusion](https://arxiv.org/abs/2312.08533).  PolyGRAD diffuses an initially random trajectory of states and actions into an on-policy trajectory, and uses the synthetic data for imagined on-policy RL training.

## Installation
1. Install [MuJoCo 2.1.0](https://github.com/deepmind/mujoco/releases) to `~/.mujoco/mujoco210`.
2. Install requirements and package.
```
cd polygrad-world-models
pip install -r requirements.txt
pip install -e .
```

Tested with Python 3.10.

## Running PolyGRAD

### Online RL Experiments
To run online RL experiments:

```
python3 scripts/online_rl.py --config config.online_rl.hopper
```

### Training World Model from Fixed Datasets
Download the datasets from [Google Drive](https://drive.google.com/drive/folders/1uyL434A4OXbqOI0wgL6uDZ9jGqSOBpfz?usp=sharing) and store them in polygrad-world-models/datasets. Train and evaluate errors for PolyGRAD world models using:

```
python3 scripts/train_diffusion_wm.py --config config.world_model_only.polygrad_mlp.hopper_h10
```
Config files for the MLP and transformer denoising network, as well as different trajectory lengths are provided in the config folder.

## Running Baselines

In this repo, we also provide implementations of the autoregressive diffusion and transformer world model baselines in the paper. Ensure that you have the datasets in polygrad-world-models/datasets. Then, to train the autoregressive diffusion world model baseline:
```
python3 scripts/train_diffusion_wm.py --config config.world_model_only.autoregressive_diffusion.hopper
```
Note that the transformer world model baseline uses a different script:
```
python3 scripts/train_transformer_wm.py --config config.world_model_only.transformer_wm.hopper
```

For the MLP ensemble baseline we used the code from [mbpo_pytorch](https://github.com/Xingyu-Lin/mbpo_pytorch). For the Dreamer-v3 baseline we used the [dreamerv3-torch](https://github.com/NM512/dreamerv3-torch) repo. Lastly, for the model-free RL baselines we used [Stable Baselines 3](https://github.com/DLR-RM/stable-baselines3).

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
Our implementation utilises code from [Diffuser](https://github.com/jannerm/diffuser), [nanoGPT](https://github.com/karpathy/nanoGPT), and [SynthER](https://github.com/conglu1997/SynthER).
