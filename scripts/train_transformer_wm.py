import polygrad.utils as utils
import os
import torch
import wandb
import importlib
import dill as pickle
import numpy as np
from polygrad.utils.envs import create_env
from polygrad.utils.timer import Timer
from polygrad.utils.datasets import reload_dataset
from polygrad.agent.transformer_wm import TransformerWM
from os.path import join
from polygrad.utils.errors import compute_traj_errors

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def update_dataset_indices(dataset, horizon):
    dataset.horizon = horizon
    for i in range(dataset.data_buffer.n_episodes):
        dataset.update_indices(i)


# -----------------------------------------------------------------------------#
# ----------------------------------- setup -----------------------------------#
# -----------------------------------------------------------------------------#


class Parser(utils.Parser):
    config: str = "config.simple_maze"


args = Parser().parse_args()

# number of steps to train diffusion model
train_diffusion_steps = args.n_train_steps

expl_env = create_env(args.env_name, args.suite)
eval_env = create_env(args.env_name, args.suite)
random_episodes = utils.rl.random_exploration(100, expl_env)
utils.set_all_seeds(args.seed)

# load all config params
configs = utils.create_configs(args, eval_env)
model = configs["model_config"]()
model.to(device)
dataset = configs["dataset_config"](random_episodes)
ac = configs["ac_config"](normalizer=dataset.normalizer)
world_model = TransformerWM(
    model, context_length=args.horizon - 1, rollout_length=args.rollout_steps
)

# load dataset and a2c from checkpoint
assert args.load_path is not None
ac_path = join(args.load_path, f"step-{args.load_step}-ac.pt")
ac.load_state_dict(torch.load(ac_path, map_location=device))
reload_dataset(join(args.load_path, f"step-{args.load_step}-dataset.npy"), dataset)
wandb.init(entity="a2i", project=args.project, group=args.group, config=args)

# -----------------------------------------------------------------------------#
# --------------------------- prepare to train --------------------------------#
# -----------------------------------------------------------------------------#

dataloader = utils.training.cycle(
    torch.utils.data.DataLoader(
        dataset,
        batch_size=args.agent_batch_size,
        num_workers=2,
        shuffle=True,
        pin_memory=True,
    )
)

# ---------------------------- Main Loop ----------------------------------#

step = 0
timer = Timer()
max_log = 256
while step < train_diffusion_steps:
    metrics = dict()

    batch = next(dataloader)
    metrics.update(world_model.train(batch))

    if step % 2000 == 0:
        (
            imag_states,
            imag_act,
            imag_rewards,
            imag_terminals,
            imag_metrics,
        ) = world_model.imagine(batch, ac.forward_actor)

        obs = dataset.normalizer.unnormalize(
            imag_states.detach().cpu().numpy(), "observations"
        )
        act = dataset.normalizer.unnormalize(imag_act.detach().cpu().numpy(), "actions")
        rew = dataset.normalizer.unnormalize(
            imag_rewards.detach().cpu().numpy(), "rewards"
        )
        error_metrics = compute_traj_errors(
            eval_env,
            obs[:max_log],
            act[:max_log],
            rew[:max_log],
            sim_states=batch.sim_states[:max_log],
        )
        metrics.update(imag_metrics)
        metrics.update(error_metrics)
        wandb.log(metrics, step=step)
        print("Error Metrics: ")
        print(error_metrics)
        print("\n")

    if step % 100 == 0:
        print("Train step: ", step)
        wandb.log(metrics, step=step)
    step += 1
