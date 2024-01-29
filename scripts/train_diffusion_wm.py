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
from os.path import join

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

print("Seed", args.seed)
utils.set_all_seeds(args.seed)

# load all config params
configs = utils.create_configs(args, eval_env)
if configs["render_config"] is not None:
    renderer = configs["render_config"]()
else:
    renderer = None
model = configs["model_config"]()
diffusion = configs["diffusion_config"](model)
dataset = configs["dataset_config"](random_episodes)
diffusion_trainer = configs["trainer_config"](diffusion, dataset, eval_env, renderer)
ac = configs["ac_config"](normalizer=dataset.normalizer)
agent = configs["agent_config"](
    diffusion_model=diffusion_trainer.ema_model,
    actor_critic=ac,
    dataset=dataset,
    env=eval_env,
    renderer=renderer,
)

# load dataset and a2c from checkpoint
assert args.load_path is not None
ac_path = join(args.load_path, f"step-{args.load_step}-ac.pt")
ac.load_state_dict(torch.load(ac_path, map_location=device))
reload_dataset(join(args.load_path, f"step-{args.load_step}-dataset.npy"), dataset)

# initialise wandb
utils.report_parameters(model)
wandb.init(entity="a2i", project=args.project, group=args.group, config=args)

# -----------------------------------------------------------------------------#
# --------------------------- prepare to train --------------------------------#
# -----------------------------------------------------------------------------#

agent_dataloader = utils.training.cycle(
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
while step < train_diffusion_steps:
    metrics = dict()

    if step % int(1 / args.train_agent_ratio) == 0:
        batch = next(agent_dataloader)
        agent_metrics = agent.training_step(batch, step, log_only=True, max_log=500)
        [
            metrics.update({f"agent/{key}": agent_metrics[key]})
            for key in agent_metrics.keys()
        ]

        diffusion_updates = int(args.train_diffusion_ratio / args.train_agent_ratio)
        diffusion_metrics = diffusion_trainer.train(diffusion_updates, step)
        [
            metrics.update({f"diffusion/{key}": diffusion_metrics[key]})
            for key in diffusion_metrics.keys()
        ]

    wandb.log(metrics, step=step)
    step += 1
