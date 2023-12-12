import polygrad.utils as utils
import os
import torch
import wandb
import importlib
import dill as pickle
import numpy as np
from polygrad.utils.evaluation import evaluate_policy
from polygrad.utils.envs import create_env
from polygrad.utils.timer import Timer
from polygrad.models.transformer_world_model import TransformerWM
from os.path import join
from polygrad.utils.errors import compute_traj_errors


def update_dataset_indices(dataset, horizon):
    dataset.horizon = horizon
    for i in range(dataset.data_buffer.n_episodes):
        dataset.update_indices(i)

#-----------------------------------------------------------------------------#
#----------------------------------- setup -----------------------------------#
#-----------------------------------------------------------------------------#

class Parser(utils.Parser):
    config: str = 'config.simple_maze'

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
model = configs["model_config"]()
model.to("cuda:0")
dataset = configs["dataset_config"](random_episodes)
ac = configs["ac_config"](normalizer=dataset.normalizer)
world_model = TransformerWM(
    model,
    context_length = args.horizon - 1,
    rollout_length = args.rollout_steps
)

ac_path = join(args.load_path, f"step-{args.load_step}-ac.pt")
ac.load_state_dict(torch.load(ac_path))

# load dataset from checkpoint.
assert args.load_path is not None

reload_dataset_path = join(args.load_path, f"step-{args.load_step}-dataset.pkl")
with open(reload_dataset_path, 'rb') as f:
    reload_dataset = pickle.load(f)
    data_buffer = reload_dataset.data_buffer

# put the loaded data into the dataset object
dataset.reset_data_buffer()
for i in range(data_buffer.n_episodes):
    path_length = data_buffer._dict['path_lengths'][i]
    episode = {
        "observations": data_buffer._dict['observations'][i][:path_length],
        "actions": data_buffer._dict['actions'][i][:path_length],
        "next_observations": data_buffer._dict['next_observations'][i][:path_length],
        "rewards": data_buffer._dict['rewards'][i][:path_length],
        "terminals": data_buffer._dict['terminals'][i][:path_length],
        "sim_states": data_buffer._dict['sim_states'][i][:path_length],
    }
    episode["timeouts"] = np.array([False] * len(episode["rewards"]))
    dataset.add_episode(episode)
dataset.update_normalizers()
print(f"Loaded dataset containing {dataset.data_buffer.n_episodes} episodes.")
wandb.init(project=args.project, group=args.group, config=args)

#-----------------------------------------------------------------------------#
#--------------------------- prepare to train --------------------------------#
#-----------------------------------------------------------------------------#

dataloader = utils.training.cycle(torch.utils.data.DataLoader(
    dataset, batch_size=args.agent_batch_size, num_workers=2, shuffle=True, pin_memory=True
))

#---------------------------- Main Loop ----------------------------------#

step = 0
timer = Timer()
max_log = 256
while step < train_diffusion_steps:
    metrics = dict()

    batch = next(dataloader)
    metrics.update(world_model.train(batch))

    if step % 2000 == 0:
        imag_states, imag_act, imag_rewards, imag_terminals, imag_metrics = world_model.imagine(batch, ac.forward_actor)
        
        obs = dataset.normalizer.unnormalize(imag_states.detach().cpu().numpy(), "observations")
        act = dataset.normalizer.unnormalize(imag_act.detach().cpu().numpy(), "actions")
        rew = dataset.normalizer.unnormalize(imag_rewards.detach().cpu().numpy(), "rewards")
        error_metrics = compute_traj_errors(
            eval_env,
            obs[:max_log],
            act[:max_log], 
            rew[:max_log],
            sim_states=batch.sim_states[:max_log]
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
