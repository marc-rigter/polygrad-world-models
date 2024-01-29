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
from os.path import join


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
    renderer=renderer
)

# load dataset and a2c from checkpoint. ensure that diffusion trainer and ac have
# correct dataset and normalizer
assert args.load_path is not None
agent.load(args.load_path, args.load_step, load_a2c=True, load_dataset=False, load_diffusion=False)

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

utils.report_parameters(model)
group = "online_rl"
wandb.init(entity="a2i", project=args.project, group=args.group, config=args)

#-----------------------------------------------------------------------------#
#--------------------------- prepare to train --------------------------------#
#-----------------------------------------------------------------------------#

agent_dataloader = utils.training.cycle(torch.utils.data.DataLoader(
    dataset, batch_size=args.agent_batch_size, num_workers=2, shuffle=True, pin_memory=True
))

#---------------------------- Main Loop ----------------------------------#

step = 0
timer = Timer()
while step < train_diffusion_steps:
    metrics = dict()

    if step % int(1 / args.train_agent_ratio) == 0:
        batch = next(agent_dataloader)
        agent_metrics = agent.training_step(batch, step, log_only=True, max_log=500)
        [metrics.update({f"agent/{key}": agent_metrics[key]}) for key in agent_metrics.keys()]

        diffusion_updates = int(args.train_diffusion_ratio / args.train_agent_ratio)
        diffusion_metrics = diffusion_trainer.train(diffusion_updates, step)
        [metrics.update({f"diffusion/{key}": diffusion_metrics[key]}) for key in diffusion_metrics.keys()]

    wandb.log(metrics, step=step)
    step += 1