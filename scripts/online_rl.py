import polygrad.utils as utils
import torch
import wandb
import numpy as np
from polygrad.utils.evaluation import evaluate_policy
from polygrad.utils.envs import create_env
from polygrad.utils.timer import Timer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# -----------------------------------------------------------------------------#
# ----------------------------------- setup -----------------------------------#
# -----------------------------------------------------------------------------#


class Parser(utils.Parser):
    config: str = "config.simple_maze"


args = Parser().parse_args()

expl_env = create_env(args.env_name, args.suite)
eval_env = create_env(args.env_name, args.suite)
random_episodes = utils.rl.random_exploration(args.n_prefill_steps, expl_env)

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

utils.report_parameters(model)
wandb.init(project=args.project, group=args.group, config=args, name=args.run_name)

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


def reset_episode():
    done = False
    state, _ = expl_env.reset()
    episode = {
        "observations": [],
        "actions": [],
        "next_observations": [],
        "rewards": [],
        "terminals": [],
        "sim_states": [],
    }
    t = 0
    return state, done, episode, t


# ---------------------------- Main Loop ----------------------------------#

state, done, episode, t = reset_episode()

step = 0
timer = Timer()
train_metrics_interval = 200
while step < args.n_environment_steps:
    log = False
    metrics = dict()

    # step the policy in the real environment
    policy_dist = ac.forward_actor(
        torch.from_numpy(state).float().to(device), normed_input=False
    )
    act = policy_dist.sample().cpu().detach().numpy()
    next_state, rew, term, trunc, info = expl_env.step(act)
    done = term or trunc
    t += 1

    episode["observations"].append(state.copy())
    episode["actions"].append(act.copy())
    episode["next_observations"].append(next_state.copy())
    episode["rewards"].append(rew.copy())
    episode["terminals"].append(term)
    if "sim_state" in info.keys():
        episode["sim_states"].append(info["sim_state"].copy())
    else:
        episode["sim_states"].append(None)

    state = next_state
    if done or t >= args.max_path_length:
        episode = {key: np.array(episode[key]) for key in episode.keys()}
        episode["timeouts"] = np.array([False] * len(episode["rewards"]))
        ret = np.sum(episode["rewards"])
        print("Episode Return: ", ret, "Length: ", len(episode["rewards"]))
        metrics.update({"expl/return": ret, "expl/length": len(episode["rewards"])})
        dataset.add_episode(episode)
        state, done, episode, t = reset_episode()

        if args.update_normalization:
            dataset.update_normalizers()

    if step % int(1 / args.train_agent_ratio) == 0:
        if step >= args.pretrain_diffusion:
            batch = next(agent_dataloader)
            agent_metrics = agent.training_step(batch, step)
            if step % train_metrics_interval == 0:
                [
                    metrics.update({f"agent/{key}": agent_metrics[key]})
                    for key in agent_metrics.keys()
                ]

        diffusion_updates = int(args.train_diffusion_ratio / args.train_agent_ratio)
        diffusion_metrics = diffusion_trainer.train(diffusion_updates, step)
        if step % train_metrics_interval == 0:
            [
                metrics.update({f"diffusion/{key}": diffusion_metrics[key]})
                for key in diffusion_metrics.keys()
            ]

    if step % args.log_interval == 0:
        dataset_metrics = dataset.get_metrics()
        [
            metrics.update({f"dataset/{key}": dataset_metrics[key]})
            for key in dataset_metrics.keys()
        ]
        metrics.update({"fps": timer.fps(step)})

    if args.save_freq is not None:
        if step % args.save_freq == 0:
            agent.save(args.savepath, step)

    if step % args.eval_interval == 0:
        eval_metrics = evaluate_policy(
            ac.forward_actor,
            eval_env,
            device,
            step,
            dataset,
            use_mean=True,
            n_episodes=20,
            renderer=renderer,
        )

    wandb.log(metrics, step=step)
    step += 1
