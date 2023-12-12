import numpy as np
import gym
from gym import Env, spaces
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

data_filename = "results/toyenv_samples.npz"
plot_filename = "toyenv_diffusion_samples.png"

# data_filename = "envs/toyenv/toyenv_dataset.npz"
# plot_filename = "toyenv_dataset.png"

steps_to_plot = 500

dataset = np.load(data_filename)

plt.xlim(-1, 1)
plt.ylim(-1, 1)
ax = plt.gca()
ax.set_aspect('equal')

states = dataset['observations']
actions = dataset['actions']
next_states = dataset['next_observations']
rewards = dataset['rewards']
terminals = dataset['terminals']
timeouts = dataset['timeouts']

# print(states[:50])
# print(actions[:50])
# print(rewards[:50])
# print(next_states[:50])
# print(terminals[:200])
print(next_states[:50] - states[:50])

# plot each trajectory
steps = min(states.shape[0], steps_to_plot)
step = 0
while step < steps:
    traj = []
    done = False
    while not done:
        traj.append(states[step])
        done = terminals[step]
        step += 1
        if step == steps:
            break

    xs = [s[0] for s in traj]
    ys = [s[1] for s in traj]
    plt.plot(xs, ys, alpha=0.5, color="k")
    plt.plot(xs[0], ys[0], alpha=1.0, marker="o", color="g")
    plt.plot(xs[-1], ys[-1], alpha=1.0, marker="o", color="r")

plt.savefig(plot_filename, dpi=300)


# plot each step
steps = min(states.shape[0], steps_to_plot)
step = 0
plt.figure()
while step < steps:
    xs = [states[step][0], next_states[step][0]]
    ys = [states[step][1], next_states[step][1]]
    plt.plot(xs, ys, alpha=0.5, color="k")
    step += 1
    # plt.plot(xs[0], ys[0], alpha=1.0, marker="o", color="g")
    # plt.plot(xs[-1], ys[-1], alpha=1.0, marker="o", color="r")

plt.savefig("plot_each_step.png", dpi=300)