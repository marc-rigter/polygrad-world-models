import numpy as np
import gym
from gym import Env, spaces
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import os
import torch

class SimpleMaze(Env):
    def __init__(self):
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,))
        self.observation_space = spaces.Box(low=np.array([-1, -1]), high=np.array([1, 1]), shape=(2,))
        self.step_size = 0.075
        self.noise_size = 0.0
        self._max_episode_steps = 100

        self.border = 0.95
        self.circle_radius = 0.3
        self.goal_loc = np.array([0.7, 0.7])
        self.goal_rad = 0.1

    def stuck_zone(self, state):
        if abs(state[0]) > self.border or abs(state[1]) > self.border:
            return True

        if np.sqrt(state[0]**2 + state[1]**2) < self.circle_radius:
            return True

        # if self.goal_dist(state) < self.goal_rad:
        #    return True

        return False

    def goal_dist(self, state):
        return np.sqrt((state[0] - self.goal_loc[0])**2 + (state[1] - self.goal_loc[1])**2)
    
    def init_cond_for_viz(self):
        init_states = np.array([
            [-0.3, -0.8],
            [-0.8, -0.3],
            [-0.7, 0.55],
            [0.55, -0.7],
            [0.0, 0.8],
            [0.8, 0.0]
        ])
        return init_states

    def step(self, action):
        action = np.clip(action, -1, 1)
        assert self.action_space.contains(action)
        prev_goal_dist = self.goal_dist(self.state)

        action = action * self.step_size
        dx = action[0]
        dy = action[1]
        # action = action * np.pi   # convert to radians
        # dx = self.step_size * np.cos(action)
        # dy = self.step_size * np.sin(action)
        new_state = self.state + np.array([dx, dy])
        if not self.stuck_zone(new_state):
          self.state = new_state

        new_goal_dist = self.goal_dist(self.state)
        reward = 1 - np.tanh(new_goal_dist * 0.5)
        #reward = (prev_goal_dist - new_goal_dist) / self.step_size

        # if self.goal_dist(self.state) < self.goal_rad:
        #   reward += 1.
        # clip state
        # self.state = np.clip(self.state, -1, 1)
        self.t += 1

        if self.t >= self._max_episode_steps:
            done = True
        else:
            done = False
        return self.state.copy(), reward.copy(), done, done, {}

    def set_state(self, state, sim_state=None):
        self.state = state

    def reset(self):
        stuck = True
        while stuck:
            state = np.random.uniform(low=-1, high=1, size=2)
            stuck = self.stuck_zone(state)
        # state = np.array([-0.5, -0.5])
        self.state = state
        self.t = 0
        return self.state.copy(), {}

if __name__ == "__main__":
    env = SimpleMaze()
    plt.figure()
    steps = 200000
    action_noise = 0.05

    actions = np.zeros((steps, 1))
    states = np.zeros((steps, 2))
    rewards = np.zeros((steps))
    next_states = np.zeros((steps, 2))
    terminals = np.zeros((steps), dtype=bool)
    timeouts = np.zeros((steps), dtype=bool)

    step = 0
    while step < steps:
        prev_state = env.reset()
        done = False

        action_mean = np.random.uniform(low=-1, high=1, size=(1)) 
        while not done:
            act = np.float32(np.clip(action_mean + np.random.normal() * action_noise, -1, 1))
            state, rew, done, info = env.step(act) 
            states[step] = prev_state
            actions[step] = act
            rewards[step] = rew
            next_states[step] = state
            timeouts[step] = done

            prev_state = state.copy()

            step += 1
            if step == steps:
                break

    print(next_states[:5] - states[:5])
    dir_path = os.path.dirname(os.path.realpath(__file__))
    fpath = os.path.join(dir_path, "simple_maze_dataset.npz")

    np.savez(
        fpath, 
        observations=states,
        actions=actions,
        next_observations=next_states,
        rewards=rewards,
        terminals=terminals,
        timeouts=timeouts
    )


class SimpleMazeRenderer:

    def __init__(self, env_name="SimpleMaze"):
        self.env = SimpleMaze()

    def composite(self, paths, actions=None, rewards=None, savepath=None, real_obs=None, real_act=None):

        fig = plt.figure(dpi=250)
        ax = plt.gca()
        # p1 = mpl.patches.Rectangle((-1, -1), 1-self.env.border, 2, color="k")
        # p2 = mpl.patches.Rectangle((self.env.border, -1), 1-self.env.border, 2, color="k")
        # p3 = mpl.patches.Rectangle((-1, -1), 2, 1-self.env.border, color="k")
        # p4 = mpl.patches.Rectangle((-1, self.env.border), 2, 1-self.env.border, color="k")
        p5 = mpl.patches.Circle((0, 0), self.env.circle_radius, color="k")
        p6 = mpl.patches.Circle(self.env.goal_loc, self.env.goal_rad, color="gold", alpha=1.0)
        # ax.add_patch(p1)
        # ax.add_patch(p2)
        # ax.add_patch(p3)
        # ax.add_patch(p4)
        ax.add_patch(p5)
        ax.add_patch(p6)

        for act, path, rew in zip(actions, paths, rewards):
            colors = np.linspace(0, 100, len(path))
            plt.scatter(path[:,0], path[:,1], c=colors, cmap='plasma', linewidth=1.5, marker="o")
            # for a, s in zip(act, path):
            #     plt.arrow(s[0], s[1], a[0] * self.env.step_size, a[1]* self.env.step_size, color="r", alpha=1.0, lw=0.5)

            # ret = np.sum(rew)
            # plt.text(path[0, 0], path[0, 1], f"{ret:.2f}", fontsize=3, color="k")

        # if (real_obs is not None) and (real_act is not None):
        #     for obss, acts in zip(real_obs, real_act):
        #         plt.plot(obss[:,0], obss[:,1], color="grey", alpha=0.2)
        #         plt.plot(obss[-1, 0], obss[-1, 1], marker="o", color="y", alpha=0.2)
        #         plt.plot(obss[0, 0], obss[0, 1], marker="o", color="g", alpha=0.2)

        #         for a, s in zip(acts, obss):
        #             plt.arrow(s[0], s[1], a[0] * self.env.step_size, a[1]* self.env.step_size, color="r", alpha=0.3, lw=0.5)
                
        plt.xlim(-1, 1)
        plt.ylim(-1, 1)
        ax.set_aspect('equal')

        # Hide X and Y axes label marks
        ax.xaxis.set_tick_params(labelbottom=False)
        ax.yaxis.set_tick_params(labelleft=False)

        # Hide X and Y axes tick marks
        ax.set_xticks([])
        ax.set_yticks([])

        if savepath is not None:
            plt.savefig(savepath, dpi=500)
            print(f'Saved {len(paths)} samples to: {savepath}')
        plt.close()
        return fig

    def render_policy(self, policy, savepath=None):
        fig = plt.figure(dpi=250)
        ax = plt.gca()
        # p1 = mpl.patches.Rectangle((-1, -1), 1-self.env.border, 2, color="k")
        # p2 = mpl.patches.Rectangle((self.env.border, -1), 1-self.env.border, 2, color="k")
        # p3 = mpl.patches.Rectangle((-1, -1), 2, 1-self.env.border, color="k")
        # p4 = mpl.patches.Rectangle((-1, self.env.border), 2, 1-self.env.border, color="k")
        p5 = mpl.patches.Circle((0, 0), self.env.circle_radius, color="k")
        p6 = mpl.patches.Circle(self.env.goal_loc, self.env.goal_rad, color="gold", alpha=1.0)
        # ax.add_patch(p1)
        # ax.add_patch(p2)
        # ax.add_patch(p3)
        # ax.add_patch(p4)
        ax.add_patch(p5)
        ax.add_patch(p6)

        for s1 in np.linspace(-0.8, 0.8, 8):
            for s2 in np.linspace(-0.8, 0.8, 8):
                if self.env.stuck_zone(np.array([s1, s2])):
                    continue
                state = np.array([s1, s2])
                policy_dist = policy(torch.from_numpy(state).float().to("cuda:0"), normed_input=False)
                policy_mean = policy_dist.mean.cpu().detach().numpy()
                plt.arrow(s1, s2, policy_mean[0] * self.env.step_size * 1.0, policy_mean[1] * self.env.step_size * 1.0, color="r", alpha=1.0, lw=3.0, head_width=0.02)
        plt.xlim(-1, 1)
        plt.ylim(-1, 1)
        ax.set_aspect('equal')

        # Hide X and Y axes label marks
        ax.xaxis.set_tick_params(labelbottom=False)
        ax.yaxis.set_tick_params(labelleft=False)

        # Hide X and Y axes tick marks
        ax.set_xticks([])
        ax.set_yticks([])

        if savepath is not None:
            plt.savefig(savepath, dpi=500)
        return fig