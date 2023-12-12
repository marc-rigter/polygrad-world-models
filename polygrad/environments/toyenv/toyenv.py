import numpy as np
import gym
from gym import Env, spaces
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

class ToyEnv(Env):
    def __init__(self):
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,))
        self.observation_space = spaces.Box(low=np.array([-1, -1]), high=np.array([1, 1]), shape=(2,))
        self.step_size = 0.02
        self.noise_size = 0.001
        self._max_episode_steps = 50

    def step(self, action):
        assert self.action_space.contains(action)

        # convert to radians
        action = action * np.pi
        dx = self.step_size * np.cos(action)
        dy = self.step_size * np.sin(action)
        self.state += np.array([dx[0], dy[0]])
        self.state += np.random.normal(size=2, scale=self.noise_size)

        # clip state
        self.state = np.clip(self.state, -1, 1)
        self.t += 1
        
        if self.t >= self._max_episode_steps:
            done = True
        else:
            done = False
        rew = action[0] * 0.1
        return self.state, rew, done, {}

    def reset(self):
        self.state = np.random.uniform(low=-0.5, high=0.5, size=2)
        self.t = 0
        return self.state

if __name__ == "__main__":
    env = ToyEnv()
    plt.figure()
    steps = 20000
    action_noise = 0.01

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

    print(next_states[:50] - states[:50])
    np.savez(
        "toyenv_dataset.npz", 
        observations=states,
        actions=actions,
        next_observations=next_states,
        rewards=rewards,
        terminals=terminals,
        timeouts=timeouts
    )


class ToyEnvRenderer:

    def __init__(self, env):
        pass

    def composite(self, savepath, paths, act, rew):

        plt.figure()
        for path in paths:
            plt.plot(path[:,0], path[:,1], alpha=0.5, color="k")
            plt.plot(path[0, 0], path[0, 1], alpha=1.0, marker="o", color="g")
            plt.plot(path[-1, 0], path[-1, 1], alpha=1.0, marker="o", color="r")
        plt.xlim(-1, 1)
        plt.ylim(-1, 1)
        ax = plt.gca()
        ax.set_aspect('equal')

        if savepath is not None:
            plt.savefig(savepath, dpi=300)
            print(f'Saved {len(paths)} samples to: {savepath}')
        plt.close()
