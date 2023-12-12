from polygrad.environments.simple_maze.simple_maze import SimpleMaze
from dm_control import suite
from gym.spaces import Box
from gym.core import Env
import gym
import numpy as np
import os
import math
from dm_env import specs

def _spec_to_box(spec, dtype=np.float32):
    def extract_min_max(s):
        assert s.dtype == np.float64 or s.dtype == np.float32
        dim = int(np.prod(s.shape))
        if type(s) == specs.Array:
            bound = np.inf * np.ones(dim, dtype=np.float32)
            return -bound, bound
        elif type(s) == specs.BoundedArray:
            zeros = np.zeros(dim, dtype=np.float32)
            return s.minimum + zeros, s.maximum + zeros
        else:
            raise NotImplementedError

    mins, maxs = [], []
    for s in spec:
        mn, mx = extract_min_max(s)
        mins.append(mn)
        maxs.append(mx)
    low = np.concatenate(mins, axis=0).astype(dtype)
    high = np.concatenate(maxs, axis=0).astype(dtype)
    assert low.shape == high.shape
    return Box(low, high, dtype=dtype)


def _flatten_obs(obs, dtype=np.float32):
    obs_pieces = []
    for v in obs.values():
        flat = np.array([v]) if np.isscalar(v) else v.ravel()
        obs_pieces.append(flat)
    return np.concatenate(obs_pieces, axis=0).astype(dtype)


class DMCGym(Env):
    def __init__(
        self,
        domain,
        task,
        task_kwargs={},
        environment_kwargs={},
        rendering="egl",
    ):
        """TODO comment up"""

        # for details see https://github.com/deepmind/dm_control
        assert rendering in ["glfw", "egl", "osmesa"]
        os.environ["MUJOCO_GL"] = rendering

        self._env = suite.load(
            domain,
            task,
            task_kwargs,
            environment_kwargs,
        )

        # placeholder to allow built in gymnasium rendering
        self.render_mode = "rgb_array"

        self._observation_space = _spec_to_box(self._env.observation_spec().values())
        self._action_space = _spec_to_box([self._env.action_spec()])
        self.domain = domain

        # set seed if provided with task_kwargs
        if "random" in task_kwargs:
            seed = task_kwargs["random"]
            self._observation_space.seed(seed)
            self._action_space.seed(seed)

    def __getattr__(self, name):
        """Add this here so that we can easily access attributes of the underlying env"""
        return getattr(self._env, name)

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    @property
    def reward_range(self):
        """DMC always has a per-step reward range of (0, 1)"""
        return 0, 1

    def step(self, action):
        if action.dtype.kind == "f":
            action = action.astype(np.float32)
        assert self._action_space.contains(action)
        sim_state = self.get_mujoco_state()
        timestep = self._env.step(action)
        observation = _flatten_obs(timestep.observation)
        reward = timestep.reward
        termination = False  # we never reach a goal
        truncation = timestep.last()
        next_sim_state = self.get_mujoco_state()
        info = {"discount": timestep.discount,
                "next_sim_state": next_sim_state,
                "sim_state": sim_state
                }
        if not isinstance(reward, np.ndarray):
            reward = np.array([reward])
            
        # assert np.allclose(self.obs_to_state(observation, required_state_info=sim_state), sim_state)
        return observation, reward, termination, truncation, info
    
    def set_state(self, obs, sim_state=None):
        """Set the state of the environment to the given observation"""
        if sim_state is None:
            sim_state = self.obs_to_state(obs)
        self._env.physics.set_state(sim_state)

    def get_mujoco_state(self):
        state = self._env.physics.get_state()
        if self.domain == "cartpole":
            while state[1] < 0:
                state[1] += 2 * np.pi
            while state[1] > 2 * np.pi:
                state[1] -= 2 * np.pi
        return state

    def reset(self, seed=None, options=None):
        timestep = self._env.reset()
        observation = _flatten_obs(timestep.observation)
        info = {}
        return observation, info

    def render(self, height, width, camera_id=0):
        return self._env.physics.render(height=height, width=width, camera_id=camera_id)
    
    def obs_to_state(self, obs, required_state_info=None):
        if self.domain == "cartpole":
            magnitude = np.sqrt(obs[1] ** 2 + obs[2] ** 2)
            angle = np.arctan2(obs[2] / magnitude, obs[1] / magnitude)
            if angle < 0:
                angle += 2 * np.pi
            state = np.array([obs[0], angle, obs[3], obs[4]])
        elif self.domain == "ball_in_cup":
            state = obs
        elif self.domain == "hopper":
            # first state is horizontal position which is not included
            # in observation. So its set arbitrarily to zero.
            # Last two obs are "touch" which are not in state.
            if required_state_info is None:
                state = np.concatenate([np.array([0.0]), obs[:-2]])
            else:
                state = np.concatenate([np.array([required_state_info[0]]), obs[:-2]])
        elif self.domain == "cheetah":
            if required_state_info is None:
                state = np.concatenate([np.array([0.0]), obs])
            else:
                state = np.concatenate([np.array([required_state_info[0]]), obs])
        else:
            raise NotImplementedError
        return state
    
class GymWrapper(Env):

    def __init__(self, env):
        self._env = env

    def __getattr__(self, name):
        return getattr(self._env, name)
    
    def sim_state_to_array(self, sim_state):
        return np.concatenate([sim_state.qpos, sim_state.qvel])
    
    def step(self, action):
        if hasattr(self._env, "sim"):
            sim_state = self._env.sim.get_state()
        else: 
            sim_state = np.concatenate([self.data.qpos.flat.copy(), self.data.qvel.flat.copy()])
        observation, reward, termination, truncation, info = self._env.step(action)
        reward = np.array([reward])
        if hasattr(self._env, "sim"):
            next_sim_state = self._env.sim.get_state()
            info.update({"next_sim_state": self.sim_state_to_array(next_sim_state),
                "sim_state": self.sim_state_to_array(sim_state)
            })
        else:
            next_sim_state = np.concatenate([self.data.qpos.flat.copy(), self.data.qvel.flat.copy()])
            info.update({"next_sim_state": next_sim_state,
                "sim_state": sim_state
            })
        return observation, reward, termination, truncation, info
    
    def set_state(self, obs, sim_state):
        self._env.set_state(sim_state[:self._env.model.nq], sim_state[self._env.model.nq:])

    def reset(self):
        observation, info = self._env.reset()
        return observation, info
    
    def get_terminals(self, states):
        if "Hopper" in self._env.spec.id:
            height = states[:, :, 0]
            angle = states[:, :, 1]
            not_done =  np.isfinite(states).all(axis=-1) \
                        * np.abs(states < 100).all(axis=-1) \
                        * (height > .7) \
                        * (np.abs(angle) < .2)
            done = ~not_done
            return done
        elif "Walker" in self._env.spec.id:
            height = states[:, :, 0]
            angle = states[:, :, 1]
            not_done =  (height > 0.8) \
                        * (height < 2.0) \
                        * (angle > -1.0) \
                        * (angle < 1.0)
            done = ~not_done
            return done
        elif "Cheetah" in self._env.spec.id:
            return np.full((states.shape[0], states.shape[1]), False)
        else:
            raise NotImplementedError
    
class ClippedActionWrapper(gym.ActionWrapper):

    def __init__(self, env):
        super().__init__(env)
        self.env = env

    def step(self, act):
        act = np.clip(act, a_min=self.env.action_space.low, a_max=self.env.action_space.high)
        return self.env.step(act)
    
    
class ActionRepeatWrapper(gym.Wrapper):
    def __init__(self, env, repeat=2):
        super().__init__(env)
        self.env = env
        self._repeat = 2
        
    def step(self, act):
        reward = 0.
        for _ in range(self._repeat):
            observation, r, termination, truncation, info = self.env.step(act)
            reward += r
            if termination:
                break
        return observation, reward, termination, truncation, info
    
class MinEpisodeLengthWrapper(gym.Wrapper):
    def __init__(self, env, min_episode_length=100):
        super().__init__(env)
        self._env = env
        self._min_episode_length = min_episode_length
        self._episode_length = 0
        
    def step(self, act):
        observation, reward, termination, truncation, info = self._env.step(act)
        self._episode_length += 1
        if termination and self._episode_length < self._min_episode_length:
            termination = False
            reward = np.array([-1.0])
        return observation, reward, termination, truncation, info
    
    def reset(self):
        self._episode_length = 0
        return self.env.reset()
            
def create_env(env_name, suite='gym'):
    if "SimpleMaze" in env_name:
        env = gym.make(env_name)
    elif suite == 'gym':
        env = gym.make(env_name)
        env = GymWrapper(env)
        # env = MinEpisodeLengthWrapper(env)
    elif suite == 'dmc':
        task = env_name.split('_')[-1]
        domain = env_name.replace(f'_{task}', '')
        env = DMCGym(domain, task)
        env = ActionRepeatWrapper(env, repeat=2)
    else:
        raise NotImplementedError
    env = ClippedActionWrapper(env)
    return env
