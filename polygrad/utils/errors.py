import numpy as np


def compute_traj_errors(env, observations, actions, rewards, sim_states):
    """
    Observations and actions are [n_traj X traj_len X dim]
    Rewards is [n_traj X traj_len]
    """

    # loop through the observations and actions
    if not hasattr(env, "set_state"):
        return None, None

    max_steps = observations.shape[1] - 1
    error_lists = dict()
    rew_errors = []

    # loop over each imagined trajectory
    for ep_obs, ep_act, ep_rew, ep_sim_state in zip(
        observations, actions, rewards, sim_states
    ):
        # compute prediction error from initial state
        init_s = ep_obs[0, :]
        init_sim_state = ep_sim_state[0, :]

        # set initial simulator state
        init_s = np.clip(
            init_s, a_min=env.observation_space.low, a_max=env.observation_space.high
        )
        env.reset()
        env.set_state(init_s, init_sim_state)

        # loop over the next num_step steps
        for current_step in range(max_steps):
            a = ep_act[current_step]
            next_s_actual, r_actual, _, _, _ = env.step(a)

            # compute error in open-loop state prediction
            next_step = current_step + 1
            next_s_pred = ep_obs[next_step, :]
            obs_error = np.square(next_s_pred - next_s_actual).mean()
            if (next_step <= 20 or next_step % 5 == 0) or next_step == max_steps:
                if next_step not in error_lists:
                    error_lists[next_step] = [obs_error]
                else:
                    error_lists[next_step].append(obs_error)

            # only compute reward error for first step
            if current_step == 0:
                r_pred = ep_rew[0]
                r_error = np.square(r_pred - r_actual).mean()
                rew_errors.append(r_error)

    metrics = dict()
    for step in error_lists:
        metrics.update(
            {
                f"errors/dynamics_mse_{step:04}_step": np.array(
                    error_lists[step]
                ).mean(),
                f"errors/dynamics_mse_std_{step:04}_step": np.array(
                    error_lists[step]
                ).std(),
            }
        )

    metrics.update(
        {
            "errors/reward_mse": np.array(rew_errors).mean(),
            "errors/reward_mse_std": np.array(rew_errors).std(),
        }
    )

    return metrics
