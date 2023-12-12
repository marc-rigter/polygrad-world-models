import numpy as np

def rollout_policy(env, policy, horizon, init_states, dataset, device):
    all_states = np.zeros((init_states.shape[0], horizon, env.observation_space.shape[0]))
    all_actions = np.zeros((init_states.shape[0], horizon, env.action_space.shape[0]))
    for i, init_state in enumerate(init_states):
        env.reset()
        env.set_state(init_state)
        state = init_state
        for t in range(horizon):
            all_states[i, t, :] = state
            action = policy(torch.from_numpy(state).float().to(device), normed_input=False).sample().cpu().detach().numpy()
            all_actions[i, t, :] = action
            state, reward, _, _, _ = env.step(action)  
    return all_states, all_actions