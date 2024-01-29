from ..base_world_model_only import base, args_to_watch, logbase

params = {
    'diffusion_method': 'autoregressive',
    'horizon': 2, # train on sequences of two states
    'rollout_steps': 300, # how far to rollout autoregressively
    'train_agent_ratio':0.0002,
}
base.update(params)