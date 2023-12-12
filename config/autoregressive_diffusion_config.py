from .base_config import base, args_to_watch, logbase

params = {
    'agent': 'agent.autoregressive_diffusion_agent.AutoregressiveDiffusionAgent',
    'suite':'gym',
    'horizon': 2, # train on sequences of two states
    'rollout_steps': 300, # how far to rollout autoregressively
    'train_agent_ratio':0.0002,
}
base.update(params)