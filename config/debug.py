from .base_config import base, args_to_watch, logbase

params = {
        'suite':'gym',
        'env_name':'Hopper-v3',
        'n_environment_steps': 10000,
        'pretrain_diffusion': 100,
        'log_interval': 1000,
        'eval_interval': 1000,
        'batch_size': 256,
        'agent_batch_size': 256,
        'hidden_dim': 128,
}
base.update(params)
