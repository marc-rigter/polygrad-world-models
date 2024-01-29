from .base_autoregressive_diffusion import base, args_to_watch, logbase

run_config = {
'env_name':'HalfCheetah-v3',
'load_path': 'datasets/HalfCheetah',
}
base.update(run_config)
