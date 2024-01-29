from .base_autoregressive_diffusion import base, args_to_watch, logbase

run_config = {
'env_name':'Hopper-v3',
'load_path': 'datasets/Hopper',
}
base.update(run_config)
