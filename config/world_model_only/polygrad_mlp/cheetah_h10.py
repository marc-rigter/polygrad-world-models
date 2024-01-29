from .base_polygrad_mlp import base, args_to_watch, logbase

run_config = {
'env_name':'HalfCheetah-v3',
'load_path': 'datasets/HalfCheetah',
'horizon': 10,
'hidden_dim': 1024,
}
base.update(run_config)
