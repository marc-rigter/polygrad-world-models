from .base_polygrad_mlp import base, args_to_watch, logbase

run_config = {
'env_name':'HalfCheetah-v3',
'load_path': 'datasets/HalfCheetah',
'horizon': 200,
'hidden_dim': 2048,
}
base.update(run_config)
