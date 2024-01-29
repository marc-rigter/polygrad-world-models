from .base_polygrad_mlp import base, args_to_watch, logbase

run_config = {
'env_name':'Walker2d-v3',
'load_path': 'datasets/Walker2d',
'horizon': 10,
'hidden_dim': 1024
}
base.update(run_config)
