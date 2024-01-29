from .base_polygrad_transformer import base, args_to_watch, logbase

run_config = {
'env_name':'Hopper-v3',
'load_path': 'datasets/Hopper',
'horizon': 200,
}
base.update(run_config)
