from .base_polygrad_transformer import base, args_to_watch, logbase

run_config = {
'env_name':'HalfCheetah-v3',
'load_path': 'datasets/HalfCheetah',
'horizon': 50,
}
base.update(run_config)
