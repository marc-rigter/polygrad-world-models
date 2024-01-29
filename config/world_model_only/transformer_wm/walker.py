from .base_transformer_wm import base, args_to_watch, logbase

run_config = {
'env_name':'Walker2d-v3',
'load_path': 'datasets/Walker2d',
}
base.update(run_config)
