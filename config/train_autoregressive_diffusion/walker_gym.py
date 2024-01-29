from ..autoregressive_diffusion_config import base, args_to_watch, logbase

run_config = {
'suite':'gym',
'env_name':'Walker2d-v3',
'load_path': 'datasets/Walker2d',
'load_step': 1000000,
}
base.update(run_config)
args_to_watch.extend([(key, key[:5]) for key in run_config.keys() if key not in ["renderer"]])
