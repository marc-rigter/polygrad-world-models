from ..base_config import base, args_to_watch, logbase

run_config = {
'suite':'gym',
'env_name':'HalfCheetah-v3',
'n_environment_steps': 1500000,
}
base.update(run_config)
args_to_watch.extend([(key, key[:5]) for key in run_config.keys() if key not in ["renderer"]])
