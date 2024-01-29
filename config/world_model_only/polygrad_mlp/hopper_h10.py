from .base_polygrad_mlp import base, args_to_watch, logbase

run_config = {
'env_name':'Hopper-v3',
'load_path': 'datasets/Hopper',
'horizon': 10,
'hidden_dim': 1024,
'noise_sched_tau': 0.1,
}
base.update(run_config)
