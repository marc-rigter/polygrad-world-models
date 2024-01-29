from ..base_config import base, args_to_watch, logbase

run_config = {
'env_name':'Hopper-v3',
'noise_sched_tau': 0.1,
'n_environment_steps': 1000000,
}
base.update(run_config)
