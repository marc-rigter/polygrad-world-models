from ..base_config import base, args_to_watch, logbase

run_config = {
'suite':'gym',
'env_name':'Hopper-v3',
'train_agent_ratio': 0.01,
'load_path': 'datasets/Hopper',
'load_step': 1000000,
'horizon': 50,
'hidden_dim': 2048,
'guidance_lr': 3e-2,
'noise_sched_tau': 0.1,
}
base.update(run_config)
args_to_watch.extend([(key, key[:5]) for key in run_config.keys() if key not in ["renderer"]])
