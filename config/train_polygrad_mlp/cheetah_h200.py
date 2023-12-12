from ..base_config import base, args_to_watch, logbase

run_config = {
'suite':'gym',
'env_name':'HalfCheetah-v3',
'train_agent_ratio': 0.01,
'load_path': 'datasets/final_datasets_nov12/final-rl-runs-lowtrainratio_seed1_HalfCheetah',
'load_step': 1000000,
'horizon': 200,
'hidden_dim': 2048,
'guidance_lr': 3e-2,
}
base.update(run_config)
args_to_watch.extend([(key, key[:5]) for key in run_config.keys() if key not in ["renderer"]])
