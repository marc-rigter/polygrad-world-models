from ..base_config import base, args_to_watch, logbase

run_config = {
'suite':'gym',
'env_name':'Hopper-v3',
'train_agent_ratio': 0.01,
'load_path': 'datasets/final_datasets_nov12/final-rl-runs-lowtrainratio_seed1_Hopper',
'load_step': 1000000,
'horizon': 50,
'guidance_lr': 3e-2,
'model': 'models.TransformerDenoiser',
'learning_rate': 1e-4,
}
base.update(run_config)
args_to_watch.extend([(key, key[:5]) for key in run_config.keys() if key not in ["renderer"]])
