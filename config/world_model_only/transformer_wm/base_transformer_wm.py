from ..base_world_model_only import base, args_to_watch, logbase

run_config = {
'model': 'models.TransformerDenoiser',
'learning_rate': 1e-4,
'horizon': 16, # context length
'rollout_steps': 300, # full rollout horizon
}
base.update(run_config)