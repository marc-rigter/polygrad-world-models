from ..base_world_model_only import base, args_to_watch, logbase

params = {
    'train_agent_ratio': 0.01,
    'guidance_lr': 3e-2,
}   
base.update(params)