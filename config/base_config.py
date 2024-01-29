import socket
import datetime

from polygrad.utils import watch

#------------------------ base ------------------------#

## automatically make experiment names for planning
## by labelling folders with these args

args_to_watch = [
    ('group', ''),
    ('timestamp', ''),
    ('seed', 'seed'),
]

logbase = 'logs'

base = {
    ## misc
    'group': 'default',
    'project': 'polygrad_world_models',
    'run_name': None,
    'seed': 0,
    'env_name': 'CustomPendulum-v0',
    'suite': 'gym',
    'log_interval': 10000,
    'timestamp': '{:%Y-%m-%d-%H:%M:%S}'.format(datetime.datetime.now()),
    'save_freq': None,
    'n_environment_steps': 1000000,
    'load_path': None,
    'load_step': None,

    ## model
    'model': 'models.ResidualMLPDenoiser',
    'diffusion': 'models.GaussianDiffusion',
    'horizon': 10,
    'n_diffusion_steps': 128,
    'action_weight': 1,
    'loss_weights': None,
    'loss_discount': 1,
    'predict_epsilon': False,
    'dim_mults': (1, 2, 4, 8),
    'attention': False,
    'renderer': None,
    'mask_obs': False,
    'dropout': 0.0,
    'embed_dim': 128,
    'hidden_dim': 1024,
    'num_layers': 6,

    ## dataset
    'loader': 'datasets.OnlineSequenceDataset',
    'normalizer': 'GaussianNormalizer',
    'preprocess_fns': [],
    'clip_denoised': False,
    'use_padding': True,
    'max_path_length': 1000,
    'norm_keys': ['observations', 'actions', 'rewards', 'terminals'],
    'update_normalization': True,

    ## serialization
    'logbase': logbase,
    'prefix': 'rl',
    'exp_name': watch(args_to_watch),

    ## diffusion training
    'train_interval': 10,
    'train_diffusion_ratio': 1.0,
    'temporal_loss_weight': 1.0,
    'loss_type': 'l2',
    'batch_size': 256,
    'learning_rate': 3e-4,
    'ema_decay': 0.99, 
    'n_saves': 5,
    'save_parallel': False,
    'n_reference': 50,
    'bucket': None,
    'n_train_steps': 1e6,
    'pretrain_diffusion': 1000,
    'noise_sched_tau': 1.0,
    'scale_obs': 1.0,
    'max_prediction_weight': 1.0,
    'clip_std': 3.0,

    ## agent
    'agent': 'agent.polygrad_wm_agent.PolygradWMAgent',
    'n_prefill_steps': 5000,
    'train_agent_ratio': 0.25,
    'agent_batch_size': 1024,
    'guidance_scale': 0.005,
    'clip_state_change': 0.25,
    'tune_guidance': True,
    'guidance_type': 'grad', # Options: ['grad', 'sample', 'none']
    'guidance_lr': 3e-3,
    'entropy_weight': 1e-5,
    'entropy_target': -1.0,
    'lr_actor': 3e-5,
    'lr_critic': 3e-4,
    'ac_grad_clip': 0.1,
    'normalize_adv': True,
    'tune_entropy': False,
    'fixed_std': False,
    'learned_std': True,
    'decay_std_steps': 500000,
    'init_std': 1.0,
    'gamma': 0.99,
    'actor_dist': 'normal_tanh',
    'min_std': 0.1,
    'lambda_gae': 0.9,
    'eval_interval': 10000,
    'ema': 0.995,
    'log_video': False,
    'ac_use_normed_inputs': False,
    'target_update': 0.01,
    'lr_schedule': 'target', # Options: ['linear', 'constant', 'target']
    'lr_decay_steps': 1500000,
    'tune_actor_lr': 3e-4,
    'update_states': False,
    'states_for_guidance': 'recon', # ['recon', 'posterior_mean']
    'rollout_steps': None,

    # linesearch actor update
    'linesearch': True,
    'linesearch_tolerance': 0.2,
    'linesearch_ratio': 0.8,

    # action noise during diffusion
    'action_guidance_noise_scale': 1.0,
    'action_condition_noise_scale': 0.0,
}
