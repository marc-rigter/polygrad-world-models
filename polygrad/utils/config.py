import os
from collections.abc import Mapping
import importlib
import pickle
import polygrad.utils as utils
import torch
import numpy as np
from polygrad.agent.a2c import ActorCritic

def import_class(_class):
    if type(_class) is not str: return _class
    ## 'diffusion' on standard installs
    repo_name = __name__.split('.')[0]
    ## eg, 'utils'
    module_name = '.'.join(_class.split('.')[:-1])
    ## eg, 'Renderer'
    class_name = _class.split('.')[-1]
    ## eg, 'diffusion.utils'
    module = importlib.import_module(f'{repo_name}.{module_name}')
    ## eg, diffusion.utils.Renderer
    _class = getattr(module, class_name)
    print(f'[ utils/config ] Imported {repo_name}.{module_name}:{class_name}')
    return _class

def set_all_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

class Config(Mapping):

    def __init__(self, _class, verbose=True, savepath=None, device=None, **kwargs):
        self._class = import_class(_class)
        self._device = device
        self._dict = {}

        for key, val in kwargs.items():
            self._dict[key] = val

        if verbose:
            print(self)

        if savepath is not None:
            savepath = os.path.join(*savepath) if type(savepath) is tuple else savepath
            pickle.dump(self, open(savepath, 'wb'))
            print(f'[ utils/config ] Saved config to: {savepath}\n')

    def __repr__(self):
        string = f'\n[utils/config ] Config: {self._class}\n'
        for key in sorted(self._dict.keys()):
            val = self._dict[key]
            string += f'    {key}: {val}\n'
        return string

    def __iter__(self):
        return iter(self._dict)

    def __getitem__(self, item):
        return self._dict[item]

    def __len__(self):
        return len(self._dict)

    def __getattr__(self, attr):
        if attr == '_dict' and '_dict' not in vars(self):
            self._dict = {}
            return self._dict
        try:
            return self._dict[attr]
        except KeyError:
            raise AttributeError(attr)

    def __call__(self, *args, **kwargs):
        instance = self._class(*args, **kwargs, **self._dict)
        if self._device:
            instance = instance.to(self._device)
        return instance


def create_configs(args, env):
    dataset_config = Config(
        args.loader,
        savepath=(args.savepath, 'dataset_config.pkl'),
        horizon=args.horizon,
        normalizer=args.normalizer,
        preprocess_fns=args.preprocess_fns,
        use_padding=args.use_padding,
        max_path_length=args.max_path_length,
        norm_keys=args.norm_keys,
    )

    if args.renderer is not None:
        render_config = Config(
            args.renderer,
            savepath=(args.savepath, 'render_config.pkl'),
            env_name=args.env_name,
        )
    else:
        render_config = None
        
    observation_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    model_config = Config(
        args.model,
        savepath=(args.savepath, 'model_config.pkl'),
        horizon=args.horizon,
        transition_dim=observation_dim + 2,
        cond_dim=action_dim,
        dim_mults=args.dim_mults,
        attention=args.attention,
        dropout=args.dropout,
        scale_obs=args.scale_obs,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        embed_dim=args.embed_dim,
    )

    diffusion_config = Config(
        args.diffusion,
        savepath=(args.savepath, 'diffusion_config.pkl'),
        horizon=args.horizon,
        observation_dim=observation_dim,
        action_dim=action_dim,
        temporal_loss_weight=args.temporal_loss_weight,
        n_timesteps=args.n_diffusion_steps,
        loss_type=args.loss_type,
        clip_denoised=args.clip_denoised,
        predict_epsilon=args.predict_epsilon,
        action_weight=args.action_weight,
        loss_weights=args.loss_weights,
        loss_discount=args.loss_discount,
        noise_sched_tau=args.noise_sched_tau,
        mask_obs=args.mask_obs,
        max_prediction_weight=args.max_prediction_weight,
        action_condition_noise_scale=args.action_condition_noise_scale,
    )

    trainer_config = Config(
        utils.Trainer,
        savepath=(args.savepath, 'trainer_config.pkl'),
        train_batch_size=args.batch_size,
        train_lr=args.learning_rate,
        ema_decay=args.ema_decay,
        sample_freq=args.log_interval,
        save_freq=args.save_freq,
        label_freq=int(args.n_train_steps // args.n_saves),
        save_parallel=args.save_parallel,
        results_folder=args.savepath,
        bucket=args.bucket,
        n_reference=args.n_reference,
    )

    agent_config = Config(
        args.agent,
        log_path=args.savepath,
        guidance_scale=args.guidance_scale,
        log_interval=args.log_interval,
        tune_guidance=args.tune_guidance,
        guidance_lr=args.guidance_lr,
        guidance_type=args.guidance_type,
        action_guidance_noise_scale=args.action_guidance_noise_scale,
        update_states=args.update_states,
        clip_std=args.clip_std,
        states_for_guidance=args.states_for_guidance,
        rollout_steps=args.rollout_steps,
        clip_state_change=args.clip_state_change,
    )

    ac_config = Config(
        ActorCritic,
        in_dim=observation_dim,
        out_actions=action_dim,
        actor_dist=args.actor_dist, 
        min_std=args.min_std,
        lambda_gae=args.lambda_gae,
        entropy_weight=args.entropy_weight,
        lr_actor=args.lr_actor,
        lr_critic=args.lr_critic,
        grad_clip=args.ac_grad_clip,
        gamma=args.gamma,
        normalize_adv=args.normalize_adv,
        fixed_std=args.fixed_std,
        learned_std=args.learned_std,
        init_std=args.init_std,
        ema=args.ema,
        ac_use_normed_inputs=args.ac_use_normed_inputs,
        target_update=args.target_update,
        actorlr_lr=args.actorlr_lr,
        update_actor_lr=args.update_actor_lr,
        linesearch=args.linesearch,
        linesearch_tolerance=args.linesearch_tolerance,
        linesearch_ratio=args.linesearch_ratio,
    )

    configs = {
        'dataset_config': dataset_config,
        'render_config': render_config,
        'model_config': model_config,
        'diffusion_config': diffusion_config,
        'trainer_config': trainer_config,
        'agent_config': agent_config,
        'ac_config': ac_config,

    }

    return configs
