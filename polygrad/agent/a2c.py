import torch
import copy
import torch.nn as nn
import copy
import torch.nn.functional as F
import torch.distributions as D
import importlib
import wandb
from torch import Tensor
from polygrad.utils.training import EMA
from .functions import *
from .common import *
from polygrad.utils.evaluation import get_standardized_stats


class ActorCritic(nn.Module):

    def __init__(self,
                 in_dim,
                 out_actions,
                 normalizer,
                 hidden_dim=256,
                 min_std=0.01,
                 fixed_std=False,
                 decay_std_steps=500000,
                 init_std=0.5,
                 hidden_layers=2,
                 layer_norm=True,
                 gamma=0.999,
                 ema=0.995,
                 lambda_gae=0.8,
                 entropy_weight=1e-3,
                 entropy_target=-1,
                 tune_entropy=True,
                 target_interval=100,
                 lr_actor=1e-4,
                 lr_critic=3e-4,
                 lr_alpha=1e-2,
                 actor_grad='reinforce',
                 actor_dist='normal_tanh',
                 normalize_adv=False,
                 grad_clip=None,
                 clip_logprob=True,
                 min_logprob=-10.0,
                 learned_std=False,
                 ac_use_normed_inputs=True,
                 target_update=0.02,
                 tune_actor_lr=3e-4,
                 lr_schedule='constant',
                 lr_decay_steps=1000000,
                 log_interval=20000,
                 linesearch=False,
                 linesearch_tolerance=0.25,
                 linesearch_ratio=0.8,
                 **kwargs
                 ):
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.in_dim = in_dim
        self.action_dim = out_actions
        self.gamma = gamma
        self.lambda_ = lambda_gae
        self.target_interval = target_interval
        self.actor_grad = actor_grad
        self.actor_dist = actor_dist
        self.min_std = min_std
        self.clip_logprob = clip_logprob
        self.normalizer = normalizer
        self.min_logprob = min_logprob * self.action_dim
        self.learned_std = learned_std
        self.fixed_std = fixed_std
        self.decay_std_steps = decay_std_steps
        self.init_std = init_std
        self.current_std = init_std
        self.use_normed_inputs = ac_use_normed_inputs
        self.lr_decay_steps = lr_decay_steps
        self.log_interval = log_interval
        self.last_log = -float('inf')

        self.linesearch = linesearch
        self.linesearch_tolerance = linesearch_tolerance
        self.linesearch_ratio = linesearch_ratio

        if not self.fixed_std and not self.learned_std:
            actor_out_dim = 2 * out_actions
        else:
            actor_out_dim = out_actions

        self.actor = MLP(in_dim, actor_out_dim, hidden_dim, hidden_layers, layer_norm).to(self.device)
        self.critic = MLP(in_dim, 1, hidden_dim, hidden_layers, layer_norm).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_target.requires_grad_(False)
        self.ema = EMA(ema)
        
        self.train_steps = 0

        if self.learned_std:
            self.logstd = AddBias((torch.ones(actor_out_dim)*np.log(self.init_std-self.min_std)).to(self.device))
            self._optimizer_actor = torch.optim.AdamW(list(self.actor.parameters()) + list(self.logstd.parameters()), lr=lr_actor)
        else:
            self._optimizer_actor = torch.optim.AdamW(self.actor.parameters(), lr=lr_actor)
        self._optimizer_critic = torch.optim.AdamW(self.critic.parameters(), lr=lr_critic)
        self.grad_clip = grad_clip
        self.normalize_adv = normalize_adv
        self.tune_entropy = tune_entropy
        self.entropy_target = entropy_target
        self.log_alpha = torch.log(torch.tensor(entropy_weight)).to(self.device)
        if self.tune_entropy:
            self.log_alpha.requires_grad_(True)
            self._optimizer_alpha = torch.optim.AdamW([self.log_alpha], lr=lr_alpha)
        
        self.lr_schedule = lr_schedule
        self.tune_actor_lr = tune_actor_lr
        self.target_update = target_update
        self.max_lr = lr_actor
        if self.lr_schedule == "target":
            self.log_actor_lr = torch.log(torch.tensor(lr_actor)).to(self.device)
            self.log_actor_lr.requires_grad_(True)
            self._optimizer_actor_lr = torch.optim.AdamW([self.log_actor_lr], lr=tune_actor_lr)

    def forward_actor(self, features: Tensor, normed_input=True) -> D.Distribution:
        """Takes as input either normalized or unnnormalized features. Outputs
        unnormalized action distribution. """

        if not normed_input and self.use_normed_inputs:
            features = self.normalizer.normalize(features, "observations")
        elif normed_input and not self.use_normed_inputs:
            features = self.normalizer.unnormalize(features, "observations")
            
        y = self.actor.forward(features).float()

        if self.actor_dist == 'normal_tanh':
            if not self.fixed_std and not self.learned_std:
                return normal_tanh(y, min_std=self.min_std)
            else:
                if len(y.shape) == 0 or y.shape[-1] != self.action_dim:
                    # TODO: Fix this.
                    y = y.unsqueeze(-1)

                if self.fixed_std:
                    std = self.current_std
                elif self.learned_std:
                    std = self.logstd(torch.zeros_like(y)).exp() + self.min_std
                else:
                    raise NotImplementedError
                return normal_tanh(y, fixed_std=std)
        else:
            raise NotImplementedError


    def forward_value(self, features: Tensor) -> Tensor:
        y = self.critic.forward(features)
        return y

    def update_alpha(self, policy_entropy):
        alpha_loss = -(self.log_alpha * (self.entropy_target - policy_entropy).detach()).mean()
        self._optimizer_alpha.zero_grad()
        alpha_loss.backward()
        self._optimizer_alpha.step()
        self.log_alpha.data = torch.clamp(self.log_alpha.data, min=-20, max=4)

    def update_actor_lr(self, update_size):
        loss = -(self.log_actor_lr * (self.target_update - update_size)).mean()
        self._optimizer_actor_lr.zero_grad()
        loss.backward()
        self._optimizer_actor_lr.step()
        self.log_actor_lr.data = torch.clamp(self.log_actor_lr.data, max=np.log(self.max_lr))
        self._optimizer_actor.param_groups[0]['lr'] = torch.exp(self.log_actor_lr).item()

    def log_action_distr(self, policy_distr, actions, step):
        policy_mean = policy_distr.mean
        policy_std = policy_distr.stddev
        std = policy_std.mean().item()
        act = (actions - policy_mean).detach().cpu().numpy().flatten()
        act_counts, act_edges  = np.histogram(act, bins=500, range=(-3.5 * std, 3.5 * std))
        act_y = act_counts / len(act) / (act_edges[1] - act_edges[0])
        act_x = (act_edges[1:] + act_edges[:-1]) / 2

        target_y = np.exp(-0.5 * act_x**2 / std**2) / (std * np.sqrt(2 * np.pi))
        metrics = {
                   f"distr/step_{step}_act_density": wandb.plot.line_series(
                       xs=act_x,
                       ys=[act_y, target_y],
                       keys=["Action distr", "Policy distr"],
                       title=f"Action Distributions Step {step}"
                    ),
                   }
        return metrics
    
    def training_step(self,
                      states,
                      actions,
                      rewards,
                      terminals,
                      env_step,
                      log_only=False
                      ):
        """
        states: [batch_size, horizon, state_dim]
        actions: [batch_size, horizon, action_dim]
        rewards: [batch_size, horizon]

        The input tensors are assumed to be normalized torch tensors on the correct device.
        """

        # unnormalize the actions as the policy is trained to output unnormalized actions
        actions = self.normalizer.unnormalize(actions, "actions")

        # if using unnormalized inputs unnormalize the states and rewards
        if not self.use_normed_inputs:
            states = self.normalizer.unnormalize(states, "observations")
            rewards = self.normalizer.unnormalize(rewards, "rewards")

        metrics = dict()

        if not log_only:
            self.ema.update_model_average(self.critic_target, self.critic)
            self.train_steps += 1

        value  = self.critic_target.forward(states)
        advantage = - value[:, :-1] + rewards[:, :-1] + self.gamma * (1.0 - terminals[:, :-1]) * value[:, 1:]
        advantage_gae = []
        agae = None
        for t in reversed(range(advantage.shape[1])):
            adv = advantage[:, t]
            term = terminals[:, t]
            if agae is None:
                agae = adv
            else:
                agae = adv + self.lambda_ * self.gamma * (1.0 - term) * agae
            advantage_gae.append(agae)
        advantage_gae.reverse()
        advantage_gae = torch.stack(advantage_gae, dim=1)
        value_target = advantage_gae + value[:, :-1]

        # When calculating losses, should ignore terminal states, or anything after
        reality_weight = (1 - terminals[:, :-1]).log().cumsum(dim=1).exp()

        # Compute normalized logprob for logging
        policy_distr = self.forward_actor(states[:, :-1, :], normed_input=self.use_normed_inputs)
        action_logprob = policy_distr.log_prob(actions[:, :-1, :])
        standard_logprob, act_stds, act_means = get_standardized_stats(policy_distr, actions[:, :-1, :])
        metrics["act_std"] = act_stds.mean().item()
        metrics["act_mean"] = act_means.mean().item()

        # log action distributions periodically
        if env_step - self.last_log >= self.log_interval:
            act_metrics = self.log_action_distr(policy_distr, actions[:, :-1, :], env_step)
            metrics.update(act_metrics)
            self.last_log = env_step

        if self.clip_logprob:
            to_keep = torch.as_tensor(standard_logprob.gt(self.min_logprob), dtype=torch.float32)
            metrics["imagine_clip_frac"] = 1 - to_keep.mean().item()
        else:
            to_keep = torch.ones_like(standard_logprob)

        # Actor loss
        if self.normalize_adv:
            advantage_gae = (advantage_gae - advantage_gae[to_keep.type(torch.bool)].mean()) / (advantage_gae[to_keep.type(torch.bool)].std() + 1e-8)

        loss_policy = - action_logprob * advantage_gae.detach() * to_keep
        standard_logprob_avg = standard_logprob[to_keep.to(torch.bool)].mean()
        standard_logprob_std = standard_logprob[to_keep.to(torch.bool)].std()

        policy_entropy = policy_distr.entropy()
        if not self.fixed_std:
            loss_actor = loss_policy - torch.exp(self.log_alpha) * policy_entropy
        else:
            loss_actor = loss_policy
        loss_actor = (loss_actor * reality_weight).mean()

        # Critic loss
        value = self.critic.forward(states)
        value = value[:, :-1]
        loss_critic = 0.5 * torch.square(value_target.detach() - value)
        loss_critic = (loss_critic * to_keep * reality_weight).mean()

        # Combined loss
        loss_combined = loss_actor + loss_critic

        if self.tune_entropy:
            self.update_alpha(policy_entropy)

        with torch.no_grad():
            metrics.update({
                "loss_critic": loss_critic.detach().item(),
                "loss_actor": loss_actor.detach().item(),
                "policy_entropy": policy_entropy.mean().item(),
                "policy_std": policy_distr.stddev.mean().item(),
                "policy_value": value[:, 0].mean().item(),  # Value of real states
                "policy_value_im": value.mean().item(),  # Value of imagined states
                "policy_reward": rewards.mean().item(),
                "policy_reward_std": rewards.std().item(),
                "alpha": torch.exp(self.log_alpha).item(),
                "standardized_logprob_avg": standard_logprob_avg.item(),
                "standardized_logprob_std": standard_logprob_std.item(), # clamp to remove outlier values from these stats
                "standardized_logprob_min": standard_logprob.min().item(),
                "standardized_logprob_max": standard_logprob.max().item(),
                "maximum_logprob": action_logprob.max().item(),
                "minimum_logprob": action_logprob.min().item(),
            })

        if not log_only:
            self._optimizer_actor.zero_grad()
            self._optimizer_critic.zero_grad()
            loss_combined.backward()

            if self.grad_clip is not None:
                actor_grad_norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
                critic_grad_norm = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)
                metrics["gradients/actor_grad_norm"] = actor_grad_norm.item()
                metrics["gradients/critic_grad_norm"] = critic_grad_norm.item()
            
            actor_state_dict = copy.deepcopy(self.actor.state_dict())
            self._optimizer_actor.step()
            self._optimizer_critic.step()

            # compute change in logprob for logging
            policy_distr = self.forward_actor(states[:, :-1, :], normed_input=self.use_normed_inputs)
            new_logprob = policy_distr.log_prob(actions[:, :-1, :])
            approx_kl = (action_logprob - new_logprob).mean().item()
            initial_update_size = (action_logprob - new_logprob).abs().mean().item()
            metrics["update_kl"] = approx_kl
            metrics["update_delta_logprob_initial"] = initial_update_size
            
            # linesearch
            if self.linesearch:
                update_size = initial_update_size
                linesearch_steps = 1
                old_lr = self._optimizer_actor.param_groups[0]['lr']
                while (update_size > self.target_update * (1 + self.linesearch_tolerance)) or (update_size < self.target_update * (1 - self.linesearch_tolerance)):
                    if update_size > self.target_update:
                        self._optimizer_actor.param_groups[0]['lr'] *= self.linesearch_ratio
                    else:
                        self._optimizer_actor.param_groups[0]['lr'] /= self.linesearch_ratio
                    if self._optimizer_actor.param_groups[0]['lr'] > self.max_lr:
                        break
                    self.actor.load_state_dict(actor_state_dict)
                    self._optimizer_actor.step()
                    policy_distr = self.forward_actor(states[:, :-1, :], normed_input=self.use_normed_inputs)
                    new_logprob = policy_distr.log_prob(actions[:, :-1, :])
                    update_size = (action_logprob - new_logprob).abs().mean().item()
                    linesearch_steps += 1
                    if linesearch_steps > 50:
                        break
                metrics["update_delta_logprob_final"] = update_size
                metrics["update_linesearch_steps"] = linesearch_steps
                self._optimizer_actor.param_groups[0]['lr'] = old_lr 

            if self.lr_schedule == "target":
                self.update_actor_lr(initial_update_size)
            elif self.lr_schedule == "linear":
                self._optimizer_actor.param_groups[0]['lr'] = self.max_lr * (1 - env_step / self.lr_decay_steps)
            metrics["lr_actor"] = self._optimizer_actor.param_groups[0]['lr']

            # if fixed std dev decay the std dev
            if self.fixed_std:
                self.update_std(env_step)
        return metrics

    def update_std(self, env_step):
        self.current_std = max(self.min_std, self.init_std * (1 - env_step / self.decay_std_steps))

