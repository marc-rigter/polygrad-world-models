# adapted from https://github.com/conglu1997/SynthER/blob/main/synther/diffusion/denoiser_network.py

import math
import torch
import torch.nn as nn
import torch.optim
from einops import rearrange
from torch.nn import functional as F


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(
            self,
            dim: int,
            is_random: bool = False,
    ):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad=not is_random)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        return fouriered
    
# Residual MLP of the form x_{L+1} = MLP(LN(x_L)) + x_L
class ResidualBlock(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, activation: str = "relu", layer_norm: bool = True):
        super().__init__()
        self.linear = nn.Linear(dim_in, dim_out, bias=True)
        if layer_norm:
            self.ln = nn.LayerNorm(dim_in)
        else:
            self.ln = torch.nn.Identity()
        self.activation = getattr(F, activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.linear(self.activation(self.ln(x)))
    
class ResidualMLP(nn.Module):
    def __init__(
            self,
            input_dim: int,
            width: int,
            depth: int,
            output_dim: int,
            activation: str = "relu",
            layer_norm: bool = False,
            dropout: float = None,
    ):
        super().__init__()

        layers = []
        layers.append(nn.Linear(input_dim, width))
        for _ in range(depth):
            layers.append(ResidualBlock(width, width, activation, layer_norm))
            layers.append(nn.Dropout(dropout) if dropout else torch.nn.Identity())
        layers.append(nn.LayerNorm(width) if layer_norm else torch.nn.Identity())
        self.network = nn.Sequential(*layers)

        self.activation = getattr(F, activation)
        self.final_linear = nn.Linear(width, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.final_linear(self.activation(self.network(x)))

class ResidualMLPDenoiser(nn.Module):
    def __init__(
            self,
            horizon,
            transition_dim,
            cond_dim,
            dim_mults,
            attention,
            scale_obs=1.0,
            dropout=None,
            embed_dim: int=128,
            hidden_dim: int=1024,
            num_layers: int=6,
            learned_sinusoidal_cond: bool=False,
            random_fourier_features: bool=True,
            learned_sinusoidal_dim: int=16,
            activation: str="relu",
            layer_norm: bool=True,
    ):
        super().__init__()
        d_in = horizon * transition_dim 
        out_dim = horizon * transition_dim
        cond_dim = horizon * cond_dim
        embed_dim = max(d_in + cond_dim, embed_dim)
        self.residual_mlp = ResidualMLP(
            input_dim=embed_dim,
            width=hidden_dim,
            depth=num_layers,
            output_dim=out_dim,
            activation=activation,
            layer_norm=layer_norm,
            dropout=dropout,
        )

        self.scale_obs = scale_obs
        self.proj = nn.Linear(d_in + cond_dim, embed_dim)

        # time embeddings
        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features
        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(embed_dim)
            fourier_dim = embed_dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(
            self,
            traj: torch.Tensor,
            act: torch.Tensor,
            timesteps: torch.Tensor,
    ) -> torch.Tensor:
        # traj[:, 1:, :-1] = traj[:, 1:, :-1] * self.scale_obs # scale obs
        # traj[:, :, -1:] = traj[:, :, -1:] * self.scale_obs # scale rew
        x = torch.cat([traj, act], dim=-1)
        b, h, d = traj.shape
        time_embed = self.time_mlp(timesteps)
        x = x.reshape(b, -1)
        x = self.proj(x) + time_embed
        y = self.residual_mlp(x)
        y = y.reshape(b, h, d)
        return y