from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


class MLPBlock(nn.Module):
    def __init__(self, d_in: int, d_out: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_out),
            nn.LayerNorm(d_out),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class ConditionalWorldModel(nn.Module):
    """Shared action-conditioned wireless world model.

    Paper mapping:
      h_t       = E_phi(s_t, c_t)
      h_{t+1}   = G_theta(h_t, a_t)
      y_{t+1}   = D_psi(h_{t+1})

    Here G_theta is implemented with a GRUCell whose hidden state is the encoded
    wireless service state and whose input is an action embedding. This is one
    concrete implementation of the paper's action-conditioned latent transition.
    """

    def __init__(
        self,
        x_dim: int = 9,
        a_dim: int = 3,
        y_dim: int = 6,
        hidden_dim: int = 256,
        latent_dim: int = 128,
        action_dim: int = 64,
        dropout: float = 0.05,
        ablate_action: bool = False,
        ablate_context: bool = False,
        ablate_aoi: bool = False,
    ):
        super().__init__()
        self.config = dict(
            x_dim=x_dim,
            a_dim=a_dim,
            y_dim=y_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            action_dim=action_dim,
            dropout=dropout,
            ablate_action=ablate_action,
            ablate_context=ablate_context,
            ablate_aoi=ablate_aoi,
        )
        self.ablate_action = ablate_action
        self.ablate_context = ablate_context
        self.ablate_aoi = ablate_aoi
        self.encoder = nn.Sequential(
            MLPBlock(x_dim, hidden_dim, dropout),
            MLPBlock(hidden_dim, hidden_dim, dropout),
            nn.Linear(hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.SiLU(),
        )
        self.action_encoder = nn.Sequential(
            MLPBlock(a_dim, action_dim, dropout),
            nn.Linear(action_dim, action_dim),
            nn.SiLU(),
        )
        self.transition = nn.GRUCell(action_dim, latent_dim)
        self.post_transition = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.SiLU(),
        )
        self.decoder = nn.Sequential(
            MLPBlock(latent_dim, hidden_dim, dropout),
            MLPBlock(hidden_dim, hidden_dim, dropout),
            nn.Linear(hidden_dim, y_dim),
        )

    def encode(self, x):
        if self.ablate_context or self.ablate_aoi:
            x = x.clone()
            if self.ablate_context:
                x[..., 4:9] = 0
            if self.ablate_aoi:
                x[..., 3] = 0
        return self.encoder(x)

    def transition_step(self, h, a):
        if self.ablate_action:
            a = torch.zeros_like(a)
        ae = self.action_encoder(a)
        h_next = self.transition(ae, h)
        return h_next + self.post_transition(h_next)

    def decode(self, h_next):
        return self.decoder(h_next)

    def forward(self, x, a):
        h = self.encode(x)
        h_next = self.transition_step(h, a)
        return self.decode(h_next)
