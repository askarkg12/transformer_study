import torch
import torch.nn as nn


class FlexibleEncoderLayer(nn.Module):
    def __init__(
        self,
        embedding_dim,
        n_heads,
        feedforward_dim,
        residual_attn: bool,
        norm_attn: bool,
        residual_ff: bool,
        norm_ff: bool,
        residual_both: bool,
        norm_both: bool,
    ) -> None:
        super().__init__()

        self.residual_attn = residual_attn
        self.norm_attn = norm_attn
        self.residual_ff = residual_ff
        self.norm_ff = norm_ff
        self.residual_both = residual_both
        self.norm_both = norm_both

        if embedding_dim % n_heads != 0:
            raise ValueError(
                f"embedding_dim must be divisible by n_heads. Got {embedding_dim} and {n_heads}"
            )
        self.head_dim = embedding_dim // n_heads

        self.Qs = nn.ModuleList(
            [nn.Linear(embedding_dim, self.head_dim) for _ in range(n_heads)]
        )
        self.Ks = nn.ModuleList(
            [nn.Linear(embedding_dim, self.head_dim) for _ in range(n_heads)]
        )
        self.Vs = nn.ModuleList(
            [nn.Linear(embedding_dim, self.head_dim) for _ in range(n_heads)]
        )
        self.projection = nn.Linear(n_heads * self.head_dim, embedding_dim)

        self.feedforward = nn.Sequential(
            nn.Linear(embedding_dim, feedforward_dim),
            nn.ReLU(),
            nn.Linear(feedforward_dim, embedding_dim),
        )

        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, x) -> torch.Tensor:
        # x: [seq_len, embedding_dim]
        residual_pre_attn = x
        x = self.self_attention(x)
        if self.residual_attn:
            x = x + residual_pre_attn
        if self.norm_attn:
            x = self.norm(x)

        residual_post_attn = x

        x = self.feedforward(x)

    def self_attention(self, x):
        heads = []
        for Q, K, V in zip(self.Qs, self.Ks, self.Vs):
            # [seq_len, head_dim]
            Qx = Q(x)
            Kx = K(x)
            Vx = V(x)

            # [seq_len, seq_len]
            attention = torch.einsum("sd,td->st", Qx, Kx) / (self.head_dim**0.5)
            attention = torch.softmax(attention, dim=-1)

            # [seq_len, head_dim]
            head = torch.einsum("st,td->sd", attention, Vx)
            heads.append(head)

        # [seq_len, n_heads * head_dim]
        heads = torch.cat(heads, dim=-1)
        # [seq_len, embedding_dim]
        x = self.projection(heads)
        return x
