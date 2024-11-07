import torch
import torch.nn as nn


class TransformerEncoderLayer(nn.Module):
    def __init__(self, embedding_dim, n_heads, feedforward_dim) -> None:
        super().__init__()

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

    def forward(self, x) -> torch.Tensor:
        # x: [seq_len, embedding_dim]
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
            head = torch.einsum("sd,st->td", Vx, attention)
            heads.append(head)
        pass
