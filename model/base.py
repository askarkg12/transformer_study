import torch
import torch.nn as nn


class BERP(nn.Module):
    def __init__(
        self, transformer_encoder: nn.Module, vocab_size: int, embedding_dim: int
    ):
        super().__init__()

        # Must return tensor of the same shape
        self.transformer_encoder = transformer_encoder

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.projection = nn.Linear(embedding_dim, vocab_size)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: [seq_len]

        # [seq_len, embedding_dim]
        embeddings = self.embedding(tokens)

        # [seq_len, embedding_dim]
        transformed = self.transformer_encoder(embeddings)

        # [seq_len, vocab_size]
        return self.projection(transformed)
