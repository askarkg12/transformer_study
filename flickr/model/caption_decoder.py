import torch
import torch.nn as nn


class CaptionDecoder(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, *, encoder_dim, decoder_dim, num_heads):
        if (not encoder_dim % num_heads) or (not decoder_dim % num_heads):
            raise ValueError(
                f"Embedding dimensions {encoder_dim} and {decoder_dim} should be divisible by number of heads {num_heads}"
            )
        self.num_heads = num_heads

        # TODO decide if its min or max
        cross_dim = max(encoder_dim, decoder_dim)

        self.enc_scaling = (encoder_dim // num_heads) ** -0.5

        self.dec_scaling = (decoder_dim // num_heads) ** -0.5

        super().__init__()

        self.q_proj = nn.Linear(decoder_dim, cross_dim)
        self.k_proj = nn.Linear(encoder_dim, cross_dim)

        self.v_proj = nn.Linear(encoder_dim, decoder_dim)
        self.out_proj = nn.Linear(decoder_dim, decoder_dim)

    def head_on_view(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        x = x.view(batch, seq_len, self.num_heads, -1)
        return x.transpose(1, 2)

    def forward(self, enc: torch.Tensor, dec: torch.Tensor) -> torch.Tensor:
        q = self.head_on_view(self.q_proj(dec))
        k = self.head_on_view(self.k_proj(enc))
        v = self.head_on_view(self.v_proj(enc))
        pass
