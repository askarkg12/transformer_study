import torch
import torch.nn as nn
from torchtune.modules import RotaryPositionalEmbeddings

# TODO apply dropout


class CaptionDecoder(nn.Module):
    def __init__(
        self,
        *,
        decoder_dim,
        encoder_dim,
        num_heads,
        vocab_size,
        transformer_layers: int,
        ff_dim: int | None = None,
    ):
        self.num_heads = num_heads

        super().__init__()

        self.emb = nn.Embedding(vocab_size, decoder_dim)
        self.pos = RotaryPositionalEmbeddings(decoder_dim // num_heads)
        self.transformer = nn.Sequential(
            *[
                CaptionDecoderLayer(
                    decoder_dim=decoder_dim,
                    encoder_dim=encoder_dim,
                    num_heads=num_heads,
                    ff_dim=ff_dim,
                )
                for _ in range(transformer_layers)
            ]
        )

    def forward(self, x: torch.Tensor, encoded_pack: torch.Tensor) -> torch.Tensor:
        # x shape [batch, seq_len]

        x = self.emb(x)

        # [batch, seq_len, embed_dim]
        x = self.pos(x.view(*x.shape[:-1], self.num_heads, -1)).view(*x.shape)

        x = self.transformer(x, encoded_pack)

        return x


class CaptionDecoderLayer(nn.Module):
    def __init__(self, *, decoder_dim, encoder_dim, num_heads, ff_dim):
        if (not encoder_dim % num_heads) or (not decoder_dim % num_heads):
            raise ValueError(
                f"Embedding dimensions {encoder_dim} and {decoder_dim} should be divisible by number of heads {num_heads}"
            )
        super().__init__()

        self.masked_attn = MaskedAttention(embed_dim=decoder_dim, num_heads=num_heads)
        self.cross_attn = CrossAttention(
            encoder_dim=encoder_dim, decoder_dim=decoder_dim, num_heads=num_heads
        )
        self.ff = nn.Sequential(
            nn.Linear(decoder_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, decoder_dim),
        )
        self.norm = nn.LayerNorm(decoder_dim)

    def forward(
        self, x: torch.Tensor, enc_pack: tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        # x shape [batch, seq_len, embed_dim]
        # lengths shape [batch, 1, 1, seq_len]
        enc, pad_mask = enc_pack

        x = self.norm(self.masked_attn(x) + x)

        x = self.norm(self.cross_attn(enc, pad_mask, x) + x)

        x = self.norm(self.ff(x) + x)

        return x


class CrossAttention(nn.Module):
    def __init__(self, *, encoder_dim, decoder_dim, num_heads):
        if (not encoder_dim % num_heads) or (not decoder_dim % num_heads):
            raise ValueError(
                f"Embedding dimensions {encoder_dim} and {decoder_dim} should be divisible by number of heads {num_heads}"
            )
        self.num_heads = num_heads

        # TODO decide if its min or max
        cross_dim = max(encoder_dim, decoder_dim)

        self.enc_scaling = (encoder_dim // num_heads) ** 0.5

        self.dec_scaling = (decoder_dim // num_heads) ** 0.5

        super().__init__()

        self.q_proj = nn.Linear(decoder_dim, cross_dim)
        self.k_proj = nn.Linear(encoder_dim, cross_dim)

        self.v_proj = nn.Linear(encoder_dim, decoder_dim)
        self.out_proj = nn.Linear(decoder_dim, decoder_dim)

    def head_on_view(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        x = x.view(batch, seq_len, self.num_heads, -1)
        return x.transpose(1, 2)

    def forward(
        self, enc: torch.Tensor, enc_mask: torch.Tensor, dec: torch.Tensor
    ) -> torch.Tensor:
        # enc_mask shape [batch, 1, 1, seq_len]

        # [batch, num_heads, seq_len, head_dim]
        q = self.head_on_view(self.q_proj(dec))
        k = self.head_on_view(self.k_proj(enc))

        a_logits: torch.Tensor = q @ k.transpose(-2, -1) / self.dec_scaling
        a = a_logits.masked_fill(enc_mask, float("-inf")).softmax(dim=-1)

        v = self.head_on_view(self.v_proj(enc))
        x = a @ v

        # [batch, seq_len, embed_dim]
        x = x.transpose(1, 2).reshape(x.shape[0], x.shape[2], -1)

        x = self.out_proj(x)

        return x


class MaskedAttention(nn.Module):
    def __init__(self, *, embed_dim, num_heads):
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"Embedding dimension {embed_dim} should be divisible by number of heads {num_heads}"
            )
        self.num_heads = num_heads
        self.head_scaling = (embed_dim // num_heads) ** 0.5
        super().__init__()

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def head_on_view(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        x = x.view(batch, seq_len, self.num_heads, -1)
        return x.transpose(1, 2)

    def forward(self, x: torch.Tensor):
        # x shape [batch, seq_len, embed_dim]

        # [batch, num_heads, seq_len, head_dim]
        q = self.head_on_view(self.q_proj(x))
        k = self.head_on_view(self.k_proj(x))

        a_logits: torch.Tensor = q @ k.transpose(-2, -1) / self.head_scaling

        mask = torch.triu(torch.ones_like(a_logits, dtype=torch.bool), diagonal=1)
        a = a_logits.masked_fill(mask, float("-inf")).softmax(dim=-1)

        v = self.head_on_view(self.v_proj(x))

        x = a @ v

        x = x.transpose(1, 2).reshape(x.shape[0], x.shape[2], -1)

        x = self.out_proj(x)

        return x
