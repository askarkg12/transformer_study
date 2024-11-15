import torch
import torch.nn as nn
from torchtune.modules import RotaryPositionalEmbeddings

# TODO apply dropout


class ImageEncoder(nn.Module):
    def __init__(self, *, embed_dim, patch_len, num_heads, transformer_layers):
        self.num_heads = num_heads

        super().__init__()

        self.emb = nn.Linear(patch_len, embed_dim)

        self.pos = RotaryPositionalEmbeddings(embed_dim // num_heads)

        self.transformer = nn.Sequential(
            *[
                ImageEncoderLayer(embed_dim=embed_dim, num_heads=num_heads)
                for _ in range(transformer_layers)
            ]
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        # x shape [batch, patch_count, patch_len]
        # lengths shape [batch]

        # [batch, patch_count, embed_dim]
        x = self.emb(x)

        # [batch, patch_count]
        positions = torch.arange(x.shape[1], device=x.device).repeat(x.shape[0], 1) - (
            lengths // 2
        ).unsqueeze(1)

        # [batch, patch_count, embed_dim]
        x = self.pos(
            x.view(*x.shape[:-1], self.num_heads, -1),
            input_pos=positions,
        ).view(*x.shape)

        # [batch, 1, 1, seq_len]
        pad_mask = (
            torch.arange(x.shape[1], device=x.device)[None, None, None, :]
            >= lengths[:, None, None, None]
        )

        x, pad_mask = self.transformer((x, pad_mask))

        return x, pad_mask


class ImageEncoderLayer(nn.Module):
    def __init__(self, *, embed_dim, num_heads, ff_dim: int | None = None):
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"Embedding dimension {embed_dim} should be divisible by number of heads {num_heads}"
            )
        self.head_dim = embed_dim // num_heads
        if ff_dim is None:
            ff_dim = 4 * embed_dim

        super().__init__()

        self.attn = SelfAttention(embed_dim=embed_dim, num_heads=num_heads)

        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim),
        )

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        # x shape [batch, seq_len, embed_dim]
        # lengths shape [batch, 1, 1, seq_len]
        x, pad_mask = x

        x = self.norm(self.attn(x, pad_mask) + x)

        x = self.norm(self.ff(x) + x)

        return x, pad_mask


class SelfAttention(nn.Module):
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
        # [batch, seq_len, embed_dim]
        batch, seq_len, _ = x.shape
        # [batch, seq_len, num_heads, head_dim]
        x = x.view(batch, seq_len, self.num_heads, -1)
        # [batch, num_heads, seq_len, head_dim]
        return x.transpose(1, 2)

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        # x shape [batch, seq_len, embed_dim]
        # lengths shape [batch, 1, 1, seq_len]

        # [batch, num_heads, seq_len, head_dim]
        q = self.head_on_view(self.q_proj(x))
        k = self.head_on_view(self.k_proj(x))
        v = self.head_on_view(self.v_proj(x))

        # [batch, num_heads, seq_len, seq_len]
        a_logits: torch.Tensor = q @ k.transpose(-2, -1) / self.head_scaling

        # [batch, num_heads, seq_len, seq_len]
        a = torch.softmax(a_logits.masked_fill(pad_mask, float("-inf")), dim=-1)

        # [batch, num_heads, seq_len, head_dim]
        x = a @ v

        # [batch, seq_len, embed_dim]
        x = x.transpose(1, 2).reshape(x.shape[0], x.shape[2], -1)

        # [batch, seq_len, embed_dim]
        x = self.out_proj(x)

        return x


if __name__ == "__main__":

    attb = ImageEncoder(embed_dim=256, patch_len=64, num_heads=4, transformer_layers=3)

    # [batch, patch_count, patch_len]
    x = torch.randn(2, 10, 64)

    x1 = x.clone()
    x2 = x.clone()

    x1[0] = torch.cat([x1[0][:5], torch.rand(5, 64)], dim=0)
    lengths = torch.tensor([5, 10])

    out1, _ = attb(x1, lengths)
    out2, _ = attb(x2, lengths)

    out1[0] = torch.cat([out1[0][:5], torch.ones(5, 256)], dim=0)
    out2[0] = torch.cat([out2[0][:5], torch.ones(5, 256)], dim=0)

    pass
