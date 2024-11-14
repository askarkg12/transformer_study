import torch
import torch.nn as nn
from torchtune.modules import RotaryPositionalEmbeddings


class ImageEncoder(nn.Module):
    def __init__(self, *, embed_dim, patch_len, num_heads):
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"Embedding dimension {embed_dim} should be divisible by number of heads {num_heads}"
            )
        self.head_dim = embed_dim // num_heads

        super().__init__()

        self.emb = nn.Linear(patch_len, embed_dim)

        self.pos = RotaryPositionalEmbeddings(embed_dim // num_heads)

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
            x.view(*x.shape[:-1], x.shape[-1] // self.head_dim, self.head_dim),
            input_pos=positions,
        ).view(*x.shape)

        return x


if __name__ == "__main__":
    img_enc = ImageEncoder(embed_dim=256, patch_len=16, num_heads=8)

    x = torch.randn(4, 64, 16)

    lengths = torch.tensor([64, 32, 16, 8])

    out = img_enc(x, lengths)

    pass
