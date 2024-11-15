import torch
import torch.nn as nn
from pathlib import Path
import sys

flickr_root = Path(__file__).parent.parent
sys.path.append(str(flickr_root))

from model.image_encoder import ImageEncoder
from model.caption_decoder import CaptionDecoder


class FlickModel(nn.Module):
    def __init__(
        self,
        *,
        img_embed_dim,
        patch_len,
        enc_num_heads,
        encoder_layers,
        caption_embed_dim,
        vocab_size,
        decoder_layers,
        decoder_num_heads
    ):
        super().__init__()

        self.image_encoder = ImageEncoder(
            embed_dim=img_embed_dim,
            patch_len=patch_len,
            num_heads=enc_num_heads,
            transformer_layers=encoder_layers,
        )
        self.caption_decoder = CaptionDecoder(
            decoder_dim=caption_embed_dim,
            encoder_dim=img_embed_dim,
            num_heads=decoder_num_heads,
            vocab_size=vocab_size,
            transformer_layers=decoder_layers,
        )

        self.projection = nn.Linear(caption_embed_dim, vocab_size)

    def forward(
        self, img_patches: torch.Tensor, patch_lens, caption_in
    ) -> torch.Tensor:

        img_encoded, pad_mask = self.image_encoder(img_patches, patch_lens)

        caption_out = self.caption_decoder(caption_in, (img_encoded, pad_mask))

        return self.projection(caption_out)
