import torch
import torch.nn as nn
from pathlib import Path
import sys

flickr_root = Path(__file__).parent.parent
sys.path.append(str(flickr_root))

from model.image_encoder import ImageEncoder
from model.caption_decoder import CaptionDecoder


class FlickModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass
