import torch
from torch import Tensor, nn
from transformers import CLIPVisionModel, CLIPModel
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from src.shared.config import VISUAL_ENCODER_MODEL, RETRIEVAL_ENCODER_MODEL
from src.shared.utils import extract_global_embedding

class CLIPVisualEncoder(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        clip_model = CLIPVisionModel.from_pretrained(VISUAL_ENCODER_MODEL, **kwargs)

        # Support both plain CLIPVisionModel and wrappers that expose .vision_model.
        self.backbone = getattr(clip_model, "vision_model", clip_model)
        self.backbone.requires_grad_(False)
        self.output_dim = self.backbone.config.hidden_size

    def forward(self, images: Tensor) -> Tensor:
        self.backbone.eval()
        with torch.no_grad():
            hidden_states = self.backbone(pixel_values=images).last_hidden_state

        # Preserve CLS and patch tokens so downstream consumers can choose either.
        return hidden_states


class CLIPImageEmbeddingEncoder(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.model = CLIPModel.from_pretrained(RETRIEVAL_ENCODER_MODEL, **kwargs)
        self.model.requires_grad_(False)
        self.output_dim = self.model.projection_dim

    def forward(self, pixel_values: Tensor) -> Tensor:
        self.model.eval()
        with torch.no_grad():
            outputs = self.model.get_image_features(pixel_values=pixel_values)
        return extract_global_embedding(outputs)


def create_clip_transform():
    return transforms.Compose(
        [
            transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC, antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            ),
        ]
    )

