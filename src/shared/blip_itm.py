"""Score image-caption pairs while reusing BLIP's visual tokens for each image."""

from pathlib import Path

from PIL import Image
import torch
from transformers import BlipForImageTextRetrieval, BlipProcessor


class BlipITMScorer:
    """Own one BLIP model on one device; keep visual features only during scoring."""

    def __init__(self, model_id, device):
        self.model_id = model_id
        self.device = torch.device(device)
        self.processor = BlipProcessor.from_pretrained(model_id)
        self.model = BlipForImageTextRetrieval.from_pretrained(model_id).to(self.device).eval()
        self.model.requires_grad_(False)
        self.revision = getattr(self.model.config, "_commit_hash", None)

    @torch.inference_mode()
    def score(self, image_path: str | Path, captions: list[str]) -> list[float]:
        """Encode one image once and share its visual tokens across captions."""
        if not captions:
            return []
        if any(not isinstance(caption, str) or not caption.strip() for caption in captions):
            raise ValueError("BLIP ITM captions must be nonempty strings.")

        with Image.open(image_path) as image, image.convert("RGB") as rgb:
            pixel_values = self.processor(images=rgb, return_tensors="pt")["pixel_values"]
        pixel_values = pixel_values.to(device=self.device, dtype=self.model.dtype)
        image_features = self.model.vision_model(
            pixel_values=pixel_values, return_dict=True,
        ).last_hidden_state

        tokens = self.processor(
            text=captions, padding=True, truncation=True,
            max_length=self.model.config.text_config.max_position_embeddings,
            return_tensors="pt",
        ).to(self.device)
        fused = self.model.text_encoder(
            input_ids=tokens["input_ids"],
            attention_mask=tokens["attention_mask"],
            encoder_hidden_states=image_features.expand(len(captions), -1, -1),
            return_dict=True,
        ).last_hidden_state
        logits = self.model.itm_head(fused[:, 0, :])
        return logits.float().softmax(dim=-1)[:, 1].cpu().tolist()

    @torch.inference_mode()
    def score_batch(self, image_paths: list[str | Path],
                    captions_per_image: list[list[str]]) -> list[list[float]]:
        """Encode images once, score all pairs, and return scores grouped by image.

        Only images with captions are encoded. Pair features are repeated in
        caption order, and scores retain the input image and caption order.
        No visual features are retained after this call.
        """
        if len(image_paths) != len(captions_per_image):
            raise ValueError("Each image must have one corresponding caption list.")
        caption_counts = [len(captions) for captions in captions_per_image]
        captions = [caption for group in captions_per_image for caption in group]
        if any(not isinstance(caption, str) or not caption.strip() for caption in captions):
            raise ValueError("BLIP ITM captions must be nonempty strings.")
        if not captions:
            return [[] for _ in image_paths]

        images = []
        try:
            for image_path, count in zip(image_paths, caption_counts):
                if count:
                    with Image.open(image_path) as image:
                        images.append(image.convert("RGB"))
            pixels_values = self.processor(images=images, return_tensors="pt")["pixel_values"]
        finally:
            for image in images:
                image.close()
        pixels_values = pixels_values.to(device=self.device, dtype=self.model.dtype)
        image_features = self.model.vision_model(
            pixel_values=pixels_values, return_dict=True,
        ).last_hidden_state
        repeats = torch.tensor([count for count in caption_counts if count], device=self.device)
        pair_features = image_features.repeat_interleave(repeats, dim=0, output_size=len(captions))

        tokens = self.processor(
            text=captions, padding=True, truncation=True,
            max_length=self.model.config.text_config.max_position_embeddings,
            return_tensors="pt",
        ).to(self.device)
        fused = self.model.text_encoder(
            input_ids=tokens["input_ids"],
            attention_mask=tokens["attention_mask"],
            encoder_hidden_states=pair_features,
            return_dict=True,
        ).last_hidden_state
        logits = self.model.itm_head(fused[:, 0, :])
        scores = logits.float().softmax(dim=-1)[:, 1].cpu()
        return [group.tolist() for group in scores.split(caption_counts)]
