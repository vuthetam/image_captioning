import torch
from transformers import AutoTokenizer, CLIPTextModelWithProjection

from src.shared.facts.dataset import normalize_fact
from src.shared.utils import extract_global_embedding


class CachedCLIPFactEncoder:
    """Frozen CLIP text tower with a lazy, process-local CPU cache.

    Keep this object outside the trainable captioner so pretrained weights and
    cache entries are not included in its optimizer or checkpoints.
    """

    def __init__(self, model_id: str, device, batch_size: int = 128):
        if batch_size < 1:
            raise ValueError("Fact text batch size must be positive.")
        self.model_id = model_id
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = CLIPTextModelWithProjection.from_pretrained(model_id)
        self.model = self.model.to(device=self.device, dtype=torch.float32).eval().requires_grad_(False)
        self.output_dim = self.model.config.projection_dim
        self.cache: dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def encode(self, facts_per_sample):
        """Encode only cache misses; return padded features and a valid-fact mask.

        No L2 normalization is applied. The cache stores only the frozen CLIP
        output, before the captioner's trainable Linear/LayerNorm projector.
        """
        groups = [[normalize_fact(fact) for fact in facts] for facts in facts_per_sample]
        if not groups or any(not facts for facts in groups):
            raise ValueError("Each sample must contain at least one fact.")
        if any(not fact for facts in groups for fact in facts):
            raise ValueError("Empty fact strings must be removed before CLIP encoding.")
        missing = list(dict.fromkeys(fact for facts in groups for fact in facts if fact not in self.cache))
        self.model.eval()
        # Cache values should not depend on the training loop's autocast context.
        with torch.autocast(device_type=self.device.type, enabled=False):
            for start in range(0, len(missing), self.batch_size):
                texts = missing[start:start + self.batch_size]
                tokens = self.tokenizer(texts, padding=True, truncation=False, return_tensors="pt")
                if tokens["input_ids"].shape[1] > self.model.config.max_position_embeddings:
                    raise ValueError("A fact exceeds the CLIP text context length; it was not truncated.")
                tokens = {key: value.to(self.device) for key, value in tokens.items()}
                output = self.model(**tokens)
                embeddings = extract_global_embedding(output)
                embeddings = embeddings.detach().float().cpu()
                if embeddings.shape != (len(texts), self.output_dim) or not torch.isfinite(embeddings).all():
                    raise ValueError("CLIP returned invalid fact embeddings.")
                for text, embedding in zip(texts, embeddings):
                    self.cache[text] = embedding.clone()

        max_facts = max(map(len, groups))
        fact_features = torch.zeros(len(groups), max_facts, self.output_dim, dtype=torch.float32)
        fact_mask = torch.zeros(len(groups), max_facts, dtype=torch.bool)
        for index, facts in enumerate(groups):
            fact_features[index, :len(facts)] = torch.stack([self.cache[fact] for fact in facts])
            fact_mask[index, :len(facts)] = True
        fact_features = fact_features.to(self.device)
        fact_mask = fact_mask.to(self.device)

        return fact_features, fact_mask
