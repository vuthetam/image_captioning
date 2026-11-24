"""A position-wise residual FFN from raw CLIP facts to decoder memory."""

from torch import Tensor, nn


class FactAdapter(nn.Module):
    def __init__(self, fact_feature_dim: int, d_model: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(fact_feature_dim)
        self.residual_projection = (
            nn.Identity() if fact_feature_dim == d_model else nn.Linear(fact_feature_dim, d_model)
        )
        self.ffn = nn.Sequential(
            nn.Linear(fact_feature_dim, 4 * d_model), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model), nn.Dropout(dropout),
        )

    def forward(self, facts: Tensor) -> Tensor:
        facts = facts.to(dtype=self.norm.weight.dtype)
        return self.residual_projection(facts) + self.ffn(self.norm(facts))
