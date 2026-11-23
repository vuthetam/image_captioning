"""A shared, position-wise residual FFN for projected fact vectors."""

from torch import Tensor, nn


class FactAdapter(nn.Module):
    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model), nn.Dropout(dropout),
        )

    def forward(self, facts: Tensor) -> Tensor:
        return facts + self.ffn(self.norm(facts))
