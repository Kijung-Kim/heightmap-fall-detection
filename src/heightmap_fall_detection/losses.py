from __future__ import annotations

import torch


class MILFallLoss(torch.nn.Module):
    """Multiple-instance loss used for video-level supervision."""

    def __init__(
        self,
        rho2: float = 0.1,
        rho3: float = 0.1,
        rho4: float = 0.1,
        rho5: float = 0.1,
        epsilon: float = 0.1,
        k_pos: int = 3,
        k_neg: int = 3,
    ) -> None:
        super().__init__()
        self.rho2 = rho2
        self.rho3 = rho3
        self.rho4 = rho4
        self.rho5 = rho5
        self.epsilon = epsilon
        self.k_pos = k_pos
        self.k_neg = k_neg

    def forward(self, scores: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        total_loss = scores.new_tensor(0.0)
        is_positive = bool(label.item() == 1)

        if is_positive:
            max_score = torch.max(scores)
            total_loss = total_loss + (1.0 - max_score) ** 2

            if len(scores) > 1:
                diffs = torch.abs(scores[1:] - scores[:-1])
                max_diff = torch.max(diffs)
                total_loss = total_loss + self.rho3 * torch.clamp(self.epsilon - max_diff, min=0.0) ** 2

            top_k_scores, _ = torch.topk(scores, k=min(self.k_pos, len(scores)))
            total_loss = total_loss + self.rho4 * (1.0 - torch.mean(top_k_scores)) ** 2
        else:
            top_k_scores, _ = torch.topk(scores, k=min(self.k_neg, len(scores)))
            avg_top_k = torch.mean(top_k_scores)
            total_loss = total_loss + avg_top_k ** 2

            if len(scores) > 1:
                diffs = scores[1:] - scores[:-1]
                total_loss = total_loss + self.rho2 * torch.sum(diffs**2)

            epsilon_log = 1e-8
            entropy = -scores * torch.log(scores + epsilon_log) - (1.0 - scores) * torch.log(
                1.0 - scores + epsilon_log
            )
            total_loss = total_loss + self.rho5 * torch.sum(entropy)

        return total_loss

