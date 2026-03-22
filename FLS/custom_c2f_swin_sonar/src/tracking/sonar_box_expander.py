from dataclasses import dataclass
import torch


@dataclass
class SonarBoxExpander:
    ratio: float = 3.0

    def expand_xyxy(self, boxes_xyxy: torch.Tensor, w: int, h: int) -> torch.Tensor:
        x1, y1, x2, y2 = boxes_xyxy.unbind(-1)
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        bw, bh = (x2 - x1).clamp(min=1e-6) * self.ratio, (y2 - y1).clamp(min=1e-6) * self.ratio
        nx1 = (cx - bw / 2).clamp(0, w - 1)
        ny1 = (cy - bh / 2).clamp(0, h - 1)
        nx2 = (cx + bw / 2).clamp(0, w - 1)
        ny2 = (cy + bh / 2).clamp(0, h - 1)
        return torch.stack([nx1, ny1, nx2, ny2], dim=-1)