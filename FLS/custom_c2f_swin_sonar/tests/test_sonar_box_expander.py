import torch
from src.tracking.sonar_box_expander import SonarBoxExpander


def test_expand_ratio_3():
    b = torch.tensor([[10.0, 10.0, 20.0, 20.0]])
    ex = SonarBoxExpander(ratio=3.0)
    out = ex.expand_xyxy(b, w=100, h=100)
    assert out.shape == (1, 4)
    assert out[0, 0] < 10 and out[0, 1] < 10
    assert out[0, 2] > 20 and out[0, 3] > 20