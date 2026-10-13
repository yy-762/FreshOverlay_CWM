import torch

from cwm.model import ConditionalWorldModel


def test_no_action_ablation_is_action_invariant():
    model = ConditionalWorldModel(ablate_action=True).eval()
    x = torch.randn(3, 9)
    a1 = torch.randn(3, 3)
    a2 = torch.randn(3, 3)
    with torch.no_grad():
        torch.testing.assert_close(model(x, a1), model(x, a2))


def test_no_context_ablation_is_context_invariant():
    model = ConditionalWorldModel(ablate_context=True).eval()
    x1 = torch.randn(3, 9)
    x2 = x1.clone()
    x2[:, 4:9] += 7
    a = torch.randn(3, 3)
    with torch.no_grad():
        torch.testing.assert_close(model(x1, a), model(x2, a))


def test_no_aoi_ablation_is_aoi_invariant():
    model = ConditionalWorldModel(ablate_aoi=True).eval()
    x1 = torch.randn(3, 9)
    x2 = x1.clone()
    x2[:, 3] += 7
    a = torch.randn(3, 3)
    with torch.no_grad():
        torch.testing.assert_close(model(x1, a), model(x2, a))
