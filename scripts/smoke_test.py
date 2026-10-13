from __future__ import annotations

import _bootstrap  # noqa: F401

import torch

from cwm.model import ConditionalWorldModel
from cwm.normalization import Normalizer
from cwm.dynamics import compose_next_x_normalized


def main():
    B, H = 16, 5
    X = torch.randn(B, 9)
    A = torch.randn(B, H, 3)
    Y = torch.randn(B, H, 6)
    # Dummy identity normalization for pure model smoke test.
    norm = Normalizer([0]*9, [1]*9, [0]*3, [1]*3, [0]*6, [1]*6)
    model = ConditionalWorldModel()
    x = X
    loss = 0.0
    for h in range(H):
        pred = model(x, A[:,h])
        loss = loss + ((pred - Y[:,h])**2).mean()
        x = compose_next_x_normalized(pred, x, norm)
    loss.backward()
    print("Smoke test OK. Loss:", float(loss))


if __name__ == "__main__":
    main()
