import numpy as np
import torch

from cwm.data import TransitionArrays, build_window_starts
from cwm.dynamics import compose_next_x_normalized
from cwm.model import ConditionalWorldModel
from cwm.normalization import Normalizer


def test_model_shape():
    m = ConditionalWorldModel()
    y = m(torch.randn(7,9), torch.randn(7,3))
    assert y.shape == (7,6)


def test_rollout_compose_shape():
    n = Normalizer([0]*9,[1]*9,[0]*3,[1]*3,[0]*6,[1]*6)
    y = torch.randn(4,6)
    x = torch.randn(4,9)
    z = compose_next_x_normalized(y,x,n)
    assert z.shape == (4,9)
    assert torch.allclose(z[:,4:7], x[:,4:7])


def test_window_starts():
    meta = np.zeros((8,11),dtype=float)
    meta[:,0]=1; meta[:,3]=1; meta[:,4]=1; meta[:,5]=np.arange(1,9)
    s = build_window_starts(meta,5)
    assert np.array_equal(s,np.array([0,1,2,3]))
