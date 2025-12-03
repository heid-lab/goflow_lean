from typing import Union, Callable

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.init import xavier_uniform_, zeros_

from goflow.gotennet.models.components.ops import Dense, shifted_softplus, str2act
from goflow.gotennet.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class SNNDense(nn.Linear):
    """Fully connected linear layer with activation function."""

    def __init__(
            self,
            in_features: int,
            out_features: int,
            bias: bool = True,
            activation: Union[Callable, nn.Module] = None,
            weight_init: Callable = xavier_uniform_,
            bias_init: Callable = zeros_,
    ):
        self.weight_init = weight_init
        self.bias_init = bias_init
        super().__init__(in_features, out_features, bias)

        self.activation = activation
        if self.activation is None:
            self.activation = nn.Identity()

    def reset_parameters(self):
        self.weight_init(self.weight)
        if self.bias is not None:
            self.bias_init(self.bias)

    def forward(self, input: torch.Tensor):
        y = F.linear(input, self.weight, self.bias)
        y = self.activation(y)
        return y


class GatedEquivariantBlock(nn.Module):
    """
    The gated equivariant block is used to obtain rotationally invariant and equivariant features
    for tensorial properties.
    """

    def __init__(self, n_sin, n_vin, n_sout, n_vout, n_hidden, activation=F.silu, sactivation=None):
        super().__init__()
        self.n_sin = n_sin
        self.n_vin = n_vin
        self.n_sout = n_sout
        self.n_vout = n_vout
        self.n_hidden = n_hidden
        self.mix_vectors = SNNDense(n_vin, 2 * n_vout, activation=None, bias=False)
        self.scalar_net = nn.Sequential(
            Dense(n_sin + n_vout, n_hidden, activation=activation),
            Dense(n_hidden, n_sout + n_vout, activation=None),
        )
        self.sactivation = sactivation

    def forward(self, scalars, vectors):
        vmix = self.mix_vectors(vectors)
        vectors_V, vectors_W = torch.split(vmix, self.n_vout, dim=-1)
        vectors_Vn = torch.norm(vectors_V, dim=-2)

        ctx = torch.cat([scalars, vectors_Vn], dim=-1)
        x = self.scalar_net(ctx)
        s_out, x = torch.split(x, [self.n_sout, self.n_vout], dim=-1)
        v_out = x.unsqueeze(-2) * vectors_W

        if self.sactivation:
            s_out = self.sactivation(s_out)

        return s_out, v_out


class Atomwise3DOut(nn.Module):
    """
    Reduced AtomwiseV3 to this to only have necessary functionality.
    """

    def __init__(
            self,
            n_in,
            n_hidden=None,
            activation=shifted_softplus
    ):
        super().__init__()

        if type(activation) is str:
            activation = str2act(activation)

        self.out_net = nn.ModuleList(
            [
                GatedEquivariantBlock(n_sin=n_in, n_vin=n_in, n_sout=n_hidden, n_vout=n_hidden, n_hidden=n_hidden,
                                      activation=activation,
                                      sactivation=activation),
                GatedEquivariantBlock(n_sin=n_hidden, n_vin=n_hidden, n_sout=1, n_vout=1,
                                      n_hidden=n_hidden, activation=activation)
            ])

    def forward(self, l0, l1):
        for eqiv_layer in self.out_net:
            l0, l1 = eqiv_layer(l0, l1)
        return l1.squeeze()
