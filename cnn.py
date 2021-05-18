from typing import Any

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from torch.nn.functional import normalize

T = torch.Tensor


class WrappedSpectral(nn.Module):
    def __init__(self, base_layer: nn.Module, ctype: str = "scalar", n_power_iterations: int = 1, dim: int = 0, eps: float = 1e-12, weight_name: str = "weight") -> None:
        super().__init__()
        self.base_layer: nn.Module
        self.add_module("base_layer", base_layer)
        self.n_power_iterations = n_power_iterations
        self.dim = dim
        self.eps = eps
        self.weight_name = weight_name
        self.ctype = ctype

        self.c: T
        if ctype == "none":
            self.register_buffer("c", torch.tensor(1.0, requires_grad=False))
        elif ctype == "scalar":
            self.c = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        elif ctype == "vector" and isinstance(base_layer, nn.Conv2d):
            self.c = nn.Parameter(torch.zeros((self.base_layer.weight.size(0), 1, 1, 1), requires_grad=True))  # type: ignore
        elif ctype == "vector" and isinstance(base_layer, nn.Linear):
            self.c = nn.Parameter(torch.zeros((self.base_layer.weight.size(0), 1), requires_grad=True))  # type: ignore
        else:
            raise NotImplementedError(f"got an unknown combination of ctype: {ctype}")

        weight = self.base_layer._parameters[weight_name]
        if isinstance(weight, torch.nn.parameter.UninitializedParameter):
            raise ValueError(
                'The module passed to `SpectralNorm` can\'t have uninitialized parameters. '
                'Make sure to run the dummy forward before applying spectral normalization')

        with torch.no_grad():
            weight_mat = self.reshape_weight_to_matrix(weight)

            h, w = weight_mat.size()
            # randomly initialize `u` and `v`
            u = normalize(weight.new_empty(h).normal_(0, 1), dim=0, eps=self.eps)
            v = normalize(weight.new_empty(w).normal_(0, 1), dim=0, eps=self.eps)

        delattr(self.base_layer, weight_name)
        self.base_layer.register_parameter(weight_name + "_orig", weight)
        # We still need to assign weight back as fn.name because all sorts of
        # things may assume that it exists, e.g., when initializing weights.
        # However, we can't directly assign as it could be an nn.Parameter and
        # gets added as a parameter. Instead, we register weight.data as a plain
        # attribute.
        setattr(self.base_layer, weight_name, weight.data)
        self.base_layer.register_buffer(weight_name + "_u", u)
        self.base_layer.register_buffer(weight_name + "_v", v)

    def reshape_weight_to_matrix(self, weight: torch.Tensor) -> torch.Tensor:
        weight_mat = weight
        if self.dim != 0:
            # permute dim to front
            weight_mat = weight_mat.permute(self.dim,
                                            *[d for d in range(weight_mat.dim()) if d != self.dim])
        height = weight_mat.size(0)
        return weight_mat.reshape(height, -1)

    def compute_weight(self) -> None:
        # NB: If `do_power_iteration` is set, the `u` and `v` vectors are
        #     updated in power iteration **in-place**. This is very important
        #     because in `DataParallel` forward, the vectors (being buffers) are
        #     broadcast from the parallelized module to each module replica,
        #     which is a new module object created on the fly. And each replica
        #     runs its own spectral norm power iteration. So simply assigning
        #     the updated vectors to the module this function runs on will cause
        #     the update to be lost forever. And the next time the parallelized
        #     module is replicated, the same randomly initialized vectors are
        #     broadcast and used!
        #
        #     Therefore, to make the change propagate back, we rely on two
        #     important behaviors (also enforced via tests):
        #       1. `DataParallel` doesn't clone storage if the broadcast tensor
        #          is already on correct device; and it makes sure that the
        #          parallelized module is already on `device[0]`.
        #       2. If the out tensor in `out=` kwarg has correct shape, it will
        #          just fill in the values.
        #     Therefore, since the same power iteration is performed on all
        #     devices, simply updating the tensors in-place will make sure that
        #     the module replica on `device[0]` will update the _u vector on the
        #     parallized module (by shared storage).
        #
        #    However, after we update `u` and `v` in-place, we need to **clone**
        #    them before using them to normalize the weight. This is to support
        #    backproping through two forward passes, e.g., the common pattern in
        #    GAN training: loss = D(real) - D(fake). Otherwise, engine will
        #    complain that variables needed to do backward for the first forward
        #    (i.e., the `u` and `v` vectors) are changed in the second forward.
        weight = getattr(self.base_layer, self.weight_name + '_orig')
        u = getattr(self.base_layer, self.weight_name + '_u')
        v = getattr(self.base_layer, self.weight_name + '_v')
        weight_mat = self.reshape_weight_to_matrix(weight)

        if self.training:
            with torch.no_grad():
                for _ in range(self.n_power_iterations):
                    # Spectral norm of weight equals to `u^T W v`, where `u` and `v`
                    # are the first left and right singular vectors.
                    # This power iteration produces approximations of `u` and `v`.
                    v = normalize(torch.mv(weight_mat.t(), u), dim=0, eps=self.eps, out=v)
                    u = normalize(torch.mv(weight_mat, v), dim=0, eps=self.eps, out=u)
                if self.n_power_iterations > 0:
                    # See above on why we need to clone
                    u = u.clone(memory_format=torch.contiguous_format)
                    v = v.clone(memory_format=torch.contiguous_format)

        sigma = torch.dot(u, torch.mv(weight_mat, v))

        weight = (weight / sigma)
        weight = weight * ((0.01 + 0.99 * F.softplus(self.c)) if self.ctype != "none" else self.c)

        setattr(self.base_layer, self.weight_name, weight)

    def forward(self, x: T) -> T:
        if self.base_layer.weight.device != self.base_layer.weight_u.device:
            self.base_layer.weight = self.base_layer.weight.to(self.base_layer.weight_u.device)  # type: ignore
        # if self.training:
        self.compute_weight()
        return self.base_layer(x)  # type: ignore


class ConvResidual(nn.Module):
    def __init__(
        self,
        ch: int,
        filters: int = 64,
        spectral: bool = False,
        padding: int = 1,
        p: float = None,
        residual: bool = False,
        stride: int = 2,
        activation: Any = nn.ReLU,
        ctype: str = "error"
    ):
        super().__init__()

        self.residual = residual
        self.padding = padding

        lyrs: Any = []
        c = nn.Conv2d(ch, filters, 3, stride=stride, padding=self.padding, bias=True)
        if spectral:
            c = WrappedSpectral(c, ctype=ctype)  # type: ignore

        lyrs.append(c)
        lyrs.append(nn.BatchNorm2d(filters, momentum=1.0, track_running_stats=False, affine=False))

        if p:
            lyrs.append(nn.Dropout(p=p))

        lyrs.append(activation())
        self.layer = nn.Sequential(*lyrs)

        self.pool: nn.Module = nn.Identity()
        if stride == 1:
            self.pool = nn.MaxPool2d(2)

    def forward(self, x: T) -> T:
        if self.residual:
            identity = x
            fx = self.layer(x)
            return self.pool(identity + fx)  # type: ignore

        return self.pool(self.layer(x))  # type: ignore


class ResidualSpectralCNN(nn.Module):
    def __init__(self, in_ch: int, filters: int = 64, ctype: str = "error") -> None:
        super().__init__()
        lyrs = []
        for i in range(4):
            lyrs.append(ConvResidual(in_ch if i == 0 else filters, filters=filters, spectral=True, stride=1, residual=i != 0, ctype=ctype))

        self.layers = nn.ModuleList(lyrs)

    def forward(self, x: T) -> T:
        for lyr in self.layers:
            x = lyr(x)
        return x


class CNN2(nn.Module):
    def __init__(self, in_ch: int, out_dim: int, filters: int = 64) -> None:
        super().__init__()

        self.in_ch = in_ch
        self.out_dim = out_dim

        lyrs = []
        for i in range(4):
            lyrs.append(ConvResidual(in_ch if i == 0 else filters, filters=filters, stride=1))

        self.layers = nn.ModuleList(lyrs)
        self.clf = torch.nn.LazyLinear(out_features=out_dim)

    def forward(self, x: T) -> T:
        for lyr in self.layers:
            x = lyr(x)

        x = x.view(x.size(0), -1)
        x = self.clf(x)
        return x


if __name__ == "__main__":
    model = ResidualSpectralCNN(1, spectral=True, residual=True, ctype="none")
    x = torch.randn(32, 1, 28, 28)
    out = model(x)
    print(out.shape)

    model = CNN(3, spectral=True, residual=True, ctype="none")
    x = torch.randn(32, 3, 84, 84)
    out = model(x)
    print(out.shape)
