import torch
from torchvision.models import resnet18
import typing as _typing
from typing import Any, Optional, Callable, List, Union, Type
from torch.nn.functional import normalize
from torch.nn import functional as F
import math
from torch import nn


T = torch.Tensor


class FcNet(torch.nn.Module):
    """Simple fully connected network
    """
    def __init__(self, dim_output: _typing.Optional[int] = None, num_hidden_units: _typing.List[int] = (32, 32)) -> None:
        """
        Args:

        """
        super(FcNet, self).__init__()

        self.dim_output = dim_output
        self.num_hidden_units = num_hidden_units

        self.fc_net = self.construct_network()

    def construct_network(self):
        """
        """
        net = torch.nn.Sequential()
        net.add_module(
            name='layer0',
            module=torch.nn.Sequential(
                torch.nn.LazyLinear(out_features=self.num_hidden_units[0]),
                torch.nn.ReLU()
            )
        )

        for i in range(1, len(self.num_hidden_units)):
            net.add_module(
                name='layer{0:d}'.format(i),
                module=torch.nn.Sequential(
                    torch.nn.Linear(in_features=self.num_hidden_units[i - 1], out_features=self.num_hidden_units[i]),
                    torch.nn.ReLU()
                )
            )
        
        net.add_module(
            name='classifier',
            module=torch.nn.Linear(in_features=self.num_hidden_units[-1], out_features=self.dim_output) if self.dim_output is not None \
                else torch.nn.Identity()
        )

        return net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """"""
        return self.fc_net(x)

class CNN(torch.nn.Module):
    """A simple convolutional module networks
    """
    def __init__(self, dim_output: _typing.Optional[int] = None, bn_affine: bool = False) -> None:
        """Initialize an instance

        Args:
            dim_output: the number of classes at the output. If None,
                the last fully-connected layer will be excluded.
            image_size: a 3-d tuple consisting of (nc, iH, iW)

        """
        super(CNN, self).__init__()

        self.dim_output = dim_output
        self.kernel_size = (3, 3)
        self.stride = (2, 2)
        self.padding = (1, 1)
        self.num_channels = (64, 64, 64, 64)
        self.bn_affine = bn_affine
        self.cnn = self.construct_network()
    
    def construct_network(self) -> torch.nn.Module:
        """Construct the network

        """
        net = torch.nn.Sequential()
        net.add_module(
            name='layer0',
            module=torch.nn.Sequential(
                torch.nn.LazyConv2d(
                    out_channels=self.num_channels[0],
                    kernel_size=self.kernel_size,
                    stride=self.stride,
                    padding=self.padding,
                    bias=not self.bn_affine
                ),
                torch.nn.BatchNorm2d(
                    num_features=self.num_channels[0],
                    momentum=1,
                    track_running_stats=False,
                    affine=self.bn_affine
                ),
                torch.nn.ReLU()
            )
        )

        for i in range(1, len(self.num_channels)):
            net.add_module(
                name='layer{0:d}'.format(i),
                module=torch.nn.Sequential(
                    torch.nn.Conv2d(
                        in_channels=self.num_channels[i - 1],
                        out_channels=self.num_channels[i],
                        kernel_size=self.kernel_size,
                        stride=self.stride,
                        padding=self.padding,
                        bias=not self.bn_affine
                    ),
                    torch.nn.BatchNorm2d(
                        num_features=self.num_channels[i],
                        momentum=1,
                        track_running_stats=False,
                        affine=self.bn_affine
                    ),
                    torch.nn.ReLU()
                )
            )

        net.add_module(name='Flatten', module=torch.nn.Flatten(start_dim=1, end_dim=-1))

        if self.dim_output is None:
            clf = torch.nn.Identity()
        else:
            clf = torch.nn.LazyLinear(out_features=self.dim_output)

        net.add_module(name='classifier', module=clf)

        return net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward"""
        return self.cnn(x)


class ResNet18(torch.nn.Module):
    """A modified version of ResNet-18 that suits meta-learning"""
    def __init__(self, dim_output: _typing.Optional[int] = None, bn_affine: bool = False) -> None:
        """
        Args:
            dim_output: the number of classes at the output. If None,
                the last fully-connected layer will be excluded.
        """
        super(ResNet18, self).__init__()

        # self.input_channel = input_channel
        self.dim_output = dim_output
        self.bn_affine = bn_affine
        self.net = self.modified_resnet18()

    def modified_resnet18(self):
        """
        """
        net = resnet18(pretrained=False)

        # modify the resnet to suit the data
        net.conv1 = torch.nn.LazyConv2d(
            out_channels=64,
            kernel_size=(7, 7),
            stride=(2, 2),
            padding=(3, 3),
            bias=not self.bn_affine
        )

        # update batch norm for meta-learning by setting momentum to 1
        net.bn1 = torch.nn.BatchNorm2d(64, momentum=1, track_running_stats=False, affine=self.bn_affine)

        net.layer1[0].bn1 = torch.nn.BatchNorm2d(64, momentum=1, track_running_stats=False, affine=self.bn_affine)
        net.layer1[0].bn2 = torch.nn.BatchNorm2d(64, momentum=1, track_running_stats=False, affine=self.bn_affine)

        net.layer1[1].bn1 = torch.nn.BatchNorm2d(64, momentum=1, track_running_stats=False, affine=self.bn_affine)
        net.layer1[1].bn2 = torch.nn.BatchNorm2d(64, momentum=1, track_running_stats=False, affine=self.bn_affine)

        net.layer2[0].bn1 = torch.nn.BatchNorm2d(128, momentum=1, track_running_stats=False, affine=self.bn_affine)
        net.layer2[0].bn2 = torch.nn.BatchNorm2d(128, momentum=1, track_running_stats=False, affine=self.bn_affine)
        net.layer2[0].downsample[1] = torch.nn.BatchNorm2d(128, momentum=1, track_running_stats=False, affine=self.bn_affine)

        net.layer2[1].bn1 = torch.nn.BatchNorm2d(128, momentum=1, track_running_stats=False, affine=self.bn_affine)
        net.layer2[1].bn2 = torch.nn.BatchNorm2d(128, momentum=1, track_running_stats=False, affine=self.bn_affine)

        net.layer3[0].bn1 = torch.nn.BatchNorm2d(256, momentum=1, track_running_stats=False, affine=self.bn_affine)
        net.layer3[0].bn2 = torch.nn.BatchNorm2d(256, momentum=1, track_running_stats=False, affine=self.bn_affine)
        net.layer3[0].downsample[1] = torch.nn.BatchNorm2d(256, momentum=1, track_running_stats=False, affine=self.bn_affine)

        net.layer3[1].bn1 = torch.nn.BatchNorm2d(256, momentum=1, track_running_stats=False, affine=self.bn_affine)
        net.layer3[1].bn2 = torch.nn.BatchNorm2d(256, momentum=1, track_running_stats=False, affine=self.bn_affine)

        net.layer4[0].bn1 = torch.nn.BatchNorm2d(512, momentum=1, track_running_stats=False, affine=self.bn_affine)
        net.layer4[0].bn2 = torch.nn.BatchNorm2d(512, momentum=1, track_running_stats=False, affine=self.bn_affine)
        net.layer4[0].downsample[1] = torch.nn.BatchNorm2d(512, momentum=1, track_running_stats=False, affine=self.bn_affine)

        net.layer4[1].bn1 = torch.nn.BatchNorm2d(512, momentum=1, track_running_stats=False, affine=self.bn_affine)
        net.layer4[1].bn2 = torch.nn.BatchNorm2d(512, momentum=1, track_running_stats=False, affine=self.bn_affine)

        # last layer
        if self.dim_output is not None:
            net.fc = torch.nn.LazyLinear(out_features=self.dim_output)
        else:
            net.fc = torch.nn.Identity()

        return net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """"""
        return self.net(x)

class MiniCNN(torch.nn.Module):
    def __init__(self, dim_output: _typing.Optional[int] = None, bn_affine: bool = False) -> None:
        """Initialize an instance

        Args:
            dim_output: the number of classes at the output. If None,
                the last fully-connected layer will be excluded.
            image_size: a 3-d tuple consisting of (nc, iH, iW)

        """
        super(MiniCNN, self).__init__()

        self.dim_output = dim_output
        self.kernel_size = (3, 3)
        self.stride = (2, 2)
        self.padding = (1, 1)
        self.num_channels = (32, 32, 32, 32)
        self.bn_affine = bn_affine
        self.cnn = self.construct_network()
    
    def construct_network(self) -> torch.nn.Module:
        """Construct the network

        """
        net = torch.nn.Sequential(
            torch.nn.LazyConv2d(
                out_channels=4,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=True
            ),
            torch.nn.BatchNorm2d(
                num_features=4,
                momentum=1.,
                affine=False,
                track_running_stats=False
            ),
            torch.nn.ReLU(),

            torch.nn.Conv2d(
                in_channels=4,
                out_channels=8,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=True
            ),
            torch.nn.BatchNorm2d(
                num_features=8,
                momentum=1.,
                affine=False,
                track_running_stats=False
            ),
            torch.nn.ReLU(),

            torch.nn.Conv2d(
                in_channels=8,
                out_channels=16,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=True
            ),
            torch.nn.BatchNorm2d(
                num_features=16,
                momentum=1.,
                affine=False,
                track_running_stats=False
            ),
            torch.nn.ReLU(),

            torch.nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=True
            ),
            torch.nn.BatchNorm2d(
                num_features=32,
                momentum=1.,
                affine=False,
                track_running_stats=False
            ),
            torch.nn.ReLU(),

            torch.nn.Flatten(start_dim=1, end_dim=-1)
        )
        if self.dim_output is None:
            clf = torch.nn.Identity()
        else:
            clf = torch.nn.LazyLinear(out_features=self.dim_output)

        net.add_module(name='classifier', module=clf)

        return net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward"""
        return self.cnn(x)


class WrappedSpectral(nn.Module):
    def __init__(self, base_layer: nn.Module, ctype: str = "scalar", n_power_iterations: int = 1, dim: int = 0, eps: float = 1e-12, weight_name: str = "weight") -> None:
        super().__init__()
        self.base_layer: nn.Module
        self.add_module("base_layer", base_layer)
        self.n_power_iterations = n_power_iterations
        self.dim = dim
        self.eps = eps
        self.weight_name = weight_name

        self.c: T
        if ctype == "none":
            self.register_buffer("c", torch.tensor(0.0, requires_grad=False))
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
        weight = weight * (0.01 + 0.99 * F.softplus(self.c))

        setattr(self.base_layer, self.weight_name, weight)

    def forward(self, x: T) -> T:
        if self.base_layer.weight.device != self.base_layer.weight_u.device:
            self.base_layer.weight = self.base_layer.weight.to(self.base_layer.weight_u.device)  # type: ignore
        # if self.training:
        self.compute_weight()
        return self.base_layer(x)  # type: ignore


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1, ctype: str = "none", spectral: bool = False) -> Any:
    """3x3 convolution with padding"""
    if spectral:
        return WrappedSpectral(nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False, dilation=dilation), ctype=ctype)
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False, dilation=dilation)
    # return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1, ctype: str = "none", spectral: bool = False) -> Any:
    """1x1 convolution"""
    if spectral:
        return WrappedSpectral(nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False), ctype=ctype)
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
    # return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        ctype: str = "none",
        momentum: float = 1.0,
        track_running_stats: bool = False
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride, ctype=ctype)
        self.bn1 = norm_layer(planes, momentum=momentum, track_running_stats=track_running_stats, affine=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, ctype=ctype)
        self.bn2 = norm_layer(planes, momentum=momentum, track_running_stats=track_running_stats, affine=False)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: T) -> T:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = identity + self.relu(out)
        # out = self.relu(identity + out)
        return out  # type: ignore


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        ctype: str = "none"
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width, ctype=ctype)
        self.bn1 = norm_layer(width, momentum=self.momentum, track_running_stats=self.track_running_stats, affine=False)

        self.conv2 = conv3x3(width, width, stride, groups, dilation, ctype=ctype)
        self.bn2 = norm_layer(width, momentum=self.momentum, track_running_stats=self.track_running_stats, affine=False)

        self.conv3 = conv1x1(width, planes * self.expansion, ctype=ctype)
        self.bn3 = norm_layer(planes * self.expansion, momentum=self.momentum, track_running_stats=self.track_running_stats, affine=False)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x: T) -> T:
        identity = x

        print(f"bottleneck in: {x.size()}")
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        print(f"bottleneck 1: {out.size()}")

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        print(f"bottleneck 2: {out.size()}")

        out = self.conv3(out)
        out = self.bn3(out)
        print(f"bottleneck 3: {out.size()}")

        if self.downsample is not None:
            identity = self.downsample(x)

        out = identity + self.relu(out)
        return out  # type: ignore


class ResNet12Base(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        in_channels: int,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        p: float = 0.01,
        ctype: str = "none",
        momentum: float = 1,
        track_running_stats: bool = False
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.ctype = ctype
        self.inplanes = 32
        self.dilation = 1
        self.in_channels = in_channels
        self.momentum = momentum
        self.track_running_stats = track_running_stats

        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]

        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))

        self.groups = groups
        self.base_width = width_per_group

        self.conv1 = conv3x3(in_channels, self.inplanes, stride=1, dilation=1, spectral=False)
        # self.conv1 = WrappedSpectral(nn.Conv2d(in_channels, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False), ctype=ctype)
        self.bn1 = norm_layer(self.inplanes, momentum=momentum, track_running_stats=track_running_stats, affine=False)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 32, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 32, layers[1], stride=1, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 32, layers[2], stride=1, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 32, layers[3], stride=1, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            # elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            #     nn.init.constant_(m.weight, 1)
            #     nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride, ctype=self.ctype),
                norm_layer(planes * block.expansion, momentum=self.momentum, track_running_stats=self.track_running_stats, affine=False),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, ctype=self.ctype))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, ctype=self.ctype))

        return nn.Sequential(*layers)

    def count_parameters(model) -> int:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def _forward_impl(self, x: T) -> T:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        # print(f"after in layer: {x.size()}")
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.maxpool(x)
        # print(f"after 1 layer: {x.size()}")
        x = self.layer2(x)
        x = self.maxpool(x)
        # print(f"after 2 layer: {x.size()}")
        x = self.layer3(x)
        x = self.maxpool(x)
        # print(f"after 3 layer: {x.size()}")
        x = self.layer4(x)
        x = self.maxpool(x)
        # print(f"after 4 layer: {x.size()}")

        # x = self.avgpool(x)
        x = torch.flatten(x, 1)
        # print(f"after pool: {x.size()}")

        return x

    def forward(self, x: T) -> T:
        return self._forward_impl(x)


def resnet12(in_ch: int, p: float = 0.01, ctype: str = "none") -> ResNet12Base:
    return ResNet12Base(BasicBlock, [2, 1, 1, 1], in_ch, p=p, ctype=ctype)


class ResNet12(torch.nn.Module):
    def __init__(self, dim_output: _typing.Optional[int] = None, bn_affine: bool = False) -> None:
        super().__init__()

        self.dim_output = dim_output
        self.kernel_size = (3, 3)
        self.stride = (2, 2)
        self.padding = (1, 1)
        self.num_channels = (64, 64, 64, 64)
        self.bn_affine = bn_affine
        self.cnn = self.construct_network()

    def construct_network(self) -> torch.nn.Module:
        """Construct the network

        """
        net = nn.Sequential(resnet12(1))
        clf = torch.nn.LazyLinear(out_features=self.dim_output)
        net.add_module(name='classifier', module=clf)

        return net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward"""
        return self.cnn(x)


if __name__ == "__main__":
    model = ResNet12Base(BasicBlock, [2, 1, 1, 1], in_channels=1)
    inputs = torch.randn(32, 1, 28, 28)
    outputs = model(inputs)
    print(outputs.size())

    model = ResNet12Base(BasicBlock, [2, 1, 1, 1], in_channels=3)
    print(model.count_parameters())
    inputs = torch.randn(32, 3, 84, 84)
    outputs = model(inputs)
    print(outputs.size())

    print(model)
    exit()

    model = _resnet(BasicBlock, [2, 2, 2, 2])
    print(model)

    inputs = torch.randn(32, 3, 64, 64)
    outputs = model(inputs)
    print(outputs.size())


