import torch
from torch import Tensor
import typing
import torch.nn as nn
from typing import Type, Any, Callable, Union, List, Optional
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


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1, spectral: bool = False, ctype: str = "error") -> nn.Module:
    """3x3 convolution with padding"""
    layer = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False, dilation=dilation)
    if spectral:
        return WrappedSpectral(layer, ctype=ctype)
    return layer


def conv1x1(in_planes: int, out_planes: int, stride: int = 1, spectral: bool = False, ctype: str = "error") -> nn.Module:
    """1x1 convolution"""
    layer = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
    if spectral:
        return WrappedSpectral(layer, ctype=ctype)
    return layer


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
        spectral: bool = False,
        ctype: str = "error"
    ) -> None:
        super(BasicBlock, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride, spectral=spectral, ctype=ctype)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, spectral=spectral, ctype=ctype)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = identity + self.relu(out)
        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(Bottleneck, self).__init__()

        raise NotImplementedError("Bottleneck has not been implemented to work with the spectral normalization...do this like basicblock before using this")
        raise NotImplementedError("fix the identities")
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 5,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        spectral: bool = False,
        ctype: str = "error"
    ) -> None:
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.spectral = spectral
        self.ctype = "ctype"

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        if self.spectral:
            self.conv1 = WrappedSpectral(self.conv1, ctype=ctype)  # type: ignore

        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, self.inplanes, layers[0])
        self.layer2 = self._make_layer(block, self.inplanes, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, self.inplanes, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, self.inplanes, layers[3], stride=2, dilate=replace_stride_with_dilation[2])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

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
                conv1x1(self.inplanes, planes * block.expansion, stride, spectral=self.spectral, ctype=self.ctype),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, spectral=self.spectral, ctype=self.ctype))

        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, spectral=self.spectral, ctype=self.ctype))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        # print(f"input: {x.size()}")
        x = self.conv1(x)
        # print(f"after conv: {x.size()}")
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # print(f"after pool: {x.size()}")

        x = self.layer1(x)
        # print(f"after 1: {x.size()}")
        x = self.layer2(x)
        # print(f"after 2: {x.size()}")
        x = self.layer3(x)
        # print(f"input: 3 {x.size()}")
        x = self.layer4(x)
        # print(f"input 4: {x.size()}")

        x = self.avgpool(x)
        # print(f"after avgpool: {x.size()}")
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _resnet(block: Type[Union[BasicBlock, Bottleneck]], layers: List[int], **kwargs: Any) -> ResNet:
    return ResNet(block, layers, **kwargs)


def resnet12(**kwargs: Any) -> ResNet:
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet(BasicBlock, [1, 1, 2, 1], **kwargs)


class ResNet12(torch.nn.Module):
    """A modified version of ResNet-18 that suits meta-learning"""
    def __init__(self, dim_output: typing.Optional[int] = None, bn_affine: bool = False) -> None:
        super().__init__()

        # self.input_channel = input_channel
        self.dim_output = dim_output
        self.bn_affine = bn_affine
        self.net = self.modified_resnet12()

    def modified_resnet12(self) -> ResNet:
        net = resnet12()
        channels = 64

        # modify the resnet to suit the data
        net.conv1 = torch.nn.LazyConv2d(
            out_channels=channels,
            kernel_size=(7, 7),
            stride=(2, 2),
            padding=(3, 3),
            bias=not self.bn_affine
        )

        # update batch norm for meta-learning by setting momentum to 1
        net.bn1 = torch.nn.BatchNorm2d(channels, momentum=1, track_running_stats=False, affine=self.bn_affine)

        net.layer1[0].bn1 = torch.nn.BatchNorm2d(channels, momentum=1, track_running_stats=False, affine=self.bn_affine)
        net.layer1[0].bn2 = torch.nn.BatchNorm2d(channels, momentum=1, track_running_stats=False, affine=self.bn_affine)

        net.layer2[0].bn1 = torch.nn.BatchNorm2d(channels, momentum=1, track_running_stats=False, affine=self.bn_affine)
        net.layer2[0].bn2 = torch.nn.BatchNorm2d(channels, momentum=1, track_running_stats=False, affine=self.bn_affine)
        net.layer2[0].downsample[1] = torch.nn.BatchNorm2d(channels, momentum=1, track_running_stats=False, affine=self.bn_affine)

        net.layer3[0].bn1 = torch.nn.BatchNorm2d(channels, momentum=1, track_running_stats=False, affine=self.bn_affine)
        net.layer3[0].bn2 = torch.nn.BatchNorm2d(channels, momentum=1, track_running_stats=False, affine=self.bn_affine)
        net.layer3[0].downsample[1] = torch.nn.BatchNorm2d(channels, momentum=1, track_running_stats=False, affine=self.bn_affine)

        net.layer3[1].bn1 = torch.nn.BatchNorm2d(channels, momentum=1, track_running_stats=False, affine=self.bn_affine)
        net.layer3[1].bn2 = torch.nn.BatchNorm2d(channels, momentum=1, track_running_stats=False, affine=self.bn_affine)

        net.layer4[0].bn1 = torch.nn.BatchNorm2d(channels, momentum=1, track_running_stats=False, affine=self.bn_affine)
        net.layer4[0].bn2 = torch.nn.BatchNorm2d(channels, momentum=1, track_running_stats=False, affine=self.bn_affine)
        net.layer4[0].downsample[1] = torch.nn.BatchNorm2d(channels, momentum=1, track_running_stats=False, affine=self.bn_affine)

        # last layer
        if self.dim_output is not None:
            net.fc = torch.nn.LazyLinear(out_features=self.dim_output)
        else:
            net.fc = torch.nn.Identity()  # type: ignore

        return net

    def count_parameters(model) -> int:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # type: ignore


if __name__ == "__main__":
    # print(model12)

    model = ResNet12(5)
    x_omniglot = torch.randn(32, 1, 28, 28)
    out = model(x_omniglot)
    print(model.count_parameters())
    print(out.size())

    model = ResNet12(5)
    x_mimgnet = torch.randn(32, 3, 84, 84)
    out = model(x_mimgnet)
    print(out.size())
