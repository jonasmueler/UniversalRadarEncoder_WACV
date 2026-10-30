"""Residual building blocks of the 3D high-resolution backbone."""

from __future__ import annotations

import torch
import torch.nn as nn


def create_conv(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int,
    order: str,
    num_groups: int,
    padding: int,
    is3d: bool,
) -> list[tuple[str, nn.Module]]:
    """Create the modules of one conv layer with optional norm and non-linearity.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Size of the convolving kernel.
        stride: Convolution stride.
        order: Layer order, one character per module: ``c`` conv, ``g``
            GroupNorm, ``b`` BatchNorm, ``r`` ReLU, ``l`` LeakyReLU, ``e`` ELU
            (e.g. ``"gcr"`` = GroupNorm, conv, ReLU).
        num_groups: Number of groups for GroupNorm (reduced to 1 if larger
            than the channel count).
        padding: Zero-padding added to all sides of the input.
        is3d: Use ``Conv3d`` if True, otherwise ``Conv2d``.

    Returns:
        List of ``(name, module)`` tuples.

    Raises:
        ValueError: If ``order`` is invalid or the channel count is not
            divisible by ``num_groups``.
    """
    if "c" not in order:
        raise ValueError("Conv layer MUST be present")
    if order[0] in "rle":
        raise ValueError("Non-linearity cannot be the first operation in the layer")

    modules = []
    for i, char in enumerate(order):
        if char == "r":
            modules.append(("ReLU", nn.ReLU(inplace=True)))
        elif char == "l":
            modules.append(("LeakyReLU", nn.LeakyReLU(inplace=True)))
        elif char == "e":
            modules.append(("ELU", nn.ELU(inplace=True)))
        elif char == "c":
            # Conv bias only when the layer has no norm.
            bias = not ("g" in order or "b" in order)
            if is3d:
                conv = nn.Conv3d(
                    in_channels, out_channels, kernel_size, stride, padding=padding, bias=bias
                )
            else:
                conv = nn.Conv2d(
                    in_channels, out_channels, kernel_size, stride, padding=padding, bias=bias
                )

            modules.append(("conv", conv))
        elif char == "g":
            is_before_conv = i < order.index("c")
            num_channels = in_channels if is_before_conv else out_channels
            if num_channels < num_groups:
                num_groups = 1
            if num_channels % num_groups != 0:
                raise ValueError(
                    "Expected number of channels in input to be divisible by num_groups. "
                    f"num_channels={num_channels}, num_groups={num_groups}"
                )
            modules.append(
                ("groupnorm", nn.GroupNorm(num_groups=num_groups, num_channels=num_channels))
            )
        elif char == "b":
            is_before_conv = i < order.index("c")
            bn = nn.BatchNorm3d if is3d else nn.BatchNorm2d
            if is_before_conv:
                modules.append(("batchnorm", bn(in_channels)))
            else:
                modules.append(("batchnorm", bn(out_channels)))
        else:
            raise ValueError(
                f"Unsupported layer type '{char}'. MUST be one of ['b', 'g', 'r', 'l', 'e', 'c']"
            )

    return modules


class SingleConv(nn.Sequential):
    """Conv layer with optional norm and non-linearity in a configurable order.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Size of the convolving kernel.
        stride: Convolution stride.
        order: Layer order string, see :func:`create_conv`.
        num_groups: Number of groups for GroupNorm.
        padding: Zero-padding added to all sides of the input.
        is3d: Use ``Conv3d`` if True, otherwise ``Conv2d``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        order: str = "gcr",
        num_groups: int = 8,
        padding: int = 1,
        is3d: bool = True,
    ) -> None:
        super().__init__()

        for name, module in create_conv(
            in_channels, out_channels, kernel_size, stride, order, num_groups, padding, is3d
        ):
            self.add_module(name, module)


class ResNetBlock(nn.Module):
    """Residual block after https://arxiv.org/pdf/1706.00120.pdf.

    A 1x1 conv first maps the input to ``out_channels`` (identity if equal);
    two 3x3 convs form the residual branch, and the non-linearity is applied
    after the residual sum.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        stride: Stride of the first 3x3 conv.
        downsample: Optional module applied to the shortcut.
        order: Layer order of the 3x3 convs, see :func:`create_conv`.
        num_groups: Number of groups for GroupNorm.
        is3d: Use 3D convolutions if True, otherwise 2D.
    """

    expansion = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
        order: str = "gcr",
        num_groups: int = 8,
        is3d: bool = True,
    ) -> None:
        super().__init__()
        kernel_size = 3
        self.downsample = downsample
        if in_channels != out_channels:
            if is3d:
                self.conv1 = nn.Conv3d(in_channels, out_channels, 1)
            else:
                self.conv1 = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.conv1 = nn.Identity()

        self.conv2 = SingleConv(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            order=order,
            num_groups=num_groups,
            is3d=is3d,
        )
        # The last conv has no non-linearity; it is applied after the residual sum.
        n_order = order
        for c in "rel":
            n_order = n_order.replace(c, "")
        self.conv3 = SingleConv(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            order=n_order,
            num_groups=num_groups,
            is3d=is3d,
        )

        if "l" in order:
            self.non_linearity = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        elif "e" in order:
            self.non_linearity = nn.ELU(inplace=True)
        else:
            self.non_linearity = nn.ReLU(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the residual block.

        Args:
            x: Input feature map ``(B, in_channels, ...)``.

        Returns:
            Output feature map ``(B, out_channels, ...)``.
        """
        residual = self.conv1(x)
        out = self.conv2(residual)
        out = self.conv3(out)
        if self.downsample is not None:
            residual = self.downsample(residual)
        out += residual
        out = self.non_linearity(out)
        return out


blocks_dict: dict[str, type[nn.Module]] = {"ResNetBlock": ResNetBlock}
