"""Multi-resolution 3D CNN backbone (3D adaptation of HRNet)."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .hrnet_blocks import blocks_dict

# The paper encoder uses "hr_tiny_feat64_zyx_l4_in64"; the feat32 variant is a
# small configuration for tests.  LAYER1.INPLANES is overwritten by the model.
HRNET_CONFIGS: dict[str, dict[str, Any]] = {
    "hr_tiny_feat64_zyx_l4_in64": {
        "LAYER1": {"INPLANES": 128, "BLOCK": "ResNetBlock"},
        "STAGE2": {
            "INPLANES": 64,
            "NUM_MODULES": 1,
            "NUM_BRANCHES": 2,
            "NUM_BLOCKS": [1, 1],
            "NUM_CHANNELS": [64, 64],
            "BLOCK": "ResNetBlock",
            "FUSE_METHOD": "SUM",
        },
        "STAGE3": {
            "NUM_MODULES": 1,
            "NUM_BRANCHES": 3,
            "NUM_BLOCKS": [1, 1, 1],
            "NUM_CHANNELS": [64, 64, 128],
            "BLOCK": "ResNetBlock",
            "FUSE_METHOD": "SUM",
        },
        "STAGE4": {
            "NUM_MODULES": 1,
            "NUM_BRANCHES": 4,
            "NUM_BLOCKS": [1, 1, 1, 1],
            "NUM_CHANNELS": [64, 64, 128, 128],
            "BLOCK": "ResNetBlock",
            "FUSE_METHOD": "SUM",
        },
    },
    "hr_tiny_feat32_zyx_l4_in32": {
        "LAYER1": {"INPLANES": 32, "BLOCK": "ResNetBlock"},
        "STAGE2": {
            "INPLANES": 32,
            "NUM_MODULES": 1,
            "NUM_BRANCHES": 2,
            "NUM_BLOCKS": [1, 1],
            "NUM_CHANNELS": [32, 32],
            "BLOCK": "ResNetBlock",
            "FUSE_METHOD": "SUM",
        },
        "STAGE3": {
            "NUM_MODULES": 1,
            "NUM_BRANCHES": 3,
            "NUM_BLOCKS": [1, 1, 1],
            "NUM_CHANNELS": [32, 32, 64],
            "BLOCK": "ResNetBlock",
            "FUSE_METHOD": "SUM",
        },
        "STAGE4": {
            "NUM_MODULES": 1,
            "NUM_BRANCHES": 4,
            "NUM_BLOCKS": [1, 1, 1, 1],
            "NUM_CHANNELS": [32, 32, 64, 64],
            "BLOCK": "ResNetBlock",
            "FUSE_METHOD": "SUM",
        },
    },
}


class HighResolutionModule(nn.Module):
    """One multi-branch HRNet module: parallel residual branches plus fusion.

    Branch ``i`` runs at ``1 / 2**i`` of the full resolution.  The fusion
    adds every branch into every output resolution: lower-resolution inputs
    via 1x1 conv and trilinear upsampling, higher-resolution inputs via
    strided 3x3 convs.

    Args:
        num_branches: Number of parallel resolution branches.
        blocks: Residual block class used inside the branches.
        num_blocks: Number of blocks per branch.
        num_inchannels: Input channels per branch (updated in place to the
            output channels).
        num_channels: Block width per branch.
        fuse_method: Fusion method name from the config (only summation is
            implemented).
        multi_scale_output: Output all resolutions (True) or only the highest.

    Raises:
        ValueError: If the per-branch lists do not match ``num_branches``.
    """

    def __init__(
        self,
        num_branches: int,
        blocks: type[nn.Module],
        num_blocks: list[int],
        num_inchannels: list[int],
        num_channels: list[int],
        fuse_method: str,
        multi_scale_output: bool = True,
    ) -> None:
        super().__init__()
        self._check_branches(
            num_branches,
            num_blocks,
            num_inchannels,
            num_channels,
        )
        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches
        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(
            num_branches,
            blocks,
            num_blocks,
            num_channels,
        )
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(inplace=False)

    @staticmethod
    def _check_branches(
        num_branches: int,
        num_blocks: list[int],
        num_inchannels: list[int],
        num_channels: list[int],
    ) -> None:
        """Check that every per-branch list has one entry per branch.

        Args:
            num_branches: Number of branches.
            num_blocks: Number of blocks per branch.
            num_inchannels: Input channels per branch.
            num_channels: Block width per branch.

        Raises:
            ValueError: If a list length differs from ``num_branches``.
        """
        for name, lst in [
            ("NUM_BLOCKS", num_blocks),
            ("NUM_CHANNELS", num_channels),
            ("NUM_INCHANNELS", num_inchannels),
        ]:
            if num_branches != len(lst):
                raise ValueError(f"NUM_BRANCHES({num_branches}) != {name}({len(lst)})")

    def _make_one_branch(
        self,
        branch_index: int,
        block: type[nn.Module],
        num_blocks: list[int],
        num_channels: list[int],
        stride: int = 1,
    ) -> nn.Sequential:
        """Build the residual block stack of one branch.

        Args:
            branch_index: Index of the branch.
            block: Residual block class.
            num_blocks: Number of blocks per branch.
            num_channels: Block width per branch.
            stride: Stride of the first block.

        Returns:
            Sequential stack of residual blocks.
        """
        downsample = None
        if (
            stride != 1
            or self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion
        ):
            downsample = nn.Sequential(
                nn.GroupNorm(8, self.num_inchannels[branch_index]),
                nn.Conv3d(
                    self.num_inchannels[branch_index],
                    num_channels[branch_index] * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
            )
        layers: list[nn.Module] = [
            block(
                self.num_inchannels[branch_index],
                num_channels[branch_index],
                stride,
                downsample,
            )
        ]
        self.num_inchannels[branch_index] = num_channels[branch_index] * block.expansion
        for _ in range(1, num_blocks[branch_index]):
            layers.append(
                block(
                    self.num_inchannels[branch_index],
                    num_channels[branch_index],
                )
            )
        return nn.Sequential(*layers)

    def _make_branches(
        self,
        num_branches: int,
        block: type[nn.Module],
        num_blocks: list[int],
        num_channels: list[int],
    ) -> nn.ModuleList:
        """Build all branches.

        Args:
            num_branches: Number of branches.
            block: Residual block class.
            num_blocks: Number of blocks per branch.
            num_channels: Block width per branch.

        Returns:
            One sequential block stack per branch.
        """
        return nn.ModuleList(
            [
                self._make_one_branch(
                    i,
                    block,
                    num_blocks,
                    num_channels,
                )
                for i in range(num_branches)
            ]
        )

    def _make_fuse_layers(self) -> nn.ModuleList | None:
        """Build the cross-resolution fusion layers.

        Entry ``[i][j]`` maps branch ``j`` to the resolution and width of
        branch ``i``: a 1x1 conv for ``j > i`` (upsampled in forward),
        ``None`` for ``j == i``, and ``i - j`` stride-2 3x3 convs for ``j < i``.

        Returns:
            Nested module list of fusion layers, or None for a single branch.
        """
        if self.num_branches == 1:
            return None
        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer: list[nn.Module | None] = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(
                        nn.Sequential(
                            nn.GroupNorm(8, num_inchannels[j]),
                            nn.Conv3d(
                                num_inchannels[j],
                                num_inchannels[i],
                                1,
                                1,
                                0,
                                bias=False,
                            ),
                        )
                    )
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            out_ch = num_inchannels[i]
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.GroupNorm(8, num_inchannels[j]),
                                    nn.Conv3d(
                                        num_inchannels[j],
                                        out_ch,
                                        3,
                                        2,
                                        1,
                                        bias=False,
                                    ),
                                )
                            )
                        else:
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.GroupNorm(8, num_inchannels[j]),
                                    nn.Conv3d(
                                        num_inchannels[j],
                                        num_inchannels[j],
                                        3,
                                        2,
                                        1,
                                        bias=False,
                                    ),
                                    nn.ReLU(inplace=False),
                                )
                            )
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))
        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self) -> list[int]:
        """Return the output channels per branch.

        Returns:
            Channels per branch after the residual blocks.
        """
        return self.num_inchannels

    def forward(self, x: list[torch.Tensor]) -> list[torch.Tensor]:
        """Run the branches and fuse them across resolutions.

        Args:
            x: One feature map ``(B, C_i, A_i, E_i, R_i)`` per branch.

        Returns:
            Fused feature maps, one per output resolution.
        """
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                elif j > i:
                    y = y + F.interpolate(
                        self.fuse_layers[i][j](x[j]),
                        size=x[i].shape[2:],
                        mode="trilinear",
                        align_corners=True,
                    )
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))
        return x_fuse


class HighResolution3DNet(nn.Module):
    """Multi-resolution 3D CNN backbone.

    A residual stem (``layer1``) at full resolution is followed by stages 2-4,
    each adding one lower-resolution branch through a transition layer and
    running a :class:`HighResolutionModule`.

    Args:
        cfg: Plain-dict HRNet config (from ``HRNET_CONFIGS``).
    """

    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__()
        self.layer1_cfg = cfg["LAYER1"]
        self.stage2_cfg = cfg["STAGE2"]

        block = blocks_dict[self.layer1_cfg["BLOCK"]]
        self.layer1 = block(
            self.layer1_cfg["INPLANES"],
            self.stage2_cfg["INPLANES"],
            order="gcr",
        )

        num_channels = self.stage2_cfg["NUM_CHANNELS"]
        block = blocks_dict[self.stage2_cfg["BLOCK"]]
        num_channels = [ch * block.expansion for ch in num_channels]
        self.transition1 = self._make_transition_layer(
            [self.stage2_cfg["INPLANES"]],
            num_channels,
        )
        self.stage2, pre_stage_channels = self._make_stage(
            self.stage2_cfg,
            num_channels,
        )

        self.stage3_cfg = cfg["STAGE3"]
        num_channels = self.stage3_cfg["NUM_CHANNELS"]
        block = blocks_dict[self.stage3_cfg["BLOCK"]]
        num_channels = [ch * block.expansion for ch in num_channels]
        self.transition2 = self._make_transition_layer(
            pre_stage_channels,
            num_channels,
        )
        self.stage3, pre_stage_channels = self._make_stage(
            self.stage3_cfg,
            num_channels,
        )

        self.stage4_cfg = cfg.get("STAGE4")
        if self.stage4_cfg is not None:
            num_channels = self.stage4_cfg["NUM_CHANNELS"]
            block = blocks_dict[self.stage4_cfg["BLOCK"]]
            num_channels = [ch * block.expansion for ch in num_channels]
            self.transition3 = self._make_transition_layer(
                pre_stage_channels,
                num_channels,
            )
            self.stage4, pre_stage_channels = self._make_stage(
                self.stage4_cfg,
                num_channels,
                multi_scale_output=True,
            )

    @staticmethod
    def _make_transition_layer(
        num_channels_pre: list[int],
        num_channels_cur: list[int],
    ) -> nn.ModuleList:
        """Build the transition from one stage's branches to the next stage's.

        Existing branches get a 3x3 conv if their width changes (else None);
        each new branch is derived from the lowest-resolution input by
        stride-2 3x3 convs.

        Args:
            num_channels_pre: Channels per branch of the previous stage.
            num_channels_cur: Channels per branch of the next stage.

        Returns:
            One transition module (or None) per branch of the next stage.
        """
        layers: list[nn.Module | None] = []
        for i in range(len(num_channels_cur)):
            if i < len(num_channels_pre):
                if num_channels_cur[i] != num_channels_pre[i]:
                    layers.append(
                        nn.Sequential(
                            nn.GroupNorm(8, num_channels_pre[i]),
                            nn.Conv3d(
                                num_channels_pre[i],
                                num_channels_cur[i],
                                3,
                                1,
                                1,
                                bias=False,
                            ),
                            nn.ReLU(inplace=False),
                        )
                    )
                else:
                    layers.append(None)
            else:
                conv3x3s = []
                for j in range(i + 1 - len(num_channels_pre)):
                    in_ch = num_channels_pre[-1]
                    out_ch = num_channels_cur[i] if j == i - len(num_channels_pre) else in_ch
                    conv3x3s.append(
                        nn.Sequential(
                            nn.GroupNorm(8, in_ch),
                            nn.Conv3d(in_ch, out_ch, 3, 2, 1, bias=False),
                            nn.ReLU(inplace=False),
                        )
                    )
                layers.append(nn.Sequential(*conv3x3s))
        return nn.ModuleList(layers)

    @staticmethod
    def _make_stage(
        layer_config: dict[str, Any],
        num_inchannels: list[int],
        multi_scale_output: bool = True,
    ) -> tuple[nn.Sequential, list[int]]:
        """Build one HRNet stage from its config.

        Args:
            layer_config: Stage config (``NUM_MODULES``, ``NUM_BRANCHES``,
                ``NUM_BLOCKS``, ``NUM_CHANNELS``, ``BLOCK``, ``FUSE_METHOD``).
            num_inchannels: Input channels per branch.
            multi_scale_output: Whether the last module outputs all resolutions.

        Returns:
            The stage and its output channels per branch.
        """
        num_modules = layer_config["NUM_MODULES"]
        num_branches = layer_config["NUM_BRANCHES"]
        num_blocks = layer_config["NUM_BLOCKS"]
        num_channels = layer_config["NUM_CHANNELS"]
        block = blocks_dict[layer_config["BLOCK"]]
        fuse_method = layer_config["FUSE_METHOD"]

        modules = []
        for i in range(num_modules):
            reset = multi_scale_output or i < num_modules - 1
            modules.append(
                HighResolutionModule(
                    num_branches,
                    block,
                    num_blocks,
                    num_inchannels,
                    num_channels,
                    fuse_method,
                    reset,
                )
            )
            num_inchannels = modules[-1].get_num_inchannels()
        return nn.Sequential(*modules), num_inchannels

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Run the backbone.

        Args:
            x: Input volume ``(B, C_in, A, E, R)``.

        Returns:
            Feature maps of all branches, highest resolution first.
        """
        x = self.layer1(x)

        x_list = []
        for i in range(self.stage2_cfg["NUM_BRANCHES"]):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(self.stage3_cfg["NUM_BRANCHES"]):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        if self.stage4_cfg is not None:
            x_list = []
            for i in range(self.stage4_cfg["NUM_BRANCHES"]):
                if self.transition3[i] is not None:
                    x_list.append(self.transition3[i](y_list[-1]))
                else:
                    x_list.append(y_list[i])
            y_list = self.stage4(x_list)

        return y_list
