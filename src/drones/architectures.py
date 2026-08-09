"""
All architectures required for ablation.

Plugin interface into extractors.py to create SB3 compatible BaseFeatureExtractors for architecture ablation

Architectures included:
    MeanPool
    CAMPool (Dual Attention)
    ECAPool (Efficient Channel Attention)
    SEPool (Squeeze & Excitation)
    WindowMLP
    WindowTransformer
    WindowLSTM

Pool Groups is essentially parameter free
Window groups are parameter matched to ensure fairness and capacity doesnt affect performance
Comparisons within a group are controlled, across groups they are confounded by capacity differences.
No normalization applied here.
"""

from dataclasses import dataclass

import torch
from torch import nn

from vendored_libs.attention_mechanisms.dual_attention import CAM
from vendored_libs.attention_mechanisms.eca import ECALayer
from vendored_libs.attention_mechanisms.se_module import SELayer

# =============== Seuquence Models over frame-stack =============== #


class MeanPool(nn.Module):
    """
    The Control. 0 params, just means across the channels (frames).
    This is to check whether the other pooling can do better than this
    """

    def __init__(self, k_frames: int, d: int):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=1)


class WindowMLP(nn.Module):
    """
    Flatten the frames and process using an MLP
    """

    def __init__(self, k_frames: int, d: int):
        super().__init__()
        self.fc = nn.Linear(k_frames * d, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x.flatten(1))


class WindowTransformer(nn.Module):
    """
    Transformer module that operates sequentially over the windows.
    Consumes the window sequences as time.

    Learned positional embeddings used for this, as otherwise the Transformer is permutation-equivariant.
    """

    def __init__(self, k_frames: int, d: int, n_heads: int = 4):
        super().__init__()

        self.pos = nn.Parameter(torch.randn(1, k_frames, d) * 0.02)
        self.net = nn.TransformerEncoderLayer(
            d_model=d, nhead=n_heads, dim_feedforward=2 * d, batch_first=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x + self.pos).mean(dim=1)


class WindowLSTM(nn.Module):
    """
    LSTM within the k-frame window, not across the episode; that is the purpose of RecurrentPPO from sb3-contrib.

    LSTM recurses over the window, but starts with h=c=0. It acts like a sequence encoder over the window, not memory.
    """

    def __init__(self, k_frames: int, d: int):
        super().__init__()

        self.lstm = nn.LSTM(d, d, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return out[:, -1]


# =============== Pooling over frame-stack =============== #
# Modules that re-weight the frames, then pool
# These modules come from the pytorch attention zoo (https://github.com/changzy00/pytorch-attention)
# These modules are under vendored_libs and their license is preserved


class FramesAsChannels(nn.Module):
    """
    Run a (B, C, H, W) vision module over the frame axis, where C=frames.

    These modules preserve shape (originally to allow for skip connections and sit between conv layers). They do not reduce K. Hence all share mean pooling; these modules just re-weight the frames differently. Hence the standard meanpool as control

    This is a shared module, each individual attention block inherits from this
    """

    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x = (B, K, d)
        y = self.module(x.unsqueeze(-1))  # (B, C=K, H=d, W=1)
        return y.squeeze(-1).mean(dim=1)


class SEPool(FramesAsChannels):
    """
    Squeeze and Excitation over frames. (https://arxiv.org/pdf/1709.01507)
    """

    def __init__(self, k_frames: int, d: int, reduction: int = 2):
        assert k_frames // reduction >= 1, (
            "Reduction resulted in layer size of 0 -> degenerate"
        )

        super().__init__(SELayer(k_frames, reduction=reduction))


class ECAPool(FramesAsChannels):
    """
    Efficient Channel attention over frames (https://arxiv.org/pdf/1910.03151)
    """

    def __init__(self, k_frames: int, d: int):
        super().__init__(ECALayer(k_frames))


class CAMPool(FramesAsChannels):
    """
    DANet dual attention channel attention over frames. (https://arxiv.org/pdf/1809.02983)
    """

    def __init__(self, k_frames: int, d: int):
        super().__init__(CAM())


# =============== Spatial Attention for Pixels =============== #
# TODO
# Meant to be used with the CNN for pixels task - waypoints

# =============== Registry =============== #


@dataclass(frozen=True)
class Arch:
    cls: type[nn.Module]
    kwargs: dict


ARCHITECTURES: dict[str, Arch] = {
    "meanpool": Arch(MeanPool, {}),
    "cam": Arch(CAMPool, {}),
    "eca": Arch(ECAPool, {}),
    "se": Arch(SEPool, {"reduction": 2}),
    "windowmlp": Arch(WindowMLP, {}),
    "transformer": Arch(WindowTransformer, {"n_heads": 4}),
    "windowlstm": Arch(WindowLSTM, {}),
}
