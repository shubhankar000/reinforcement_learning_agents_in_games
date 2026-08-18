"""
SB3 gives only 1 injection point for custom architectures; the `feature_extractor_class`, and a kwargs dict. So encoder, temporal architecture
"""

import gymnasium as gym
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

from src.drones.config import VECTOR_DIM


class PerFrameMLP(nn.Module):
    """
    Encode every frame of a stacked observation with the same small MLP before passing it on to different architectures for ablation. Only to be used for the Pole Balancing task, which works on Kinematic Telemetry (KIN).
    Key here is the arch consumes the frames (and hence the time history) as memory for the network.

    obs (B, K, 20) -> (B*K, 20) -> PerFrameMLP (B*K, d) -> (B, K, d) -> arch (B, d)
    Here B is batch size, K is the frame-stack, d is the dim of the MLP, set to 64 in this case.

    For LSTM, there is no frame stack, obs is (20,) instead of (k, 20), MLP encoder runs once.

    FRAME_RANK tells FoldedExtractor how many trailing dims in a frame so it can infer K. Rank 1 is 1 frame (20,) or (8, 20)
    """

    FRAME_RANK: int = 1

    def __init__(self, frame_shape, d=64):
        super().__init__()

        (obs_dim,) = frame_shape
        self.net = nn.Sequential(
            nn.Linear(obs_dim, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
        )  # 20 -> 64 -> 64

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PerFrameCNN(nn.Module):
    """
    Per Frame CNN feature extractor.
    obs (B*K, 96, 96) -> (B*K, d)

    obs already arrives normalised (/255) by SB3's preprocessing

    Modify NatureCNN, add a 1x1 bottleneck at the end to keep CNN params low, so that aggregators can be bulk of the full NN's param counts. Without the bottleneck, the MLP would be 4096x512, which would be 2.1M params just for 1 MLP layer, would dominate the feature extractor.
    """

    FRAME_RANK: int = 2

    def __init__(self, shape, d=512) -> None:
        super().__init__()

        h, w = shape

        assert (h, w) == (96, 96), (
            f"drone camera initialised to camera res not (96, 96) but {(h, w)}"
        )

        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 8, 4),  # 96 -> 23
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2),  # 23 -> 10
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1),  # 10 -> 8
            nn.ReLU(),
            nn.Conv2d(64, 16, 1),  # 1x1 bottleneck
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1024, d),  # Wouldve been 4096 if not for 1x1 bottleneck
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.unsqueeze(1))


class DictFoldedExtractor(BaseFeaturesExtractor):
    """
    Folded extractor for Dict{pixels, vector} for Task B

    pixels (B, K, H, W) -> perframeCNN (B, K, d) -> architecture -> (B, d) -> LayerNorm
    vector (B, 30) concat after aggregator (only 1 entry, at latest point in time, no stack)
    """

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        encoder_cls: type[PerFrameMLP] | type[PerFrameCNN],
        architecture_cls=None,
        encoder_kwargs=None,
        architecture_kwargs=None,
        d: int = 512,
    ):
        super().__init__(observation_space, features_dim=d + VECTOR_DIM)

        encoder_kwargs = encoder_kwargs if encoder_kwargs else {}
        architecture_kwargs = architecture_kwargs if architecture_kwargs else {}

        shape = tuple(observation_space["pixels"].shape)
        rank = encoder_cls.FRAME_RANK

        if len(shape) == rank:
            self.k_frames, self.frame_shape = 1, shape
        elif len(shape) == rank + 1:
            self.k_frames, self.frame_shape = shape[0], shape[1:]
        else:
            raise ValueError(f"Unsupported obs shape {shape}")

        if self.k_frames > 1:
            if architecture_cls is None:
                raise ValueError(
                    "Frame stack in obs but no architecture class provided"
                )
            # build from class not instance, else weights are tied in parallel seed runs
            self.architecture = architecture_cls(
                k_frames=self.k_frames, d=d, **architecture_kwargs
            )
        else:
            if architecture_cls is not None:
                raise ValueError("Obs is 1 frame but architecture class provided")
            self.architecture = None

        self.norm = nn.LayerNorm(d)
        self.encoder = encoder_cls(self.frame_shape, d, **encoder_kwargs)

    def forward(self, observations: dict[str, torch.Tensor]):
        px = observations["pixels"]
        B = px.shape[0]

        x = self.encoder(px.reshape(B * self.k_frames, *self.frame_shape))

        if self.architecture is not None:
            x = self.architecture(x.reshape(B, self.k_frames, -1))

        return torch.cat([self.norm(x), observations["vector"]], dim=1)


class FoldedExtractor(BaseFeaturesExtractor):
    """
    Run a shared encoder (MLP or CNN) then consume the frame axis.

    Supports encoder swapping for different tasks

    K is derived by comparing obs rank against the encoders `FRAME_RANK`.

    This is the final class passed into SB3 once the encoder and arch is declared.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        encoder_cls: type[PerFrameMLP] | type[PerFrameCNN],
        architecture_cls=None,
        encoder_kwargs=None,
        architecture_kwargs=None,
        d: int = 64,
    ):
        super().__init__(observation_space, features_dim=d)

        encoder_kwargs = encoder_kwargs if encoder_kwargs else {}
        architecture_kwargs = architecture_kwargs if architecture_kwargs else {}

        shape = tuple(observation_space.shape)
        rank = encoder_cls.FRAME_RANK

        if len(shape) == rank:
            self.k_frames, self.frame_shape = 1, shape
        elif len(shape) == rank + 1:
            self.k_frames, self.frame_shape = shape[0], shape[1:]
        else:
            raise ValueError(f"Unsupported obs shape {shape}")

        if self.k_frames > 1:
            if architecture_cls is None:
                raise ValueError(
                    "Frame stack in obs but no architecture class provided"
                )
            # build from class not instance, else weights are tied in parallel seed runs
            self.architecture = architecture_cls(
                k_frames=self.k_frames, d=d, **architecture_kwargs
            )
        else:
            if architecture_cls is not None:
                raise ValueError("Obs is 1 frame but architecture class provided")
            self.architecture = None

        self.norm = nn.LayerNorm(d)
        self.encoder = encoder_cls(self.frame_shape, d, **encoder_kwargs)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        if self.architecture is None:
            return self.norm(self.encoder(observations))

        B = observations.shape[0]
        x = self.encoder(observations.reshape(B * self.k_frames, *self.frame_shape))
        x = self.architecture(x.reshape(B, self.k_frames, -1))

        return self.norm(x)
