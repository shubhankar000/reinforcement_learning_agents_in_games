"""
SB3 gives only 1 injection point for custom architectures; the `feature_extractor_class`, and a kwargs dict. So encoder, temporal aggregator
"""

from typing import Any

import gymnasium as gym
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn


class PerFrameMLP(nn.Module):
    """
    Encode every frame of a stacked observation with the same small MLP before passing it on to different aggregators for architecture ablation. Only to be used for the Pole Balancing task, which works on Kinematic Telemetry (KIN).
    Key here is the aggregator consumes the frames (and hence the time history) as memory for the network.

    obs (B, K, 20) -> (B*K, 20) -> PerFrameMLP (B*K, d) -> (B, K, d) -> aggregator (B, d)
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
    Per Frame CNN feature extractor. TODO this is a stub
    """

    FRAME_RANK: int = 2

    def __init__(self) -> None:
        super().__init__()
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class FoldedExtractor(BaseFeaturesExtractor):
    """
    Run a shared encoder (MLP or CNN) then consume the frame axis.

    Supports encoder swapping for different tasks

    K is derived by comparing obs rank against the encoders `FRAME_RANK`.

    This is the final class passed into SB3 once the encoder and aggregator is declared.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        encoder_cls: PerFrameMLP | PerFrameCNN,
        aggregator_cls=None,
        encoder_kwargs=None,
        aggregator_kwargs=None,
        d: int = 64,
    ):
        super().__init__(observation_space, features_dim=d)

        encoder_kwargs = encoder_kwargs if encoder_kwargs else {}
        aggregator_kwargs = aggregator_kwargs if aggregator_kwargs else {}

        shape = tuple(observation_space.shape)
        rank = encoder_cls.FRAME_RANK

        if len(shape) == rank:
            self.k_frames, self.frame_shape = 1, shape
        elif len(shape) == rank + 1:
            self.k_frames, self.frame_shape = shape[0], shape[1:]
        else:
            raise ValueError(f"Unsupported obs shape {shape}")

        if self.k_frames > 1:
            if aggregator_cls is None:
                raise ValueError("Frame stack in obs but no aggregator class provided")
            # build from class not instance, else weights are tied in parallel seed runs
            self.aggregator = aggregator_cls(
                k_frames=self.k_frames, d=d, **aggregator_kwargs
            )
        else:  # This path required param-free aggregators
            if aggregator_cls is not None:
                raise ValueError("Obs is 1 frame but aggregator class provided")
            self.aggregator = None

        self.norm = nn.LayerNorm(d)
        self.encoder = encoder_cls(self.frame_shape, d, **encoder_kwargs)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        if self.aggregator is None:
            return self.norm(self.encoder(observations))

        B = observations.shape[0]
        x = self.encoder(observations.reshape(B * self.k_frames, *self.frame_shape))
        x = self.aggregator(x.reshape(B, self.k_frames, -1))

        return self.norm(x)
