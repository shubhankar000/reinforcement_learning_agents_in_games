"""
Make SB3 write to wandb as well.
"""

import sys
import numpy as np
import wandb
from stable_baselines3.common.logger import HumanOutputFormat, KVWriter, Logger


class WandBWriter(KVWriter):
    def write(self, key_values: dict, key_excluded: dict, step: int = 0):
        payload = {}
        for key, value in key_values.items():
            if "wandb" in (key_excluded.get(key) or ()):
                continue
            if isinstance(
                value, (int, float, np.integer, np.floating)
            ) and not isinstance(value, bool):
                payload[key] = float(value)

        if payload:
            wandb.log(payload, step=step)

    def close(self):
        pass


def patch_wandb(model):
    """
    Replace model's logger with stdout + wandb
    """
    model.set_logger(
        Logger(
            folder=None, output_formats=[HumanOutputFormat(sys.stdout), WandBWriter()]
        )
    )
