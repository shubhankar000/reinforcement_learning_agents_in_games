"""
Generic config class to be used by all learner classes.
"""

import json

import numpy as np

DEFAULT_SEED = 67


class BaseConfig:
    def to_dict(self) -> dict:
        """
        Plain-python dict of all public attributes, recursing into nested configs.
        """
        return {
            key: self._encode(value)
            for key, value in vars(self).items()
            if not key.startswith("_")  # skip _env and friends
        }

    @staticmethod
    def _encode(value):
        if isinstance(value, BaseConfig):  # nested config -> recurse
            return value.to_dict()
        if isinstance(value, (list, tuple)):
            return [BaseConfig._encode(v) for v in value]
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, dict):
            return {k: BaseConfig._encode(v) for k, v in value.items()}
        return value  # int / float / str / bool / None

    def json(self, **kwargs) -> str:
        """
        Serialize to a JSON string. Extra kwargs pass through to json.dumps
        """
        return json.dumps(self.to_dict(), default=str, **kwargs)
