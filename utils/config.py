"""Configuration loader: reads YAML config and provides dot-access."""

import yaml
import os
import copy


class Config(dict):
    """Nested dict with attribute access."""

    def __getattr__(self, key):
        try:
            val = self[key]
        except KeyError:
            raise AttributeError(f"Config has no attribute '{key}'")
        if isinstance(val, dict) and not isinstance(val, Config):
            val = Config(val)
            self[key] = val
        return val

    def __setattr__(self, key, val):
        self[key] = val

    def __delattr__(self, key):
        del self[key]

    def merge(self, other):
        for k, v in other.items():
            if k in self and isinstance(self[k], dict) and isinstance(v, dict):
                Config(self[k]).merge(v)
            else:
                self[k] = v


def load_config(path):
    """Load YAML config file and return Config object."""
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)
    return Config(cfg)
