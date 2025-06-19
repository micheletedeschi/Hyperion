import yaml
import dataclasses
from enum import Enum
from .base_config import HyperionV2Config


def load_config(path: str) -> HyperionV2Config:
    """Load ``HyperionV2Config`` from a YAML file."""
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    # Pydantic dataclasses validate fields on creation
    return HyperionV2Config(**data)


def _serialize(obj):
    """Recursively convert dataclasses and Enums to plain Python objects."""
    if dataclasses.is_dataclass(obj):
        return {k: _serialize(getattr(obj, k)) for k in obj.__dataclass_fields__}
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, list):
        return [_serialize(i) for i in obj]
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    return obj


def save_config(config: HyperionV2Config, path: str) -> None:
    """Save configuration to YAML."""
    with open(path, "w") as f:
        yaml.safe_dump(_serialize(config), f)


__all__ = ["load_config", "save_config"]
