from dataclasses import dataclass


@dataclass
class ScalpingStrategy:
    timeframe: str = "1m"
    hold_time: int = 300


@dataclass
class SwingStrategy:
    timeframe: str = "15m"
    hold_time: int = 86400


@dataclass
class PositionStrategy:
    timeframe: str = "4h"
    hold_time: int = 604800


__all__ = ["ScalpingStrategy", "SwingStrategy", "PositionStrategy"]
