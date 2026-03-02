# src/memory/methods/empty.py
from dataclasses import dataclass
from src.memory.base import MASMemoryBase


@dataclass
class EmptyMemory(MASMemoryBase):
    """No-memory baseline。所有 cross-trial 操作均为空。"""

    def __post_init__(self):
        super().__post_init__()