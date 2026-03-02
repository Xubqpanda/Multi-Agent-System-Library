# src/registry/registry.py
from typing import Type

from src.mas.base import MetaMAS
from src.mas.autogen import AutoGen
from src.mas.macnet import MacNet
from src.mas.dylan import DyLAN
from src.mas.reasoning import ReasoningBase, ReasoningIO

from src.memory.base import MASMemoryBase
from src.memory.methods import (
    EmptyMemory,
    GenerativeMASMemory,
    VoyagerMASMemory,
    MemoryBankMASMemory,
    ChatDevMASMemory,
    MetaGPTMASMemory,
    GMemory,
)


# ─── MAS  ────────────────────────────────────────────────────────────

MAS_REGISTRY: dict[str, Type[MetaMAS]] = {
    "autogen": AutoGen,
    "macnet": MacNet,
    "dylan": DyLAN,
}

# ─── Memory ─────────────────────────────────────────────────────────

MEMORY_REGISTRY: dict[str, Type[MASMemoryBase]] = {
    "empty": EmptyMemory,
    "generative": GenerativeMASMemory,
    "voyager": VoyagerMASMemory,
    "memorybank": MemoryBankMASMemory,
    "chatdev": ChatDevMASMemory,
    "metagpt": MetaGPTMASMemory,
    "g-memory": GMemory,
}

# ─── Reasoning  ──────────────────────────────────────────────────────

REASONING_REGISTRY: dict[str, Type[ReasoningBase]] = {
    "io": ReasoningIO,
}


# ─── get functions ──────────────────────────────────────────────────────────────────

def get_mas_cls(name: str) -> Type[MetaMAS]:
    if name not in MAS_REGISTRY:
        raise ValueError(f"Unknown MAS framework '{name}'. Available: {list(MAS_REGISTRY)}")
    return MAS_REGISTRY[name]


def get_memory_cls(name: str) -> Type[MASMemoryBase]:
    if name not in MEMORY_REGISTRY:
        raise ValueError(f"Unknown memory method '{name}'. Available: {list(MEMORY_REGISTRY)}")
    return MEMORY_REGISTRY[name]


def get_reasoning_cls(name: str) -> Type[ReasoningBase]:
    if name not in REASONING_REGISTRY:
        raise ValueError(f"Unknown reasoning type '{name}'. Available: {list(REASONING_REGISTRY)}")
    return REASONING_REGISTRY[name]