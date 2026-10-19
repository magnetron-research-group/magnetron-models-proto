# +---------------------------------------------------------------------+
# | (c) 2026 Mario Sieg <mario.sieg.64@gmail.com>                       |
# | Licensed under the Apache License, Version 2.0                      |
# |                                                                     |
# | Website : https://mariosieg.com                                     |
# | GitHub  : https://github.com/MarioSieg                              |
# | License : https://www.apache.org/licenses/LICENSE-2.0               |
# +---------------------------------------------------------------------+


from abc import ABC
from dataclasses import fields
from typing import get_type_hints, get_args, get_origin, Any, Callable
from enum import Enum
from magnetron import nn, context, Tensor
import gc


class ConfigBase(ABC):
    @staticmethod
    def decode(config_cls: Callable[..., Any], data: dict[str, Any]) -> object:
        hints = get_type_hints(config_cls)

        def _decode_inner(hint: object, value: object) -> object:
            for candidate in (hint, *get_args(hint)):
                if value is not None and isinstance(candidate, type) and issubclass(candidate, Enum):
                    return candidate(value)
            if get_origin(hint) is frozenset:
                return frozenset(value)
            if get_origin(hint) is tuple and isinstance(value, list):
                return tuple(value)
            return value

        return config_cls(**{f.name: _decode_inner(hints[f.name], data[f.name]) for f in fields(config_cls) if f.name in data})


class SnapshotModule(nn.Module):
    cfg: Any

    def __init__(self) -> None:
        super().__init__()
        self.snapshot_metadata: dict[str, Any] = {}

    def load_tensors(self, tensors: dict[str, Tensor], metadata: dict[str, Any], writable: bool = False) -> None:
        self.snapshot_metadata = metadata
        src_repo: str | None = self.snapshot_metadata.get('src_repo')
        if src_repo is not None and src_repo != self.cfg.repo_id:
            raise RuntimeError(f'Snapshot was converted from {src_repo} but this model is configured for {self.cfg.repo_id}')
        device: str = context.get_default_device()
        for name, param in self.named_parameters():
            tensor = tensors.pop(name, None)
            if tensor is None:
                raise KeyError(f'Snapshot has no tensor named {name}')
            if tuple(tensor.shape) != tuple(param.shape):
                raise RuntimeError(f'Shape mismatch for {name}: {tensor.shape} != {param.shape}')
            if tensor.dtype != param.dtype:
                raise RuntimeError(f'Dtype mismatch for {name}: {tensor.dtype} != {param.dtype}')
            if writable:
                tensor = tensor.clone()
            param.data = tensor.transfer(device)
        del tensors
        gc.collect()


class ModelBase(SnapshotModule, ABC):
    def __init__(self, cfg: ConfigBase, is_trainable: bool) -> None:
        self.cfg = cfg
        self._is_trainable = is_trainable

    @property
    def is_trainable(self) -> bool:
        return self._is_trainable
