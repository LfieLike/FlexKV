from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Union, List, Optional, Any, Dict, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from flexkv.common.config import LayerGroupSpec
    from flexkv.common.memory_handle import TensorSharedHandle


def _tensor_shared_handle_type():
    # Keep lightweight layout/config imports usable without loading CUDA IPC,
    # pyzmq, or libcudart. Worker paths import the handle type on first use.
    from flexkv.common.memory_handle import TensorSharedHandle
    return TensorSharedHandle


class AccessHandleType(Enum):
    TENSOR = auto()  # single tensor or tensor list
    FILE = auto()  # single file or file list
    TENSOR_HANDLE = auto()  # single tensor handle or tensor handle list
    GDS_MANAGER = auto()

# NOTE: GPU layout depends on the vLLM version's non-MLA KV cache shape:
#   vLLM <= 0.21: (kv, num_blocks, ...)  -> LAYERFIRST
#   vLLM >= 0.23: (num_blocks, kv, ...)  -> LAYERBLOCK
# CPU, SSD, remote layout should be the same, either LAYERFIRST or BLOCKFIRST.
class KVCacheLayoutType(Enum):
    LAYERFIRST = "LAYERFIRST"
    BLOCKFIRST = "BLOCKFIRST"
    LAYERBLOCK = "LAYERBLOCK"

@dataclass
class KVCacheLayout:
    type: KVCacheLayoutType
    num_layer: int
    num_block: int
    tokens_per_block: int
    num_head: int
    head_size: int
    is_mla: bool
    _kv_shape: Optional[torch.Size] = None
    # Heterogeneous layouts use a byte-flat BLOCKFIRST buffer. Each group can
    # carry a different head shape, dtype, and tokens-per-block compression.
    layer_groups: Optional[List[LayerGroupSpec]] = None
    # Number of TP slices stored in each CPU/SSD block for multi-group layouts.
    tp_size: int = 1

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, KVCacheLayout):
            return NotImplemented
        return (self.type == other.type and
                self.num_layer == other.num_layer and
                self.num_block == other.num_block and
                self.tokens_per_block == other.tokens_per_block and
                self.num_head == other.num_head and
                self.head_size == other.head_size and
                self.is_mla == other.is_mla and
                self.layer_groups == other.layer_groups and
                self.tp_size == other.tp_size and
                self.kv_shape == other.kv_shape)

    @property
    def kv_dim(self) -> int:
        return 2 if not self.is_mla else 1

    @property
    def kv_shape(self) -> torch.Size:
        if self._kv_shape is None:
            self._compute_kv_shape()
        assert self._kv_shape is not None
        return self._kv_shape

    def __post_init__(self) -> None:
        self._compute_kv_shape()

    def _compute_kv_shape(self) -> None:
        if self._kv_shape is None:
            if self.layer_groups is not None:
                if self.type != KVCacheLayoutType.BLOCKFIRST:
                    raise ValueError(
                        "Multi-group KVCacheLayout currently requires "
                        f"BLOCKFIRST, got {self.type}"
                    )
                if self.tp_size < 1:
                    raise ValueError(
                        f"Multi-group KVCacheLayout requires tp_size >= 1, got {self.tp_size}"
                    )
                if not self.layer_groups:
                    raise ValueError("layer_groups must not be empty")

                for gi, group in enumerate(self.layer_groups):
                    if group.num_layers < 1:
                        raise ValueError(
                            f"layer_groups[{gi}].num_layers must be positive, "
                            f"got {group.num_layers}"
                        )
                    if group.num_layers != len(group.layer_indices):
                        raise ValueError(
                            f"layer_groups[{gi}].num_layers={group.num_layers} "
                            "does not match len(layer_indices)="
                            f"{len(group.layer_indices)}"
                        )
                    if group.num_kv_heads < 1 or group.head_size < 1:
                        raise ValueError(
                            f"layer_groups[{gi}] requires positive "
                            "num_kv_heads/head_size"
                        )
                    if len(set(group.layer_indices)) != len(group.layer_indices):
                        raise ValueError(
                            f"layer_groups[{gi}].layer_indices contains duplicates"
                        )
                    if any(
                        layer_id < 0 or layer_id >= self.num_layer
                        for layer_id in group.layer_indices
                    ):
                        raise ValueError(
                            f"layer_groups[{gi}] contains a layer outside "
                            f"[0, {self.num_layer})"
                        )
                    if group.dtype is None:
                        raise ValueError(
                            "Multi-group KVCacheLayout requires every group dtype "
                            f"to be resolved; layer_groups[{gi}].dtype is None"
                        )
                    if group.compress_ratio < 1:
                        raise ValueError(
                            f"layer_groups[{gi}].compress_ratio must be >= 1, "
                            f"got {group.compress_ratio}"
                        )
                    if self.tokens_per_block % group.compress_ratio != 0:
                        raise ValueError(
                            f"layer_groups[{gi}].compress_ratio={group.compress_ratio} "
                            f"does not divide tokens_per_block={self.tokens_per_block}"
                        )

                # The second dimension is bytes, not elements of one shared
                # dtype. CPU/SSD storage uses a uint8 view for this layout.
                bytes_per_block = self.tp_size * sum(
                    group.num_layers
                    * self.kv_dim
                    * (self.tokens_per_block // group.compress_ratio)
                    * group.num_kv_heads
                    * group.head_size
                    * group.dtype.itemsize
                    for group in self.layer_groups
                )
                self._kv_shape = torch.Size([self.num_block, bytes_per_block])
            elif self.type == KVCacheLayoutType.LAYERFIRST:  # for Layerwise transfer
                self._kv_shape = torch.Size([self.num_layer,
                                             self.kv_dim,
                                             self.num_block,
                                             self.tokens_per_block,
                                             self.num_head,
                                             self.head_size])
            elif self.type == KVCacheLayoutType.BLOCKFIRST:
                self._kv_shape = torch.Size([self.num_block,
                                             self.num_layer,
                                             self.kv_dim,
                                             self.tokens_per_block,
                                             self.num_head,
                                             self.head_size])
            elif self.type == KVCacheLayoutType.LAYERBLOCK:  # vLLM >= 0.23 non-MLA GPU layout
                self._kv_shape = torch.Size([self.num_layer,
                                             self.num_block,
                                             self.kv_dim,
                                             self.tokens_per_block,
                                             self.num_head,
                                             self.head_size])
            else:
                raise ValueError(f"Invalid KVCacheLayoutType: {self.type}")

    def div_block(self, num_chunks: int, padding: bool = False) -> 'KVCacheLayout':
        if padding:
            num_blocks = (self.num_block + num_chunks - 1) // num_chunks
        else:
            assert self.num_block % num_chunks == 0, \
                f"num_block {self.num_block} must be divisible by num_chunks {num_chunks}"
            num_blocks = self.num_block // num_chunks
        new_layout = KVCacheLayout(
            type=self.type,
            num_layer=self.num_layer,
            num_block=num_blocks,
            tokens_per_block=self.tokens_per_block,
            num_head=self.num_head,
            head_size=self.head_size,
            is_mla=self.is_mla,
            layer_groups=self.layer_groups,
            tp_size=self.tp_size,
        )
        return new_layout

    def div_layer(self, num_chunks: int) -> 'KVCacheLayout':
        if self.layer_groups is not None:
            raise ValueError(
                "div_layer() is ambiguous for a multi-group layout; "
                "partition LayerGroupSpec entries explicitly"
            )
        assert self.num_layer % num_chunks == 0, \
            f"num_layer {self.num_layer} must be divisible by num_chunks {num_chunks}"
        new_layout = KVCacheLayout(
            type=self.type,
            num_layer=self.num_layer // num_chunks,
            num_block=self.num_block,
            tokens_per_block=self.tokens_per_block,
            num_head=self.num_head,
            head_size=self.head_size,
            is_mla=self.is_mla,
        )
        return new_layout

    def div_head(self, num_chunks: int) -> 'KVCacheLayout':
        if self.layer_groups is not None:
            raise ValueError(
                "div_head() is not valid for a multi-group layout; "
                "group head counts are already per-rank"
            )
        assert self.num_head % num_chunks == 0, \
            f"num_head {self.num_head} must be divisible by num_chunks {num_chunks}"
        new_layout = KVCacheLayout(
            type=self.type,
            num_layer=self.num_layer,
            num_block=self.num_block,
            tokens_per_block=self.tokens_per_block,
            num_head=self.num_head // num_chunks,
            head_size=self.head_size,
            is_mla=self.is_mla,
        )
        return new_layout

    def get_chunk_size(self) -> int:
        if self.layer_groups is not None:
            raise ValueError(
                "get_chunk_size() is not valid for a multi-group layout; "
                "use get_group_strides()"
            )
        return self.tokens_per_block * self.num_head * self.head_size

    def get_layer_stride(self) -> int:
        if self.layer_groups is not None:
            raise ValueError(
                "get_layer_stride() is not valid for a multi-group layout; "
                "use get_group_strides()"
            )
        if self.type == KVCacheLayoutType.LAYERFIRST:
            return self.kv_shape[1:].numel()
        elif self.type == KVCacheLayoutType.BLOCKFIRST:
            return self.kv_shape[2:].numel()
        elif self.type == KVCacheLayoutType.LAYERBLOCK:
            return self.kv_shape[1:].numel()
        else:
            raise ValueError(f"Invalid KVCacheLayoutType: {self.type}")

    def get_block_stride(self) -> int:
        if self.layer_groups is not None:
            # kv_shape[1] is already byte-sized for the uint8 backing buffer.
            return self.kv_shape[1]
        if self.type == KVCacheLayoutType.LAYERFIRST:
            return self.kv_shape[3:].numel()
        elif self.type == KVCacheLayoutType.BLOCKFIRST:
            return self.kv_shape[1:].numel()
        elif self.type == KVCacheLayoutType.LAYERBLOCK:
            return self.kv_shape[2:].numel()
        else:
            raise ValueError(f"Invalid KVCacheLayoutType: {self.type}")

    def get_kv_stride(self) -> int:
        if self.layer_groups is not None:
            raise ValueError(
                "get_kv_stride() is not valid for a multi-group layout; "
                "use get_group_strides()"
            )
        if self.type == KVCacheLayoutType.LAYERFIRST:
            return self.kv_shape[2:].numel()
        elif self.type == KVCacheLayoutType.BLOCKFIRST:
            return self.kv_shape[3:].numel()
        elif self.type == KVCacheLayoutType.LAYERBLOCK:
            return self.kv_shape[3:].numel()
        else:
            raise ValueError(f"Invalid KVCacheLayoutType: {self.type}")

    def get_group_strides(self) -> List[Dict[str, Any]]:
        """Return per-group offsets and strides inside one TP slice.

        Values are expressed in the group's native dtype elements. The backing
        multi-group block itself is byte-flat; callers convert each group's
        values with ``group.dtype.itemsize``.
        """
        if self.layer_groups is None:
            raise ValueError("get_group_strides() requires layer_groups")
        if self.type != KVCacheLayoutType.BLOCKFIRST:
            raise ValueError("get_group_strides() only supports BLOCKFIRST")

        result: List[Dict[str, Any]] = []
        offset_bytes = 0
        for group in self.layer_groups:
            assert group.dtype is not None
            group_tokens = self.tokens_per_block // group.compress_ratio
            chunk_size = group_tokens * group.num_kv_heads * group.head_size
            kv_stride = chunk_size
            layer_stride = self.kv_dim * kv_stride
            result.append({
                "num_layers": group.num_layers,
                "num_kv_heads": group.num_kv_heads,
                "head_size": group.head_size,
                "layer_indices": group.layer_indices,
                "dtype": group.dtype,
                "compress_ratio": group.compress_ratio,
                "offset_bytes": offset_bytes,
                "layer_stride": layer_stride,
                "kv_stride": kv_stride,
                "chunk_size": chunk_size,
            })
            offset_bytes += group.num_layers * layer_stride * group.dtype.itemsize
        return result

    def get_total_elements(self) -> int:
        return self.kv_shape.numel()

    def get_elements_per_block(self) -> int:
        return self.get_total_elements() // self.num_block


@dataclass
class StorageHandle:
    handle_type: AccessHandleType
    # The actual handle data
    data: Union[List[torch.Tensor],
                torch.Tensor,
                List[str],
                List[TensorSharedHandle],  # for shared gpu tensors
                Dict[int, List[str]]  # for ssd files: ssd_device_id -> file_paths
                ]
    kv_layout: KVCacheLayout
    dtype: torch.dtype
    # Optional metadata
    num_blocks_per_file: Optional[int] = None
    gpu_device_id: Optional[int] = None
    remote_config_custom: Optional[Dict[str, Any]] = None
    worker_data: Optional[Any] = None

    def get_tensor_list(self) -> List[torch.Tensor]:
        tensor_shared_handle = _tensor_shared_handle_type()
        assert isinstance(self.data, list) and \
                (all(isinstance(x, torch.Tensor) for x in self.data) or \
                all(isinstance(x, tensor_shared_handle) for x in self.data)), \
                "handle data must be List[Tensor] or List[TensorWrapper]"
        if self.handle_type == AccessHandleType.TENSOR:
            return self.data  # type: ignore
        elif self.handle_type == AccessHandleType.TENSOR_HANDLE:
            assert all(isinstance(x, tensor_shared_handle) for x in self.data), \
                "All elements must be TensorSharedHandle for TENSOR_HANDLE type"
            return [x.get_tensor() for x in self.data]  # type: ignore
        else:
            raise ValueError(f"Invalid handle type: {self.handle_type}, expected TENSOR or TENSOR_HANDLE")

    def get_tensor(self) -> torch.Tensor:
        assert isinstance(self.data, torch.Tensor), \
            "handle data must be torch.Tensor"
        if self.handle_type == AccessHandleType.TENSOR:
            return self.data
        else:
            raise ValueError(f"Invalid handle type: {self.handle_type}, expected TENSOR")

    def get_worker_tensor(self) -> Any:
        if self.worker_data is not None:
            return self.worker_data
        return self.get_tensor()

    def get_file_list(self) -> Union[List[str], Dict[int, List[str]]]:
        if self.handle_type == AccessHandleType.FILE:
            return self.data  # type: ignore
        else:
            raise ValueError(f"Invalid handle type: {self.handle_type}, expected FILE")

    def get_tensor_handle_list(self) -> List[TensorSharedHandle]:
        tensor_shared_handle = _tensor_shared_handle_type()
        assert isinstance(self.data, list) and \
                (all(isinstance(x, torch.Tensor) for x in self.data) or \
                all(isinstance(x, tensor_shared_handle) for x in self.data)), \
                "handle data must be List[Tensor] or List[TensorWrapper]"
        if self.handle_type == AccessHandleType.TENSOR_HANDLE:
            assert all(isinstance(x, tensor_shared_handle) for x in self.data), \
                "All elements must be TensorSharedHandle for TENSOR_HANDLE type"
            return self.data  # type: ignore
        elif self.handle_type == AccessHandleType.TENSOR:
            assert all(isinstance(x, torch.Tensor) for x in self.data), \
                "All elements must be torch.Tensor for TENSOR type"
            return [tensor_shared_handle(x) for x in self.data]  # type: ignore
        else:
            raise ValueError(f"Invalid handle type: {self.handle_type}, expected TENSOR_HANDLE or TENSOR")
