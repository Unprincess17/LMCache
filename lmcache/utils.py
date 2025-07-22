# Copyright 2024-2025 LMCache Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import hashlib
import threading
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from nvtx import annotate  # type: ignore
import sys

# Type definition
KVCache = Tuple[Tuple[torch.Tensor, torch.Tensor], ...]


@dataclass
class DiskCacheMetadata:
    path: str
    size: int  # in bytes
    shape: Optional[torch.Size] = None
    dtype: Optional[torch.dtype] = None
    is_pin: bool = False

    def pin(self) -> bool:
        self.is_pin = True
        return True

    def unpin(self) -> bool:
        self.is_pin = False
        return True

    @property
    def is_pinned(self) -> bool:
        return self.is_pin


TORCH_DTYPE_TO_STR_DTYPE = {
    torch.half: "half",
    torch.float16: "half",
    torch.bfloat16: "bfloat16",
    torch.float: "float",
    torch.float32: "float",
    torch.float64: "double",
    torch.double: "double",
    torch.uint8: "fp8",
    torch.float8_e4m3fn: "fp8_e4m3",
    torch.float8_e5m2: "fp8_e5m2",
}


@dataclass(order=True)
class CacheEngineKey:
    fmt: str
    model_name: str
    world_size: int
    worker_id: int
    chunk_hash: str

    def __hash__(self):
        return hash((
            self.fmt,
            self.model_name,
            self.world_size,
            self.worker_id,
            self.chunk_hash,
        ))

    def to_string(self):
        return f"{self.fmt}@{self.model_name}@{self.world_size}"\
            f"@{self.worker_id}@{self.chunk_hash}"

    def split_layers(self, num_layers: int) -> List["LayerCacheEngineKey"]:
        """ Split the key into multiple keys for each layer """
        keys = []
        for layer_id in range(num_layers):
            keys.append(
                LayerCacheEngineKey(self.fmt, self.model_name, self.world_size,
                                    self.worker_id, self.chunk_hash, layer_id))
        return keys

    def get_first_layer(self) -> "LayerCacheEngineKey":
        """ Return the key for the first layer """
        key = LayerCacheEngineKey(self.fmt, self.model_name, self.world_size,
                                  self.worker_id, self.chunk_hash, 0)
        return key

    @staticmethod
    def from_string(s):
        parts = s.split("@")
        if len(parts) != 5:
            raise ValueError(f"Invalid key string: {s}")
        return CacheEngineKey(parts[0], parts[1], int(parts[2]), int(parts[3]),
                              parts[4])

    def to_dict(self):
        # Note(Kuntai): this is used for serializing CacheEngineKey via msgpack.
        return {
            "__type__": "CacheEngineKey",
            "fmt": self.fmt,
            "model_name": self.model_name,
            "world_size": self.world_size,
            "worker_id": self.worker_id,
            "chunk_hash": self.chunk_hash
        }

    @staticmethod
    def from_dict(d):
        return CacheEngineKey(fmt=d["fmt"],
                              model_name=d["model_name"],
                              world_size=d["world_size"],
                              worker_id=d["worker_id"],
                              chunk_hash=d["chunk_hash"])


@dataclass(order=True)
class LayerCacheEngineKey(CacheEngineKey):
    """ A key for the layer cache engine """
    layer_id: int
    chunk_num: int = 1

    def __hash__(self):
        return hash((
            self.fmt,
            self.model_name,
            self.world_size,
            self.worker_id,
            self.chunk_hash,
            self.layer_id,
            self.chunk_num,
        ))

    def to_string(self):
        return f"{self.fmt}@{self.model_name}@{self.world_size}"\
            f"@{self.worker_id}@{self.chunk_hash}@{self.layer_id}@{self.chunk_num}"

    @staticmethod
    def from_string(s):
        parts = s.split("@")
        if len(parts) != 7:
            raise ValueError(f"Invalid key string: {s}")
        return LayerCacheEngineKey(parts[0], parts[1], int(parts[2]),
                                   int(parts[3]), parts[4], int(parts[5]),
                                   int(parts[6]))

    def to_dict(self):
        return {
            "__type__": "LayerCacheEngineKey",
            "fmt": self.fmt,
            "model_name": self.model_name,
            "world_size": self.world_size,
            "worker_id": self.worker_id,
            "chunk_hash": self.chunk_hash,
            "layer_id": self.layer_id,
            "chunk_num": self.chunk_num
        }

    @staticmethod
    def from_dict(d):
        return LayerCacheEngineKey(fmt=d["fmt"],
                                   model_name=d["model_name"],
                                   world_size=d["world_size"],
                                   worker_id=d["worker_id"],
                                   chunk_hash=d["chunk_hash"],
                                   layer_id=d["layer_id"],
                                   chunk_num=d["chunk_num"])


@dataclass(order=True)
class ChunkMappingInfo:
    """Information about how individual chunks map to combined objects."""
    chunk_key: str  # String representation of the individual chunk key
    offset_start: int  # Start offset within combined object
    offset_end: int  # End offset within combined object

    def to_dict(self):
        return {
            "__type__": "ChunkMappingInfo",
            "chunk_key": self.chunk_key,
            "offset_start": self.offset_start,
            "offset_end": self.offset_end
        }

    @staticmethod
    def from_dict(d):
        return ChunkMappingInfo(chunk_key=d["chunk_key"],
                                offset_start=d["offset_start"],
                                offset_end=d["offset_end"])


@dataclass(order=True)
class CombinedLayerCacheEngineKey(LayerCacheEngineKey):
    """
    A specialized key for combined layer objects with chunk mapping information.
    
    This key represents a combined memory object that contains multiple individual 
    chunks from the same layer. It includes mapping information that allows 
    retrieval of individual chunks from different offsets within the combined object.
    
    Used in disaggregated inference scenarios where multiple small chunks are 
    combined into larger objects for efficient RDMA transfer.
    """
    chunk_mappings: List[
        ChunkMappingInfo] = None  # Mapping from individual chunks to offsets
    total_tokens: int = 0  # Total number of tokens in the combined object
    is_combined: bool = True  # Flag to identify this as a combined key

    def __post_init__(self):
        assert self.chunk_mappings is not None, \
            "The chunk mappings are None."

    def __hash__(self):
        # Include chunk mappings in hash for uniqueness
        mapping_hash = hash(
            tuple((mapping.chunk_key, mapping.offset_start, mapping.offset_end)
                  for mapping in self.chunk_mappings))
        return hash((
            self.fmt,
            self.model_name,
            self.world_size,
            self.worker_id,
            self.chunk_hash,
            self.layer_id,
            self.chunk_num,
            mapping_hash,
            self.total_tokens,
        ))

    def to_string(self):
        # Extended string format to include combined object info
        base_str = super().to_string()
        mapping_str = "|".join([
            f"{m.chunk_key}:{m.offset_start}-{m.offset_end}"
            for m in self.chunk_mappings
        ])
        return f"{base_str}@combined@{self.total_tokens}@[{mapping_str}]"

    def add_chunk_mapping(self, chunk_key: str, offset_start: int,
                          offset_end: int):
        """Add a new chunk mapping to this combined key."""
        mapping = ChunkMappingInfo(chunk_key, offset_start, offset_end)
        self.chunk_mappings.append(mapping)

    def get_chunk_mapping(self, chunk_key: str) -> Optional[ChunkMappingInfo]:
        """Get the mapping information for a specific chunk key."""
        for mapping in self.chunk_mappings:
            if mapping.chunk_key == chunk_key:
                return mapping
        return None

    def get_individual_chunk_keys(self) -> List[str]:
        """Get all individual chunk keys contained in this combined object."""
        return [mapping.chunk_key for mapping in self.chunk_mappings]

    @staticmethod
    def from_string(s):
        # Parse the extended string format for combined keys
        # Format: base@combined@total_tokens@[chunk_mappings]
        if "@combined@" not in s:
            raise ValueError(f"Invalid combined key string: {s}")

        parts = s.split("@combined@")
        if len(parts) != 2:
            raise ValueError(f"Invalid combined key string: {s}")

        # Parse base LayerCacheEngineKey part
        base_str = parts[0]
        base_parts = base_str.split("@")
        if len(base_parts) != 7:
            raise ValueError(f"Invalid base key in combined string: {s}")

        # Parse combined-specific parts
        combined_parts = parts[1].split("@[")
        if len(combined_parts) != 2:
            raise ValueError(f"Invalid combined format: {s}")

        total_tokens = int(combined_parts[0])
        mapping_str = combined_parts[1].rstrip("]")

        # Parse chunk mappings
        chunk_mappings = []
        if mapping_str:  # Only parse if not empty
            for mapping_part in mapping_str.split("|"):
                if mapping_part:  # Skip empty parts
                    key_and_range = mapping_part.split(":")
                    if len(key_and_range) == 2:
                        chunk_key = key_and_range[0]
                        range_part = key_and_range[1].split("-")
                        if len(range_part) == 2:
                            offset_start = int(range_part[0])
                            offset_end = int(range_part[1])
                            chunk_mappings.append(
                                ChunkMappingInfo(chunk_key, offset_start,
                                                 offset_end))

        return CombinedLayerCacheEngineKey(fmt=base_parts[0],
                                           model_name=base_parts[1],
                                           world_size=int(base_parts[2]),
                                           worker_id=int(base_parts[3]),
                                           chunk_hash=base_parts[4],
                                           layer_id=int(base_parts[5]),
                                           chunk_num=int(base_parts[6]),
                                           chunk_mappings=chunk_mappings,
                                           total_tokens=total_tokens)

    def to_dict(self):
        return {
            "__type__":
            "CombinedLayerCacheEngineKey",
            "fmt":
            self.fmt,
            "model_name":
            self.model_name,
            "world_size":
            self.world_size,
            "worker_id":
            self.worker_id,
            "chunk_hash":
            self.chunk_hash,
            "layer_id":
            self.layer_id,
            "chunk_num":
            self.chunk_num,
            "chunk_mappings":
            [mapping.to_dict() for mapping in self.chunk_mappings],
            "total_tokens":
            self.total_tokens,
            "is_combined":
            self.is_combined
        }

    @staticmethod
    def from_dict(d):
        # msgpack's object_hook deserializes nested objects recursively.
        # By the time this method is called, the items in d["chunk_mappings"]
        # have already been converted from dicts to ChunkMappingInfo objects,
        # so we can use them directly.
        chunk_mappings = d.get("chunk_mappings", [])

        return CombinedLayerCacheEngineKey(
            fmt=d["fmt"],
            model_name=d["model_name"],
            world_size=d["world_size"],
            worker_id=d["worker_id"],
            chunk_hash=d["chunk_hash"],
            layer_id=d["layer_id"],
            chunk_num=d["chunk_num"],
            chunk_mappings=chunk_mappings,
            total_tokens=d.get("total_tokens", 0),
            is_combined=d.get("is_combined", True))

    @classmethod
    def create_combined_key(cls, chunks_with_metadata: List[Tuple[
        int, int, LayerCacheEngineKey]],
                            layer_id: int) -> "CombinedLayerCacheEngineKey":
        """
        Factory method to create a combined key from multiple individual keys.
        
        Args:
            chunks_with_metadata: List of (start, end, layer_key) tuples to combine
            layer_id: The layer ID for this combined object
            
        Returns:
            A new CombinedLayerCacheEngineKey with proper chunk mappings
        """
        if not chunks_with_metadata:
            raise ValueError("Cannot create combined key from empty list")

        # Use the first key as the base for metadata
        base_key = chunks_with_metadata[0][2]

        # Create a deterministic hash from all individual keys
        import hashlib
        chunk_hashes = [key.chunk_hash for _, _, key in chunks_with_metadata]
        combined_hash = hashlib.sha256('|'.join(
            sorted(chunk_hashes)).encode()).hexdigest()

        # Populate chunk mappings and calculate total tokens
        chunk_mappings = []
        total_tokens = 0
        current_offset = 0
        for start, end, layer_key in chunks_with_metadata:
            num_chunk_tokens = end - start
            mapping = ChunkMappingInfo(chunk_key=layer_key.to_string(),
                                       offset_start=current_offset,
                                       offset_end=current_offset +
                                       num_chunk_tokens)
            chunk_mappings.append(mapping)
            current_offset += num_chunk_tokens
            total_tokens += num_chunk_tokens

        # Create the combined key
        combined_key = cls(
            fmt=base_key.fmt,
            model_name=base_key.model_name,
            world_size=base_key.world_size,
            worker_id=base_key.worker_id,
            chunk_hash=f"combined_{layer_id}_{combined_hash}",
            layer_id=layer_id,
            chunk_num=len(chunks_with_metadata),  # Number of chunks combined
            chunk_mappings=chunk_mappings,
            total_tokens=total_tokens,
            is_combined=True)

        return combined_key


##### NVTX annotation #####
_NVTX_COLORS = ["green", "blue", "purple", "rapids"]


def _get_color_for_nvtx(name):
    m = hashlib.sha256()
    m.update(name.encode())
    hash_value = int(m.hexdigest(), 16)
    idx = hash_value % len(_NVTX_COLORS)
    return _NVTX_COLORS[idx]


def _lmcache_nvtx_annotate(func, domain="lmcache"):
    """Decorator for applying nvtx annotations to methods in lmcache."""
    return annotate(
        message=func.__qualname__,
        color=_get_color_for_nvtx(func.__qualname__),
        domain=domain,
    )(func)


def _lmcache_nvtx_annotate_generator(func, domain="lmcache"):
    """Decorator for applying nvtx annotations to generator functions."""

    def wrapper(*args, **kwargs):
        gen = func(*args, **kwargs)
        func_name = func.__qualname__

        try:
            # Annotate the generator setup
            with annotate(message=f"{func_name}_setup",
                          color=_get_color_for_nvtx(func_name),
                          domain=domain):
                value = next(gen)

            iteration = 0
            while True:
                try:
                    # Annotate each yield/iteration
                    with annotate(message=f"{func_name}_iter_{iteration}",
                                  color=_get_color_for_nvtx(func_name),
                                  domain=domain):
                        # Send value to generator and get next
                        sent_value = yield value
                        value = gen.send(sent_value)
                    iteration += 1
                except StopIteration as e:
                    return e.value

        except GeneratorExit:
            gen.close()
            raise
        except Exception:
            gen.throw(*sys.exc_info())
            raise

    return wrapper


class NVTXContext:
    """Context manager for fine-grained NVTX annotations"""

    def __init__(self, message: str, domain: str = "lmcache"):
        self.message = message
        self.domain = domain
        self.color = _get_color_for_nvtx(message)

    def __enter__(self):
        from nvtx import push_range
        push_range(message=self.message, color=self.color, domain=self.domain)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        from nvtx import pop_range
        pop_range(domain=self.domain)


##### Threading related #####
def thread_safe(func):
    lock = threading.Lock()

    def wrapper(*args, **kwargs):
        with lock:
            return func(*args, **kwargs)

    return wrapper
