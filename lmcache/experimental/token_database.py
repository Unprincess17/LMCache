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

import abc
import array
import hashlib
from typing import Iterable, List, Optional, Tuple, Union

import torch
from transformers import AutoTokenizer

from lmcache.config import LMCacheEngineMetadata
from lmcache.experimental.config import LMCacheEngineConfig
from lmcache.utils import CacheEngineKey, LayerCacheEngineKey, NVTXContext, _lmcache_nvtx_annotate


class TokenDatabase(metaclass=abc.ABCMeta):
    """TokenDatabase is used to convert input tokens into list of
    cache engine keys. There are multiple ways to implement this:

    - ChunkedTokenDatabase: It processes tokens into chunks and convert 
    each chunk into a cache engine key using prefix hash.

    - SegmentTokenDatabase: It processes tokens into segments based on
    special separators and convert each segment into a cache engine key.
    """

    @abc.abstractmethod
    def process_tokens(
        self,
        tokens: Union[torch.Tensor, List[int]],
        mask: Optional[torch.Tensor] = None,
        make_key: bool = True,
    ) -> Iterable[Tuple[int, int, Union[CacheEngineKey, str]]]:
        """Process the tokens and return the corresponding cache engine keys.

        :param Union[torch.Tensor, List[int]] tokens: The tokens to process.

        :param Optional[torch.Tensor] mask: The mask for the tokens. Should 
            have the same length as tokens. And the mask should ALWAYS be like
            FFFFFTTTTTTT, where True means the tokens needs to be matched, 
            and the Falses will ALWAYS be at the PREFIX of the tensor.

        :returns: A iterable of tuples with three elements. The first element
            is the start index of the tokens for the key. The second element
            is the end index of the tokens for the key. The third element is
            the cache engine key (or hash) for the tokens.
        """

        raise NotImplementedError


class ChunkedTokenDatabase(TokenDatabase):

    def __init__(self,
                 config: Optional[LMCacheEngineConfig] = None,
                 metadata: Optional[LMCacheEngineMetadata] = None):
        if config is not None:
            self.chunk_size = config.chunk_size
        self.metadata = metadata

    def _make_key_by_hash(self,
                          chunk_hash: str,
                          layer_id: Optional[int] = None):
        assert self.metadata is not None
        return CacheEngineKey(self.metadata.fmt, self.metadata.model_name,
                              self.metadata.world_size,
                              self.metadata.worker_id, chunk_hash)

    def _get_init_hash(self) -> str:
        return ""

    def _hash(
        self,
        tokens: Union[torch.Tensor, List[int]],
        prefix_hash: str,
    ) -> str:
        # TODO: change it to a more efficient hash function
        if isinstance(tokens, torch.Tensor):
            # Assume tokens are already on CPU - no .cpu() call needed
            tokens_bytes = tokens.to(torch.uint32).numpy().tobytes()
        elif isinstance(tokens, list):
            tokens_bytes = array.array('I', tokens).tobytes()
        return hashlib.sha256(prefix_hash.encode("ascii") +
                              tokens_bytes).hexdigest()

    def _chunk_tokens(
        self,
        tokens: Union[torch.Tensor, List[int]],
    ) -> Iterable[Union[torch.Tensor, List[int]]]:
        """
        Chunk the tokens into chunks of size self.chunk_size.

        :param tokens: the input tokens, with shape [seq_len]
            device: the target device after chunking

        :return: a generator of chunks of tokens, each with 
                shape [chunk_size]
        """
        for i in range(0, len(tokens), self.chunk_size):
            yield tokens[i:i + self.chunk_size]

    def _prefix_hash(
        self,
        token_chunks: Iterable[Union[torch.Tensor, List[int]]],
    ) -> Iterable[str]:
        prefix_hash = self._get_init_hash()
        for token_chunk in token_chunks:
            prefix_hash = self._hash(token_chunk, prefix_hash)
            yield prefix_hash

    def process_tokens(
        self,
        tokens: Union[torch.Tensor, List[int]],
        mask: Optional[torch.Tensor] = None,
        make_key: bool = True,
    ) -> Iterable[Tuple[int, int, Union[CacheEngineKey, str]]]:
        """Process the tokens and return the corresponding cache engine keys.

        :param Union[torch.Tensor, List[int]] tokens: The tokens to process.

        :param Optional[torch.Tensor] mask: The mask for the tokens. Should 
            have the same length as tokens. And the mask should ALWAYS be like
            FFFFFTTTTTTT, where True means the tokens needs to be matched, 
            and the Falses will ALWAYS be at the PREFIX of the tensor.

        :param bool make_key: Whether to make the cache engine key or not.
            If False, the hash value will be returned instead.
        
        :returns: A iterable of tuples with three elements. The first element
            is the start index of the tokens for the key. The second element
            is the end index of the tokens for the key. The third element is
            the cache engine key (or hash) for the tokens.

        :raises: ValueError if the number of Falses in the mask is not a 
            multiple of the chunk size.
        """
        if mask is not None:
            num_falses = mask.numel() - mask.long().sum().item()
        else:
            num_falses = 0

        if num_falses % self.chunk_size != 0:
            raise ValueError("The number of Falses in the mask is not a "
                             "multiple of the chunk size.")
        total_len = len(tokens)

        if isinstance(tokens, torch.Tensor) and tokens.is_cuda:
            tokens = tokens.cpu()

        token_chunks = self._chunk_tokens(tokens)
        prefix_hashes = self._prefix_hash(token_chunks)

        start_idx = 0
        for chunk_id, hash_val in enumerate(prefix_hashes):
            start_idx = chunk_id * self.chunk_size
            end_idx = min(start_idx + self.chunk_size, total_len)
            if start_idx < num_falses:
                continue
            else:
                if make_key:
                    yield start_idx, end_idx, self._make_key_by_hash(hash_val)
                else:
                    yield start_idx, end_idx, hash_val


class SegmentTokenDatabase(TokenDatabase):
    """
    Currently, we still use special separators to identify chunks.
    In the future, we might need to implement a fast substring match.
    """

    def __init__(self, config: LMCacheEngineConfig,
                 metadata: LMCacheEngineMetadata):
        self.tokenizer = AutoTokenizer.from_pretrained(metadata.model_name)

        # TODO (Jiayi): figure out how to decide when
        # to use `1:` (whether there's a special starting token
        # in the beginning)
        self.sep_tokens = self.tokenizer.encode(config.blend_special_str)[1:]
        self.sep_tokens = torch.tensor(self.sep_tokens, device="cpu")
        self.sep_len = len(self.sep_tokens)
        self.metadata = metadata

    def _make_key_by_hash(self, chunk_hash: str):
        return CacheEngineKey(self.metadata.fmt, self.metadata.model_name,
                              self.metadata.world_size,
                              self.metadata.worker_id, chunk_hash)

    def _hash(
        self,
        tokens: Union[torch.Tensor, List[int]],
    ) -> str:
        # TODO: change it to a more efficient hash function
        if isinstance(tokens, torch.Tensor):
            tokens_bytes = tokens.cpu().to(torch.uint32).numpy().tobytes()
        elif isinstance(tokens, list):
            tokens_bytes = array.array('I', tokens).tobytes()
        return hashlib.sha256(tokens_bytes).hexdigest()

    def _fast_split_by_subtensor(
            self, tokens: torch.Tensor) -> Iterable[torch.Tensor]:
        """Match the `sep_tokens` with sliding windows"""

        if self.sep_len == 0 or len(tokens) < self.sep_len:
            yield tokens

        # Unfold into sliding windows
        # shape: (num_tokens-sep_len+1, sep_len)
        windows = tokens.unfold(0, self.sep_len, 1)

        # Compare each window with sep_tokens
        matches = (windows == self.sep_tokens).all(dim=1).nonzero(
            as_tuple=True)[0].tolist()

        # Split based on matches
        start = 0
        for idx in matches:
            yield tokens[start:idx]
            start = idx + self.sep_len

    def process_tokens(
        self,
        tokens: Union[torch.Tensor, List[int]],
        mask: Optional[torch.Tensor] = None,
        make_key: bool = True,
    ) -> Iterable[Tuple[int, int, Union[CacheEngineKey, str]]]:
        """Process the tokens and return the corresponding cache engine keys.

        :param Union[torch.Tensor, List[int]] tokens: The tokens to process.

        :param Optional[torch.Tensor] mask: The mask for the tokens. Should 
            have the same length as tokens. And the mask should ALWAYS be like
            FFFFFTTTTTTT, where True means the tokens needs to be matched, 
            and the Falses will ALWAYS be at the PREFIX of the tensor.

        :returns: A iterable of tuples with three elements. The first element
            is the start index of the tokens for the key. The second element
            is the end index of the tokens for the key. The third element is
            the cache engine key for the tokens.

        """

        assert isinstance(tokens, torch.Tensor), \
            "Only tokens in tensor format are supported for now."
        if mask is not None:
            num_falses = mask.numel() - mask.long().sum().item()
        else:
            num_falses = 0
        assert num_falses < len(tokens), \
            ("The number of Falses in the mask shouldn't "
            "be less than the length of tokens.")
        token_chunks = self._fast_split_by_subtensor(tokens)
        start_idx = 0
        for idx, token_chunk in enumerate(token_chunks):
            token_chunk_len = len(token_chunk)
            end_idx = start_idx + token_chunk_len
            if idx > 0:
                start_idx += self.sep_len
                end_idx += self.sep_len
            if start_idx >= num_falses:
                if make_key:
                    yield start_idx, end_idx, self._make_key_by_hash(
                        self._hash(token_chunk))
                else:
                    yield start_idx, end_idx, self._hash(token_chunk)
            start_idx = end_idx


class LayerFirstTokenDataBase(ChunkedTokenDatabase):
    """
    This token database is used to convert tokens into layer first cache engine keys.
    It processes tokens in a layer-first manner, where each layer's tokens are processed
    separately and returned as a list of (start_idx, end_idx, key) tuples.
    
    Improved hash calculation design:
    1. First compute chunk hashes: chunk_hash_i = hash(prefix_hash, chunk_i)
    2. Then derive layer hashes: layer_hash = hash(chunk_hash, layer_id)
    
    This creates a hierarchical structure that's compatible with prefix caching in LLM serving systems.
    """

    def __init__(self, config: LMCacheEngineConfig,
                 metadata: LMCacheEngineMetadata):
        super().__init__(config, metadata)
        self.num_layers = metadata.kv_shape[0]

    def _make_key_by_hash(self,
                          chunk_layer_hash: str,
                          layer_id: Optional[int] = None):
        """Override to create layer-specific keys"""
        assert layer_id is not None, "LayerFirstTokenDatabase requires layer_id to be specified"
        base_key = super()._make_key_by_hash(chunk_layer_hash)
        return LayerCacheEngineKey(base_key.fmt, base_key.model_name,
                                   base_key.world_size, base_key.worker_id,
                                   base_key.chunk_hash, layer_id)

    def _layer_hash(self, chunk_hash: str, layer_id: int) -> str:
        """
        Compute layer-specific hash from chunk hash.
        
        Args:
            chunk_hash: Base chunk hash
            layer_id: Layer ID to include in hash
            
        Returns:
            Layer-specific hash string
        """
        hasher = hashlib.sha256()
        hasher.update(chunk_hash.encode("ascii"))
        hasher.update(str(layer_id).encode("ascii"))
        return hasher.hexdigest()

    def _prefix_chunk_hash(
        self,
        token_chunks: Iterable[Union[torch.Tensor, List[int]]],
    ) -> Iterable[str]:
        """
        Compute progressive chunk hashes without layer information.
        This creates the base chunk hash chain that can be reused across layers.
        
        Args:
            token_chunks: Iterable of token chunks
            
        Yields:
            Progressive chunk hash for each chunk
        """
        return super()._prefix_hash(token_chunks)

    def process_tokens(
        self,
        tokens: Union[torch.Tensor, List[int]],
        mask: Optional[torch.Tensor] = None,
        make_key: bool = True
    ) -> Iterable[Tuple[int, int, Union[CacheEngineKey, str]]]:
        """
        Process the tokens in a layer-first manner using improved hash calculation.
        
        Args:
            tokens: Input tokens to process
            mask: Optional mask for the tokens
            make_key: Whether to create CacheEngineKey or just return hash string
            
        Returns:
            An iterable of (start_idx, end_idx, key) tuples for each layer
        """
        if isinstance(tokens, list):
            tokens = torch.tensor(tokens)

        # Batch CPU transfer - do this once at the beginning
        if isinstance(tokens, torch.Tensor) and tokens.is_cuda:
            tokens = tokens.cpu()

        # Calculate num_falses once outside the loop
        num_falses = 0
        if mask is not None:
            num_falses = mask.numel() - mask.long().sum().item()

        with NVTXContext("token_sub_processing_setup"):
            # Process tokens in chunks using parent class's chunking
            token_chunks = self._chunk_tokens(tokens)

        # First, compute all chunk hashes (can be reused across layers)
        base_chunk_hashes = self._prefix_chunk_hash(token_chunks)

        start_idx = 0
        # Iterate over chunks, computing layer-specific hashes efficiently
        for chunk_id, base_chunk_hash in enumerate(base_chunk_hashes):
            with NVTXContext("per chunk token key"):
                start_idx = chunk_id * self.chunk_size
                end_idx = min(start_idx + self.chunk_size, len(tokens))

                # Skip chunks that are masked out
                if mask is not None and start_idx < num_falses:
                    continue

                with NVTXContext("per layer token key"):
                    # For each layer, compute the layer-specific hash from chunk hash
                    for layer_id in range(self.num_layers):
                        layer_hash = self._layer_hash(base_chunk_hash, layer_id)

                        if make_key:
                            yield start_idx, end_idx, self._make_key_by_hash(
                                layer_hash, layer_id)
                        else:
                            yield start_idx, end_idx, layer_hash
