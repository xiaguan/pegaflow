"""
Shared types and helpers for the PegaFlow vLLM connector.
"""

import enum
import hashlib
import os
import uuid
from dataclasses import dataclass

from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata

from pegaflow.logging_utils import get_connector_logger
from pegaflow.pegaflow import EngineRpcClient

logger = get_connector_logger()

# Engine server endpoint (gRPC URL)
ENGINE_ENDPOINT = os.environ.get("PEGAFLOW_ENGINE_ENDPOINT",
                                 "http://127.0.0.1:50055")


@dataclass(frozen=True)
class ConnectorContext:
    """Shared configuration for scheduler/worker connectors."""
    instance_id: str
    namespace: str
    block_size: int
    num_layers: int
    tp_size: int
    tp_rank: int | None
    device_id: int | None
    engine_client: EngineRpcClient


class RequestPhase(enum.Enum):
    """Lifecycle phase of a request in the KV connector."""
    LOOKUP = "lookup"  # Waiting for lookup result from external storage
    LOADING = "loading"  # Need to load KV from external storage
    ACTIVE = "active"  # Actively generating (may be saving concurrently)
    DRAINING = "draining"  # Generation done, waiting for async save to complete
    DONE = "done"  # Fully completed


@dataclass(frozen=True)
class LoadIntent:
    """Intent for a KV load operation."""
    block_ids: tuple[int, ...]
    block_hashes: tuple[bytes, ...]
    num_tokens: int


@dataclass(frozen=True)
class SaveIntent:
    """Intent for a KV save operation."""
    block_ids: tuple[int, ...]
    block_hashes: tuple[bytes, ...]


class RequestTracker:
    """
    Tracks the KV cache state for a single request.
    """

    __slots__ = (
        'request_id',
        '_block_hashes',
        '_block_size',
        # Lookup state
        '_hit_blocks',
        '_computed_blocks',
        '_lookup_done',
        # Allocation state
        '_allocated_blocks',
        '_external_tokens',
        # Progress tracking
        '_scheduled_tokens',
        '_stored_blocks',
        # Save tracking
        '_total_layers',
        '_saved_layers',
        # Flags
        '_load_consumed',
        '_finished',
    )

    def __init__(
        self,
        request_id: str,
        block_hashes: list[bytes],
        block_size: int,
        num_layers: int,
    ):
        self.request_id = request_id
        self._block_hashes = tuple(block_hashes)
        self._block_size = block_size

        # Lookup state
        self._hit_blocks: int = 0
        self._computed_blocks: int = 0
        self._lookup_done: bool = False

        # Allocation state
        self._allocated_blocks: list[int] = []
        self._external_tokens: int = 0

        # Progress tracking
        self._scheduled_tokens: int = 0
        self._stored_blocks: int = 0

        # Save tracking
        self._total_layers = num_layers
        self._saved_layers: int = 0

        # Flags
        self._load_consumed: bool = False
        self._finished: bool = False

    # ========== Properties ==========
    @property
    def phase(self) -> RequestPhase:
        """
        Current lifecycle phase (derived).

        Matches the pre-refactor behavior to avoid regressions in scheduler
        expectations.
        """
        if not self._lookup_done:
            return RequestPhase.LOOKUP
        if self._needs_load and not self._load_consumed:
            return RequestPhase.LOADING
        if not self._finished:
            return RequestPhase.ACTIVE
        if self._saved_layers < self._total_layers:
            return RequestPhase.DRAINING
        return RequestPhase.DONE

    @property
    def num_blocks(self) -> int:
        return len(self._block_hashes)

    @property
    def _needs_load(self) -> bool:
        """Return True if we should load KV from external storage."""
        return self._external_tokens > 0 and self._hit_blocks > self._computed_blocks

    # ========== Events ==========
    def on_lookup(self, hit_blocks: int, computed_blocks: int) -> None:
        self._hit_blocks = hit_blocks
        self._computed_blocks = computed_blocks
        self._lookup_done = True

    def on_alloc(self, block_ids: list[int], num_external_tokens: int) -> None:
        # Preserve previously allocated block IDs; vLLM may allocate in chunks.
        self._allocated_blocks.extend(block_ids)
        if num_external_tokens > 0:
            self._external_tokens = num_external_tokens

    def on_scheduled(self, num_tokens: int) -> None:
        # Accumulate tokens across scheduler steps to mirror old behavior.
        self._scheduled_tokens += num_tokens

    def on_layer_saved(self) -> None:
        self._saved_layers += 1

    def on_finished(self) -> None:
        self._finished = True

    # ========== Intent consumption ==========
    def consume_load_intent(self) -> LoadIntent | None:
        if self._load_consumed or not self._needs_load:
            return None

        num_blocks = min(
            self._hit_blocks,
            len(self._allocated_blocks),
            len(self._block_hashes),
        )
        load_blocks = num_blocks - self._computed_blocks
        if load_blocks <= 0:
            return None

        self._load_consumed = True
        start = self._computed_blocks
        end = start + load_blocks
        return LoadIntent(
            block_ids=tuple(self._allocated_blocks[start:end]),
            block_hashes=self._block_hashes[start:end],
            num_tokens=load_blocks * self._block_size,
        )

    def consume_save_intent(self) -> SaveIntent | None:
        saveable = min(
            len(self._block_hashes),
            len(self._allocated_blocks),
            self._scheduled_tokens // self._block_size,
        )
        new_blocks = saveable - self._stored_blocks
        if new_blocks <= 0:
            return None

        start = self._stored_blocks
        end = start + new_blocks
        self._stored_blocks = end
        return SaveIntent(
            block_ids=tuple(self._allocated_blocks[start:end]),
            block_hashes=self._block_hashes[start:end],
        )

    # ========== Queries ==========
    def should_hold_blocks(self) -> bool:
        return (self._finished and self._stored_blocks > 0 and
                self._saved_layers < self._total_layers)

    def is_done(self) -> bool:
        return self.phase == RequestPhase.DONE

    def __repr__(self) -> str:
        return (
            f"RequestTracker(id={self.request_id}, phase={self.phase.value}, "
            f"hit={self._hit_blocks}, computed={self._computed_blocks}, "
            f"allocated={len(self._allocated_blocks)}, stored={self._stored_blocks}, "
            f"saved_layers={self._saved_layers}/{self._total_layers})")


class PegaConnectorMetadata(KVConnectorMetadata):
    """Metadata passed from scheduler to worker for KV cache operations."""

    def __init__(
        self,
        load_intents: dict[str, LoadIntent] | None = None,
        save_intents: dict[str, SaveIntent] | None = None,
        finished_requests: set[str] | None = None,
    ):
        super().__init__()
        # Maps request_id -> intent
        self.load_intents: dict[str, LoadIntent] = load_intents or {}
        self.save_intents: dict[str, SaveIntent] = save_intents or {}
        # Requests that have finished generation and are waiting for save to complete
        self.finished_requests: set[str] = finished_requests or set()

    def __repr__(self) -> str:
        return (f"PegaConnectorMetadata(loads={len(self.load_intents)}, "
                f"saves={len(self.save_intents)}, finished={len(self.finished_requests)})")


def resolve_instance_id(vllm_config, dp_rank_suffix: bool = True) -> str:
    """Resolve or generate connector instance_id with optional DP rank suffix."""
    instance_id = vllm_config.kv_transfer_config.engine_id
    if instance_id:
        logger.info(
            "[PegaKVConnector] Using kv_transfer_config.engine_id: %s",
            instance_id)
        return instance_id

    instance_id = vllm_config.instance_id or os.environ.get(
        "PEGAFLOW_INSTANCE_ID", "")
    if not instance_id:
        instance_id = uuid.uuid4().hex
        logger.info(
            "[PegaKVConnector] No instance_id from vLLM; generated fallback %s",
            instance_id)

    if dp_rank_suffix:
        parallel_config = vllm_config.parallel_config
        if parallel_config.data_parallel_size > 1:
            local_dp_rank = parallel_config.data_parallel_rank_local
            if local_dp_rank is not None:
                instance_id = f"{instance_id}_dp{local_dp_rank}"
                logger.info(
                    "[PegaKVConnector] Appended DP rank to instance_id: %s (dp_size=%d, local_dp_rank=%d)",
                    instance_id,
                    parallel_config.data_parallel_size,
                    local_dp_rank,
                )

    return instance_id


def derive_namespace(vllm_config, tp_size: int) -> str:
    """
    Derive namespace for storage isolation.
    """
    model_config = vllm_config.model_config
    cache_config = vllm_config.cache_config

    factors = {
        "model": model_config.model,
        "dtype": str(model_config.dtype),
        "tp_size": tp_size,
        "num_kv_heads": model_config.get_total_num_kv_heads(),
        "head_size": model_config.get_head_size(),
        "num_hidden_layers": model_config.get_total_num_hidden_layers(),
        "cache_dtype": str(cache_config.cache_dtype),
    }

    factor_str = str(sorted(factors.items()))
    hash_suffix = hashlib.sha256(factor_str.encode()).hexdigest()[:8]
    return f"{hash_suffix}"


__all__ = [
    "ConnectorContext",
    "ENGINE_ENDPOINT",
    "LoadIntent",
    "PegaConnectorMetadata",
    "RequestPhase",
    "RequestTracker",
    "SaveIntent",
    "derive_namespace",
    "logger",
    "resolve_instance_id",
]
