pub mod allocator;
pub mod pinned_pool;
mod storage;
mod transfer;

pub use pinned_pool::PinnedAllocation;

// ============================================================================
// PegaEngine currently prioritizes vLLM's layer-first (KV-first) tensor layout.
// This means all K segments are contiguous, followed by all V segments, so the
// GPU memory picture looks like:
//
//   +---------------------------------------------------------------+
//   |  Layer0: KKKKKKKK.... | Layer0: VVVVVVVV.... | Layer1: K ...  |
//   +---------------------------------------------------------------+
//          ^ contiguous K blocks        ^ contiguous V blocks
//
// As long as vLLM keeps this layout we must respect its stride-based view and
// fall back to strided transfers; future refactors can add dedicated handling
// for other layouts without breaking this contract.
//
// To support efficient batching during "load" (CPU -> GPU), we now avoid
// storing K and V interleaved in a single contiguous block. Instead, we allocate
// all K segments for a saved batch in one contiguous CPU region, and all V segments
// in another. This Split-Storage approach ensures that when we load the batch back,
// the K source pointers are contiguous and can be merged into a single cuMemcpy,
// significantly improving PCIe bandwidth utilization compared to strided copies.
// ============================================================================

use cudarc::driver::{CudaContext, CudaEvent, CudaStream};
use std::{
    collections::{hash_map::Entry, HashMap},
    fmt,
    sync::{Arc, Mutex, RwLock},
    time::Instant,
};
use tracing::{debug, info, instrument};

use crate::storage::{Block, StorageEngine};

const DEFAULT_PINNED_POOL_BYTES: usize = 30 * 1024 * 1024 * 1024; // 10GB

#[derive(Debug)]
pub enum EngineError {
    ContextMissing(String),
    InvalidArgument(String),
    CudaInit(String),
    Storage(String),
    Poisoned(&'static str),
}

impl fmt::Display for EngineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EngineError::ContextMissing(ctx) => write!(f, "context {ctx} not registered"),
            EngineError::InvalidArgument(msg) => write!(f, "invalid argument: {msg}"),
            EngineError::CudaInit(msg) => write!(f, "failed to initialize CUDA: {msg}"),
            EngineError::Storage(msg) => write!(f, "storage error: {msg}"),
            EngineError::Poisoned(what) => write!(f, "internal lock poisoned: {what}"),
        }
    }
}

impl std::error::Error for EngineError {}

pub struct PegaEngine {
    contexts: RwLock<HashMap<String, EngineContext>>,
    /// Storage engine responsible for pinned allocations + block cache
    storage: StorageEngine,
}

#[derive(Debug, Clone)]
pub struct KVCacheRegistration {
    pub data_ptr: u64,
    pub size_bytes: usize,
    pub num_blocks: usize,
    pub bytes_per_block: usize,
    /// Distance in bytes between K and V segments when KV-first layout is used.
    /// Zero when the layout stores a single segment per block.
    pub kv_stride_bytes: usize,
    /// Number of segments per block (1 for blocks-first, 2 for KV-first).
    pub segments: usize,
}

struct EngineMetadata {
    kv_caches: HashMap<String, KVCacheRegistration>,
    layer_name_to_id: HashMap<String, usize>,
    layer_names: Vec<String>,
}

impl EngineMetadata {
    fn new() -> Self {
        Self {
            kv_caches: HashMap::new(),
            layer_name_to_id: HashMap::new(),
            layer_names: Vec::new(),
        }
    }
}

struct EngineContext {
    /// KV metadata associated with this context (mutated only during registration)
    metadata: EngineMetadata,
    /// Single stream for all transfers to ensure sequential execution (Layer0 -> Layer1...)
    stream: Arc<CudaStream>,
    /// Track per-layer completion events for async loading
    layer_events: Mutex<HashMap<String, CudaEvent>>,
    /// Hold CUDA context for the lifetime of the inference context
    _cuda_ctx: Arc<CudaContext>,
    device_id: i32,
}

impl EngineContext {
    fn new(cuda_ctx: Arc<CudaContext>, device_id: i32) -> Self {
        let stream = cuda_ctx
            .new_stream()
            .expect("Failed to create stream for engine context");
        Self {
            metadata: EngineMetadata::new(),
            stream,
            layer_events: Mutex::new(HashMap::new()),
            _cuda_ctx: cuda_ctx,
            device_id,
        }
    }

    fn register_layer(&mut self, layer_name: String, registration: KVCacheRegistration) {
        if !self.metadata.layer_name_to_id.contains_key(&layer_name) {
            let layer_id = self.metadata.layer_names.len();
            self.metadata
                .layer_name_to_id
                .insert(layer_name.clone(), layer_id);
            self.metadata.layer_names.push(layer_name.clone());
        }

        self.metadata.kv_caches.insert(layer_name, registration);
    }

    fn get_layer_id(&self, layer_name: &str) -> Option<usize> {
        self.metadata.layer_name_to_id.get(layer_name).copied()
    }

    fn num_layers(&self) -> usize {
        self.metadata.layer_names.len()
    }

    fn get_registration(&self, layer_name: &str) -> Option<KVCacheRegistration> {
        self.metadata.kv_caches.get(layer_name).cloned()
    }

    fn stream(&self) -> Arc<CudaStream> {
        self.stream.clone()
    }

    fn record_layer_event(&self, layer_name: &str, event: CudaEvent) {
        let mut guard = self.layer_events.lock().expect("layer events map poisoned");
        guard.insert(layer_name.to_string(), event);
    }

    fn take_layer_event(&self, layer_name: &str) -> Option<CudaEvent> {
        let mut guard = self.layer_events.lock().expect("layer events map poisoned");
        guard.remove(layer_name)
    }
}

impl PegaEngine {
    /// Create a new PegaEngine instance
    #[instrument(level = "info")]
    pub fn new() -> Self {
        Self::new_with_pool_size(DEFAULT_PINNED_POOL_BYTES)
    }

    /// Create a new PegaEngine instance with a custom pinned memory pool size
    pub fn new_with_pool_size(pool_size: usize) -> Self {
        let storage = StorageEngine::new(pool_size);
        PegaEngine {
            contexts: RwLock::new(HashMap::new()),
            storage,
        }
    }

    fn ensure_context(&self, context_id: &str, device_id: i32) -> Result<(), EngineError> {
        if device_id < 0 {
            return Err(EngineError::InvalidArgument(format!(
                "device_id must be >= 0 (got {device_id})"
            )));
        }

        let mut contexts = self
            .contexts
            .write()
            .map_err(|_| EngineError::Poisoned("contexts write"))?;

        match contexts.entry(context_id.to_string()) {
            Entry::Occupied(existing) => {
                if existing.get().device_id != device_id {
                    return Err(EngineError::InvalidArgument(format!(
                        "context {context_id} already bound to device {}, requested device {device_id}",
                        existing.get().device_id
                    )));
                }
            }
            Entry::Vacant(vacant) => {
                let cuda_ctx = CudaContext::new(device_id as usize)
                    .map_err(|e| EngineError::CudaInit(format!("{e:?}")))?;
                let ctx = EngineContext::new(cuda_ctx, device_id);
                vacant.insert(ctx);
            }
        }

        Ok(())
    }

    fn with_context<F, R>(&self, context_id: &str, f: F) -> Result<R, EngineError>
    where
        F: FnOnce(&EngineContext) -> Result<R, EngineError>,
    {
        let contexts = self
            .contexts
            .read()
            .map_err(|_| EngineError::Poisoned("contexts read"))?;
        let ctx = contexts
            .get(context_id)
            .ok_or_else(|| EngineError::ContextMissing(context_id.to_string()))?;
        f(ctx)
    }

    fn with_context_mut<F, R>(&self, context_id: &str, f: F) -> Result<R, EngineError>
    where
        F: FnOnce(&mut EngineContext) -> Result<R, EngineError>,
    {
        let mut contexts = self
            .contexts
            .write()
            .map_err(|_| EngineError::Poisoned("contexts write"))?;
        let ctx = contexts
            .get_mut(context_id)
            .ok_or_else(|| EngineError::ContextMissing(context_id.to_string()))?;
        f(ctx)
    }

    /// Register a KV cache region with its layout info
    #[instrument(
        level = "debug",
        skip(self),
        fields(layer = %layer_name, size_bytes, num_blocks, bytes_per_block)
    )]
    pub fn register_context_layer(
        &self,
        context_id: &str,
        device_id: i32,
        layer_name: String,
        data_ptr: u64,
        size_bytes: usize,
        num_blocks: usize,
        bytes_per_block: usize,
        kv_stride_bytes: usize,
        segments: usize,
    ) -> Result<(), EngineError> {
        if bytes_per_block == 0 || num_blocks == 0 || segments == 0 {
            return Err(EngineError::InvalidArgument(format!(
                "invalid KV cache layout for layer {layer_name}"
            )));
        }

        let registration = KVCacheRegistration {
            data_ptr,
            size_bytes,
            num_blocks,
            bytes_per_block,
            kv_stride_bytes,
            segments,
        };

        self.ensure_context(context_id, device_id)?;
        self.with_context_mut(context_id, |ctx| {
            ctx.register_layer(layer_name, registration);
            Ok(())
        })
    }

    /// Unregister all KV cache handles
    #[instrument(level = "info", skip(self))]
    pub fn unregister_context(&self, context_id: &str) -> Result<(), EngineError> {
        let removed = self
            .contexts
            .write()
            .map_err(|_| EngineError::Poisoned("contexts write"))?
            .remove(context_id);

        if removed.is_none() {
            return Err(EngineError::ContextMissing(context_id.to_string()));
        }

        Ok(())
    }

    #[instrument(
        level = "debug",
        skip(self, block_ids, block_hashes),
        fields(layer = %layer_name, blocks = %block_ids.len(), hashes = %block_hashes.len()),
    )]
    pub fn save_kv_blocks_from_ipc(
        &self,
        context_id: &str,
        layer_name: &str,
        block_ids: Vec<i32>,
        block_hashes: Vec<Vec<u8>>,
    ) -> Result<(), EngineError> {
        if block_ids.len() != block_hashes.len() {
            return Err(EngineError::InvalidArgument(format!(
                "block_ids length {} does not match block_hashes {}",
                block_ids.len(),
                block_hashes.len()
            )));
        }

        let (layer_id, registration, total_layers) = self.with_context(context_id, |ctx| {
            let layer_id = ctx.get_layer_id(layer_name).ok_or_else(|| {
                EngineError::InvalidArgument(format!("layer {layer_name} not registered"))
            })?;
            let registration = ctx.get_registration(layer_name).ok_or_else(|| {
                EngineError::InvalidArgument(format!("layer {layer_name} not registered"))
            })?;
            Ok((layer_id, registration, ctx.num_layers()))
        })?;

        // Collect blocks that need to be saved
        let mut blocks_to_save = Vec::with_capacity(block_ids.len());

        for (block_id, block_hash) in block_ids.iter().zip(block_hashes.iter()) {
            if *block_id < 0 {
                continue;
            }
            let block_idx = *block_id as usize;
            if block_idx >= registration.num_blocks {
                return Err(EngineError::InvalidArgument(format!(
                    "block {block_idx} out of range for layer {layer_name} ({} blocks registered)",
                    registration.num_blocks
                )));
            }

            // Check if this block_hash already has data for this layer
            let needs_save = !self.storage.layer_has_block(block_hash, layer_id);

            if needs_save {
                blocks_to_save.push((block_idx, block_hash.clone()));
            }
        }

        if blocks_to_save.is_empty() {
            return Ok(());
        }

        let block_size = transfer::block_size(&registration).unwrap();
        let num_blocks = blocks_to_save.len();

        // For layer-first layout with KV stride, allocate separate regions for K and V
        if registration.segments == 2 && registration.kv_stride_bytes > registration.bytes_per_block
        {
            let segment_size = registration.bytes_per_block;
            let k_total_size = segment_size * num_blocks;
            let v_total_size = segment_size * num_blocks;

            // Allocate separate regions for K and V segments
            let k_allocation = self.storage.allocate(k_total_size);
            let v_allocation = self.storage.allocate(v_total_size);
            let k_base_ptr = k_allocation.as_mut_ptr();
            let v_base_ptr = v_allocation.as_mut_ptr();

            // Calculate GPU offsets for batching
            let mut k_offsets_with_idx = Vec::with_capacity(num_blocks);
            let mut v_offsets_with_idx = Vec::with_capacity(num_blocks);

            for (i, (block_idx, _)) in blocks_to_save.iter().enumerate() {
                let k_offset = transfer::segment_offset(&registration, *block_idx, 0).unwrap();
                let v_offset = transfer::segment_offset(&registration, *block_idx, 1).unwrap();
                k_offsets_with_idx.push((k_offset, i));
                v_offsets_with_idx.push((v_offset, i));
            }

            // Sort by GPU offset to find contiguous ranges
            k_offsets_with_idx.sort_by_key(|&(offset, _)| offset);
            v_offsets_with_idx.sort_by_key(|&(offset, _)| offset);

            // Batch copy K segments
            transfer::batch_copy_segments(
                &k_offsets_with_idx,
                k_base_ptr,
                segment_size,
                &registration,
            )
            .unwrap();

            // Batch copy V segments
            transfer::batch_copy_segments(
                &v_offsets_with_idx,
                v_base_ptr,
                segment_size,
                &registration,
            )
            .unwrap();

            // Create Block objects after all copying is done
            for (i, (_, block_hash)) in blocks_to_save.into_iter().enumerate() {
                let k_ptr = unsafe { k_base_ptr.add(i * segment_size) };
                let v_ptr = unsafe { v_base_ptr.add(i * segment_size) };

                // We now keep K and V data in separate allocations during their lifetime
                // This avoids the memory overwrite bug and keeps data contiguous for better batching next time
                let block = Arc::new(Block::new_split(
                    k_ptr,
                    v_ptr,
                    block_size,
                    Arc::clone(&k_allocation),
                    Arc::clone(&v_allocation),
                ));

                self.storage
                    .insert_block(block_hash, layer_id, block, total_layers);
            }
        } else {
            // Original logic for contiguous or single-segment layouts
            let total_size = block_size * num_blocks;
            let allocation = self.storage.allocate(total_size);
            let base_ptr = allocation.as_mut_ptr();

            // Copy blocks and create Block objects
            for (i, (block_idx, block_hash)) in blocks_to_save.into_iter().enumerate() {
                let cpu_ptr = unsafe { base_ptr.add(i * block_size) };
                transfer::copy_block_gpu_to_cpu(&registration, block_idx, cpu_ptr).unwrap();

                let block = Arc::new(Block::new_contiguous(
                    cpu_ptr,
                    block_size,
                    Arc::clone(&allocation),
                ));

                self.storage
                    .insert_block(block_hash, layer_id, block, total_layers);
            }
        }
        Ok(())
    }

    /// Count how many blocks from the prefix are available in CPU storage
    ///
    /// Returns the number of contiguous blocks available from the start.
    /// Stops counting at the first unavailable block.
    /// Uses the per-block completion status so schedulers only see fully saved blocks.
    ///
    /// Args:
    ///   - block_hashes: List of block hashes to check
    ///
    /// Returns:
    ///   - usize: Number of contiguous blocks available from the prefix
    #[instrument(
        level = "info",
        skip(self, block_hashes),
        fields(requested = %block_hashes.len()),
        ret
    )]
    pub fn count_prefix_hit_blocks(
        &self,
        context_id: &str,
        block_hashes: &[Vec<u8>],
    ) -> Result<usize, EngineError> {
        let total_layers = self.with_context(context_id, |ctx| Ok(ctx.num_layers()))?;
        if total_layers == 0 {
            return Ok(0);
        }

        let mut hit_count = 0;

        for block_hash in block_hashes.iter() {
            if !self.storage.block_is_complete(block_hash) {
                break;
            }
            hit_count += 1;
        }

        debug!(
            hit_count,
            total = block_hashes.len(),
            "Counted prefix hit blocks"
        );

        Ok(hit_count)
    }

    /// Batch load KV blocks for multiple layers with shared block mapping
    ///
    /// This method optimizes loading the same blocks across multiple layers by:
    /// 1. Looking up all block_hashes in storage ONCE
    /// 2. For each layer, extracting blocks from the cached LayerBlocks
    /// 3. Performing transfers for each layer
    ///
    /// This reduces hash table lookups from O(layers × blocks) to O(blocks)
    ///
    /// Args:
    ///   - layer_names: List of layer names to load
    ///   - block_ids: GPU block IDs to load into (shared across all layers)
    ///   - block_hashes: Content hashes for each block (shared across all layers)
    ///
    /// Returns:
    ///   - Vec of (layer_name, bytes_transferred) for each successfully loaded layer
    #[instrument(
        level = "debug",
        skip(self, block_ids, block_hashes),
        fields(layers = %layer_names.len(), blocks = %block_ids.len(), hashes = %block_hashes.len(),context_id = %context_id),
    )]
    pub fn batch_load_kv_blocks_multi_layer(
        &self,
        context_id: &str,
        layer_names: &[&str],
        block_ids: &[i32],
        block_hashes: &[Vec<u8>],
    ) -> Result<Vec<(String, usize)>, EngineError> {
        let start_time = Instant::now();

        // Step 1: Lookup all block_hashes ONCE and cache the LayerBlocks
        let layer_blocks_cache = self
            .storage
            .lookup_many(block_hashes)
            .map_err(EngineError::Storage)?;

        let stream = self.with_context(context_id, |ctx| Ok(ctx.stream()))?;

        // Step 2: For each layer, extract blocks and perform transfer
        let mut results = Vec::with_capacity(layer_names.len());

        for layer_name in layer_names {
            let layer_start = Instant::now();

            let (layer_id, registration) = match self.with_context(context_id, |ctx| {
                let layer_id = ctx.get_layer_id(layer_name).ok_or_else(|| {
                    EngineError::InvalidArgument(format!("layer {layer_name} not registered"))
                })?;
                let registration = ctx.get_registration(layer_name).ok_or_else(|| {
                    EngineError::InvalidArgument(format!("layer {layer_name} not registered"))
                })?;
                Ok((layer_id, registration))
            }) {
                Ok(data) => data,
                Err(err @ EngineError::InvalidArgument(_)) => {
                    info!("Layer {} not registered, skipping ({err})", layer_name);
                    continue;
                }
                Err(err) => return Err(err),
            };

            // Collect valid blocks to load for this layer
            let mut blocks_to_load = Vec::with_capacity(block_ids.len());

            for (block_id, layer_blocks_arc) in block_ids.iter().zip(layer_blocks_cache.iter()) {
                let block_idx = *block_id as usize;

                let blocks = layer_blocks_arc.lock_blocks();
                if let Some(block) = blocks.get(layer_id).and_then(|opt| opt.as_ref()) {
                    blocks_to_load.push((block_idx, block.clone()));
                }
            }

            if blocks_to_load.is_empty() {
                info!("No blocks to load for layer {}", layer_name);
                continue;
            }

            // Perform transfer using existing logic
            let mut total_transfer = 0;

            // Optimize for layer-first layout with KV stride
            if registration.segments == 2
                && registration.kv_stride_bytes > registration.bytes_per_block
            {
                let segment_size = registration.bytes_per_block;

                // Prepare K and V segments with their GPU destinations
                let mut k_transfers = Vec::with_capacity(blocks_to_load.len());
                let mut v_transfers = Vec::with_capacity(blocks_to_load.len());

                for (block_idx, block) in &blocks_to_load {
                    let k_gpu_offset = match transfer::segment_offset(&registration, *block_idx, 0)
                    {
                        Ok(offset) => offset,
                        Err(e) => {
                            info!("Failed to get K offset for layer {}: {}", layer_name, e);
                            continue;
                        }
                    };
                    let v_gpu_offset = match transfer::segment_offset(&registration, *block_idx, 1)
                    {
                        Ok(offset) => offset,
                        Err(e) => {
                            info!("Failed to get V offset for layer {}: {}", layer_name, e);
                            continue;
                        }
                    };

                    let k_cpu_ptr = block.k_ptr() as *const u8;
                    let v_cpu_ptr = if let Some(v_ptr) = block.v_ptr() {
                        v_ptr as *const u8
                    } else {
                        // If it was stored contiguously (e.g. old format), V follows K
                        unsafe { k_cpu_ptr.add(segment_size) }
                    };

                    k_transfers.push((k_gpu_offset, k_cpu_ptr));
                    v_transfers.push((v_gpu_offset, v_cpu_ptr));
                }

                // Sort by GPU offset for batching
                k_transfers.sort_by_key(|&(offset, _)| offset);
                v_transfers.sort_by_key(|&(offset, _)| offset);

                // Batch copy K segments
                if let Err(e) = transfer::batch_copy_segments_to_gpu(
                    &k_transfers,
                    segment_size,
                    &registration,
                    &stream,
                ) {
                    info!("Failed to copy K segments for layer {}: {}", layer_name, e);
                    continue;
                }

                // Batch copy V segments
                if let Err(e) = transfer::batch_copy_segments_to_gpu(
                    &v_transfers,
                    segment_size,
                    &registration,
                    &stream,
                ) {
                    info!("Failed to copy V segments for layer {}: {}", layer_name, e);
                    continue;
                }

                total_transfer = blocks_to_load.len() * segment_size * 2;
            } else {
                // Original logic for contiguous or single-segment layouts
                for (block_idx, block) in blocks_to_load {
                    match transfer::copy_block_cpu_to_gpu(
                        &registration,
                        block_idx,
                        block.k_ptr() as *const u8,
                        &stream,
                    ) {
                        Ok(_) => {
                            total_transfer += block.size();
                        }
                        Err(e) => {
                            info!(
                                "Failed to copy block {} for layer {}: {}",
                                block_idx, layer_name, e
                            );
                        }
                    }
                }
            }

            // Record event for this layer
            match stream.record_event(None) {
                Ok(event) => {
                    self.with_context(context_id, |ctx| {
                        ctx.record_layer_event(layer_name, event);
                        Ok(())
                    })?;
                }
                Err(e) => {
                    info!(
                        "Failed to record CUDA event for layer {}: {:?}",
                        layer_name, e
                    );
                }
            }

            let layer_elapsed = (Instant::now() - layer_start).as_secs_f64();
            let bandwidth = if layer_elapsed > 0.0 {
                total_transfer as f64 / layer_elapsed
            } else {
                0.0
            };
            debug!(
                layer = layer_name,
                total_transfer,
                elapsed_us = (Instant::now() - layer_start).as_micros(),
                bandwidth_gbps = bandwidth / 1e9,
                "Completed layer transfer"
            );

            results.push((layer_name.to_string(), total_transfer));
        }

        let total_elapsed = (Instant::now() - start_time).as_secs_f64();
        info!(
            "batch_load_kv_blocks_multi_layer: completed {} layers in {:.3}s",
            results.len(),
            total_elapsed
        );

        Ok(results)
    }

    /// Block until the most recent async transfer for a layer finishes.
    pub fn wait_for_layer_transfer(
        &self,
        context_id: &str,
        layer_name: &str,
    ) -> Result<(), EngineError> {
        let event = self.with_context(context_id, |ctx| Ok(ctx.take_layer_event(layer_name)))?;

        if let Some(event) = event {
            event.synchronize().map_err(|e| {
                EngineError::Storage(format!("failed to sync layer {layer_name}: {e:?}"))
            })?;
        }
        Ok(())
    }
}

impl Default for PegaEngine {
    fn default() -> Self {
        Self::new()
    }
}

// Safety: PegaEngine can be safely sent between threads
// - PinnedMemoryPool owns the CUDA allocation
// - CUDA context is thread-safe (Arc<CudaContext>)
unsafe impl Send for PegaEngine {}
unsafe impl Sync for PegaEngine {}
