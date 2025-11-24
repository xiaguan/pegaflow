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
    collections::HashMap,
    sync::{Arc, Mutex, RwLock},
    time::Instant,
};
use tracing::{debug, info, instrument};

use crate::storage::{Block, StorageEngine};

const DEFAULT_PINNED_POOL_BYTES: usize = 20 * 1024 * 1024 * 1024; // 10GB

pub struct PegaEngine {
    /// Stateful context describing the active inference graph
    context: EngineContext,
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

struct EngineContext {
    /// Store registered KV cache pointers (new IPC wrapper): layer_name -> registration
    kv_caches: RwLock<HashMap<String, KVCacheRegistration>>,
    /// Map layer names to layer IDs for efficient indexing
    layer_name_to_id: RwLock<HashMap<String, usize>>,
    /// Ordered list of layer names (layer_id is the index into this vec)
    layer_names: RwLock<Vec<String>>,
    /// Single stream for all transfers to ensure sequential execution (Layer0 -> Layer1...)
    stream: Arc<CudaStream>,
    /// Track per-layer completion events for async loading
    layer_events: Mutex<HashMap<String, CudaEvent>>,
    /// Hold CUDA context for the lifetime of the inference context
    _cuda_ctx: Arc<CudaContext>,
}

impl EngineContext {
    fn new(cuda_ctx: Arc<CudaContext>) -> Self {
        let stream = cuda_ctx
            .new_stream()
            .expect("Failed to create stream for engine context");
        Self {
            kv_caches: RwLock::new(HashMap::new()),
            layer_name_to_id: RwLock::new(HashMap::new()),
            layer_names: RwLock::new(Vec::new()),
            stream,
            layer_events: Mutex::new(HashMap::new()),
            _cuda_ctx: cuda_ctx,
        }
    }

    fn register_layer(&self, layer_name: String, registration: KVCacheRegistration) {
        let mut layer_name_to_id = self
            .layer_name_to_id
            .write()
            .expect("layer_name_to_id lock poisoned");
        let mut layer_names = self.layer_names.write().expect("layer_names lock poisoned");
        let mut kv_caches = self.kv_caches.write().expect("kv_caches lock poisoned");

        if !layer_name_to_id.contains_key(&layer_name) {
            let layer_id = layer_names.len();
            layer_name_to_id.insert(layer_name.clone(), layer_id);
            layer_names.push(layer_name.clone());
        }

        kv_caches.insert(layer_name, registration);
    }

    fn clear(&self) {
        let mut kv_caches = self.kv_caches.write().expect("kv_caches lock poisoned");
        let mut layer_name_to_id = self
            .layer_name_to_id
            .write()
            .expect("layer_name_to_id lock poisoned");
        let mut layer_names = self.layer_names.write().expect("layer_names lock poisoned");

        kv_caches.clear();
        layer_name_to_id.clear();
        layer_names.clear();
        self.layer_events
            .lock()
            .expect("layer events map poisoned")
            .clear();
    }

    fn get_layer_id(&self, layer_name: &str) -> Option<usize> {
        let map = self
            .layer_name_to_id
            .read()
            .expect("layer_name_to_id lock poisoned");
        map.get(layer_name).copied()
    }

    fn num_layers(&self) -> usize {
        let names = self.layer_names.read().expect("layer_names lock poisoned");
        names.len()
    }

    fn get_registration(&self, layer_name: &str) -> Option<KVCacheRegistration> {
        let kv_caches = self.kv_caches.read().expect("kv_caches lock poisoned");
        kv_caches.get(layer_name).cloned()
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
        // TODO: hard code device 0 for now
        let cuda_ctx = cudarc::driver::CudaContext::new(0).unwrap();
        let context = EngineContext::new(cuda_ctx);

        PegaEngine { context, storage }
    }

    /// Register a KV cache region with its layout info
    #[instrument(
        level = "debug",
        skip(self),
        fields(layer = %layer_name, size_bytes, num_blocks, bytes_per_block)
    )]
    pub fn register_context_layer(
        &self,
        layer_name: String,
        data_ptr: u64,
        size_bytes: usize,
        num_blocks: usize,
        bytes_per_block: usize,
        kv_stride_bytes: usize,
        segments: usize,
    ) {
        if bytes_per_block == 0 || num_blocks == 0 || segments == 0 {
            panic!("Invalid KV cache layout for layer {}", layer_name);
        }

        let registration = KVCacheRegistration {
            data_ptr,
            size_bytes,
            num_blocks,
            bytes_per_block,
            kv_stride_bytes,
            segments,
        };

        self.context.register_layer(layer_name, registration);
    }

    /// Unregister all KV cache handles
    #[instrument(level = "info", skip(self))]
    pub fn unregister_context(&self) {
        self.context.clear();
    }

    /// Get the layer_id for a given layer_name
    fn get_layer_id(&self, layer_name: &str) -> Option<usize> {
        self.context.get_layer_id(layer_name)
    }

    /// Get the total number of layers
    fn num_layers(&self) -> usize {
        self.context.num_layers()
    }

    fn record_layer_event(&self, layer_name: &str, event: CudaEvent) {
        self.context.record_layer_event(layer_name, event);
    }

    #[instrument(
        level = "debug",
        skip(self, block_ids, block_hashes),
        fields(layer = %layer_name, blocks = %block_ids.len(), hashes = %block_hashes.len()),
    )]
    pub fn save_kv_blocks_from_ipc(
        &self,
        layer_name: String,
        block_ids: Vec<i32>,
        block_hashes: Vec<Vec<u8>>,
    ) {
        assert_eq!(
            block_ids.len(),
            block_hashes.len(),
            "block_ids and block_hashes must have equal length"
        );

        let layer_id = self
            .get_layer_id(&layer_name)
            .unwrap_or_else(|| panic!("Layer {} not registered", layer_name));

        // Acquire the registration metadata for this layer
        let registration = self
            .context
            .get_registration(&layer_name)
            .unwrap_or_else(|| panic!("Layer {} not registered", layer_name));

        // Collect blocks that need to be saved
        let mut blocks_to_save = Vec::with_capacity(block_ids.len());

        for (block_id, block_hash) in block_ids.iter().zip(block_hashes.iter()) {
            if *block_id < 0 {
                continue;
            }
            let block_idx = *block_id as usize;
            assert!(
                block_idx < registration.num_blocks,
                "Block {} out of range for layer {} ({} blocks registered)",
                block_idx,
                layer_name,
                registration.num_blocks
            );

            // Check if this block_hash already has data for this layer
            let needs_save = !self.storage.layer_has_block(block_hash, layer_id);

            if needs_save {
                blocks_to_save.push((block_idx, block_hash.clone()));
            }
        }

        if blocks_to_save.is_empty() {
            return;
        }

        let block_size = transfer::block_size(&registration).unwrap();
        self.storage.initialize_layer_count(self.num_layers());
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

                self.storage.insert_block(block_hash, layer_id, block);
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

                self.storage.insert_block(block_hash, layer_id, block);
            }
        }
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
    pub fn count_prefix_hit_blocks(&self, block_hashes: &[Vec<u8>]) -> usize {
        if self.num_layers() == 0 {
            return 0;
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

        hit_count
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
        fields(layers = %layer_names.len(), blocks = %block_ids.len(), hashes = %block_hashes.len()),
    )]
    pub fn batch_load_kv_blocks_multi_layer(
        &self,
        layer_names: &[&str],
        block_ids: &[i32],
        block_hashes: &[Vec<u8>],
    ) -> Result<Vec<(String, usize)>, String> {
        let start_time = Instant::now();

        // Step 1: Lookup all block_hashes ONCE and cache the LayerBlocks
        let layer_blocks_cache = self
            .storage
            .lookup_many(block_hashes)
            .map_err(|e| format!("Failed to lookup KV blocks: {e}"))?;

        // Step 2: For each layer, extract blocks and perform transfer
        let mut results = Vec::with_capacity(layer_names.len());

        for layer_name in layer_names {
            let layer_start = Instant::now();

            let layer_id = match self.get_layer_id(layer_name) {
                Some(id) => id,
                None => {
                    info!("Layer {} not registered, skipping", layer_name);
                    continue;
                }
            };

            // Acquire registration metadata for this layer
            let registration = match self.context.get_registration(layer_name) {
                Some(reg) => reg,
                None => {
                    info!("Layer {} not registered, skipping", layer_name);
                    continue;
                }
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
            let stream = self.context.stream();

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
                    self.record_layer_event(layer_name, event);
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
    pub fn wait_for_layer_transfer(&self, layer_name: &str) -> Result<(), String> {
        let event = { self.context.take_layer_event(layer_name) };

        if let Some(event) = event {
            event
                .synchronize()
                .map_err(|e| format!("Failed to sync layer {layer_name}: {e:?}"))?;
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
