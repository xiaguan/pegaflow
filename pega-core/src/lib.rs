pub mod allocator;
pub mod pinned_pool;
mod transfer;

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

use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
    time::Instant,
};

use allocator::Allocation;
use cudarc::driver::{CudaContext, CudaEvent, CudaStream};
use moka::sync::Cache;
use tracing::{debug, info, instrument};

use crate::pinned_pool::PinnedMemoryPool;

const DEFAULT_PINNED_POOL_BYTES: usize = 20 * 1024 * 1024 * 1024; // 10GB
const CACHE_USAGE_RATIO: f64 = 0.90;

type BlockKey = Vec<u8>;

pub struct PegaEngine {
    /// Store registered KV cache pointers (new IPC wrapper): layer_name -> registration
    kv_caches: HashMap<String, KVCacheRegistration>,
    /// Store saved KV blocks (layer_name, block_hash) -> block data
    kv_storage: Cache<BlockKey, Arc<Block>>,
    /// Pinned memory pool for zero-copy GPU transfers
    pinned_pool: Arc<PinnedMemoryPool>,
    /// Single stream for all transfers to ensure sequential execution (Layer0 -> Layer1...)
    stream: Arc<CudaStream>,
    /// Track per-layer completion events for async loading
    layer_events: Mutex<HashMap<String, CudaEvent>>,
    _cuda_ctx: Arc<CudaContext>,
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

pub struct Block {
    /// Pointer to K segment (or combined data if contiguous)
    k_ptr: *mut u8,
    /// Pointer to V segment (if stored separately)
    v_ptr: Option<*mut u8>,
    size: usize,
    /// Allocation handle for K memory
    k_allocation: Arc<Allocation>,
    /// Allocation handle for V memory (if separate from K)
    v_allocation: Option<Arc<Allocation>>,
    pool: Arc<PinnedMemoryPool>,
}

impl Block {
    fn new_contiguous(
        ptr: *mut u8,
        size: usize,
        allocation: Arc<Allocation>,
        pool: Arc<PinnedMemoryPool>,
    ) -> Self {
        Self {
            k_ptr: ptr,
            v_ptr: None,
            size,
            k_allocation: allocation,
            v_allocation: None,
            pool,
        }
    }

    fn new_split(
        k_ptr: *mut u8,
        v_ptr: *mut u8,
        size: usize,
        k_allocation: Arc<Allocation>,
        v_allocation: Arc<Allocation>,
        pool: Arc<PinnedMemoryPool>,
    ) -> Self {
        Self {
            k_ptr,
            v_ptr: Some(v_ptr),
            size,
            k_allocation,
            v_allocation: Some(v_allocation),
            pool,
        }
    }

    fn k_ptr(&self) -> *mut u8 {
        self.k_ptr
    }

    fn v_ptr(&self) -> Option<*mut u8> {
        self.v_ptr
    }

    fn size(&self) -> usize {
        self.size
    }

    fn weight(&self) -> u32 {
        u32::try_from(self.size)
            .expect("KV block larger than u32::MAX bytes is not supported for caching")
    }
}

impl Drop for Block {
    fn drop(&mut self) {
        // Free K allocation if last reference
        if Arc::strong_count(&self.k_allocation) == 1 {
            self.pool.free(*self.k_allocation);
        }
        // Free V allocation if exists and is last reference
        if let Some(ref v_allocation) = self.v_allocation {
            if Arc::strong_count(v_allocation) == 1 {
                self.pool.free(**v_allocation);
            }
        }
    }
}

// Safety: Block wraps a read-only pinned allocation; freeing is handled through its Drop.
unsafe impl Send for Block {}
unsafe impl Sync for Block {}

impl PegaEngine {
    /// Create a new PegaEngine instance
    #[instrument(level = "info")]
    pub fn new() -> Self {
        Self::new_with_pool_size(DEFAULT_PINNED_POOL_BYTES)
    }

    /// Create a new PegaEngine instance with a custom pinned memory pool size
    pub fn new_with_pool_size(pool_size: usize) -> Self {
        let pinned_pool = Arc::new(PinnedMemoryPool::new(pool_size));
        let cache_capacity = ((pool_size as f64) * CACHE_USAGE_RATIO).floor().max(1.0) as u64;
        let kv_storage = Cache::builder()
            .max_capacity(cache_capacity)
            .weigher(|_, block: &Arc<Block>| block.weight())
            .build();

        // TODO: hard code device 0 for now
        let cuda_ctx = cudarc::driver::CudaContext::new(0).unwrap();
        let stream = cuda_ctx.new_stream().expect("Failed to create stream");

        PegaEngine {
            kv_caches: HashMap::new(),
            kv_storage,
            pinned_pool,
            stream: stream,
            layer_events: Mutex::new(HashMap::new()),
            _cuda_ctx: cuda_ctx,
        }
    }

    /// Register a KV cache region with its layout info
    #[instrument(
        level = "debug",
        skip(self),
        fields(layer = %layer_name, size_bytes, num_blocks, bytes_per_block)
    )]
    pub fn register_kv_cache(
        &mut self,
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

        self.kv_caches.insert(layer_name, registration);
    }

    /// Unregister all KV cache handles
    #[instrument(level = "info", skip(self))]
    pub fn unregister_all_kv_caches(&mut self) {
        self.kv_caches.clear();
    }

    /// Get the number of registered KV caches
    #[instrument(level = "debug", skip(self), ret)]
    pub fn num_registered_kv_caches(&self) -> usize {
        self.kv_caches.len()
    }

    /// Allocate pinned memory from the pool. Panics when the allocation cannot be satisfied.
    fn allocate_pinned(&self, size: usize) -> (Allocation, *mut u8) {
        self.pinned_pool.allocate(size)
    }

    /// Get pinned memory usage statistics
    pub fn get_pinned_memory_usage(&self) -> (usize, usize) {
        self.pinned_pool.usage()
    }

    fn encode_key_to_buffer(layer_name: &str, block_hash: &[u8], buffer: &mut Vec<u8>) {
        buffer.clear();
        buffer.reserve(layer_name.len() + 1 + block_hash.len());
        buffer.extend_from_slice(layer_name.as_bytes());
        buffer.push(0);
        buffer.extend_from_slice(block_hash);
    }

    fn record_layer_event(&self, layer_name: &str, event: CudaEvent) {
        let mut guard = self.layer_events.lock().expect("layer events map poisoned");

        guard.insert(layer_name.to_string(), event);
    }

    #[instrument(
        level = "debug",
        skip(self, block_ids, block_hashes),
        fields(layer = %layer_name, blocks = %block_ids.len(), hashes = %block_hashes.len()),
    )]
    pub fn save_kv_blocks_from_ipc(
        &mut self,
        layer_name: String,
        block_ids: Vec<i32>,
        block_hashes: Vec<Vec<u8>>,
    ) {
        assert_eq!(
            block_ids.len(),
            block_hashes.len(),
            "block_ids and block_hashes must have equal length"
        );

        let registration = self
            .kv_caches
            .get(&layer_name)
            .unwrap_or_else(|| panic!("Layer {} not registered", layer_name));

        // Collect blocks that need to be saved
        let mut blocks_to_save = Vec::with_capacity(block_ids.len());
        let mut key_buffer = Vec::new();

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

            Self::encode_key_to_buffer(&layer_name, block_hash, &mut key_buffer);
            if !self.kv_storage.contains_key(key_buffer.as_slice()) {
                blocks_to_save.push((block_idx, key_buffer.clone()));
            }
        }

        if blocks_to_save.is_empty() {
            return;
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
            let (k_allocation, k_base_ptr) = self.allocate_pinned(k_total_size);
            let (v_allocation, v_base_ptr) = self.allocate_pinned(v_total_size);
            let k_shared_allocation = Arc::new(k_allocation);
            let v_shared_allocation = Arc::new(v_allocation);

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
            for (i, (_, key)) in blocks_to_save.into_iter().enumerate() {
                let k_ptr = unsafe { k_base_ptr.add(i * segment_size) };
                let v_ptr = unsafe { v_base_ptr.add(i * segment_size) };

                // We now keep K and V data in separate allocations during their lifetime
                // This avoids the memory overwrite bug and keeps data contiguous for better batching next time
                let block = Arc::new(Block::new_split(
                    k_ptr,
                    v_ptr,
                    block_size,
                    Arc::clone(&k_shared_allocation),
                    Arc::clone(&v_shared_allocation),
                    Arc::clone(&self.pinned_pool),
                ));
                self.kv_storage.insert(key, block);
            }
        } else {
            // Original logic for contiguous or single-segment layouts
            let total_size = block_size * num_blocks;
            let (allocation, base_ptr) = self.allocate_pinned(total_size);
            let shared_allocation = Arc::new(allocation);

            // Copy blocks and create Block objects
            for (i, (block_idx, key)) in blocks_to_save.into_iter().enumerate() {
                let cpu_ptr = unsafe { base_ptr.add(i * block_size) };
                transfer::copy_block_gpu_to_cpu(&registration, block_idx, cpu_ptr).unwrap();

                let block = Arc::new(Block::new_contiguous(
                    cpu_ptr,
                    block_size,
                    Arc::clone(&shared_allocation),
                    Arc::clone(&self.pinned_pool),
                ));
                self.kv_storage.insert(key, block);
            }
        }
    }

    /// Get storage statistics
    /// Returns (num_blocks, total_bytes)
    #[instrument(level = "info", skip(self), ret)]
    pub fn get_storage_stats(&self) -> (usize, usize) {
        let num_blocks = usize::try_from(self.kv_storage.entry_count()).unwrap_or(usize::MAX);
        let total_bytes = usize::try_from(self.kv_storage.weighted_size()).unwrap_or(usize::MAX);
        (num_blocks, total_bytes)
    }

    /// Check which KV blocks are available in CPU storage
    ///
    /// Args:
    ///   - layer_name: Name of the layer
    ///   - block_hashes: List of block hashes to check
    ///
    /// Returns:
    ///   - Vec<bool>: For each hash, true if available in storage
    #[instrument(
        level = "debug",
        skip(self, block_hashes),
        fields(layer = %layer_name, requested = %block_hashes.len()),
        ret
    )]
    pub fn check_kv_blocks_availability(
        &self,
        layer_name: String,
        block_hashes: Vec<Vec<u8>>,
    ) -> Vec<bool> {
        let mut availability = Vec::with_capacity(block_hashes.len());
        let mut key_buffer = Vec::new();

        for (idx, block_hash) in block_hashes.iter().enumerate() {
            Self::encode_key_to_buffer(&layer_name, block_hash, &mut key_buffer);
            let available = self.kv_storage.contains_key(key_buffer.as_slice());
            availability.push(available);

            let hash_preview: Vec<u8> = block_hash.iter().copied().take(8).collect();
            debug!(
                block_index = idx,
                available,
                hash_prefix = ?hash_preview,
                "Checked KV block availability"
            );
        }

        let num_available = availability.iter().filter(|&&x| x).count();
        debug!(
            num_available,
            total = block_hashes.len(),
            "Completed KV block availability check"
        );

        availability
    }

    /// Load KV blocks from CPU memory to GPU via IPC handle
    ///
    /// Args:
    ///   - layer_name: Name of the layer
    ///   - block_ids: GPU block IDs to load into
    ///   - block_hashes: Content hashes for each block
    #[instrument(
        level = "debug",
        skip(self, block_ids, block_hashes),
        fields(layer = %layer_name, blocks = %block_ids.len(), hashes = %block_hashes.len()),
        err
    )]
    pub fn load_kv_blocks_to_ipc(
        &self,
        layer_name: &str,
        block_ids: &[i32],
        block_hashes: &[Vec<u8>],
    ) -> Result<usize, String> {
        let start_time = Instant::now();
        let registration = self
            .kv_caches
            .get(layer_name)
            .ok_or_else(|| format!("Layer {} not registered", layer_name))?;

        // Collect valid blocks to load
        let mut blocks_to_load = Vec::with_capacity(block_ids.len());
        let mut key_buffer = Vec::new();

        for (block_id, block_hash) in block_ids.iter().zip(block_hashes.iter()) {
            let block_idx = *block_id as usize;

            Self::encode_key_to_buffer(layer_name, block_hash, &mut key_buffer);
            let Some(block) = self.kv_storage.get(key_buffer.as_slice()) else {
                return Err(format!("Missing KV block for layer {}", layer_name));
            };

            blocks_to_load.push((block_idx, block));
        }

        let end_time = Instant::now();
        info!(
            "load_kv_blocks_to_ipc: lookup time: {} us",
            (end_time - start_time).as_micros()
        );

        let mut total_transfer = 0;
        let stream = self.stream.clone();

        // Optimize for layer-first layout with KV stride
        if registration.segments == 2 && registration.kv_stride_bytes > registration.bytes_per_block
        {
            let segment_size = registration.bytes_per_block;

            // Prepare K and V segments with their GPU destinations
            let mut k_transfers = Vec::with_capacity(blocks_to_load.len());
            let mut v_transfers = Vec::with_capacity(blocks_to_load.len());

            for (block_idx, block) in &blocks_to_load {
                let k_gpu_offset = transfer::segment_offset(&registration, *block_idx, 0)?;
                let v_gpu_offset = transfer::segment_offset(&registration, *block_idx, 1)?;

                // K segment is at k_ptr, V segment is at v_ptr
                // In the new split layout, they are in separate memory regions provided by Block::new_split
                // In the legacy or different layout, check logic. But for this branch (segments == 2 && stride > block),
                // we assume it's the split layout we just optimized.
                let k_cpu_ptr = block.k_ptr() as *const u8;

                // Fallback for old blocks if any, or correct new blocks
                // If v_ptr is None, it means it was stored as contiguous block (maybe old version data or different path block?)
                // But if we are in this branch, we expect split handling.
                // However, let's support "contiguous block loaded into split GPU layout" just in case?
                // No, for now we assume data saved with new logic has v_ptr.
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
            transfer::batch_copy_segments_to_gpu(
                &k_transfers,
                segment_size,
                &registration,
                &stream,
            )?;

            // Batch copy V segments
            transfer::batch_copy_segments_to_gpu(
                &v_transfers,
                segment_size,
                &registration,
                &stream,
            )?;

            total_transfer = blocks_to_load.len() * segment_size * 2;
        } else {
            // Original logic for contiguous or single-segment layouts
            for (block_idx, block) in blocks_to_load {
                transfer::copy_block_cpu_to_gpu(
                    &registration,
                    block_idx,
                    block.k_ptr() as *const u8,
                    &stream,
                )?;
                total_transfer += block.size();
            }
        }

        let event = stream
            .record_event(None)
            .map_err(|e| format!("Failed to record CUDA event for {layer_name}: {e:?}"))?;
        self.record_layer_event(layer_name, event);

        let end_time = Instant::now();
        // print cost
        let elapsed = (end_time - start_time).as_secs_f64();
        let bandwidth = if elapsed > 0.0 {
            total_transfer as f64 / elapsed
        } else {
            0.0
        };
        debug!(
            total_transfer,
            elapsed_us = (end_time - start_time).as_micros(),
            bandwidth_gbps = bandwidth / 1e9,
            "Completed load_kv_blocks_to_ipc"
        );
        Ok(total_transfer)
    }

    /// Block until the most recent async transfer for a layer finishes.
    pub fn wait_for_layer_transfer(&self, layer_name: &str) -> Result<(), String> {
        let event = {
            let mut guard = self.layer_events.lock().expect("layer events map poisoned");
            guard.remove(layer_name)
        };

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
