use std::{
    num::NonZeroU64,
    ptr::NonNull,
    sync::{Arc, Mutex},
};

use bytesize::ByteSize;
use tracing::{error, info};

use crate::allocator::{Allocation, ScaledOffsetAllocator};
use crate::metrics::core_metrics;
use crate::pinned_mem::PinnedMemory;

/// RAII guard for a pinned memory allocation.
/// Automatically frees the allocation when dropped.
pub struct PinnedAllocation {
    allocation: Allocation,
    ptr: NonNull<u8>,
    pool: Arc<PinnedMemoryPool>,
}

// SAFETY: PinnedAllocation points to CUDA pinned memory which is fixed in physical
// memory and safe to access from any thread. The NonNull<u8> is just a pointer to
// this pinned memory region.
unsafe impl Send for PinnedAllocation {}
unsafe impl Sync for PinnedAllocation {}

impl PinnedAllocation {
    /// Get a const pointer to the allocated memory
    pub fn as_ptr(&self) -> *const u8 {
        self.ptr.as_ptr()
    }

    /// Get a mutable pointer to the allocated memory
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.ptr.as_ptr()
    }

    /// Get the size of the allocation in bytes
    pub fn size(&self) -> NonZeroU64 {
        self.allocation.size_bytes
    }
}

impl Drop for PinnedAllocation {
    fn drop(&mut self) {
        // Automatically free the allocation when the guard is dropped
        self.pool.free_internal(&self.allocation);
    }
}

/// Manages a CUDA pinned memory pool and a byte-addressable allocator.
pub struct PinnedMemoryPool {
    /// Backing pinned memory (handles mmap + cudaHostRegister)
    backing: PinnedMemory,
    allocator: Mutex<ScaledOffsetAllocator>,
}

impl PinnedMemoryPool {
    /// Upper bound for simultaneous allocations in the pinned pool.
    const MAX_ALLOCS: u32 = 4_000_000;

    /// Allocate a new pinned memory pool of `pool_size` bytes.
    ///
    /// If `use_hugepages` is true, uses huge pages (requires system config).
    pub fn new(pool_size: usize, use_hugepages: bool) -> Self {
        if pool_size == 0 {
            panic!("Pinned memory pool size must be greater than zero");
        }

        let backing = if use_hugepages {
            info!("Allocating pinned memory pool with huge pages");
            PinnedMemory::allocate_hugepages(pool_size)
                .expect("Failed to allocate pinned memory pool with huge pages")
        } else {
            info!("Allocating pinned memory pool with regular pages");
            PinnedMemory::allocate(pool_size).expect("Failed to allocate pinned memory pool")
        };

        let actual_size = backing.size();
        let allocator =
            ScaledOffsetAllocator::new_with_max_allocs(actual_size as u64, Self::MAX_ALLOCS)
                .expect("Failed to create memory allocator");

        Self {
            backing,
            allocator: Mutex::new(allocator),
        }
    }

    /// Allocate pinned memory from the pool. Returns None when the allocation cannot be satisfied.
    /// Returns a RAII guard that automatically frees the allocation when dropped.
    pub fn allocate(self: &Arc<Self>, size: NonZeroU64) -> Option<PinnedAllocation> {
        let mut allocator = self.allocator.lock().unwrap();

        let allocation = match allocator.allocate(size.get()) {
            Ok(Some(allocation)) => allocation,
            Ok(None) => {
                return None; // Pool exhausted, caller can retry after eviction
            }
            Err(err) => {
                error!(
                    requested_bytes = size.get(),
                    "Pinned memory allocation error: {} (requested {})",
                    err,
                    ByteSize(size.get())
                );
                return None;
            }
        };

        let metrics = core_metrics();

        let offset: usize = allocation
            .offset_bytes
            .try_into()
            .expect("allocation offset exceeds usize");
        let ptr = unsafe { self.backing.as_ptr().add(offset) };
        let ptr = NonNull::new(ptr as *mut u8).expect("PinnedMemoryPool returned null pointer");

        let size_bytes = allocation.size_bytes.get();
        if let Ok(size_i64) = i64::try_from(size_bytes) {
            metrics.pool_used_bytes.add(size_i64, &[]);
        }

        Some(PinnedAllocation {
            allocation,
            ptr,
            pool: Arc::clone(self),
        })
    }

    /// Internal method to free a pinned memory allocation.
    /// This is called automatically by PinnedAllocation's Drop implementation.
    /// Users should not call this directly - use PinnedAllocation RAII instead.
    pub(crate) fn free_internal(&self, allocation: &Allocation) {
        let mut allocator = self.allocator.lock().unwrap();
        allocator.free(allocation);

        let metrics = core_metrics();
        let size_bytes = allocation.size_bytes.get();
        if let Ok(size_i64) = i64::try_from(size_bytes) {
            metrics.pool_used_bytes.add(-size_i64, &[]);
        }
    }

    /// Get (used_bytes, total_bytes) for the pool.
    pub fn usage(&self) -> (u64, u64) {
        let allocator = self.allocator.lock().unwrap();
        let report = allocator.storage_report();
        let total = allocator.total_bytes();
        let used = total - report.total_free_bytes;
        (used, total)
    }
}

// PinnedMemory handles cleanup in its Drop impl, no manual Drop needed here.

// SAFETY: The pool owns a PinnedMemory backing that remains valid for the lifetime
// of the pool. All mutations of the allocator state are guarded by the internal
// `Mutex`. CUDA pinned host memory can be accessed from any host thread.
unsafe impl Send for PinnedMemoryPool {}
unsafe impl Sync for PinnedMemoryPool {}
