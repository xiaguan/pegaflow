//! Shared-memory synchronization state for layer-by-layer async KV cache loading.
//!
//! This module provides `LayerSyncState`, a cross-process synchronization primitive
//! that allows the connector worker to wait for layer transfers without ZMQ round-trips.
//!
//! # Architecture
//!
//! ```text
//! Connector Worker                    Engine Server (async thread)
//! ─────────────────                   ────────────────────────────
//! create LayerSyncState ──shm_name──▶ attach LayerSyncState
//!                                     
//! wait_layer(0) ◄─────────────────── mark_layer_done(0)
//! wait_layer(1) ◄─────────────────── mark_layer_done(1)
//! ...
//! ```

use shared_memory::{Shmem, ShmemConf, ShmemError};
use std::sync::atomic::{AtomicU8, Ordering};
use tracing::info;
use uuid::Uuid;

/// Flag values for layer synchronization state
const FLAG_NOT_STARTED: u8 = 0;
const FLAG_COMPLETED: u8 = 1;

/// Shared-memory based synchronization state for async layer loading.
///
/// The connector creates this, passes the `shm_name` to the server,
/// and the server attaches to the same shared memory region.
pub struct LayerSyncState {
    shmem: Shmem,
    num_layers: usize,
}

// SAFETY: LayerSyncState uses atomic operations for cross-process synchronization.
// The shared memory region is accessed via atomic operations only.
unsafe impl Send for LayerSyncState {}
unsafe impl Sync for LayerSyncState {}

impl LayerSyncState {
    /// Create a new LayerSyncState with the given number of layers.
    ///
    /// This creates a new shared memory region with a unique name.
    /// The name can be retrieved via `shm_name()` and passed to the server.
    pub fn new(num_layers: usize) -> Result<Self, ShmemError> {
        let shm_name = format!("pega_sync_{}", Uuid::new_v4().as_simple());
        let size = num_layers;

        let shmem = ShmemConf::new().os_id(&shm_name).size(size).create()?;

        // Initialize all flags to NOT_STARTED
        let ptr = shmem.as_ptr() as *mut AtomicU8;
        for i in 0..num_layers {
            unsafe {
                (*ptr.add(i)).store(FLAG_NOT_STARTED, Ordering::Relaxed);
            }
        }

        Ok(Self { shmem, num_layers })
    }

    /// Attach to an existing shared memory region by name.
    ///
    /// Used by the server to attach to the region created by the connector.
    pub fn attach(shm_name: &str, num_layers: usize) -> Result<Self, ShmemError> {
        let shmem = ShmemConf::new().os_id(shm_name).open()?;

        if shmem.len() < num_layers {
            return Err(ShmemError::MapSizeZero);
        }

        Ok(Self { shmem, num_layers })
    }

    /// Get the shared memory identifier.
    ///
    /// Pass this to the server so it can attach to the same region.
    pub fn shm_name(&self) -> &str {
        self.shmem.get_os_id()
    }

    /// Get the number of layers.
    pub fn num_layers(&self) -> usize {
        self.num_layers
    }

    /// Reset all flags to NOT_STARTED.
    ///
    /// Call this before starting a new load batch.
    pub fn reset(&self) {
        let ptr = self.shmem.as_ptr() as *mut AtomicU8;
        for i in 0..self.num_layers {
            unsafe {
                (*ptr.add(i)).store(FLAG_NOT_STARTED, Ordering::Release);
            }
        }
    }

    /// Mark a layer as completed.
    ///
    /// Called by the server's async thread after the CUDA transfer is done.
    pub fn mark_layer_done(&self, layer_id: usize) {
        if layer_id >= self.num_layers {
            return;
        }
        info!("layer {} done", layer_id);
        let ptr = self.shmem.as_ptr() as *mut AtomicU8;
        unsafe {
            (*ptr.add(layer_id)).store(FLAG_COMPLETED, Ordering::Release);
        }
    }

    /// Wait until a layer is completed.
    ///
    /// Called by the connector worker before executing attention for that layer.
    /// Uses spin-wait for simplicity.
    pub fn wait_layer(&self, layer_id: usize) {
        if layer_id >= self.num_layers {
            return;
        }
        let ptr = self.shmem.as_ptr() as *const AtomicU8;
        loop {
            let flag = unsafe { (*ptr.add(layer_id)).load(Ordering::Acquire) };
            if flag == FLAG_COMPLETED {
                return;
            }
            std::hint::spin_loop();
        }
    }

    /// Check if a layer is completed (non-blocking).
    pub fn is_layer_done(&self, layer_id: usize) -> bool {
        if layer_id >= self.num_layers {
            return true;
        }
        let ptr = self.shmem.as_ptr() as *const AtomicU8;
        let flag = unsafe { (*ptr.add(layer_id)).load(Ordering::Acquire) };
        flag == FLAG_COMPLETED
    }
}
