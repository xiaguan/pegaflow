// ============================================================================
// Seal Offload: Background task for write-through to SSD
//
// Receives notifications when blocks are sealed and offloads them to SSD.
// Uses Weak<SealedBlock> to avoid holding blocks in memory if evicted.
// ============================================================================

use bytesize::ByteSize;
use crossbeam::channel::Receiver;
use tracing::{debug, info};

use crate::storage::SealNotification;

/// Spawn a tokio task that consumes seal notifications.
/// Currently just logs for verification; will be extended to write to SSD.
pub fn spawn_seal_offload_task(rx: Receiver<SealNotification>) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        info!("Seal offload task started");
        loop {
            // Use recv() in a blocking manner via spawn_blocking to avoid busy-wait
            let notification = {
                let rx = rx.clone();
                tokio::task::spawn_blocking(move || rx.recv())
                    .await
                    .expect("spawn_blocking panicked")
            };

            match notification {
                Ok((key, weak_block)) => {
                    if let Some(block) = weak_block.upgrade() {
                        debug!(
                            namespace = %key.namespace,
                            hash_len = key.hash.len(),
                            footprint = %ByteSize(block.memory_footprint()),
                            "Sealed block received for offload"
                        );
                    } else {
                        debug!(
                            namespace = %key.namespace,
                            "Sealed block already evicted before offload"
                        );
                    }
                }
                Err(_) => {
                    info!("Seal notification channel closed, stopping offload task");
                    break;
                }
            }
        }
    })
}
