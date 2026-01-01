// ============================================================================
// DFS Offload: Background write-through to distributed filesystem
//
// When blocks are sealed, this module offloads them to DFS for persistence.
// Uses Weak<SealedBlock> to avoid holding blocks in memory if evicted early.
//
// Data flow:
//   SealNotification (batch) → group by namespace
//                            → write_packed_file() per namespace
//                            → update Redis metadata
//
// File packing:
//   Multiple blocks are packed into one file: {namespace}/{uuid}.bin
//   Each block's offset is recorded in BlockMeta.offset
//
// Redis schema:
//   pega:block:{ns}:{hash}  → BlockMeta JSON (file, offset, slots)
//   pega:file:{ns}:{uuid}   → FileMeta JSON (blocks list, size)
//   pega:files              → ZSET of ns:uuid scored by timestamp (LRU)
//   pega:usage              → total bytes counter for quota
// ============================================================================

use bytesize::ByteSize;
use dashmap::DashMap;
use redis::aio::MultiplexedConnection;
use redis::{AsyncCommands, Script};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;
use tokio::fs;
use tokio::sync::mpsc::UnboundedReceiver;
use tokio::time::Instant;
use tracing::{debug, error, info, warn};
use uuid::Uuid;

use crate::storage::{BlockKey, LayerBlock, SealNotification, SealedBlock, StorageEngine};

// ============================================================================
// Configuration & Types
// ============================================================================

/// Configuration for DFS offload
#[derive(Debug, Clone)]
pub struct DfsOffloadConfig {
    /// Root directory on DFS (e.g., "/mnt/dfs/pega")
    pub dfs_root: PathBuf,
    /// Hard limit in bytes (triggers eviction when exceeded)
    pub quota_bytes: u64,
    /// Quota scan interval in milliseconds
    pub scan_interval_ms: u64,
    /// Max blocks per packed file
    pub max_pack_size: usize,
    /// Batch size for eviction queries (how many files to fetch per ZRANGE)
    pub evict_batch_size: usize,
}

impl Default for DfsOffloadConfig {
    fn default() -> Self {
        Self {
            dfs_root: PathBuf::from("/mnt/dfs/pega"),
            quota_bytes: 100 * 1024 * 1024 * 1024, // 100GB
            scan_interval_ms: 100,
            max_pack_size: 64,
            evict_batch_size: 100,
        }
    }
}

/// Per-slot metadata (one slot = one layer's KV cache)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlotMeta {
    /// K and V stored separately (split) or together (contiguous)
    pub is_split: bool,
    /// Total size in bytes (K + V combined)
    pub size: u64,
}

/// Block metadata stored in Redis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockMeta {
    /// File name ({uuid}.bin)
    pub file: String,
    /// Offset within file
    pub offset: u64,
    /// Per-slot layout info
    pub slots: Vec<SlotMeta>,
    /// Total slots (layers * tp_size) for rebuilding SealedBlock
    pub total_slots: usize,
}

impl BlockMeta {
    pub fn total_size(&self) -> u64 {
        self.slots.iter().map(|s| s.size).sum()
    }
}

/// File metadata in Redis (tracks blocks referencing this file)
#[derive(Debug, Clone, Serialize, Deserialize)]
struct FileMeta {
    blocks: Vec<String>, // block hashes (hex)
    size: u64,
}

/// A block ready for writing (after upgrading Weak)
struct ReadyBlock {
    key: BlockKey,
    block: Arc<SealedBlock>,
}

// ============================================================================
// Redis Keys
// ============================================================================

fn redis_block_key(namespace: &str, hash: &[u8]) -> String {
    format!("pega:block:{}:{}", namespace, hex::encode(hash))
}

fn redis_file_key(namespace: &str, uuid: &Uuid) -> String {
    format!("pega:file:{}:{}", namespace, uuid)
}

fn redis_files_zset_key() -> &'static str {
    "pega:files"
}

fn redis_usage_key() -> &'static str {
    "pega:usage"
}

// ============================================================================
// Lua Scripts
// ============================================================================

/// Atomic batch write: SET multiple blocks, SET file meta, ZADD files, INCRBY usage
/// KEYS: [file_key, files_zset, usage_key]
/// ARGV: [file_meta_json, zset_member, file_size, timestamp, block_key1, block_meta1, hash1, ...]
const LUA_WRITE_BATCH: &str = r#"
redis.call('SET', KEYS[1], ARGV[1])
redis.call('ZADD', KEYS[2], ARGV[4], ARGV[2])
redis.call('INCRBY', KEYS[3], tonumber(ARGV[3]))
local i = 5
while i <= #ARGV do
    redis.call('SET', ARGV[i], ARGV[i+1])
    i = i + 3
end
return 1
"#;

/// Atomic file delete: DEL blocks, DEL file, ZREM files, DECRBY usage
const LUA_DELETE_FILE: &str = r#"
for i = 3, #ARGV do
    redis.call('DEL', ARGV[i])
end
redis.call('DEL', KEYS[1])
redis.call('ZREM', KEYS[2], ARGV[1])
redis.call('DECRBY', KEYS[3], tonumber(ARGV[2]))
return #ARGV - 2
"#;

// ============================================================================
// Public API
// ============================================================================

/// Spawn DFS offload and quota scan tasks.
///
/// Returns JoinHandles for (offload_task, quota_task).
pub async fn spawn_dfs_offload_task(
    rx: UnboundedReceiver<SealNotification>,
    redis_url: &str,
    config: DfsOffloadConfig,
) -> Result<(tokio::task::JoinHandle<()>, tokio::task::JoinHandle<()>), redis::RedisError> {
    let client = redis::Client::open(redis_url)?;
    let conn = client.get_multiplexed_async_connection().await?;
    let quota_conn = client.get_multiplexed_async_connection().await?;

    info!(
        redis_url = %redis_url,
        dfs_root = %config.dfs_root.display(),
        quota = %ByteSize(config.quota_bytes),
        max_pack_size = config.max_pack_size,
        "DFS offload started"
    );

    let quota_config = config.clone();
    let offload_handle = tokio::spawn(offload_loop(rx, conn, config));
    let quota_handle = tokio::spawn(quota_loop(quota_conn, quota_config));

    Ok((offload_handle, quota_handle))
}

/// Read block metadata from Redis.
pub async fn get_block_meta(
    conn: &mut MultiplexedConnection,
    namespace: &str,
    hash: &[u8],
) -> Result<Option<BlockMeta>, redis::RedisError> {
    let key = redis_block_key(namespace, hash);
    let json: Option<String> = conn.get(&key).await?;

    Ok(json.and_then(|j| serde_json::from_str(&j).ok()))
}

/// Batch lookup block metadata from Redis.
pub async fn batch_get_block_meta(
    conn: &mut MultiplexedConnection,
    namespace: &str,
    hashes: &[Vec<u8>],
) -> Result<Vec<Option<BlockMeta>>, redis::RedisError> {
    if hashes.is_empty() {
        return Ok(vec![]);
    }

    let keys: Vec<String> = hashes
        .iter()
        .map(|h| redis_block_key(namespace, h))
        .collect();

    let values: Vec<Option<String>> = conn.mget(&keys).await?;

    Ok(values
        .into_iter()
        .map(|opt| opt.and_then(|j| serde_json::from_str(&j).ok()))
        .collect())
}

// Legacy alias
pub use spawn_dfs_offload_task as spawn_seal_offload_task;

// ============================================================================
// Offload Loop (batch receive + pack by namespace)
// ============================================================================

async fn offload_loop(
    mut rx: UnboundedReceiver<SealNotification>,
    mut conn: MultiplexedConnection,
    config: DfsOffloadConfig,
) {
    let write_script = Script::new(LUA_WRITE_BATCH);

    loop {
        // Wait for first notification
        let Some(first) = rx.recv().await else {
            break;
        };

        // Collect batch (non-blocking)
        let mut notifications = vec![first];
        while notifications.len() < config.max_pack_size {
            match rx.try_recv() {
                Ok(n) => notifications.push(n),
                Err(_) => break,
            }
        }

        // Upgrade Weak refs, filter evicted blocks
        let ready: Vec<ReadyBlock> = notifications
            .into_iter()
            .filter_map(|(key, weak)| weak.upgrade().map(|block| ReadyBlock { key, block }))
            .collect();

        if ready.is_empty() {
            continue;
        }

        // Group by namespace
        let mut by_namespace: HashMap<String, Vec<ReadyBlock>> = HashMap::new();
        for rb in ready {
            by_namespace
                .entry(rb.key.namespace.clone())
                .or_default()
                .push(rb);
        }

        // Write each namespace's blocks to a packed file
        for (namespace, blocks) in by_namespace {
            if let Err(e) = write_and_register(
                &mut conn,
                &write_script,
                &namespace,
                blocks,
                &config.dfs_root,
            )
            .await
            {
                error!(namespace = %namespace, "Failed to write packed file: {}", e);
            }
        }
    }

    info!("Offload loop stopped");
}

/// Write blocks to a packed file and register in Redis.
async fn write_and_register(
    conn: &mut MultiplexedConnection,
    script: &Script,
    namespace: &str,
    blocks: Vec<ReadyBlock>,
    dfs_root: &Path,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let block_count = blocks.len();
    let total_size: u64 = blocks.iter().map(|b| b.block.memory_footprint()).sum();

    debug!(
        namespace = %namespace,
        blocks = block_count,
        size = %ByteSize(total_size),
        "Writing packed file"
    );

    // Write to DFS
    let (file_uuid, block_metas) = write_packed_file(namespace, &blocks, dfs_root).await?;

    // Prepare Redis update
    let file_name = format!("{}.bin", file_uuid);
    let file_key = redis_file_key(namespace, &file_uuid);
    let zset_member = format!("{}:{}", namespace, file_uuid);

    let now_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64;

    // Build FileMeta
    let file_meta = FileMeta {
        blocks: blocks.iter().map(|b| hex::encode(&b.key.hash)).collect(),
        size: total_size,
    };
    let file_meta_json = serde_json::to_string(&file_meta)?;

    // Build script args
    let mut invocation = script.key(&file_key);
    invocation
        .key(redis_files_zset_key())
        .key(redis_usage_key())
        .arg(&file_meta_json)
        .arg(&zset_member)
        .arg(total_size)
        .arg(now_ms);

    // Add each block's metadata
    for (rb, meta) in blocks.iter().zip(block_metas.iter()) {
        let block_key = redis_block_key(namespace, &rb.key.hash);
        let meta_json = serde_json::to_string(meta)?;
        invocation
            .arg(&block_key)
            .arg(&meta_json)
            .arg(hex::encode(&rb.key.hash));
    }

    // Execute
    if let Err(e) = invocation.invoke_async::<i32>(conn).await {
        // Cleanup orphaned file
        let file_path = dfs_root.join(namespace).join(&file_name);
        let _ = fs::remove_file(&file_path).await;
        return Err(e.into());
    }

    info!(
        namespace = %namespace,
        file = %file_name,
        blocks = block_count,
        size = %ByteSize(total_size),
        "Packed file written"
    );

    Ok(())
}

// ============================================================================
// Packed File Write (zero-copy from pinned memory)
// ============================================================================

/// Write multiple blocks to a single packed file.
/// Returns (file_uuid, Vec<BlockMeta>) with each block's offset.
#[tracing::instrument(level = "info", skip(blocks), fields(blocks = blocks.len()))]
async fn write_packed_file(
    namespace: &str,
    blocks: &[ReadyBlock],
    dfs_root: &Path,
) -> std::io::Result<(Uuid, Vec<BlockMeta>)> {
    let dir = dfs_root.join(namespace);
    fs::create_dir_all(&dir).await?;

    let uuid = Uuid::new_v4();
    let file_name = format!("{}.bin", uuid);
    let file_path = dir.join(&file_name);

    // Get total_slots from first block (all blocks in a namespace have same topology)
    let total_slots = blocks.first().map(|b| b.block.slots().len()).unwrap_or(0);

    // Collect metadata for each block (computed before write)
    let mut block_metas: Vec<BlockMeta> = Vec::with_capacity(blocks.len());
    let mut current_offset = 0u64;

    for rb in blocks {
        let slots = rb.block.slots();
        let slot_metas: Vec<SlotMeta> = slots
            .iter()
            .map(|s| SlotMeta {
                is_split: s.v_ptr().is_some(),
                size: s.size() as u64,
            })
            .collect();
        let block_size: u64 = slot_metas.iter().map(|s| s.size).sum();

        block_metas.push(BlockMeta {
            file: file_name.clone(),
            offset: current_offset,
            slots: slot_metas,
            total_slots,
        });

        current_offset += block_size;
    }

    // Clone Arcs for the blocking task
    let block_arcs: Vec<Arc<SealedBlock>> = blocks.iter().map(|rb| Arc::clone(&rb.block)).collect();

    // Write directly to target file — no rename needed since Redis index
    // is the entry point; no reader can discover this UUID before Redis write
    tokio::task::spawn_blocking(move || {
        use std::io::Write;

        let mut file = std::fs::File::create(&file_path)?;

        for block in &block_arcs {
            for slot in block.slots().iter() {
                let size = slot.size();
                if let Some(v_ptr) = slot.v_ptr() {
                    let half = size / 2;
                    file.write_all(unsafe { std::slice::from_raw_parts(slot.k_ptr(), half) })?;
                    file.write_all(unsafe { std::slice::from_raw_parts(v_ptr, half) })?;
                } else {
                    file.write_all(unsafe { std::slice::from_raw_parts(slot.k_ptr(), size) })?;
                }
            }
        }

        file.sync_all()
    })
    .await
    .map_err(std::io::Error::other)??;

    Ok((uuid, block_metas))
}

// ============================================================================
// Quota Management & Eviction
// ============================================================================

async fn quota_loop(mut conn: MultiplexedConnection, config: DfsOffloadConfig) {
    let interval = Duration::from_millis(config.scan_interval_ms);

    loop {
        tokio::time::sleep(interval).await;

        if let Err(e) = check_quota(&mut conn, &config).await {
            error!("Quota check failed: {}", e);
        }
    }
}

async fn check_quota(
    conn: &mut MultiplexedConnection,
    config: &DfsOffloadConfig,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let usage: i64 = conn.get(redis_usage_key()).await.unwrap_or(0);
    let usage = usage.max(0) as u64;

    if usage <= config.quota_bytes {
        return Ok(());
    }

    let to_free = usage.saturating_sub(config.quota_bytes);
    info!(
        usage = %ByteSize(usage),
        quota = %ByteSize(config.quota_bytes),
        to_free = %ByteSize(to_free),
        "Over quota, evicting"
    );

    evict_oldest(conn, config, to_free).await?;
    Ok(())
}

#[tracing::instrument(level = "info", skip(conn, config), fields(target = %ByteSize(target)))]
async fn evict_oldest(
    conn: &mut MultiplexedConnection,
    config: &DfsOffloadConfig,
    target: u64,
) -> Result<u64, Box<dyn std::error::Error + Send + Sync>> {
    let start = Instant::now();
    let delete_script = Script::new(LUA_DELETE_FILE);
    let mut freed = 0u64;
    // ZRANGE is inclusive [0, end], so fetch evict_batch_size items
    let zrange_end = (config.evict_batch_size.saturating_sub(1)) as isize;

    while freed < target {
        let oldest: Vec<String> = conn.zrange(redis_files_zset_key(), 0, zrange_end).await?;
        if oldest.is_empty() {
            break;
        }

        for member in oldest {
            if freed >= target {
                break;
            }

            let Some((namespace, uuid)) = parse_zset_member(&member) else {
                error!(member = %member, "Corrupt ZSET member: failed to parse namespace:uuid, removing");
                let _: Result<(), _> = conn.zrem(redis_files_zset_key(), &member).await;
                continue;
            };

            let file_key = redis_file_key(&namespace, &uuid);
            let Some(meta_json): Option<String> = conn.get(&file_key).await.ok().flatten() else {
                error!(member = %member, "Orphaned ZSET member: file metadata missing, removing");
                let _: Result<(), _> = conn.zrem(redis_files_zset_key(), &member).await;
                continue;
            };

            let Ok(meta) = serde_json::from_str::<FileMeta>(&meta_json) else {
                error!(member = %member, meta = %meta_json, "Corrupt file metadata: JSON parse failed, removing");
                let _: Result<(), _> = conn.zrem(redis_files_zset_key(), &member).await;
                let _: Result<(), _> = conn.del::<_, ()>(&file_key).await;
                continue;
            };

            let block_keys: Vec<String> = meta
                .blocks
                .iter()
                .filter_map(|h| hex::decode(h).ok())
                .map(|h| redis_block_key(&namespace, &h))
                .collect();

            // Delete file from DFS
            let file_path = config
                .dfs_root
                .join(&namespace)
                .join(format!("{}.bin", uuid));
            let _ = fs::remove_file(&file_path).await;

            // Delete Redis keys atomically
            let mut inv = delete_script.key(&file_key);
            inv.key(redis_files_zset_key())
                .key(redis_usage_key())
                .arg(&member)
                .arg(meta.size);
            for k in &block_keys {
                inv.arg(k);
            }
            let _: Result<i32, _> = inv.invoke_async(conn).await;

            freed += meta.size;
            info!(file = %uuid, blocks = block_keys.len(), freed = %ByteSize(freed), "Evicted");
        }
    }

    // Cleanup empty namespace dirs
    if let Ok(mut entries) = fs::read_dir(&config.dfs_root).await {
        while let Ok(Some(e)) = entries.next_entry().await {
            let _ = fs::remove_dir(e.path()).await;
        }
    }

    info!(freed = %ByteSize(freed), elapsed = ?start.elapsed(), "Eviction complete");
    Ok(freed)
}

fn parse_zset_member(member: &str) -> Option<(String, Uuid)> {
    let (ns, uuid_str) = member.rsplit_once(':')?;
    Some((ns.to_string(), Uuid::parse_str(uuid_str).ok()?))
}

// ============================================================================
// Prefetch: Read from DFS into cache
// ============================================================================

/// Result of checking prefix hits with prefetch support
#[derive(Debug, Clone)]
pub enum PrefetchStatus {
    /// All requested blocks are in local cache
    Ready(usize),
    /// Some blocks are being prefetched from DFS
    Prefetching {
        /// Blocks already in cache
        ready: usize,
        /// Blocks currently being loaded from DFS
        loading: usize,
    },
    /// Some blocks don't exist in DFS either
    PartialMiss {
        /// Blocks available (in cache or will be loaded)
        ready: usize,
        /// Blocks not found anywhere
        missing: usize,
    },
}

/// Tracks in-flight prefetch tasks. Thread-safe.
pub struct PrefetchTracker {
    /// Currently prefetching block keys
    pending: DashMap<BlockKey, ()>,
    /// Semaphore to limit concurrent prefetch tasks
    semaphore: Arc<tokio::sync::Semaphore>,
    /// Maximum concurrent prefetch tasks
    max_concurrent: usize,
}

impl PrefetchTracker {
    pub fn new(max_concurrent: usize) -> Self {
        Self {
            pending: DashMap::new(),
            semaphore: Arc::new(tokio::sync::Semaphore::new(max_concurrent)),
            max_concurrent,
        }
    }

    /// Check if a block is currently being prefetched
    pub fn is_pending(&self, namespace: &str, hash: &[u8]) -> bool {
        let key = BlockKey::new(namespace.to_string(), hash.to_vec());
        self.pending.contains_key(&key)
    }

    /// Try to acquire a prefetch slot. Returns None if at capacity.
    pub fn try_acquire(&self) -> Option<tokio::sync::OwnedSemaphorePermit> {
        Arc::clone(&self.semaphore).try_acquire_owned().ok()
    }

    /// Mark blocks as pending prefetch
    fn mark_pending(&self, keys: &[BlockKey]) {
        for key in keys {
            self.pending.insert(key.clone(), ());
        }
    }

    /// Remove blocks from pending set
    fn clear_pending(&self, keys: &[BlockKey]) {
        for key in keys {
            self.pending.remove(key);
        }
    }

    /// Get current pending count
    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }

    /// Get available permits
    pub fn available_permits(&self) -> usize {
        self.semaphore.available_permits()
    }
}

/// Spawn a background prefetch task from DFS into storage cache.
///
/// Returns immediately after spawning. Use tracker.is_pending() to check status.
pub fn prefetch_blocks_from_dfs(
    storage: Arc<StorageEngine>,
    conn: MultiplexedConnection,
    tracker: Arc<PrefetchTracker>,
    dfs_root: PathBuf,
    namespace: String,
    hashes: Vec<Vec<u8>>,
) {
    if hashes.is_empty() {
        return;
    }

    tokio::spawn(async move {
        let mut conn = conn;
        if let Err(e) =
            prefetch_blocks_inner(&storage, &mut conn, &tracker, &dfs_root, &namespace, hashes)
                .await
        {
            error!("Background prefetch failed: {}", e);
        }
    });
}

/// Internal prefetch implementation
async fn prefetch_blocks_inner(
    storage: &StorageEngine,
    conn: &mut MultiplexedConnection,
    tracker: &PrefetchTracker,
    dfs_root: &Path,
    namespace: &str,
    hashes: Vec<Vec<u8>>,
) -> Result<usize, Box<dyn std::error::Error + Send + Sync>> {
    if hashes.is_empty() {
        return Ok(0);
    }

    let start = Instant::now();
    // Try to acquire a permit (max 16 concurrent)
    let Some(permit) = tracker.try_acquire() else {
        debug!(
            "Prefetch rejected: at capacity ({})",
            tracker.max_concurrent
        );
        return Ok(0);
    };

    // Filter out blocks already in cache or pending
    let to_fetch: Vec<Vec<u8>> = hashes
        .into_iter()
        .filter(|h| !storage.cache_contains(namespace, h) && !tracker.is_pending(namespace, h))
        .collect();

    if to_fetch.is_empty() {
        drop(permit);
        return Ok(0);
    }

    // Query Redis for block metadata
    let metas = batch_get_block_meta(conn, namespace, &to_fetch).await?;

    // Collect blocks that have metadata in Redis
    let mut blocks_to_load: Vec<(Vec<u8>, BlockMeta)> = Vec::new();
    for (hash, meta_opt) in to_fetch.iter().zip(metas.into_iter()) {
        if let Some(meta) = meta_opt {
            blocks_to_load.push((hash.clone(), meta));
        }
    }

    if blocks_to_load.is_empty() {
        drop(permit);
        return Ok(0);
    }

    // Mark as pending before spawning
    let block_keys: Vec<BlockKey> = blocks_to_load
        .iter()
        .map(|(h, _)| BlockKey::new(namespace.to_string(), h.clone()))
        .collect();
    tracker.mark_pending(&block_keys);

    info!(
        namespace = %namespace,
        blocks = blocks_to_load.len(),
        "Starting DFS prefetch"
    );

    // Group by file for efficient reading
    let mut by_file: HashMap<String, Vec<(Vec<u8>, BlockMeta)>> = HashMap::new();
    for (hash, meta) in blocks_to_load {
        by_file
            .entry(meta.file.clone())
            .or_default()
            .push((hash, meta));
    }

    let mut total_loaded = 0usize;
    let dfs_root = dfs_root.to_path_buf();
    let ns = namespace.to_string();

    for (file_name, file_blocks) in by_file {
        match read_file_blocks(storage, &dfs_root, &ns, &file_name, file_blocks).await {
            Ok(count) => total_loaded += count,
            Err(e) => {
                error!(file = %file_name, "Failed to read from DFS: {}", e);
            }
        }
    }

    // Clear pending status
    tracker.clear_pending(&block_keys);
    drop(permit);

    let elapsed = start.elapsed();
    info!(
        namespace = %namespace,
        loaded = total_loaded,
        elapsed_ms = elapsed.as_secs_f64() * 1000.0,
        "DFS prefetch complete"
    );

    Ok(total_loaded)
}

/// Read blocks from a single packed file and insert into cache
async fn read_file_blocks(
    storage: &StorageEngine,
    dfs_root: &Path,
    namespace: &str,
    file_name: &str,
    blocks: Vec<(Vec<u8>, BlockMeta)>,
) -> Result<usize, std::io::Error> {
    use std::num::NonZeroU64;
    use std::os::unix::fs::FileExt;

    let file_path = dfs_root.join(namespace).join(file_name);

    // Sort by offset for sequential read pattern
    let mut blocks = blocks;
    blocks.sort_by_key(|(_, m)| m.offset);

    let mut loaded = 0;

    for (hash, meta) in blocks {
        let block_size = meta.total_size();
        let total_slots = meta.total_slots;

        if total_slots == 0 || block_size == 0 {
            warn!(
                hash = %hex::encode(&hash),
                "Skipping block with zero slots or size"
            );
            continue;
        }

        // Allocate pinned memory
        let Some(allocation) = storage.allocate(NonZeroU64::new(block_size).unwrap()) else {
            error!("Failed to allocate pinned memory for prefetch");
            continue;
        };

        // Read from file into pinned memory
        let file_path_clone = file_path.clone();
        let offset = meta.offset;

        let read_result = tokio::task::spawn_blocking({
            let allocation = Arc::clone(&allocation);
            move || {
                let start_time = Instant::now();
                let file = std::fs::File::open(&file_path_clone)?;
                let buf = unsafe {
                    std::slice::from_raw_parts_mut(
                        allocation.as_ptr() as *mut u8,
                        block_size as usize,
                    )
                };
                file.read_exact_at(buf, offset)?;
                let elapsed = start_time.elapsed();
                // calculate bandwitdth
                let bandwidth = (block_size as f64 / 1e6) / elapsed.as_secs_f64();
                info!(
                    "Read block from file: {} , {} ms, {} MB/s",
                    ByteSize(block_size),
                    elapsed.as_millis(),
                    bandwidth
                );
                Ok::<_, std::io::Error>(())
            }
        })
        .await
        .map_err(std::io::Error::other)?;

        if let Err(e) = read_result {
            error!(file = %file_name, offset = offset, "Failed to read block: {}", e);
            continue;
        }

        // Rebuild SealedBlock from the read data
        let sealed = rebuild_sealed_block(allocation, &meta)?;

        // Insert into cache
        storage.cache_insert(namespace, hash, Arc::new(sealed));
        loaded += 1;
    }

    Ok(loaded)
}

/// Rebuild a SealedBlock from raw data and metadata
fn rebuild_sealed_block(
    allocation: Arc<crate::pinned_pool::PinnedAllocation>,
    meta: &BlockMeta,
) -> Result<SealedBlock, std::io::Error> {
    let mut layer_blocks = Vec::with_capacity(meta.slots.len());
    let base_ptr = allocation.as_ptr() as *mut u8;
    let mut current_offset = 0usize;

    for slot_meta in &meta.slots {
        let slot_size = slot_meta.size as usize;

        let layer_block = if slot_meta.is_split {
            // Split storage: K and V are stored sequentially in the file
            // but we need separate pointers
            let half = slot_size / 2;
            let k_ptr = unsafe { base_ptr.add(current_offset) };
            let v_ptr = unsafe { base_ptr.add(current_offset + half) };

            Arc::new(LayerBlock::new_split(
                k_ptr,
                v_ptr,
                slot_size,
                Arc::clone(&allocation),
                Arc::clone(&allocation), // Same allocation for both K and V
            ))
        } else {
            // Contiguous storage
            let ptr = unsafe { base_ptr.add(current_offset) };
            Arc::new(LayerBlock::new_contiguous(
                ptr,
                slot_size,
                Arc::clone(&allocation),
            ))
        };

        layer_blocks.push(layer_block);
        current_offset += slot_size;
    }

    Ok(SealedBlock::from_slots(layer_blocks))
}
