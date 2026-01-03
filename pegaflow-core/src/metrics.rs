use opentelemetry::{
    global,
    metrics::{Counter, Histogram, Meter, UpDownCounter},
};
use std::sync::OnceLock;

pub(crate) struct CoreMetrics {
    pub pool_used_bytes: UpDownCounter<i64>,
    pub pool_alloc_failures: Counter<u64>,

    pub cache_block_hits: Counter<u64>,
    pub cache_block_misses: Counter<u64>,
    pub cache_block_insertions: Counter<u64>,
    pub cache_block_admission_rejections: Counter<u64>,
    pub cache_block_evictions: Counter<u64>,

    pub save_bytes: Counter<u64>,
    pub save_duration_ms: Histogram<f64>,

    pub load_bytes: Counter<u64>,
    pub load_duration_ms: Histogram<f64>,
    pub load_failures: Counter<u64>,
}

fn init_meter() -> Meter {
    global::meter("pegaflow-core")
}

pub(crate) fn core_metrics() -> &'static CoreMetrics {
    static METRICS: OnceLock<CoreMetrics> = OnceLock::new();
    METRICS.get_or_init(|| {
        let meter = init_meter();

        CoreMetrics {
            pool_used_bytes: meter
                .i64_up_down_counter("pegaflow_pool_used_bytes")
                .with_unit("bytes")
                .with_description("Current pinned pool usage in bytes")
                .build(),
            pool_alloc_failures: meter
                .u64_counter("pegaflow_pool_alloc_failures_total")
                .with_description("Pinned pool allocation failures after eviction retries")
                .build(),

            cache_block_hits: meter
                .u64_counter("pegaflow_cache_block_hits_total")
                .with_description("Complete blocks found in cache (cache hit)")
                .build(),
            cache_block_misses: meter
                .u64_counter("pegaflow_cache_block_misses_total")
                .with_description("Complete blocks not found in cache (cache miss)")
                .build(),
            cache_block_insertions: meter
                .u64_counter("pegaflow_cache_block_insertions_total")
                .with_description("New blocks inserted into cache")
                .build(),
            cache_block_admission_rejections: meter
                .u64_counter("pegaflow_cache_block_admission_rejections_total")
                .with_description("Blocks rejected by cache admission policy")
                .build(),
            cache_block_evictions: meter
                .u64_counter("pegaflow_cache_block_evictions_total")
                .with_description("Blocks evicted from cache due to memory pressure")
                .build(),

            save_bytes: meter
                .u64_counter("pegaflow_save_bytes_total")
                .with_unit("bytes")
                .with_description("Total bytes saved from GPU to CPU storage")
                .build(),
            save_duration_ms: meter
                .f64_histogram("pegaflow_save_duration_ms")
                .with_unit("ms")
                .with_description("Save operation latency in milliseconds")
                .build(),

            load_bytes: meter
                .u64_counter("pegaflow_load_bytes_total")
                .with_unit("bytes")
                .with_description("Total bytes loaded from CPU storage to GPU")
                .build(),
            load_duration_ms: meter
                .f64_histogram("pegaflow_load_duration_ms")
                .with_unit("ms")
                .with_description("Load operation latency in milliseconds")
                .build(),
            load_failures: meter
                .u64_counter("pegaflow_load_failures_total")
                .with_description("Load operation failures (e.g., transfer errors)")
                .build(),
        }
    })
}
