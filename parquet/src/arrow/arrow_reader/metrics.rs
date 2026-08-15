// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! [ArrowReaderMetrics] for collecting metrics about the Arrow reader

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::Instant;

use crate::file::serialized_reader::PageDecompressionMetrics;

// The PC-1c attribution harness observes at most hundreds of calls per site
// and milliseconds of accumulated time per scan. Packing both values into one
// relaxed atomic update removes observer work without changing a boundary.
const PC1C_PACKED_COUNT_BITS: u32 = 24;
const PC1C_PACKED_COUNT_MASK: u64 = (1 << PC1C_PACKED_COUNT_BITS) - 1;
const PC1C_PACKED_MAX_NS: u64 = u64::MAX >> PC1C_PACKED_COUNT_BITS;

/// Decision counts for the experimental per-column output reader.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PerColumnDecisionMetrics {
    /// Scope or uniform-strategy decisions that stayed on Auto32.
    pub fallback_auto: usize,
    /// Uniform dictionary overrides lowered to one forced standard cursor.
    pub fallback_forced: usize,
    /// Genuine per-column executions constructed.
    pub engaged: usize,
    /// Auto32 fallbacks caused specifically by loaded-page row ranges.
    pub loaded_row_ranges_fallback: usize,
}

/// Differential timing counters for the Arrow reader hot path.
///
/// Durations are inclusive. In particular, page decompression happens inside
/// `selection_decode_ns`; callers that need an exclusive breakdown should
/// subtract `page_decompression_ns` from it once.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ArrowReaderDecompositionMetrics {
    /// Number of top-level `ArrayReader::skip_records` calls.
    pub skip_records_calls: usize,
    /// Logical rows skipped by top-level `ArrayReader::skip_records` calls.
    pub skip_records_rows: usize,
    /// Number of top-level `ArrayReader::read_records` calls.
    pub read_records_calls: usize,
    /// Logical rows decoded by top-level `ArrayReader::read_records` calls.
    pub read_records_rows: usize,
    /// Time spent executing selection cursor and array-reader decode work.
    pub selection_decode_ns: u64,
    /// Number of output-batch decode loops measured.
    pub selection_decode_calls: usize,
    /// Calls to `ArrayReader::read_selection` across the root and Struct fanout.
    pub read_selection_calls: usize,
    /// Selector-backed output batches dispatched through `read_selection`.
    pub driver_batches: usize,
    /// Time spent inside compression codecs while decoding pages.
    pub page_decompression_ns: u64,
    /// Number of compressed pages passed to a codec.
    pub page_decompression_pages: usize,
    /// Compressed payload bytes passed to codecs.
    pub page_decompression_bytes: usize,
    /// Time spent filtering decoded Mask batches.
    pub filter_record_batch_ns: u64,
    /// Number of decoded Mask batches filtered.
    pub filter_record_batch_calls: usize,
    /// Time spent converting selector-backed selections to boolean masks.
    pub selectors_to_mask_ns: u64,
    /// Number of selector-to-mask conversions.
    pub selectors_to_mask_calls: usize,
    /// Time spent in top-level `ArrayReader::consume_batch` calls.
    pub consume_batch_ns: u64,
    /// Number of top-level `ArrayReader::consume_batch` calls.
    pub consume_batch_calls: usize,
    /// PC-1c B1: time spent constructing the selected row-group reader shape.
    pub pc1c_reader_build_ns: u64,
    /// Number of row-group reader constructions measured by PC-1c B1.
    pub pc1c_reader_build_calls: usize,
    /// PC-1c/PC-2 B2: time spent compiling the shared per-column plan.
    pub pc1c_window_ns: u64,
    /// Number of output-window calculations measured by PC-1c B2.
    pub pc1c_window_calls: usize,
    /// PC-1c B3: time spent in the coarse skip/read driver and its decode calls.
    pub pc1c_dispatch_ns: u64,
    /// Number of column, strategy-group, or standard-batch driver invocations measured by B3.
    pub pc1c_dispatch_calls: usize,
    /// PC-1c/PC-2 B4: time spent building and applying the shared filter plan.
    pub pc1c_filter_ns: u64,
    /// Number of filter applications measured by PC-1c B4.
    pub pc1c_filter_calls: usize,
    /// PC-1c B5: time spent consuming one intermediate column/batch.
    pub pc1c_consume_ns: u64,
    /// Number of intermediate/final consumes measured by PC-1c B5.
    pub pc1c_consume_calls: usize,
    /// PC-1c B6: time spent assembling the extra per-column output batch.
    pub pc1c_batch_assembly_ns: u64,
    /// Number of final per-column batch assemblies measured by PC-1c B6.
    pub pc1c_batch_assembly_calls: usize,
    /// Per-column engagement and fallback counts.
    pub per_column_decisions: PerColumnDecisionMetrics,
}

/// Coarse, mutually exclusive boundaries used by the PC-1c attribution run.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Pc1cAttributionSite {
    ReaderBuild,
    Window,
    Dispatch,
    Filter,
    Consume,
    BatchAssembly,
}

/// Outcome recorded once for every attempted per-column reader construction.
#[derive(Clone, Copy, Debug)]
pub(crate) enum PerColumnDecisionKind {
    FallbackAuto,
    FallbackForced,
    Engaged,
}

/// This enum represents the state of Arrow reader metrics collection.
///
/// The inner metrics are stored in an `Arc<ArrowReaderMetricsInner>`
/// so cloning the `ArrowReaderMetrics` enum will not clone the inner metrics.
///
/// To access metrics, create an `ArrowReaderMetrics` via [`ArrowReaderMetrics::enabled()`]
/// and configure the `ArrowReaderBuilder` with a clone.
#[derive(Debug, Clone)]
pub enum ArrowReaderMetrics {
    /// Metrics are not collected (default)
    Disabled,
    /// Metrics are collected and stored in an `Arc`.
    ///
    /// Create this via [`ArrowReaderMetrics::enabled()`].
    Enabled(Arc<ArrowReaderMetricsInner>),
}

impl ArrowReaderMetrics {
    /// Creates a new instance of [`ArrowReaderMetrics::Disabled`]
    pub fn disabled() -> Self {
        Self::Disabled
    }

    /// Creates a new instance of [`ArrowReaderMetrics::Enabled`]
    pub fn enabled() -> Self {
        Self::Enabled(Arc::new(ArrowReaderMetricsInner::new(false)))
    }

    /// Enables only the coarse PC-1c attribution boundaries.
    ///
    /// General decomposition and page-decompression counters remain disabled
    /// so their observer cost cannot contaminate the PC-1c overhead gate.
    pub fn pc1c_attribution() -> Self {
        Self::Enabled(Arc::new(ArrowReaderMetricsInner::new(true)))
    }

    /// Predicate Cache: number of records read directly from the inner reader
    ///
    /// This is the total number of records read from the inner reader (that is
    /// actually decoding). It measures the amount of work that could not be
    /// avoided with caching.
    ///
    /// It returns the number of records read across all columns, so if you read
    /// 2 columns each with 100 records, this will return 200.
    ///
    ///
    /// Returns None if metrics are disabled.
    pub fn records_read_from_inner(&self) -> Option<usize> {
        match self {
            Self::Disabled => None,
            Self::Enabled(inner) if !inner.pc1c_only => Some(
                inner
                    .records_read_from_inner
                    .load(std::sync::atomic::Ordering::Relaxed),
            ),
            Self::Enabled(_) => None,
        }
    }

    /// Predicate Cache: number of records read from the cache
    ///
    /// This is the total number of records read from the cache actually
    /// decoding). It measures the amount of work that was avoided with caching.
    ///
    /// It returns the number of records read across all columns, so if you read
    /// 2 columns each with 100 records from the cache, this will return 200.
    ///
    /// Returns None if metrics are disabled.
    pub fn records_read_from_cache(&self) -> Option<usize> {
        match self {
            Self::Disabled => None,
            Self::Enabled(inner) if !inner.pc1c_only => Some(
                inner
                    .records_read_from_cache
                    .load(std::sync::atomic::Ordering::Relaxed),
            ),
            Self::Enabled(_) => None,
        }
    }

    /// Returns a snapshot of the differential reader timings.
    ///
    /// Returns `None` if metrics are disabled.
    pub fn decomposition(&self) -> Option<ArrowReaderDecompositionMetrics> {
        let Self::Enabled(inner) = self else {
            return None;
        };
        let page = inner.page_decompression.snapshot();
        let (pc1c_reader_build_ns, pc1c_reader_build_calls) =
            inner.pc1c_values(Pc1cAttributionSite::ReaderBuild);
        let (pc1c_window_ns, pc1c_window_calls) = inner.pc1c_values(Pc1cAttributionSite::Window);
        let (pc1c_dispatch_ns, pc1c_dispatch_calls) =
            inner.pc1c_values(Pc1cAttributionSite::Dispatch);
        let (pc1c_filter_ns, pc1c_filter_calls) = inner.pc1c_values(Pc1cAttributionSite::Filter);
        let (pc1c_consume_ns, pc1c_consume_calls) = inner.pc1c_values(Pc1cAttributionSite::Consume);
        let (pc1c_batch_assembly_ns, pc1c_batch_assembly_calls) =
            inner.pc1c_values(Pc1cAttributionSite::BatchAssembly);
        Some(ArrowReaderDecompositionMetrics {
            skip_records_calls: inner.skip_records_calls.load(Ordering::Relaxed),
            skip_records_rows: inner.skip_records_rows.load(Ordering::Relaxed),
            read_records_calls: inner.read_records_calls.load(Ordering::Relaxed),
            read_records_rows: inner.read_records_rows.load(Ordering::Relaxed),
            selection_decode_ns: inner.selection_decode_ns.load(Ordering::Relaxed),
            selection_decode_calls: inner.selection_decode_calls.load(Ordering::Relaxed),
            read_selection_calls: inner.read_selection_calls.load(Ordering::Relaxed),
            driver_batches: inner.driver_batches.load(Ordering::Relaxed),
            page_decompression_ns: page.ns,
            page_decompression_pages: page.pages,
            page_decompression_bytes: page.bytes,
            filter_record_batch_ns: inner.filter_record_batch_ns.load(Ordering::Relaxed),
            filter_record_batch_calls: inner.filter_record_batch_calls.load(Ordering::Relaxed),
            selectors_to_mask_ns: inner.selectors_to_mask_ns.load(Ordering::Relaxed),
            selectors_to_mask_calls: inner.selectors_to_mask_calls.load(Ordering::Relaxed),
            consume_batch_ns: inner.consume_batch_ns.load(Ordering::Relaxed),
            consume_batch_calls: inner.consume_batch_calls.load(Ordering::Relaxed),
            pc1c_reader_build_ns,
            pc1c_reader_build_calls,
            pc1c_window_ns,
            pc1c_window_calls,
            pc1c_dispatch_ns,
            pc1c_dispatch_calls,
            pc1c_filter_ns,
            pc1c_filter_calls,
            pc1c_consume_ns,
            pc1c_consume_calls,
            pc1c_batch_assembly_ns,
            pc1c_batch_assembly_calls,
            per_column_decisions: PerColumnDecisionMetrics {
                fallback_auto: inner.per_column_fallback_auto.load(Ordering::Relaxed),
                fallback_forced: inner.per_column_fallback_forced.load(Ordering::Relaxed),
                engaged: inner.per_column_engaged.load(Ordering::Relaxed),
                loaded_row_ranges_fallback: inner
                    .per_column_loaded_row_ranges_fallback
                    .load(Ordering::Relaxed),
            },
        })
    }

    #[inline]
    pub(crate) fn record_per_column_decision(&self, kind: PerColumnDecisionKind) {
        let Self::Enabled(inner) = self else {
            return;
        };
        let counter = match kind {
            PerColumnDecisionKind::FallbackAuto => &inner.per_column_fallback_auto,
            PerColumnDecisionKind::FallbackForced => &inner.per_column_fallback_forced,
            PerColumnDecisionKind::Engaged => &inner.per_column_engaged,
        };
        counter.fetch_add(1, Ordering::Relaxed);
    }

    #[inline]
    pub(crate) fn record_per_column_loaded_row_ranges_fallback(&self) {
        let Self::Enabled(inner) = self else {
            return;
        };
        inner
            .per_column_loaded_row_ranges_fallback
            .fetch_add(1, Ordering::Relaxed);
    }

    /// Increments the count of records read from the inner reader
    pub(crate) fn increment_inner_reads(&self, count: usize) {
        let Self::Enabled(inner) = self else {
            return;
        };
        if inner.pc1c_only {
            return;
        }
        inner
            .records_read_from_inner
            .fetch_add(count, std::sync::atomic::Ordering::Relaxed);
    }

    /// Increments the count of records read from the cache
    pub(crate) fn increment_cache_reads(&self, count: usize) {
        let Self::Enabled(inner) = self else {
            return;
        };
        if inner.pc1c_only {
            return;
        }

        inner
            .records_read_from_cache
            .fetch_add(count, std::sync::atomic::Ordering::Relaxed);
    }

    #[inline]
    pub(crate) fn start_timing(&self) -> Option<Instant> {
        matches!(self, Self::Enabled(_)).then(Instant::now)
    }

    #[inline]
    pub(crate) fn start_general_timing(&self) -> Option<Instant> {
        matches!(self, Self::Enabled(inner) if !inner.pc1c_only).then(Instant::now)
    }

    #[inline]
    pub(crate) fn record_selection_decode(
        &self,
        started: Option<Instant>,
        skip_calls: usize,
        skip_rows: usize,
        read_calls: usize,
        read_rows: usize,
    ) {
        let (Self::Enabled(inner), Some(started)) = (self, started) else {
            return;
        };
        if inner.pc1c_only {
            return;
        }
        inner
            .selection_decode_ns
            .fetch_add(elapsed_ns(started), Ordering::Relaxed);
        inner.selection_decode_calls.fetch_add(1, Ordering::Relaxed);
        inner
            .skip_records_calls
            .fetch_add(skip_calls, Ordering::Relaxed);
        inner
            .skip_records_rows
            .fetch_add(skip_rows, Ordering::Relaxed);
        inner
            .read_records_calls
            .fetch_add(read_calls, Ordering::Relaxed);
        inner
            .read_records_rows
            .fetch_add(read_rows, Ordering::Relaxed);
    }

    #[inline]
    pub(crate) fn record_read_selection_root_call(&self) {
        let Self::Enabled(inner) = self else {
            return;
        };
        if inner.pc1c_only {
            return;
        }
        inner.driver_batches.fetch_add(1, Ordering::Relaxed);
        inner.read_selection_calls.fetch_add(1, Ordering::Relaxed);
    }

    #[inline]
    pub(crate) fn record_read_selection_child_call(&self) {
        let Self::Enabled(inner) = self else {
            return;
        };
        if inner.pc1c_only {
            return;
        }
        inner.read_selection_calls.fetch_add(1, Ordering::Relaxed);
    }

    #[inline]
    pub(crate) fn record_filter_record_batch(&self, started: Option<Instant>) {
        let (Self::Enabled(inner), Some(started)) = (self, started) else {
            return;
        };
        if inner.pc1c_only {
            return;
        }
        inner
            .filter_record_batch_ns
            .fetch_add(elapsed_ns(started), Ordering::Relaxed);
        inner
            .filter_record_batch_calls
            .fetch_add(1, Ordering::Relaxed);
    }

    #[inline]
    pub(crate) fn record_selectors_to_mask(&self, started: Option<Instant>) {
        let (Self::Enabled(inner), Some(started)) = (self, started) else {
            return;
        };
        if inner.pc1c_only {
            return;
        }
        inner
            .selectors_to_mask_ns
            .fetch_add(elapsed_ns(started), Ordering::Relaxed);
        inner
            .selectors_to_mask_calls
            .fetch_add(1, Ordering::Relaxed);
    }

    #[inline]
    pub(crate) fn record_consume_batch(&self, started: Option<Instant>) {
        let (Self::Enabled(inner), Some(started)) = (self, started) else {
            return;
        };
        if inner.pc1c_only {
            return;
        }
        inner
            .consume_batch_ns
            .fetch_add(elapsed_ns(started), Ordering::Relaxed);
        inner.consume_batch_calls.fetch_add(1, Ordering::Relaxed);
    }

    #[inline]
    pub(crate) fn record_pc1c_attribution(
        &self,
        site: Pc1cAttributionSite,
        started: Option<Instant>,
    ) {
        let (Self::Enabled(inner), Some(started)) = (self, started) else {
            return;
        };
        let elapsed = elapsed_ns(started);
        if inner.pc1c_only {
            inner
                .pc1c_packed(site)
                .fetch_add(pack_pc1c_sample(elapsed), Ordering::Relaxed);
            return;
        }
        let (nanoseconds, calls) = match site {
            Pc1cAttributionSite::ReaderBuild => {
                (&inner.pc1c_reader_build_ns, &inner.pc1c_reader_build_calls)
            }
            Pc1cAttributionSite::Window => (&inner.pc1c_window_ns, &inner.pc1c_window_calls),
            Pc1cAttributionSite::Dispatch => (&inner.pc1c_dispatch_ns, &inner.pc1c_dispatch_calls),
            Pc1cAttributionSite::Filter => (&inner.pc1c_filter_ns, &inner.pc1c_filter_calls),
            Pc1cAttributionSite::Consume => (&inner.pc1c_consume_ns, &inner.pc1c_consume_calls),
            Pc1cAttributionSite::BatchAssembly => (
                &inner.pc1c_batch_assembly_ns,
                &inner.pc1c_batch_assembly_calls,
            ),
        };
        nanoseconds.fetch_add(elapsed, Ordering::Relaxed);
        calls.fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn page_decompression_metrics(&self) -> Option<Arc<PageDecompressionMetrics>> {
        match self {
            Self::Disabled => None,
            Self::Enabled(inner) if !inner.pc1c_only => Some(Arc::clone(&inner.page_decompression)),
            Self::Enabled(_) => None,
        }
    }
}

#[inline]
fn elapsed_ns(started: Instant) -> u64 {
    started.elapsed().as_nanos().min(u128::from(u64::MAX)) as u64
}

#[inline]
fn pack_pc1c_sample(nanoseconds: u64) -> u64 {
    (nanoseconds.min(PC1C_PACKED_MAX_NS) << PC1C_PACKED_COUNT_BITS) | 1
}

#[inline]
fn unpack_pc1c_samples(packed: u64) -> (u64, usize) {
    (
        packed >> PC1C_PACKED_COUNT_BITS,
        (packed & PC1C_PACKED_COUNT_MASK) as usize,
    )
}

/// Holds the actual metrics for the Arrow reader.
///
/// Please see [`ArrowReaderMetrics`] for the public interface.
#[derive(Debug)]
pub struct ArrowReaderMetricsInner {
    /// True when only the coarse PC-1c attribution boundaries are active.
    pc1c_only: bool,
    // Metrics for Predicate Cache
    /// Total number of records read from the inner reader (uncached)
    records_read_from_inner: AtomicUsize,
    /// Total number of records read from previously cached pages
    records_read_from_cache: AtomicUsize,
    skip_records_calls: AtomicUsize,
    skip_records_rows: AtomicUsize,
    read_records_calls: AtomicUsize,
    read_records_rows: AtomicUsize,
    selection_decode_ns: AtomicU64,
    selection_decode_calls: AtomicUsize,
    read_selection_calls: AtomicUsize,
    driver_batches: AtomicUsize,
    page_decompression: Arc<PageDecompressionMetrics>,
    filter_record_batch_ns: AtomicU64,
    filter_record_batch_calls: AtomicUsize,
    selectors_to_mask_ns: AtomicU64,
    selectors_to_mask_calls: AtomicUsize,
    consume_batch_ns: AtomicU64,
    consume_batch_calls: AtomicUsize,
    pc1c_reader_build_ns: AtomicU64,
    pc1c_reader_build_calls: AtomicUsize,
    pc1c_window_ns: AtomicU64,
    pc1c_window_calls: AtomicUsize,
    pc1c_dispatch_ns: AtomicU64,
    pc1c_dispatch_calls: AtomicUsize,
    pc1c_filter_ns: AtomicU64,
    pc1c_filter_calls: AtomicUsize,
    pc1c_consume_ns: AtomicU64,
    pc1c_consume_calls: AtomicUsize,
    pc1c_batch_assembly_ns: AtomicU64,
    pc1c_batch_assembly_calls: AtomicUsize,
    per_column_fallback_auto: AtomicUsize,
    per_column_fallback_forced: AtomicUsize,
    per_column_engaged: AtomicUsize,
    per_column_loaded_row_ranges_fallback: AtomicUsize,
}

impl ArrowReaderMetricsInner {
    fn pc1c_packed(&self, site: Pc1cAttributionSite) -> &AtomicU64 {
        match site {
            Pc1cAttributionSite::ReaderBuild => &self.pc1c_reader_build_ns,
            Pc1cAttributionSite::Window => &self.pc1c_window_ns,
            Pc1cAttributionSite::Dispatch => &self.pc1c_dispatch_ns,
            Pc1cAttributionSite::Filter => &self.pc1c_filter_ns,
            Pc1cAttributionSite::Consume => &self.pc1c_consume_ns,
            Pc1cAttributionSite::BatchAssembly => &self.pc1c_batch_assembly_ns,
        }
    }

    fn pc1c_values(&self, site: Pc1cAttributionSite) -> (u64, usize) {
        if self.pc1c_only {
            return unpack_pc1c_samples(self.pc1c_packed(site).load(Ordering::Relaxed));
        }
        let (nanoseconds, calls) = match site {
            Pc1cAttributionSite::ReaderBuild => {
                (&self.pc1c_reader_build_ns, &self.pc1c_reader_build_calls)
            }
            Pc1cAttributionSite::Window => (&self.pc1c_window_ns, &self.pc1c_window_calls),
            Pc1cAttributionSite::Dispatch => (&self.pc1c_dispatch_ns, &self.pc1c_dispatch_calls),
            Pc1cAttributionSite::Filter => (&self.pc1c_filter_ns, &self.pc1c_filter_calls),
            Pc1cAttributionSite::Consume => (&self.pc1c_consume_ns, &self.pc1c_consume_calls),
            Pc1cAttributionSite::BatchAssembly => (
                &self.pc1c_batch_assembly_ns,
                &self.pc1c_batch_assembly_calls,
            ),
        };
        (
            nanoseconds.load(Ordering::Relaxed),
            calls.load(Ordering::Relaxed),
        )
    }

    /// Creates a new instance of `ArrowReaderMetricsInner`
    pub(crate) fn new(pc1c_only: bool) -> Self {
        Self {
            pc1c_only,
            records_read_from_inner: AtomicUsize::new(0),
            records_read_from_cache: AtomicUsize::new(0),
            skip_records_calls: AtomicUsize::new(0),
            skip_records_rows: AtomicUsize::new(0),
            read_records_calls: AtomicUsize::new(0),
            read_records_rows: AtomicUsize::new(0),
            selection_decode_ns: AtomicU64::new(0),
            selection_decode_calls: AtomicUsize::new(0),
            read_selection_calls: AtomicUsize::new(0),
            driver_batches: AtomicUsize::new(0),
            page_decompression: Arc::new(PageDecompressionMetrics::default()),
            filter_record_batch_ns: AtomicU64::new(0),
            filter_record_batch_calls: AtomicUsize::new(0),
            selectors_to_mask_ns: AtomicU64::new(0),
            selectors_to_mask_calls: AtomicUsize::new(0),
            consume_batch_ns: AtomicU64::new(0),
            consume_batch_calls: AtomicUsize::new(0),
            pc1c_reader_build_ns: AtomicU64::new(0),
            pc1c_reader_build_calls: AtomicUsize::new(0),
            pc1c_window_ns: AtomicU64::new(0),
            pc1c_window_calls: AtomicUsize::new(0),
            pc1c_dispatch_ns: AtomicU64::new(0),
            pc1c_dispatch_calls: AtomicUsize::new(0),
            pc1c_filter_ns: AtomicU64::new(0),
            pc1c_filter_calls: AtomicUsize::new(0),
            pc1c_consume_ns: AtomicU64::new(0),
            pc1c_consume_calls: AtomicUsize::new(0),
            pc1c_batch_assembly_ns: AtomicU64::new(0),
            pc1c_batch_assembly_calls: AtomicUsize::new(0),
            per_column_fallback_auto: AtomicUsize::new(0),
            per_column_fallback_forced: AtomicUsize::new(0),
            per_column_engaged: AtomicUsize::new(0),
            per_column_loaded_row_ranges_fallback: AtomicUsize::new(0),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        ArrowReaderMetrics, PC1C_PACKED_MAX_NS, PerColumnDecisionKind, pack_pc1c_sample,
        unpack_pc1c_samples,
    };

    #[test]
    fn pc1c_packed_samples_accumulate_time_and_calls() {
        let packed = pack_pc1c_sample(123) + pack_pc1c_sample(456);
        assert_eq!(unpack_pc1c_samples(packed), (579, 2));
    }

    #[test]
    fn pc1c_packed_sample_saturates_individual_duration() {
        assert_eq!(
            unpack_pc1c_samples(pack_pc1c_sample(u64::MAX)),
            (PC1C_PACKED_MAX_NS, 1)
        );
    }

    #[test]
    fn per_column_decisions_snapshot_independently() {
        let metrics = ArrowReaderMetrics::enabled();
        metrics.record_per_column_decision(PerColumnDecisionKind::FallbackAuto);
        metrics.record_per_column_decision(PerColumnDecisionKind::FallbackForced);
        metrics.record_per_column_decision(PerColumnDecisionKind::Engaged);
        metrics.record_per_column_loaded_row_ranges_fallback();
        let decisions = metrics.decomposition().unwrap().per_column_decisions;
        assert_eq!(decisions.fallback_auto, 1);
        assert_eq!(decisions.fallback_forced, 1);
        assert_eq!(decisions.engaged, 1);
        assert_eq!(decisions.loaded_row_ranges_fallback, 1);
    }
}
