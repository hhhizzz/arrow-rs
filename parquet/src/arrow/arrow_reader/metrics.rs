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

/// Differential timing counters for the Arrow reader hot path.
///
/// Durations are inclusive. In particular, page decompression happens inside
/// `skip_records` or `read_records`; callers that need an exclusive breakdown
/// should subtract `page_decompression_ns` from those two counters once.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ArrowReaderDecompositionMetrics {
    /// Time spent in top-level `ArrayReader::skip_records` calls.
    pub skip_records_ns: u64,
    /// Number of top-level `ArrayReader::skip_records` calls.
    pub skip_records_calls: usize,
    /// Logical rows skipped by top-level `ArrayReader::skip_records` calls.
    pub skip_records_rows: usize,
    /// Time spent in top-level `ArrayReader::read_records` calls.
    pub read_records_ns: u64,
    /// Number of top-level `ArrayReader::read_records` calls.
    pub read_records_calls: usize,
    /// Logical rows decoded by top-level `ArrayReader::read_records` calls.
    pub read_records_rows: usize,
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
        Self::Enabled(Arc::new(ArrowReaderMetricsInner::new()))
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
            Self::Enabled(inner) => Some(
                inner
                    .records_read_from_inner
                    .load(std::sync::atomic::Ordering::Relaxed),
            ),
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
            Self::Enabled(inner) => Some(
                inner
                    .records_read_from_cache
                    .load(std::sync::atomic::Ordering::Relaxed),
            ),
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
        Some(ArrowReaderDecompositionMetrics {
            skip_records_ns: inner.skip_records_ns.load(Ordering::Relaxed),
            skip_records_calls: inner.skip_records_calls.load(Ordering::Relaxed),
            skip_records_rows: inner.skip_records_rows.load(Ordering::Relaxed),
            read_records_ns: inner.read_records_ns.load(Ordering::Relaxed),
            read_records_calls: inner.read_records_calls.load(Ordering::Relaxed),
            read_records_rows: inner.read_records_rows.load(Ordering::Relaxed),
            page_decompression_ns: page.ns,
            page_decompression_pages: page.pages,
            page_decompression_bytes: page.bytes,
            filter_record_batch_ns: inner.filter_record_batch_ns.load(Ordering::Relaxed),
            filter_record_batch_calls: inner.filter_record_batch_calls.load(Ordering::Relaxed),
            selectors_to_mask_ns: inner.selectors_to_mask_ns.load(Ordering::Relaxed),
            selectors_to_mask_calls: inner.selectors_to_mask_calls.load(Ordering::Relaxed),
            consume_batch_ns: inner.consume_batch_ns.load(Ordering::Relaxed),
            consume_batch_calls: inner.consume_batch_calls.load(Ordering::Relaxed),
        })
    }

    /// Increments the count of records read from the inner reader
    pub(crate) fn increment_inner_reads(&self, count: usize) {
        let Self::Enabled(inner) = self else {
            return;
        };
        inner
            .records_read_from_inner
            .fetch_add(count, std::sync::atomic::Ordering::Relaxed);
    }

    /// Increments the count of records read from the cache
    pub(crate) fn increment_cache_reads(&self, count: usize) {
        let Self::Enabled(inner) = self else {
            return;
        };

        inner
            .records_read_from_cache
            .fetch_add(count, std::sync::atomic::Ordering::Relaxed);
    }

    #[inline]
    pub(crate) fn start_timing(&self) -> Option<Instant> {
        matches!(self, Self::Enabled(_)).then(Instant::now)
    }

    #[inline]
    pub(crate) fn record_skip_records(&self, started: Option<Instant>, rows: usize) {
        let (Self::Enabled(inner), Some(started)) = (self, started) else {
            return;
        };
        inner
            .skip_records_ns
            .fetch_add(elapsed_ns(started), Ordering::Relaxed);
        inner.skip_records_calls.fetch_add(1, Ordering::Relaxed);
        inner.skip_records_rows.fetch_add(rows, Ordering::Relaxed);
    }

    #[inline]
    pub(crate) fn record_read_records(&self, started: Option<Instant>, rows: usize) {
        let (Self::Enabled(inner), Some(started)) = (self, started) else {
            return;
        };
        inner
            .read_records_ns
            .fetch_add(elapsed_ns(started), Ordering::Relaxed);
        inner.read_records_calls.fetch_add(1, Ordering::Relaxed);
        inner.read_records_rows.fetch_add(rows, Ordering::Relaxed);
    }

    #[inline]
    pub(crate) fn record_filter_record_batch(&self, started: Option<Instant>) {
        let (Self::Enabled(inner), Some(started)) = (self, started) else {
            return;
        };
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
        inner
            .consume_batch_ns
            .fetch_add(elapsed_ns(started), Ordering::Relaxed);
        inner.consume_batch_calls.fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn page_decompression_metrics(&self) -> Option<Arc<PageDecompressionMetrics>> {
        match self {
            Self::Disabled => None,
            Self::Enabled(inner) => Some(Arc::clone(&inner.page_decompression)),
        }
    }
}

#[inline]
fn elapsed_ns(started: Instant) -> u64 {
    started.elapsed().as_nanos().min(u128::from(u64::MAX)) as u64
}

/// Holds the actual metrics for the Arrow reader.
///
/// Please see [`ArrowReaderMetrics`] for the public interface.
#[derive(Debug)]
pub struct ArrowReaderMetricsInner {
    // Metrics for Predicate Cache
    /// Total number of records read from the inner reader (uncached)
    records_read_from_inner: AtomicUsize,
    /// Total number of records read from previously cached pages
    records_read_from_cache: AtomicUsize,
    skip_records_ns: AtomicU64,
    skip_records_calls: AtomicUsize,
    skip_records_rows: AtomicUsize,
    read_records_ns: AtomicU64,
    read_records_calls: AtomicUsize,
    read_records_rows: AtomicUsize,
    page_decompression: Arc<PageDecompressionMetrics>,
    filter_record_batch_ns: AtomicU64,
    filter_record_batch_calls: AtomicUsize,
    selectors_to_mask_ns: AtomicU64,
    selectors_to_mask_calls: AtomicUsize,
    consume_batch_ns: AtomicU64,
    consume_batch_calls: AtomicUsize,
}

impl ArrowReaderMetricsInner {
    /// Creates a new instance of `ArrowReaderMetricsInner`
    pub(crate) fn new() -> Self {
        Self {
            records_read_from_inner: AtomicUsize::new(0),
            records_read_from_cache: AtomicUsize::new(0),
            skip_records_ns: AtomicU64::new(0),
            skip_records_calls: AtomicUsize::new(0),
            skip_records_rows: AtomicUsize::new(0),
            read_records_ns: AtomicU64::new(0),
            read_records_calls: AtomicUsize::new(0),
            read_records_rows: AtomicUsize::new(0),
            page_decompression: Arc::new(PageDecompressionMetrics::default()),
            filter_record_batch_ns: AtomicU64::new(0),
            filter_record_batch_calls: AtomicUsize::new(0),
            selectors_to_mask_ns: AtomicU64::new(0),
            selectors_to_mask_calls: AtomicUsize::new(0),
            consume_batch_ns: AtomicU64::new(0),
            consume_batch_calls: AtomicUsize::new(0),
        }
    }
}
