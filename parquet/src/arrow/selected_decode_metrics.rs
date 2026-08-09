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

//! Process-wide coverage counters for the experimental selected-decode path
//! (experiment `arrow-selected-decode-reader-wiring-v26`, gate G-W2).
//!
//! G-W2 requires reporting how much of a real workload actually reaches the
//! selected path **as a counter, not an inference** — a leaf-level speedup on
//! 3% of decoded rows is not a workload-level claim, and a path that is never
//! reached at all would otherwise look like "no effect" for a wiring reason
//! rather than a real result.
//!
//! These are deliberately process-global and unsynchronised-relaxed: they are
//! measurement scaffolding for an experiment, not a production feature. They
//! are only ever written from the Mask execution path, so a build that never
//! enables `selected_decode` pays two never-taken branches.
//!
//! The benchmark harness reads [`snapshot`] after each query and prints the
//! result, giving per-query coverage.

use std::sync::atomic::{AtomicU64, Ordering};

static SELECTED_ROWS: AtomicU64 = AtomicU64::new(0);
static FALLBACK_ROWS: AtomicU64 = AtomicU64::new(0);
static SELECTED_CHUNKS: AtomicU64 = AtomicU64::new(0);
static FALLBACK_CHUNKS: AtomicU64 = AtomicU64::new(0);
static SELECTED_BATCHES: AtomicU64 = AtomicU64::new(0);
static FALLBACK_BATCHES: AtomicU64 = AtomicU64::new(0);

/// A point-in-time reading of the coverage counters.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SelectedDecodeCoverage {
    /// Rows emitted by the selected-decode path (already filtered).
    pub selected_rows: u64,
    /// Rows emitted by the ordinary decode-then-filter path.
    pub fallback_rows: u64,
    /// Mask chunks served by the selected path.
    pub selected_chunks: u64,
    /// Mask chunks served by the ordinary path.
    pub fallback_chunks: u64,
    /// Output batches assembled entirely from the selected path.
    pub selected_batches: u64,
    /// Output batches assembled from the ordinary path.
    pub fallback_batches: u64,
}

impl SelectedDecodeCoverage {
    /// Fraction of emitted rows that came from the selected path, in `[0, 1]`.
    /// Returns 0.0 when no rows were emitted at all, so callers never divide by
    /// zero and an untouched workload reports 0% rather than an error.
    pub fn selected_row_fraction(&self) -> f64 {
        let total = self.selected_rows + self.fallback_rows;
        if total == 0 {
            return 0.0;
        }
        self.selected_rows as f64 / total as f64
    }
}

pub(crate) fn record_selected_chunk(rows: usize) {
    SELECTED_ROWS.fetch_add(rows as u64, Ordering::Relaxed);
    SELECTED_CHUNKS.fetch_add(1, Ordering::Relaxed);
}

pub(crate) fn record_fallback_chunk(rows: usize) {
    FALLBACK_ROWS.fetch_add(rows as u64, Ordering::Relaxed);
    FALLBACK_CHUNKS.fetch_add(1, Ordering::Relaxed);
}

pub(crate) fn record_batch(used_selected: bool) {
    if used_selected {
        SELECTED_BATCHES.fetch_add(1, Ordering::Relaxed);
    } else {
        FALLBACK_BATCHES.fetch_add(1, Ordering::Relaxed);
    }
}

/// Read the counters accumulated so far in this process.
pub fn snapshot() -> SelectedDecodeCoverage {
    SelectedDecodeCoverage {
        selected_rows: SELECTED_ROWS.load(Ordering::Relaxed),
        fallback_rows: FALLBACK_ROWS.load(Ordering::Relaxed),
        selected_chunks: SELECTED_CHUNKS.load(Ordering::Relaxed),
        fallback_chunks: FALLBACK_CHUNKS.load(Ordering::Relaxed),
        selected_batches: SELECTED_BATCHES.load(Ordering::Relaxed),
        fallback_batches: FALLBACK_BATCHES.load(Ordering::Relaxed),
    }
}

/// Reset the counters. Used by the harness between queries so each query's
/// coverage is reported independently rather than cumulatively.
pub fn reset() {
    SELECTED_ROWS.store(0, Ordering::Relaxed);
    FALLBACK_ROWS.store(0, Ordering::Relaxed);
    SELECTED_CHUNKS.store(0, Ordering::Relaxed);
    FALLBACK_CHUNKS.store(0, Ordering::Relaxed);
    SELECTED_BATCHES.store(0, Ordering::Relaxed);
    FALLBACK_BATCHES.store(0, Ordering::Relaxed);
}
