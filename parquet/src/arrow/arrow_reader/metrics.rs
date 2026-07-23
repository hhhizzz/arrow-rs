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

use crate::arrow::arrow_reader::selection::{
    CostModelDecisionReason, RowGroupExecutionMode, RowSelectionStrategyDecision,
    RowSelectionStrategyReason,
};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

#[derive(Clone, Copy, Debug)]
pub(crate) enum ArrowReaderPhase {
    PredicateRangePlanning,
    PredicateDecode,
    PredicateEvaluate,
    PredicateSelectionBuild,
    PredicateSelectionMerge,
    OutputRangePlanning,
    OutputSelectionResolve,
    OutputSkipRecords,
    OutputReadRecords,
    OutputConsumeBatch,
    OutputMaskFilter,
    OutputConcatBatches,
    OutputBatchBuild,
    PostFilterPredicateProject,
    PostFilterPredicateEvaluate,
    PostFilterApplyFilter,
    PostFilterOutputProject,
}

impl ArrowReaderPhase {
    const COUNT: usize = 17;
    const ALL: [Self; Self::COUNT] = [
        Self::PredicateRangePlanning,
        Self::PredicateDecode,
        Self::PredicateEvaluate,
        Self::PredicateSelectionBuild,
        Self::PredicateSelectionMerge,
        Self::OutputRangePlanning,
        Self::OutputSelectionResolve,
        Self::OutputSkipRecords,
        Self::OutputReadRecords,
        Self::OutputConsumeBatch,
        Self::OutputMaskFilter,
        Self::OutputConcatBatches,
        Self::OutputBatchBuild,
        Self::PostFilterPredicateProject,
        Self::PostFilterPredicateEvaluate,
        Self::PostFilterApplyFilter,
        Self::PostFilterOutputProject,
    ];

    fn index(self) -> usize {
        match self {
            Self::PredicateRangePlanning => 0,
            Self::PredicateDecode => 1,
            Self::PredicateEvaluate => 2,
            Self::PredicateSelectionBuild => 3,
            Self::PredicateSelectionMerge => 4,
            Self::OutputRangePlanning => 5,
            Self::OutputSelectionResolve => 6,
            Self::OutputSkipRecords => 7,
            Self::OutputReadRecords => 8,
            Self::OutputConsumeBatch => 9,
            Self::OutputMaskFilter => 10,
            Self::OutputConcatBatches => 11,
            Self::OutputBatchBuild => 12,
            Self::PostFilterPredicateProject => 13,
            Self::PostFilterPredicateEvaluate => 14,
            Self::PostFilterApplyFilter => 15,
            Self::PostFilterOutputProject => 16,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::PredicateRangePlanning => "predicate_range_planning",
            Self::PredicateDecode => "predicate_decode",
            Self::PredicateEvaluate => "predicate_evaluate",
            Self::PredicateSelectionBuild => "predicate_selection_build",
            Self::PredicateSelectionMerge => "predicate_selection_merge",
            Self::OutputRangePlanning => "output_range_planning",
            Self::OutputSelectionResolve => "output_selection_resolve",
            Self::OutputSkipRecords => "output_skip_records",
            Self::OutputReadRecords => "output_read_records",
            Self::OutputConsumeBatch => "output_consume_batch",
            Self::OutputMaskFilter => "output_mask_filter",
            Self::OutputConcatBatches => "output_concat_batches",
            Self::OutputBatchBuild => "output_batch_build",
            Self::PostFilterPredicateProject => "post_filter_predicate_project",
            Self::PostFilterPredicateEvaluate => "post_filter_predicate_evaluate",
            Self::PostFilterApplyFilter => "post_filter_apply_filter",
            Self::PostFilterOutputProject => "post_filter_output_project",
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum ArrowReaderRangePlanning {
    Predicate,
    Output,
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct PredicateDenseFetchDiagnostics {
    pub(crate) not_predicate_planning_count: usize,
    pub(crate) env_not_parsed_count: usize,
    pub(crate) below_range_threshold_count: usize,
    pub(crate) no_dense_candidate_count: usize,
    pub(crate) ratio_guard_failed_count: usize,
    pub(crate) extra_bytes_guard_failed_count: usize,
    pub(crate) materialization_unavailable_count: usize,
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

    /// Creates metrics with phase profiling enabled.
    pub fn enabled_with_phase_profile() -> Self {
        Self::Enabled(Arc::new(ArrowReaderMetricsInner::new(true)))
    }

    #[inline]
    pub(crate) fn phase_profile_enabled(&self) -> bool {
        matches!(self, Self::Enabled(inner) if inner.phase_profile_enabled)
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
            Self::Enabled(inner) => Some(inner.records_read_from_cache.load(Ordering::Relaxed)),
        }
    }

    /// Row Selection: number of selected rows recorded in planned selections
    pub fn row_selection_selected_rows(&self) -> Option<usize> {
        self.load(|inner| &inner.row_selection_selected_rows)
    }

    /// Row Selection: number of skipped rows recorded in planned selections
    pub fn row_selection_skipped_rows(&self) -> Option<usize> {
        self.load(|inner| &inner.row_selection_skipped_rows)
    }

    /// Row Selection: number of non-empty selectors recorded in planned selections
    pub fn row_selection_selector_count(&self) -> Option<usize> {
        self.load(|inner| &inner.row_selection_selector_count)
    }

    /// Row Selection: number of selected runs recorded in planned selections
    pub fn row_selection_selected_run_count(&self) -> Option<usize> {
        self.load(|inner| &inner.row_selection_selected_run_count)
    }

    /// Row Selection: number of skipped runs recorded in planned selections
    pub fn row_selection_skipped_run_count(&self) -> Option<usize> {
        self.load(|inner| &inner.row_selection_skipped_run_count)
    }

    /// Row Selection: number of output pages touched by cost-model observations
    pub fn row_selection_output_pages_touched(&self) -> Option<usize> {
        self.load(|inner| &inner.row_selection_output_pages_touched)
    }

    /// Row Selection: number of output pages available to cost-model observations
    pub fn row_selection_output_pages_total(&self) -> Option<usize> {
        self.load(|inner| &inner.row_selection_output_pages_total)
    }

    /// Row Selection: whether output page touch metrics were observable
    pub fn row_selection_output_page_touch_available(&self) -> Option<usize> {
        self.load(|inner| &inner.row_selection_output_page_touch_available)
    }

    /// Row Selection: compressed bytes for output pages touched by row selections
    pub fn row_selection_output_page_bytes_touched(&self) -> Option<usize> {
        self.load(|inner| &inner.row_selection_output_page_bytes_touched)
    }

    /// Row Selection: compressed bytes for output pages available to row selections
    pub fn row_selection_output_page_bytes_total(&self) -> Option<usize> {
        self.load(|inner| &inner.row_selection_output_page_bytes_total)
    }

    /// Row Selection: sparse selected fetch range count before any dense fallback
    pub fn row_selection_sparse_range_count(&self) -> Option<usize> {
        self.load(|inner| &inner.row_selection_sparse_range_count)
    }

    /// Row Selection: sparse selected fetch range bytes before any dense fallback
    pub fn row_selection_sparse_range_bytes(&self) -> Option<usize> {
        self.load(|inner| &inner.row_selection_sparse_range_bytes)
    }

    /// Row Selection: number of dense fetch fallbacks applied
    pub fn row_selection_dense_fetch_fallback_count(&self) -> Option<usize> {
        self.load(|inner| &inner.row_selection_dense_fetch_fallback_count)
    }

    /// Row Selection: dense fetch range count after fallback
    pub fn row_selection_dense_fetch_range_count(&self) -> Option<usize> {
        self.load(|inner| &inner.row_selection_dense_fetch_range_count)
    }

    /// Row Selection: dense fetch range bytes after fallback
    pub fn row_selection_dense_fetch_range_bytes(&self) -> Option<usize> {
        self.load(|inner| &inner.row_selection_dense_fetch_range_bytes)
    }

    /// Predicate range planning: number of non-empty fetch requests
    pub fn predicate_fetch_request_count(&self) -> Option<usize> {
        self.load(|inner| &inner.predicate_fetch_request_count)
    }

    /// Predicate range planning: number of non-empty fetch requests
    pub fn predicate_request_count(&self) -> Option<usize> {
        self.predicate_fetch_request_count()
    }

    /// Predicate range planning: number of requests containing exactly one fetch range
    pub fn predicate_single_range_request_count(&self) -> Option<usize> {
        self.load(|inner| &inner.predicate_single_range_request_count)
    }

    /// Predicate range planning: number of fetch ranges requested
    pub fn predicate_fetch_range_count(&self) -> Option<usize> {
        self.load(|inner| &inner.predicate_fetch_range_count)
    }

    /// Predicate range planning: bytes in fetch ranges requested
    pub fn predicate_fetch_range_bytes(&self) -> Option<usize> {
        self.load(|inner| &inner.predicate_fetch_range_bytes)
    }

    /// Predicate evaluation: number of times row-filter predicates were evaluated
    pub fn predicate_evaluate_call_count(&self) -> Option<usize> {
        self.load(|inner| &inner.predicate_evaluate_call_count)
    }

    /// Predicate evaluation: number of input rows passed to row-filter predicates
    pub fn predicate_evaluate_input_row_count(&self) -> Option<usize> {
        self.load(|inner| &inner.predicate_evaluate_input_row_count)
    }

    /// Predicate evaluation: number of rows selected by row-filter predicates
    pub fn predicate_evaluate_selected_row_count(&self) -> Option<usize> {
        self.load(|inner| &inner.predicate_evaluate_selected_row_count)
    }

    /// Predicate range planning: number of dense fetch fallbacks applied
    pub fn predicate_dense_fetch_fallback_count(&self) -> Option<usize> {
        self.load(|inner| &inner.predicate_dense_fetch_fallback_count)
    }

    /// Predicate range planning: dense fetch range count after fallback
    pub fn predicate_dense_fetch_range_count(&self) -> Option<usize> {
        self.load(|inner| &inner.predicate_dense_fetch_range_count)
    }

    /// Predicate range planning: dense fetch range bytes after fallback
    pub fn predicate_dense_fetch_range_bytes(&self) -> Option<usize> {
        self.load(|inner| &inner.predicate_dense_fetch_range_bytes)
    }

    /// Predicate request batching: number of request batches emitted
    pub fn predicate_request_batch_count(&self) -> Option<usize> {
        self.load(|inner| &inner.predicate_request_batch_count)
    }

    /// Predicate request batching: number of fetch ranges emitted by batches
    pub fn predicate_batched_range_count(&self) -> Option<usize> {
        self.load(|inner| &inner.predicate_batched_range_count)
    }

    /// Predicate request batching: bytes in fetch ranges emitted by batches
    pub fn predicate_batched_range_bytes(&self) -> Option<usize> {
        self.load(|inner| &inner.predicate_batched_range_bytes)
    }

    /// Predicate request batching: bytes fetched beyond original predicate requests
    pub fn predicate_request_batch_extra_bytes(&self) -> Option<usize> {
        self.load(|inner| &inner.predicate_request_batch_extra_bytes)
    }

    /// Predicate request batching: total nanoseconds spent planning future row-group requests
    pub fn predicate_request_batch_plan_time_nanos(&self) -> Option<u64> {
        self.load_u64(|inner| &inner.predicate_request_batch_plan_time_nanos)
    }

    /// Predicate request batching: number of timed future row-group request planning calls
    pub fn predicate_request_batch_plan_count(&self) -> Option<usize> {
        self.load(|inner| &inner.predicate_request_batch_plan_count)
    }

    /// Predicate request batching: total nanoseconds spent pushing candidate requests into batches
    pub fn predicate_request_batch_try_push_time_nanos(&self) -> Option<u64> {
        self.load_u64(|inner| &inner.predicate_request_batch_try_push_time_nanos)
    }

    /// Predicate request batching: number of timed batch try-push calls
    pub fn predicate_request_batch_try_push_count(&self) -> Option<usize> {
        self.load(|inner| &inner.predicate_request_batch_try_push_count)
    }

    /// Predicate range planning: dense fetch skipped because request was not predicate-planning
    pub fn predicate_dense_fetch_not_predicate_planning_count(&self) -> Option<usize> {
        self.load(|inner| &inner.predicate_dense_fetch_not_predicate_planning_count)
    }

    /// Predicate range planning: dense fetch skipped because env/config was not parsed
    pub fn predicate_dense_fetch_env_not_parsed_count(&self) -> Option<usize> {
        self.load(|inner| &inner.predicate_dense_fetch_env_not_parsed_count)
    }

    /// Predicate range planning: dense fetch skipped because range count was under threshold
    pub fn predicate_dense_fetch_below_range_threshold_count(&self) -> Option<usize> {
        self.load(|inner| &inner.predicate_dense_fetch_below_range_threshold_count)
    }

    /// Predicate range planning: dense fetch skipped because no denser candidate existed
    pub fn predicate_dense_fetch_no_dense_candidate_count(&self) -> Option<usize> {
        self.load(|inner| &inner.predicate_dense_fetch_no_dense_candidate_count)
    }

    /// Predicate range planning: dense fetch skipped because the ratio guard failed
    pub fn predicate_dense_fetch_ratio_guard_failed_count(&self) -> Option<usize> {
        self.load(|inner| &inner.predicate_dense_fetch_ratio_guard_failed_count)
    }

    /// Predicate range planning: dense fetch skipped because the extra-bytes guard failed
    pub fn predicate_dense_fetch_extra_bytes_guard_failed_count(&self) -> Option<usize> {
        self.load(|inner| &inner.predicate_dense_fetch_extra_bytes_guard_failed_count)
    }

    /// Predicate range planning: dense fetch skipped because materialization split was unavailable
    pub fn predicate_dense_fetch_materialization_unavailable_count(&self) -> Option<usize> {
        self.load(|inner| &inner.predicate_dense_fetch_materialization_unavailable_count)
    }

    /// Output range planning: number of non-empty fetch requests
    pub fn output_fetch_request_count(&self) -> Option<usize> {
        self.load(|inner| &inner.output_fetch_request_count)
    }

    /// Output range planning: number of fetch ranges requested
    pub fn output_fetch_range_count(&self) -> Option<usize> {
        self.load(|inner| &inner.output_fetch_range_count)
    }

    /// Output range planning: bytes in fetch ranges requested
    pub fn output_fetch_range_bytes(&self) -> Option<usize> {
        self.load(|inner| &inner.output_fetch_range_bytes)
    }

    /// Row Selection: number of plans using mask materialization
    pub fn row_selection_mask_plan_count(&self) -> Option<usize> {
        self.load(|inner| &inner.row_selection_mask_plan_count)
    }

    /// Row Selection: number of plans using selector materialization
    pub fn row_selection_selector_plan_count(&self) -> Option<usize> {
        self.load(|inner| &inner.row_selection_selector_plan_count)
    }

    /// Row Selection: number of plans forced to masks
    pub fn row_selection_forced_mask_plan_count(&self) -> Option<usize> {
        self.load(|inner| &inner.row_selection_forced_mask_plan_count)
    }

    /// Row Selection: number of plans forced to selectors
    pub fn row_selection_forced_selector_plan_count(&self) -> Option<usize> {
        self.load(|inner| &inner.row_selection_forced_selector_plan_count)
    }

    /// Row Selection: number of Auto plans choosing masks for empty selections
    pub fn row_selection_auto_mask_empty_plan_count(&self) -> Option<usize> {
        self.load(|inner| &inner.row_selection_auto_mask_empty_plan_count)
    }

    /// Row Selection: number of Auto plans choosing masks for short runs
    pub fn row_selection_auto_mask_short_run_plan_count(&self) -> Option<usize> {
        self.load(|inner| &inner.row_selection_auto_mask_short_run_plan_count)
    }

    /// Row Selection: number of Auto plans choosing masks for fragmented selected rows
    pub fn row_selection_auto_mask_fragmented_plan_count(&self) -> Option<usize> {
        self.load(|inner| &inner.row_selection_auto_mask_fragmented_plan_count)
    }

    /// Row Selection: number of Auto plans choosing masks for high selected-row ratio
    pub fn row_selection_auto_mask_high_ratio_plan_count(&self) -> Option<usize> {
        self.load(|inner| &inner.row_selection_auto_mask_high_ratio_plan_count)
    }

    /// Row Selection: number of Auto plans choosing selectors for clustered selected rows
    pub fn row_selection_auto_selector_clustered_plan_count(&self) -> Option<usize> {
        self.load(|inner| &inner.row_selection_auto_selector_clustered_plan_count)
    }

    /// Row Selection: number of Auto plans choosing selectors for long runs
    pub fn row_selection_auto_selector_long_run_plan_count(&self) -> Option<usize> {
        self.load(|inner| &inner.row_selection_auto_selector_long_run_plan_count)
    }

    /// Cost model: number of row groups included in the observation window
    pub fn cost_model_observed_row_group_count(&self) -> Option<usize> {
        self.load(|inner| &inner.cost_model_observed_row_group_count)
    }

    /// Cost model: number of row groups executed with pushdown
    pub fn cost_model_pushdown_row_group_count(&self) -> Option<usize> {
        self.load(|inner| &inner.cost_model_pushdown_row_group_count)
    }

    /// Cost model: number of row groups executed with post-filter
    pub fn cost_model_post_filter_row_group_count(&self) -> Option<usize> {
        self.load(|inner| &inner.cost_model_post_filter_row_group_count)
    }

    /// Cost model contract: number of post-filter attempts denied by support checks
    pub fn cost_model_post_filter_supported_denied_count(&self) -> Option<usize> {
        self.load(|inner| &inner.cost_model_post_filter_supported_denied_count)
    }

    /// Cost model contract: number of row groups that started in post-filter mode
    pub fn cost_model_started_with_post_filter_count(&self) -> Option<usize> {
        self.load(|inner| &inner.cost_model_started_with_post_filter_count)
    }

    /// Cost model contract: number of adaptive decisions that switched to post-filter
    pub fn cost_model_adaptive_switched_to_post_filter_count(&self) -> Option<usize> {
        self.load(|inner| &inner.cost_model_adaptive_switched_to_post_filter_count)
    }

    /// Cost model contract: number of adaptive decisions that kept pushdown
    pub fn cost_model_adaptive_kept_pushdown_count(&self) -> Option<usize> {
        self.load(|inner| &inner.cost_model_adaptive_kept_pushdown_count)
    }

    /// Cost model: number of incomplete observation-window decisions
    pub fn cost_model_observation_incomplete_count(&self) -> Option<usize> {
        self.load(|inner| &inner.cost_model_observation_incomplete_count)
    }

    /// Cost model: number of times pushdown remained preferred
    pub fn cost_model_pushdown_still_preferred_count(&self) -> Option<usize> {
        self.load(|inner| &inner.cost_model_pushdown_still_preferred_count)
    }

    /// Cost model: number of high-selectivity no-pruning triggers
    pub fn cost_model_high_selectivity_no_pruning_count(&self) -> Option<usize> {
        self.load(|inner| &inner.cost_model_high_selectivity_no_pruning_count)
    }

    /// Cost model: number of low-selectivity high page-touch triggers
    pub fn cost_model_low_selectivity_high_page_touch_count(&self) -> Option<usize> {
        self.load(|inner| &inner.cost_model_low_selectivity_high_page_touch_count)
    }

    /// Cost model: number of projected-predicate moderate-selectivity triggers
    pub fn cost_model_projected_predicate_moderate_selectivity_count(&self) -> Option<usize> {
        self.load(|inner| &inner.cost_model_projected_predicate_moderate_selectivity_count)
    }

    /// Cost model: number of projected-predicate sparse-fragmented triggers
    pub fn cost_model_projected_predicate_sparse_fragmented_count(&self) -> Option<usize> {
        self.load(|inner| &inner.cost_model_projected_predicate_sparse_fragmented_count)
    }

    /// Cost model: number of fragmented moderate-selectivity triggers
    pub fn cost_model_fragmented_moderate_selectivity_count(&self) -> Option<usize> {
        self.load(|inner| &inner.cost_model_fragmented_moderate_selectivity_count)
    }

    /// Cost model: number of fragmented high-selectivity triggers
    pub fn cost_model_fragmented_high_selectivity_count(&self) -> Option<usize> {
        self.load(|inner| &inner.cost_model_fragmented_high_selectivity_count)
    }

    /// Increments the count of records read from the inner reader
    pub(crate) fn increment_inner_reads(&self, count: usize) {
        let Self::Enabled(inner) = self else {
            return;
        };
        inner
            .records_read_from_inner
            .fetch_add(count, Ordering::Relaxed);
    }

    /// Increments the count of records read from the cache
    pub(crate) fn increment_cache_reads(&self, count: usize) {
        let Self::Enabled(inner) = self else {
            return;
        };

        inner
            .records_read_from_cache
            .fetch_add(count, Ordering::Relaxed);
    }

    pub(crate) fn record_row_selection(&self, decision: RowSelectionStrategyDecision) {
        let Self::Enabled(inner) = self else {
            return;
        };

        let shape = decision.shape;
        inner
            .row_selection_selected_rows
            .fetch_add(shape.selected_rows, Ordering::Relaxed);
        inner
            .row_selection_skipped_rows
            .fetch_add(shape.skipped_rows, Ordering::Relaxed);
        inner
            .row_selection_selector_count
            .fetch_add(shape.selector_count, Ordering::Relaxed);
        inner
            .row_selection_selected_run_count
            .fetch_add(shape.selected_run_count, Ordering::Relaxed);
        inner
            .row_selection_skipped_run_count
            .fetch_add(shape.skipped_run_count, Ordering::Relaxed);

        let strategy_count = if decision.uses_mask() {
            &inner.row_selection_mask_plan_count
        } else {
            &inner.row_selection_selector_plan_count
        };
        strategy_count.fetch_add(1, Ordering::Relaxed);

        let decision_count = match decision.reason {
            RowSelectionStrategyReason::ForcedMask => &inner.row_selection_forced_mask_plan_count,
            RowSelectionStrategyReason::ForcedSelectors => {
                &inner.row_selection_forced_selector_plan_count
            }
            RowSelectionStrategyReason::AutoMaskEmptySelection => {
                &inner.row_selection_auto_mask_empty_plan_count
            }
            RowSelectionStrategyReason::AutoMaskShortRuns => {
                &inner.row_selection_auto_mask_short_run_plan_count
            }
            RowSelectionStrategyReason::AutoMaskFragmentedSelection => {
                &inner.row_selection_auto_mask_fragmented_plan_count
            }
            RowSelectionStrategyReason::AutoMaskHighSelectedRatio => {
                &inner.row_selection_auto_mask_high_ratio_plan_count
            }
            RowSelectionStrategyReason::AutoSelectorClusteredSelection => {
                &inner.row_selection_auto_selector_clustered_plan_count
            }
            RowSelectionStrategyReason::AutoSelectorLongRuns => {
                &inner.row_selection_auto_selector_long_run_plan_count
            }
        };
        decision_count.fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn record_row_selection_output_page_touch(
        &self,
        pages_touched: usize,
        pages_total: usize,
        bytes_touched: usize,
        bytes_total: usize,
    ) {
        let Self::Enabled(inner) = self else {
            return;
        };

        if pages_total == 0 {
            return;
        }

        inner
            .row_selection_output_page_touch_available
            .store(1, Ordering::Relaxed);
        inner
            .row_selection_output_pages_touched
            .fetch_add(pages_touched, Ordering::Relaxed);
        inner
            .row_selection_output_pages_total
            .fetch_add(pages_total, Ordering::Relaxed);
        inner
            .row_selection_output_page_bytes_touched
            .fetch_add(bytes_touched, Ordering::Relaxed);
        inner
            .row_selection_output_page_bytes_total
            .fetch_add(bytes_total, Ordering::Relaxed);
    }

    pub(crate) fn record_row_selection_fetch_ranges(
        &self,
        sparse_range_count: usize,
        sparse_range_bytes: usize,
        dense_fetch_fallback_count: usize,
        dense_fetch_range_count: usize,
        dense_fetch_range_bytes: usize,
    ) {
        let Self::Enabled(inner) = self else {
            return;
        };

        inner
            .row_selection_sparse_range_count
            .fetch_add(sparse_range_count, Ordering::Relaxed);
        inner
            .row_selection_sparse_range_bytes
            .fetch_add(sparse_range_bytes, Ordering::Relaxed);
        inner
            .row_selection_dense_fetch_fallback_count
            .fetch_add(dense_fetch_fallback_count, Ordering::Relaxed);
        inner
            .row_selection_dense_fetch_range_count
            .fetch_add(dense_fetch_range_count, Ordering::Relaxed);
        inner
            .row_selection_dense_fetch_range_bytes
            .fetch_add(dense_fetch_range_bytes, Ordering::Relaxed);
    }

    pub(crate) fn record_range_planning_fetch_ranges(
        &self,
        planning: ArrowReaderRangePlanning,
        range_count: usize,
        range_bytes: usize,
    ) {
        let Self::Enabled(inner) = self else {
            return;
        };
        if range_count == 0 {
            return;
        }

        let (request_count, range_count_counter, range_bytes_counter) = match planning {
            ArrowReaderRangePlanning::Predicate => (
                &inner.predicate_fetch_request_count,
                &inner.predicate_fetch_range_count,
                &inner.predicate_fetch_range_bytes,
            ),
            ArrowReaderRangePlanning::Output => (
                &inner.output_fetch_request_count,
                &inner.output_fetch_range_count,
                &inner.output_fetch_range_bytes,
            ),
        };

        request_count.fetch_add(1, Ordering::Relaxed);
        range_count_counter.fetch_add(range_count, Ordering::Relaxed);
        range_bytes_counter.fetch_add(range_bytes, Ordering::Relaxed);
        if matches!(planning, ArrowReaderRangePlanning::Predicate) && range_count == 1 {
            inner
                .predicate_single_range_request_count
                .fetch_add(1, Ordering::Relaxed);
        }
    }

    pub(crate) fn record_predicate_evaluate(&self, input_rows: usize, selected_rows: usize) {
        let Self::Enabled(inner) = self else {
            return;
        };

        inner
            .predicate_evaluate_call_count
            .fetch_add(1, Ordering::Relaxed);
        inner
            .predicate_evaluate_input_row_count
            .fetch_add(input_rows, Ordering::Relaxed);
        inner
            .predicate_evaluate_selected_row_count
            .fetch_add(selected_rows, Ordering::Relaxed);
    }

    #[allow(dead_code)]
    pub(crate) fn record_predicate_request_batch(
        &self,
        range_count: usize,
        range_bytes: usize,
        extra_bytes: usize,
    ) {
        let Self::Enabled(inner) = self else {
            return;
        };
        if range_count == 0 {
            return;
        }

        inner
            .predicate_request_batch_count
            .fetch_add(1, Ordering::Relaxed);
        inner
            .predicate_batched_range_count
            .fetch_add(range_count, Ordering::Relaxed);
        inner
            .predicate_batched_range_bytes
            .fetch_add(range_bytes, Ordering::Relaxed);
        inner
            .predicate_request_batch_extra_bytes
            .fetch_add(extra_bytes, Ordering::Relaxed);
    }

    pub(crate) fn record_predicate_request_batch_plan_time(&self, duration: Duration) {
        let Self::Enabled(inner) = self else {
            return;
        };

        inner
            .predicate_request_batch_plan_time_nanos
            .fetch_add(duration_nanos_for_metric(duration), Ordering::Relaxed);
        inner
            .predicate_request_batch_plan_count
            .fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn record_predicate_request_batch_try_push_time(&self, duration: Duration) {
        let Self::Enabled(inner) = self else {
            return;
        };

        inner
            .predicate_request_batch_try_push_time_nanos
            .fetch_add(duration_nanos_for_metric(duration), Ordering::Relaxed);
        inner
            .predicate_request_batch_try_push_count
            .fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn record_predicate_dense_fetch_ranges(
        &self,
        fallback_count: usize,
        range_count: usize,
        range_bytes: usize,
    ) {
        let Self::Enabled(inner) = self else {
            return;
        };

        inner
            .predicate_dense_fetch_fallback_count
            .fetch_add(fallback_count, Ordering::Relaxed);
        inner
            .predicate_dense_fetch_range_count
            .fetch_add(range_count, Ordering::Relaxed);
        inner
            .predicate_dense_fetch_range_bytes
            .fetch_add(range_bytes, Ordering::Relaxed);
    }

    pub(crate) fn record_predicate_dense_fetch_diagnostics(
        &self,
        diagnostics: PredicateDenseFetchDiagnostics,
    ) {
        let Self::Enabled(inner) = self else {
            return;
        };

        inner
            .predicate_dense_fetch_not_predicate_planning_count
            .fetch_add(diagnostics.not_predicate_planning_count, Ordering::Relaxed);
        inner
            .predicate_dense_fetch_env_not_parsed_count
            .fetch_add(diagnostics.env_not_parsed_count, Ordering::Relaxed);
        inner
            .predicate_dense_fetch_below_range_threshold_count
            .fetch_add(diagnostics.below_range_threshold_count, Ordering::Relaxed);
        inner
            .predicate_dense_fetch_no_dense_candidate_count
            .fetch_add(diagnostics.no_dense_candidate_count, Ordering::Relaxed);
        inner
            .predicate_dense_fetch_ratio_guard_failed_count
            .fetch_add(diagnostics.ratio_guard_failed_count, Ordering::Relaxed);
        inner
            .predicate_dense_fetch_extra_bytes_guard_failed_count
            .fetch_add(
                diagnostics.extra_bytes_guard_failed_count,
                Ordering::Relaxed,
            );
        inner
            .predicate_dense_fetch_materialization_unavailable_count
            .fetch_add(
                diagnostics.materialization_unavailable_count,
                Ordering::Relaxed,
            );
    }

    pub(crate) fn record_cost_model_observed_row_group(&self) {
        let Self::Enabled(inner) = self else {
            return;
        };
        inner
            .cost_model_observed_row_group_count
            .fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn record_cost_model_row_group(&self, mode: RowGroupExecutionMode) {
        let Self::Enabled(inner) = self else {
            return;
        };

        let counter = match mode {
            RowGroupExecutionMode::Pushdown(_) => &inner.cost_model_pushdown_row_group_count,
            RowGroupExecutionMode::PostFilter => &inner.cost_model_post_filter_row_group_count,
        };
        counter.fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn record_cost_model_post_filter_supported_denied(&self) {
        let Self::Enabled(inner) = self else {
            return;
        };
        inner
            .cost_model_post_filter_supported_denied_count
            .fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn record_cost_model_started_with_post_filter(&self) {
        let Self::Enabled(inner) = self else {
            return;
        };
        inner
            .cost_model_started_with_post_filter_count
            .fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn record_cost_model_adaptive_switched_to_post_filter(&self) {
        let Self::Enabled(inner) = self else {
            return;
        };
        inner
            .cost_model_adaptive_switched_to_post_filter_count
            .fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn record_cost_model_adaptive_kept_pushdown(&self) {
        let Self::Enabled(inner) = self else {
            return;
        };
        inner
            .cost_model_adaptive_kept_pushdown_count
            .fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn record_cost_model_trigger(&self, reason: CostModelDecisionReason) {
        let Self::Enabled(inner) = self else {
            return;
        };

        let counter = match reason {
            CostModelDecisionReason::HighSelectivityNoPruning => {
                &inner.cost_model_high_selectivity_no_pruning_count
            }
            CostModelDecisionReason::ProjectedPredicateModerateSelectivity => {
                &inner.cost_model_projected_predicate_moderate_selectivity_count
            }
            CostModelDecisionReason::FragmentedModerateSelectivity => {
                &inner.cost_model_fragmented_moderate_selectivity_count
            }
            CostModelDecisionReason::FragmentedHighSelectivity => {
                &inner.cost_model_fragmented_high_selectivity_count
            }
            CostModelDecisionReason::ObservationIncomplete => {
                &inner.cost_model_observation_incomplete_count
            }
            CostModelDecisionReason::PushdownStillPreferred => {
                &inner.cost_model_pushdown_still_preferred_count
            }
        };
        counter.fetch_add(1, Ordering::Relaxed);
    }

    #[inline]
    pub(crate) fn time_phase<T>(&self, phase: ArrowReaderPhase, f: impl FnOnce() -> T) -> T {
        let Self::Enabled(inner) = self else {
            return f();
        };
        if !inner.phase_profile_enabled {
            return f();
        }

        let start = Instant::now();
        let result = f();
        inner.record_phase(phase, start.elapsed());
        result
    }

    /// Returns `(phase_name, total_nanoseconds, count)` for every reader phase.
    pub fn phase_profile(&self) -> Option<Vec<(&'static str, u64, usize)>> {
        let Self::Enabled(inner) = self else {
            return None;
        };
        if !inner.phase_profile_enabled {
            return None;
        }

        Some(
            ArrowReaderPhase::ALL
                .iter()
                .map(|phase| {
                    let idx = phase.index();
                    (
                        phase.name(),
                        inner.phase_ns[idx].load(Ordering::Relaxed),
                        inner.phase_counts[idx].load(Ordering::Relaxed),
                    )
                })
                .collect(),
        )
    }

    #[cfg(all(test, feature = "async"))]
    pub(crate) fn phase_profile_report(&self) -> Option<String> {
        let Self::Enabled(inner) = self else {
            return None;
        };
        if !inner.phase_profile_enabled {
            return None;
        }

        let mut lines = vec!["phase,total_ms,count,avg_us".to_string()];
        for phase in ArrowReaderPhase::ALL {
            let idx = phase.index();
            let total_ns = inner.phase_ns[idx].load(Ordering::Relaxed);
            let count = inner.phase_counts[idx].load(Ordering::Relaxed);
            if count == 0 {
                continue;
            }

            let total_ms = total_ns as f64 / 1_000_000.0;
            let avg_us = total_ns as f64 / count as f64 / 1_000.0;
            lines.push(format!(
                "{},{total_ms:.3},{count},{avg_us:.3}",
                phase.name()
            ));
        }
        Some(lines.join("\n"))
    }

    fn load(&self, metric: fn(&ArrowReaderMetricsInner) -> &AtomicUsize) -> Option<usize> {
        match self {
            Self::Disabled => None,
            Self::Enabled(inner) => Some(metric(inner).load(Ordering::Relaxed)),
        }
    }

    fn load_u64(&self, metric: fn(&ArrowReaderMetricsInner) -> &AtomicU64) -> Option<u64> {
        match self {
            Self::Disabled => None,
            Self::Enabled(inner) => Some(metric(inner).load(Ordering::Relaxed)),
        }
    }
}

fn duration_nanos_for_metric(duration: Duration) -> u64 {
    u64::try_from(duration.as_nanos())
        .unwrap_or(u64::MAX)
        .max(1)
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
    /// Total selected rows in planned row selections
    row_selection_selected_rows: AtomicUsize,
    /// Total skipped rows in planned row selections
    row_selection_skipped_rows: AtomicUsize,
    /// Total non-empty selectors in planned row selections
    row_selection_selector_count: AtomicUsize,
    /// Total selected runs in planned row selections
    row_selection_selected_run_count: AtomicUsize,
    /// Total skipped runs in planned row selections
    row_selection_skipped_run_count: AtomicUsize,
    /// Output pages touched by row selections during cost-model observation
    row_selection_output_pages_touched: AtomicUsize,
    /// Output pages available during cost-model observation
    row_selection_output_pages_total: AtomicUsize,
    /// Whether output page touch metrics were observable
    row_selection_output_page_touch_available: AtomicUsize,
    /// Compressed bytes of output pages touched during row-selection observation
    row_selection_output_page_bytes_touched: AtomicUsize,
    /// Compressed bytes of output pages available during row-selection observation
    row_selection_output_page_bytes_total: AtomicUsize,
    /// Sparse selected fetch range count before any dense fallback
    row_selection_sparse_range_count: AtomicUsize,
    /// Sparse selected fetch range bytes before any dense fallback
    row_selection_sparse_range_bytes: AtomicUsize,
    /// Number of dense fetch fallbacks applied
    row_selection_dense_fetch_fallback_count: AtomicUsize,
    /// Dense fetch range count after fallback
    row_selection_dense_fetch_range_count: AtomicUsize,
    /// Dense fetch range bytes after fallback
    row_selection_dense_fetch_range_bytes: AtomicUsize,
    /// Non-empty predicate range-planning fetch requests
    predicate_fetch_request_count: AtomicUsize,
    /// Predicate range-planning fetch requests with exactly one range
    predicate_single_range_request_count: AtomicUsize,
    /// Predicate range-planning fetch range count
    predicate_fetch_range_count: AtomicUsize,
    /// Predicate range-planning fetch range bytes
    predicate_fetch_range_bytes: AtomicUsize,
    /// Predicate evaluation call count
    predicate_evaluate_call_count: AtomicUsize,
    /// Predicate evaluation input rows
    predicate_evaluate_input_row_count: AtomicUsize,
    /// Predicate evaluation selected rows
    predicate_evaluate_selected_row_count: AtomicUsize,
    /// Number of predicate dense fetch fallbacks applied
    predicate_dense_fetch_fallback_count: AtomicUsize,
    /// Predicate dense fetch range count after fallback
    predicate_dense_fetch_range_count: AtomicUsize,
    /// Predicate dense fetch range bytes after fallback
    predicate_dense_fetch_range_bytes: AtomicUsize,
    /// Predicate request batches emitted
    predicate_request_batch_count: AtomicUsize,
    /// Predicate fetch ranges emitted by request batches
    predicate_batched_range_count: AtomicUsize,
    /// Predicate fetch range bytes emitted by request batches
    predicate_batched_range_bytes: AtomicUsize,
    /// Bytes fetched beyond original predicate requests by request batches
    predicate_request_batch_extra_bytes: AtomicUsize,
    /// Nanoseconds spent planning future row-group predicate requests for batching
    predicate_request_batch_plan_time_nanos: AtomicU64,
    /// Count of future row-group predicate request planning calls for batching
    predicate_request_batch_plan_count: AtomicUsize,
    /// Nanoseconds spent pushing candidate requests into predicate request batches
    predicate_request_batch_try_push_time_nanos: AtomicU64,
    /// Count of predicate request batch try-push calls
    predicate_request_batch_try_push_count: AtomicUsize,
    /// Predicate dense fetch skipped because request was not predicate-planning
    predicate_dense_fetch_not_predicate_planning_count: AtomicUsize,
    /// Predicate dense fetch skipped because env/config was not parsed
    predicate_dense_fetch_env_not_parsed_count: AtomicUsize,
    /// Predicate dense fetch skipped because range count was under threshold
    predicate_dense_fetch_below_range_threshold_count: AtomicUsize,
    /// Predicate dense fetch skipped because no denser candidate existed
    predicate_dense_fetch_no_dense_candidate_count: AtomicUsize,
    /// Predicate dense fetch skipped because the ratio guard failed
    predicate_dense_fetch_ratio_guard_failed_count: AtomicUsize,
    /// Predicate dense fetch skipped because the extra-bytes guard failed
    predicate_dense_fetch_extra_bytes_guard_failed_count: AtomicUsize,
    /// Predicate dense fetch skipped because materialization split was unavailable
    predicate_dense_fetch_materialization_unavailable_count: AtomicUsize,
    /// Non-empty output range-planning fetch requests
    output_fetch_request_count: AtomicUsize,
    /// Output range-planning fetch range count
    output_fetch_range_count: AtomicUsize,
    /// Output range-planning fetch range bytes
    output_fetch_range_bytes: AtomicUsize,
    /// Number of plans materialized with masks
    row_selection_mask_plan_count: AtomicUsize,
    /// Number of plans materialized with selectors
    row_selection_selector_plan_count: AtomicUsize,
    /// Number of plans forced to masks
    row_selection_forced_mask_plan_count: AtomicUsize,
    /// Number of plans forced to selectors
    row_selection_forced_selector_plan_count: AtomicUsize,
    /// Number of Auto plans choosing masks for empty selections
    row_selection_auto_mask_empty_plan_count: AtomicUsize,
    /// Number of Auto plans choosing masks for short runs
    row_selection_auto_mask_short_run_plan_count: AtomicUsize,
    /// Number of Auto plans using masks for fragmented selected rows
    row_selection_auto_mask_fragmented_plan_count: AtomicUsize,
    /// Number of Auto plans using masks for high selected-row ratio
    row_selection_auto_mask_high_ratio_plan_count: AtomicUsize,
    /// Number of Auto plans using selectors for clustered selected rows
    row_selection_auto_selector_clustered_plan_count: AtomicUsize,
    /// Number of Auto plans choosing selectors for long runs
    row_selection_auto_selector_long_run_plan_count: AtomicUsize,
    /// Number of row groups included in cost-model observation
    cost_model_observed_row_group_count: AtomicUsize,
    /// Number of cost-model eligible row groups executed with pushdown
    cost_model_pushdown_row_group_count: AtomicUsize,
    /// Number of row groups executed with post-filter
    cost_model_post_filter_row_group_count: AtomicUsize,
    /// Number of post-filter attempts denied by support checks
    cost_model_post_filter_supported_denied_count: AtomicUsize,
    /// Number of row groups that started directly in post-filter mode
    cost_model_started_with_post_filter_count: AtomicUsize,
    /// Number of adaptive decisions that switched to post-filter
    cost_model_adaptive_switched_to_post_filter_count: AtomicUsize,
    /// Number of adaptive decisions that kept pushdown
    cost_model_adaptive_kept_pushdown_count: AtomicUsize,
    /// Number of incomplete cost-model observations
    cost_model_observation_incomplete_count: AtomicUsize,
    /// Number of cost-model decisions that kept pushdown
    cost_model_pushdown_still_preferred_count: AtomicUsize,
    /// Number of high-selectivity no-pruning cost-model triggers
    cost_model_high_selectivity_no_pruning_count: AtomicUsize,
    /// Number of low-selectivity high page-touch cost-model triggers
    cost_model_low_selectivity_high_page_touch_count: AtomicUsize,
    /// Number of projected-predicate moderate-selectivity cost-model triggers
    cost_model_projected_predicate_moderate_selectivity_count: AtomicUsize,
    /// Number of projected-predicate sparse-fragmented cost-model triggers
    cost_model_projected_predicate_sparse_fragmented_count: AtomicUsize,
    /// Number of fragmented moderate-selectivity cost-model triggers
    cost_model_fragmented_moderate_selectivity_count: AtomicUsize,
    /// Number of fragmented high-selectivity cost-model triggers
    cost_model_fragmented_high_selectivity_count: AtomicUsize,
    phase_profile_enabled: bool,
    phase_ns: [AtomicU64; ArrowReaderPhase::COUNT],
    phase_counts: [AtomicUsize; ArrowReaderPhase::COUNT],
}

impl ArrowReaderMetricsInner {
    /// Creates a new instance of `ArrowReaderMetricsInner`
    pub(crate) fn new(phase_profile_enabled: bool) -> Self {
        Self {
            records_read_from_inner: AtomicUsize::new(0),
            records_read_from_cache: AtomicUsize::new(0),
            row_selection_selected_rows: AtomicUsize::new(0),
            row_selection_skipped_rows: AtomicUsize::new(0),
            row_selection_selector_count: AtomicUsize::new(0),
            row_selection_selected_run_count: AtomicUsize::new(0),
            row_selection_skipped_run_count: AtomicUsize::new(0),
            row_selection_output_pages_touched: AtomicUsize::new(0),
            row_selection_output_pages_total: AtomicUsize::new(0),
            row_selection_output_page_touch_available: AtomicUsize::new(0),
            row_selection_output_page_bytes_touched: AtomicUsize::new(0),
            row_selection_output_page_bytes_total: AtomicUsize::new(0),
            row_selection_sparse_range_count: AtomicUsize::new(0),
            row_selection_sparse_range_bytes: AtomicUsize::new(0),
            row_selection_dense_fetch_fallback_count: AtomicUsize::new(0),
            row_selection_dense_fetch_range_count: AtomicUsize::new(0),
            row_selection_dense_fetch_range_bytes: AtomicUsize::new(0),
            predicate_fetch_request_count: AtomicUsize::new(0),
            predicate_single_range_request_count: AtomicUsize::new(0),
            predicate_fetch_range_count: AtomicUsize::new(0),
            predicate_fetch_range_bytes: AtomicUsize::new(0),
            predicate_evaluate_call_count: AtomicUsize::new(0),
            predicate_evaluate_input_row_count: AtomicUsize::new(0),
            predicate_evaluate_selected_row_count: AtomicUsize::new(0),
            predicate_dense_fetch_fallback_count: AtomicUsize::new(0),
            predicate_dense_fetch_range_count: AtomicUsize::new(0),
            predicate_dense_fetch_range_bytes: AtomicUsize::new(0),
            predicate_request_batch_count: AtomicUsize::new(0),
            predicate_batched_range_count: AtomicUsize::new(0),
            predicate_batched_range_bytes: AtomicUsize::new(0),
            predicate_request_batch_extra_bytes: AtomicUsize::new(0),
            predicate_request_batch_plan_time_nanos: AtomicU64::new(0),
            predicate_request_batch_plan_count: AtomicUsize::new(0),
            predicate_request_batch_try_push_time_nanos: AtomicU64::new(0),
            predicate_request_batch_try_push_count: AtomicUsize::new(0),
            predicate_dense_fetch_not_predicate_planning_count: AtomicUsize::new(0),
            predicate_dense_fetch_env_not_parsed_count: AtomicUsize::new(0),
            predicate_dense_fetch_below_range_threshold_count: AtomicUsize::new(0),
            predicate_dense_fetch_no_dense_candidate_count: AtomicUsize::new(0),
            predicate_dense_fetch_ratio_guard_failed_count: AtomicUsize::new(0),
            predicate_dense_fetch_extra_bytes_guard_failed_count: AtomicUsize::new(0),
            predicate_dense_fetch_materialization_unavailable_count: AtomicUsize::new(0),
            output_fetch_request_count: AtomicUsize::new(0),
            output_fetch_range_count: AtomicUsize::new(0),
            output_fetch_range_bytes: AtomicUsize::new(0),
            row_selection_mask_plan_count: AtomicUsize::new(0),
            row_selection_selector_plan_count: AtomicUsize::new(0),
            row_selection_forced_mask_plan_count: AtomicUsize::new(0),
            row_selection_forced_selector_plan_count: AtomicUsize::new(0),
            row_selection_auto_mask_empty_plan_count: AtomicUsize::new(0),
            row_selection_auto_mask_short_run_plan_count: AtomicUsize::new(0),
            row_selection_auto_mask_fragmented_plan_count: AtomicUsize::new(0),
            row_selection_auto_mask_high_ratio_plan_count: AtomicUsize::new(0),
            row_selection_auto_selector_clustered_plan_count: AtomicUsize::new(0),
            row_selection_auto_selector_long_run_plan_count: AtomicUsize::new(0),
            cost_model_observed_row_group_count: AtomicUsize::new(0),
            cost_model_pushdown_row_group_count: AtomicUsize::new(0),
            cost_model_post_filter_row_group_count: AtomicUsize::new(0),
            cost_model_post_filter_supported_denied_count: AtomicUsize::new(0),
            cost_model_started_with_post_filter_count: AtomicUsize::new(0),
            cost_model_adaptive_switched_to_post_filter_count: AtomicUsize::new(0),
            cost_model_adaptive_kept_pushdown_count: AtomicUsize::new(0),
            cost_model_observation_incomplete_count: AtomicUsize::new(0),
            cost_model_pushdown_still_preferred_count: AtomicUsize::new(0),
            cost_model_high_selectivity_no_pruning_count: AtomicUsize::new(0),
            cost_model_low_selectivity_high_page_touch_count: AtomicUsize::new(0),
            cost_model_projected_predicate_moderate_selectivity_count: AtomicUsize::new(0),
            cost_model_projected_predicate_sparse_fragmented_count: AtomicUsize::new(0),
            cost_model_fragmented_moderate_selectivity_count: AtomicUsize::new(0),
            cost_model_fragmented_high_selectivity_count: AtomicUsize::new(0),
            phase_profile_enabled,
            phase_ns: std::array::from_fn(|_| AtomicU64::new(0)),
            phase_counts: std::array::from_fn(|_| AtomicUsize::new(0)),
        }
    }

    fn record_phase(&self, phase: ArrowReaderPhase, duration: Duration) {
        let idx = phase.index();
        self.phase_ns[idx].fetch_add(duration.as_nanos() as u64, Ordering::Relaxed);
        self.phase_counts[idx].fetch_add(1, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rich_metrics_accessors_report_recorded_values() {
        let metrics = ArrowReaderMetrics::enabled();
        metrics.record_row_selection_output_page_touch(3, 10, 100, 400);
        metrics.record_row_selection_fetch_ranges(7, 700, 1, 2, 2_000);
        metrics.record_range_planning_fetch_ranges(ArrowReaderRangePlanning::Predicate, 1, 100);
        metrics.record_range_planning_fetch_ranges(ArrowReaderRangePlanning::Predicate, 4, 400);
        metrics.record_predicate_request_batch(3, 350, 25);
        metrics.record_predicate_request_batch_plan_time(Duration::from_nanos(10));
        metrics.record_predicate_request_batch_try_push_time(Duration::from_nanos(20));
        metrics.record_predicate_dense_fetch_ranges(1, 1, 550);
        metrics.record_predicate_dense_fetch_diagnostics(PredicateDenseFetchDiagnostics {
            not_predicate_planning_count: 2,
            env_not_parsed_count: 3,
            below_range_threshold_count: 4,
            no_dense_candidate_count: 5,
            ratio_guard_failed_count: 6,
            extra_bytes_guard_failed_count: 7,
            materialization_unavailable_count: 8,
        });
        metrics.record_range_planning_fetch_ranges(ArrowReaderRangePlanning::Output, 2, 200);

        assert_eq!(metrics.row_selection_output_pages_touched(), Some(3));
        assert_eq!(metrics.row_selection_output_pages_total(), Some(10));
        assert_eq!(metrics.row_selection_output_page_touch_available(), Some(1));
        assert_eq!(metrics.row_selection_output_page_bytes_touched(), Some(100));
        assert_eq!(metrics.row_selection_output_page_bytes_total(), Some(400));
        assert_eq!(metrics.row_selection_sparse_range_count(), Some(7));
        assert_eq!(metrics.row_selection_sparse_range_bytes(), Some(700));
        assert_eq!(metrics.row_selection_dense_fetch_fallback_count(), Some(1));
        assert_eq!(metrics.row_selection_dense_fetch_range_count(), Some(2));
        assert_eq!(metrics.row_selection_dense_fetch_range_bytes(), Some(2_000));
        assert_eq!(metrics.predicate_fetch_request_count(), Some(2));
        assert_eq!(metrics.predicate_request_count(), Some(2));
        assert_eq!(metrics.predicate_single_range_request_count(), Some(1));
        assert_eq!(metrics.predicate_fetch_range_count(), Some(5));
        assert_eq!(metrics.predicate_fetch_range_bytes(), Some(500));
        assert_eq!(metrics.predicate_request_batch_count(), Some(1));
        assert_eq!(metrics.predicate_batched_range_count(), Some(3));
        assert_eq!(metrics.predicate_batched_range_bytes(), Some(350));
        assert_eq!(metrics.predicate_request_batch_extra_bytes(), Some(25));
        assert_eq!(metrics.predicate_request_batch_plan_time_nanos(), Some(10));
        assert_eq!(metrics.predicate_request_batch_plan_count(), Some(1));
        assert_eq!(
            metrics.predicate_request_batch_try_push_time_nanos(),
            Some(20)
        );
        assert_eq!(metrics.predicate_request_batch_try_push_count(), Some(1));
        assert_eq!(metrics.predicate_dense_fetch_fallback_count(), Some(1));
        assert_eq!(metrics.predicate_dense_fetch_range_count(), Some(1));
        assert_eq!(metrics.predicate_dense_fetch_range_bytes(), Some(550));
        assert_eq!(
            metrics.predicate_dense_fetch_not_predicate_planning_count(),
            Some(2)
        );
        assert_eq!(
            metrics.predicate_dense_fetch_env_not_parsed_count(),
            Some(3)
        );
        assert_eq!(
            metrics.predicate_dense_fetch_below_range_threshold_count(),
            Some(4)
        );
        assert_eq!(
            metrics.predicate_dense_fetch_no_dense_candidate_count(),
            Some(5)
        );
        assert_eq!(
            metrics.predicate_dense_fetch_ratio_guard_failed_count(),
            Some(6)
        );
        assert_eq!(
            metrics.predicate_dense_fetch_extra_bytes_guard_failed_count(),
            Some(7)
        );
        assert_eq!(
            metrics.predicate_dense_fetch_materialization_unavailable_count(),
            Some(8)
        );
        assert_eq!(metrics.output_fetch_request_count(), Some(1));
        assert_eq!(metrics.output_fetch_range_count(), Some(2));
        assert_eq!(metrics.output_fetch_range_bytes(), Some(200));
        assert_eq!(
            metrics.cost_model_low_selectivity_high_page_touch_count(),
            Some(0)
        );
        assert_eq!(
            metrics.cost_model_projected_predicate_sparse_fragmented_count(),
            Some(0)
        );
    }

    #[test]
    fn cost_model_contract_trace_accessors_report_recorded_values() {
        let metrics = ArrowReaderMetrics::enabled();
        metrics.record_cost_model_observed_row_group();
        metrics.record_cost_model_row_group(RowGroupExecutionMode::PostFilter);
        metrics.record_cost_model_row_group(RowGroupExecutionMode::Pushdown(
            crate::arrow::arrow_reader::selection::RowSelectionStrategy::Mask,
        ));

        for reason in [
            CostModelDecisionReason::HighSelectivityNoPruning,
            CostModelDecisionReason::ProjectedPredicateModerateSelectivity,
            CostModelDecisionReason::FragmentedModerateSelectivity,
            CostModelDecisionReason::FragmentedHighSelectivity,
            CostModelDecisionReason::ObservationIncomplete,
            CostModelDecisionReason::PushdownStillPreferred,
        ] {
            metrics.record_cost_model_trigger(reason);
        }

        metrics.record_cost_model_post_filter_supported_denied();
        metrics.record_cost_model_started_with_post_filter();
        metrics.record_cost_model_adaptive_switched_to_post_filter();
        metrics.record_cost_model_adaptive_kept_pushdown();

        assert_eq!(metrics.cost_model_observed_row_group_count(), Some(1));
        assert_eq!(metrics.cost_model_post_filter_row_group_count(), Some(1));
        assert_eq!(metrics.cost_model_pushdown_row_group_count(), Some(1));
        assert_eq!(
            metrics.cost_model_high_selectivity_no_pruning_count(),
            Some(1)
        );
        assert_eq!(
            metrics.cost_model_projected_predicate_moderate_selectivity_count(),
            Some(1)
        );
        assert_eq!(
            metrics.cost_model_fragmented_moderate_selectivity_count(),
            Some(1)
        );
        assert_eq!(
            metrics.cost_model_fragmented_high_selectivity_count(),
            Some(1)
        );
        assert_eq!(metrics.cost_model_observation_incomplete_count(), Some(1));
        assert_eq!(metrics.cost_model_pushdown_still_preferred_count(), Some(1));
        assert_eq!(
            metrics.cost_model_post_filter_supported_denied_count(),
            Some(1)
        );
        assert_eq!(metrics.cost_model_started_with_post_filter_count(), Some(1));
        assert_eq!(
            metrics.cost_model_adaptive_switched_to_post_filter_count(),
            Some(1)
        );
        assert_eq!(metrics.cost_model_adaptive_kept_pushdown_count(), Some(1));
    }

    #[test]
    fn phase_profile_exports_named_phases() {
        let metrics = ArrowReaderMetrics::enabled_with_phase_profile();
        metrics.time_phase(ArrowReaderPhase::OutputReadRecords, || {});

        let profile = metrics.phase_profile().expect("phase profile enabled");
        assert!(
            profile
                .iter()
                .any(|(name, _, count)| *name == "output_read_records" && *count == 1)
        );
    }
}
