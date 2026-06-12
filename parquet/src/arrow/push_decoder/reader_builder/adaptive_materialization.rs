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

//! Adaptive row filter materialization for push decoder row groups.
//!
//! Adaptive materialization chooses how a filtered scan should materialize
//! rows after evaluating predicates. There are two ways to enter post-filter
//! execution:
//!
//! * a narrow static rule starts there for variable-width predicate columns
//!   that are not already part of the output projection, where building
//!   fragmented pushdown selections is commonly expensive
//! * the first eligible row group runs predicate pushdown, records the actual
//!   `RowSelection` shape, and lets later row groups use post-filter if the
//!   shape suggests pushdown is doing extra work without pruning enough rows.
//!   When predicate columns are already part of the output projection, the
//!   observed selected-row ratio can also choose post-filter without requiring
//!   fragmented selected runs.
//!
//! ```text
//! Start
//!   |
//!   v
//! Observing -- incomplete observation --> Observing
//!   |
//!   +-- pushdown still preferred ------> UsePushdown
//!   |
//!   +-- post-filter preferred + supported --> UsePostFilter
//! ```
//!
//! Adaptive materialization only applies to `Auto`. Explicit `Mask` and
//! `Selectors` are treated as user intent and are not overridden here.

use super::{RowBudget, RowGroupReaderBuilder};
use crate::arrow::ProjectionMask;
use crate::arrow::arrow_reader::RowFilter;
use crate::arrow::arrow_reader::RowSelectionPolicy;
use crate::arrow::arrow_reader::selection::{
    AdaptiveMaterializationObservation, AdaptiveMaterializationReason, RowSelectionShape,
    RowSelectionStrategyDecision,
};
use crate::arrow::push_decoder::reader_builder::projection_profile::ProjectionReadProfile;
use crate::arrow::schema::{ParquetField, ParquetFieldType};

#[derive(Debug)]
pub(super) enum AdaptiveMaterializationState {
    /// Collect row-selection shape from early row groups before choosing a mode.
    Observing {
        observation: AdaptiveMaterializationObservation,
    },
    /// Predicate pushdown remains the execution mode for this reader.
    UsePushdown,
    /// Later row groups should decode once and evaluate predicates after decode.
    UsePostFilter,
}

impl Default for AdaptiveMaterializationState {
    fn default() -> Self {
        Self::Observing {
            observation: AdaptiveMaterializationObservation::default(),
        }
    }
}

#[derive(Debug)]
struct PostFilterProjectionRoles {
    /// Columns required to evaluate all predicates.
    predicate_projection: ProjectionMask,
    /// Columns decoded by post-filter execution.
    read_projection: ProjectionMask,
    /// True when predicate columns are already part of the caller output.
    predicate_already_projected: bool,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
enum StaticPostFilterDecision {
    UsePushdown(StaticPostFilterDecisionReason),
    UsePostFilter(StaticPostFilterDecisionReason),
}

impl StaticPostFilterDecision {
    fn uses_post_filter(self) -> bool {
        matches!(self, Self::UsePostFilter(_))
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
enum StaticPostFilterDecisionReason {
    UnsupportedContext,
    UnsupportedProjection,
    NoStaticBenefit,
    FixedWidthPrefixBeforeVariableWidthPredicate,
    UnprojectedVariableWidthPredicate,
    CheapProjectedFixedWidthRead,
    CacheableProjectedPredicate,
}

impl RowGroupReaderBuilder {
    const CHEAP_FIXED_WIDTH_READ_BYTES_PER_ROW: f64 = 24.0;
    const PROJECTED_PREDICATE_MAX_AVERAGE_SELECTED_RUN_LENGTH: f64 = 10.0;

    pub(super) fn should_use_post_filter_by_adaptive_materialization(
        &self,
        budget: RowBudget,
    ) -> bool {
        matches!(
            self.adaptive_materialization_state,
            AdaptiveMaterializationState::UsePostFilter
        ) && self.adaptive_materialization_context_supported(budget)
    }

    fn adaptive_materialization_context_supported(&self, budget: RowBudget) -> bool {
        // Keep the runtime switch narrow:
        //
        // * `Auto` means the caller allowed the reader to choose.
        // * `limit` and `offset` are applied during row-group planning; moving
        //   predicates after decode changes where short-circuiting can happen.
        // * virtual columns are not read from Parquet pages and need their
        //   existing projection path.
        self.adaptive_materialization_enabled
            && matches!(self.row_selection_policy, RowSelectionPolicy::Auto { .. })
            && budget.is_unbounded()
            && !self.has_virtual_columns()
    }

    pub(super) fn post_filter_read_projection(
        &self,
        filter: &RowFilter,
        budget: RowBudget,
    ) -> Option<ProjectionMask> {
        if !self.should_use_post_filter_by_adaptive_materialization(budget) {
            return None;
        }

        Some(self.post_filter_projection_roles(filter)?.read_projection)
    }

    pub(super) fn post_filter_read_projection_for_filter(
        &self,
        filter: &RowFilter,
        budget: RowBudget,
    ) -> Option<ProjectionMask> {
        if !self.adaptive_materialization_context_supported(budget) {
            return None;
        }

        Some(self.post_filter_projection_roles(filter)?.read_projection)
    }

    pub(super) fn should_start_with_post_filter(
        &self,
        filter: &RowFilter,
        row_group_idx: usize,
        budget: RowBudget,
    ) -> bool {
        self.static_post_filter_decision(filter, row_group_idx, budget)
            .uses_post_filter()
    }

    fn static_post_filter_decision(
        &self,
        filter: &RowFilter,
        row_group_idx: usize,
        budget: RowBudget,
    ) -> StaticPostFilterDecision {
        if !self.adaptive_materialization_context_supported(budget) {
            return StaticPostFilterDecision::UsePushdown(
                StaticPostFilterDecisionReason::UnsupportedContext,
            );
        }

        let Some(projections) = self.post_filter_projection_roles(filter) else {
            return StaticPostFilterDecision::UsePushdown(
                StaticPostFilterDecisionReason::UnsupportedProjection,
            );
        };

        if let Some(decision) = self.static_unprojected_variable_width_predicate_decision(
            filter,
            &projections,
            row_group_idx,
        ) {
            return decision;
        }

        if let Some(decision) =
            self.static_projected_predicate_decision(filter, &projections, row_group_idx)
        {
            return decision;
        }

        StaticPostFilterDecision::UsePushdown(StaticPostFilterDecisionReason::NoStaticBenefit)
    }

    fn static_unprojected_variable_width_predicate_decision(
        &self,
        filter: &RowFilter,
        projections: &PostFilterProjectionRoles,
        row_group_idx: usize,
    ) -> Option<StaticPostFilterDecision> {
        if projections.predicate_already_projected
            || !self.projection_has_variable_width_leaf(
                row_group_idx,
                &projections.predicate_projection,
            )
        {
            return None;
        }

        if self.has_cheap_fixed_width_predicate_prefix_before_first_variable_width_predicate(
            filter,
            row_group_idx,
        ) {
            Some(StaticPostFilterDecision::UsePushdown(
                StaticPostFilterDecisionReason::FixedWidthPrefixBeforeVariableWidthPredicate,
            ))
        } else {
            Some(StaticPostFilterDecision::UsePostFilter(
                StaticPostFilterDecisionReason::UnprojectedVariableWidthPredicate,
            ))
        }
    }

    fn static_projected_predicate_decision(
        &self,
        filter: &RowFilter,
        projections: &PostFilterProjectionRoles,
        row_group_idx: usize,
    ) -> Option<StaticPostFilterDecision> {
        // If predicate columns are already in the output projection, pushdown
        // cannot save a deferred output read for those columns. For cheap
        // fixed-width reads, starting directly with post-filter avoids building
        // a row selection just to decode the same values again.
        //
        // Do not apply this to deferred output columns: sparse predicates can
        // still win by reading only a handful of output values.
        if !projections.predicate_already_projected {
            return None;
        }

        // Cacheable predicate columns need one pushdown row group to reveal
        // whether selection is sparse. Starting post-filter here bypasses the
        // predicate cache before the adaptive model can observe that shape.
        if self.has_cacheable_projected_predicate(filter) {
            return Some(StaticPostFilterDecision::UsePushdown(
                StaticPostFilterDecisionReason::CacheableProjectedPredicate,
            ));
        }

        if self
            .projection_read_profile(row_group_idx, &projections.read_projection)
            .is_cheap_fixed_width_read(Self::CHEAP_FIXED_WIDTH_READ_BYTES_PER_ROW)
        {
            Some(StaticPostFilterDecision::UsePostFilter(
                StaticPostFilterDecisionReason::CheapProjectedFixedWidthRead,
            ))
        } else {
            None
        }
    }

    fn has_cacheable_projected_predicate(&self, filter: &RowFilter) -> bool {
        let Some(cache_projection) = self.compute_cache_projection_inner(filter) else {
            return false;
        };

        let schema = self.metadata.file_metadata().schema_descr();
        (0..schema.num_columns()).any(|leaf_idx| cache_projection.leaf_included(leaf_idx))
    }

    fn post_filter_projection_roles(
        &self,
        filter: &RowFilter,
    ) -> Option<PostFilterProjectionRoles> {
        // Post-filter execution decodes each row once, so it needs both:
        //
        // * output columns, which will be returned to the caller
        // * predicate columns, which are needed to evaluate the RowFilter
        //
        // The final reader projects back to the original output projection
        // after predicate evaluation.
        let predicate_projection = filter.union_projection()?;
        let mut read_projection = self.projection.clone();
        read_projection.union(&predicate_projection);

        if !self.post_filter_supports_batch_projection(&self.projection) {
            return None;
        }

        // The combined read projection may be whole-root even when an individual
        // predicate asks for one nested child that is completed by the output
        // projection. Check every batch projection that `PostFilterState` will
        // materialize, not only their union.
        if !filter
            .predicates()
            .iter()
            .all(|predicate| self.post_filter_supports_batch_projection(predicate.projection()))
        {
            return None;
        }

        if !self.post_filter_supports_batch_projection(&read_projection) {
            return None;
        }

        let predicate_already_projected =
            self.projection_includes_all(&self.projection, &predicate_projection);

        Some(PostFilterProjectionRoles {
            predicate_projection,
            read_projection,
            predicate_already_projected,
        })
    }

    fn post_filter_supports_batch_projection(&self, projection: &ProjectionMask) -> bool {
        // Post-filter projects decoded record batches by top-level Arrow field
        // index. A nested root is safe when it is selected as a whole root:
        // the decoded batch then contains exactly one top-level field for that
        // root and can be projected without recursively trimming children.
        //
        // Partial nested projections, such as `struct.a` without `struct.b`,
        // still need recursive array projection and remain on the pushdown path.
        let schema = self.metadata.file_metadata().schema_descr();
        projection.selects_whole_root_columns(schema)
    }

    fn projection_has_variable_width_leaf(
        &self,
        row_group_idx: usize,
        projection: &ProjectionMask,
    ) -> bool {
        self.projection_read_profile(row_group_idx, projection)
            .has_variable_width_leaf()
    }

    fn has_cheap_fixed_width_predicate_prefix_before_first_variable_width_predicate(
        &self,
        filter: &RowFilter,
        row_group_idx: usize,
    ) -> bool {
        let mut has_cheap_fixed_width_prefix = false;
        for predicate in filter.predicates() {
            let projection = predicate.projection();
            if self.projection_has_variable_width_leaf(row_group_idx, projection) {
                return has_cheap_fixed_width_prefix;
            }

            has_cheap_fixed_width_prefix |=
                self.projection_is_cheap_fixed_width_read(row_group_idx, projection);
        }

        false
    }

    fn projection_is_cheap_fixed_width_read(
        &self,
        row_group_idx: usize,
        projection: &ProjectionMask,
    ) -> bool {
        self.projection_read_profile(row_group_idx, projection)
            .is_cheap_fixed_width_read(Self::CHEAP_FIXED_WIDTH_READ_BYTES_PER_ROW)
    }

    fn projection_read_profile(
        &self,
        row_group_idx: usize,
        projection: &ProjectionMask,
    ) -> ProjectionReadProfile {
        ProjectionReadProfile::from_projection(self.metadata.row_group(row_group_idx), projection)
    }

    fn read_profile_for_leaves(
        &self,
        row_group_idx: usize,
        leaf_included: impl FnMut(usize) -> bool,
    ) -> ProjectionReadProfile {
        ProjectionReadProfile::from_leaves(self.metadata.row_group(row_group_idx), leaf_included)
    }

    fn projection_includes_all(&self, projection: &ProjectionMask, other: &ProjectionMask) -> bool {
        let schema = self.metadata.file_metadata().schema_descr();
        (0..schema.num_columns())
            .all(|leaf_idx| !other.leaf_included(leaf_idx) || projection.leaf_included(leaf_idx))
    }

    pub(super) fn observe_adaptive_materialization_candidate(
        &mut self,
        decision: RowSelectionStrategyDecision,
        row_group_idx: usize,
        row_count: usize,
        budget: RowBudget,
    ) {
        if !matches!(self.row_selection_policy, RowSelectionPolicy::Auto { .. }) {
            return;
        }

        let (observation, shape) = {
            let AdaptiveMaterializationState::Observing { observation } =
                &mut self.adaptive_materialization_state
            else {
                return;
            };

            let mut shape = decision.shape;
            if shape.total_rows() == 0 {
                // `None` selection means the predicate kept the whole row group.
                // Represent it as one selected run so the adaptive materialization can
                // treat "no pruning" as an observed high-selectivity case.
                shape = RowSelectionShape {
                    selected_rows: row_count,
                    skipped_rows: 0,
                    selector_count: 1,
                    selected_run_count: 1,
                    skipped_run_count: 0,
                };
            }

            observation.observed_row_groups += 1;
            observation.shape.add_assign(shape);
            (*observation, shape)
        };
        self.metrics
            .record_adaptive_materialization_observed_row_group();
        self.metrics
            .record_adaptive_materialization_observation_shape(shape);

        let reason = self
            .adaptive_materialization_reason_with_projection_context(observation, row_group_idx);
        if matches!(reason, AdaptiveMaterializationReason::ObservationIncomplete) {
            self.metrics.record_adaptive_materialization_trigger(reason);
            return;
        }

        let prefers_post_filter = observation.prefers_post_filter()
            || matches!(
                reason,
                AdaptiveMaterializationReason::ProjectedPredicateModerateSelectivity
            );
        self.metrics.record_adaptive_materialization_trigger(reason);

        if prefers_post_filter && self.adaptive_materialization_supported(budget) {
            self.adaptive_materialization_state = AdaptiveMaterializationState::UsePostFilter;
        } else {
            self.adaptive_materialization_state = AdaptiveMaterializationState::UsePushdown;
        }
    }

    fn adaptive_materialization_reason_with_projection_context(
        &self,
        observation: AdaptiveMaterializationObservation,
        row_group_idx: usize,
    ) -> AdaptiveMaterializationReason {
        let reason = observation.trigger_reason();
        if !matches!(
            reason,
            AdaptiveMaterializationReason::PushdownStillPreferred
        ) {
            return reason;
        }

        let Some(filter) = self.filter.as_ref() else {
            return reason;
        };
        let Some(predicate_projection) = filter.union_projection() else {
            return reason;
        };

        let selected_ratio = observation.shape.selected_ratio();
        // Projected predicates can reuse decoded predicate values, but sparse
        // or clustered filters can still win with page pruning. Keep this
        // shortcut to moderate selectivity before switching to post-filter.
        //
        // A TPC-DS Q2-shaped projected predicate plus one deferred fixed-width
        // output column still favors post-filter once selectivity is moderate:
        // the saved output decode is smaller than the row-selection and cache
        // overhead. Sparse projected predicates stay below this range.
        if self.projection_includes_all(&self.projection, &predicate_projection)
            && self
                .projected_predicate_deferred_output_is_cheap(row_group_idx, &predicate_projection)
            && Self::projected_predicate_shape_is_fragmented_enough(observation.shape)
            && (AdaptiveMaterializationObservation::PROJECTED_PREDICATE_MIN_RATIO
                ..AdaptiveMaterializationObservation::PROJECTED_PREDICATE_MAX_RATIO)
                .contains(&selected_ratio)
        {
            AdaptiveMaterializationReason::ProjectedPredicateModerateSelectivity
        } else {
            reason
        }
    }

    fn projected_predicate_shape_is_fragmented_enough(shape: RowSelectionShape) -> bool {
        shape.average_selected_run_length()
            <= Self::PROJECTED_PREDICATE_MAX_AVERAGE_SELECTED_RUN_LENGTH
    }

    fn projected_predicate_deferred_output_is_cheap(
        &self,
        row_group_idx: usize,
        predicate_projection: &ProjectionMask,
    ) -> bool {
        let profile = self.read_profile_for_leaves(row_group_idx, |leaf_idx| {
            self.projection.leaf_included(leaf_idx) && !predicate_projection.leaf_included(leaf_idx)
        });
        if profile.row_count() == 0 {
            return true;
        }

        profile.is_cheap_fixed_width_read(Self::CHEAP_FIXED_WIDTH_READ_BYTES_PER_ROW)
    }

    pub(super) fn adaptive_materialization_supported(&self, budget: RowBudget) -> bool {
        let Some(filter) = self.filter.as_ref() else {
            return false;
        };
        self.post_filter_supports_filter(filter, budget)
    }

    fn post_filter_supports_filter(&self, filter: &RowFilter, budget: RowBudget) -> bool {
        self.adaptive_materialization_context_supported(budget)
            && self.post_filter_projection_roles(filter).is_some()
    }

    fn has_virtual_columns(&self) -> bool {
        self.fields
            .as_deref()
            .is_some_and(parquet_field_has_virtual_columns)
    }
}

fn parquet_field_has_virtual_columns(field: &ParquetField) -> bool {
    match &field.field_type {
        ParquetFieldType::Primitive { .. } => false,
        ParquetFieldType::Group { children } => {
            children.iter().any(parquet_field_has_virtual_columns)
        }
        ParquetFieldType::Virtual(_) => true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arrow::ArrowWriter;
    use crate::arrow::arrow_reader::ArrowPredicateFn;
    use crate::arrow::arrow_reader::metrics::ArrowReaderMetrics;
    use crate::file::metadata::{ParquetMetaData, ParquetMetaDataReader};
    use crate::file::properties::WriterProperties;
    use crate::util::push_buffers::PushBuffers;
    use arrow_array::{ArrayRef, Int64Array, RecordBatch, StringViewArray};
    use bytes::Bytes;
    use std::sync::Arc;

    #[test]
    fn static_decision_keeps_pushdown_when_fixed_width_prefix_precedes_variable_width() {
        let builder = fixed_prefix_builder(["a", "c"], ["b"]);
        let filter = row_filter(&builder, ["a", "c"]);

        assert_eq!(
            builder.static_post_filter_decision(&filter, 0, RowBudget::new(None, None)),
            StaticPostFilterDecision::UsePushdown(
                StaticPostFilterDecisionReason::FixedWidthPrefixBeforeVariableWidthPredicate
            )
        );
    }

    #[test]
    fn static_decision_starts_post_filter_when_variable_width_predicate_is_first() {
        let builder = fixed_prefix_builder(["c", "a"], ["b"]);
        let filter = row_filter(&builder, ["c", "a"]);

        assert_eq!(
            builder.static_post_filter_decision(&filter, 0, RowBudget::new(None, None)),
            StaticPostFilterDecision::UsePostFilter(
                StaticPostFilterDecisionReason::UnprojectedVariableWidthPredicate
            )
        );
    }

    #[test]
    fn static_decision_keeps_pushdown_without_variable_width_predicate() {
        let builder = fixed_prefix_builder(["a"], ["b"]);
        let filter = row_filter(&builder, ["a"]);

        assert_eq!(
            builder.static_post_filter_decision(&filter, 0, RowBudget::new(None, None)),
            StaticPostFilterDecision::UsePushdown(StaticPostFilterDecisionReason::NoStaticBenefit)
        );
    }

    #[test]
    fn static_decision_starts_post_filter_without_fixed_width_prefix() {
        let builder = fixed_prefix_builder(["c"], ["b"]);
        let filter = row_filter(&builder, ["c"]);

        assert_eq!(
            builder.static_post_filter_decision(&filter, 0, RowBudget::new(None, None)),
            StaticPostFilterDecision::UsePostFilter(
                StaticPostFilterDecisionReason::UnprojectedVariableWidthPredicate
            )
        );
    }

    #[test]
    fn static_decision_starts_post_filter_for_cheap_projected_fixed_width_read() {
        let builder = fixed_prefix_builder(["a"], ["a"]);
        let filter = row_filter(&builder, ["a"]);

        assert_eq!(
            builder.static_post_filter_decision(&filter, 0, RowBudget::new(None, None)),
            StaticPostFilterDecision::UsePostFilter(
                StaticPostFilterDecisionReason::CheapProjectedFixedWidthRead
            )
        );
    }

    #[test]
    fn static_decision_keeps_pushdown_for_cacheable_projected_predicate() {
        let builder = fixed_prefix_builder_with_cache(["a"], ["a"], 1024);
        let filter = row_filter(&builder, ["a"]);

        assert_eq!(
            builder.static_post_filter_decision(&filter, 0, RowBudget::new(None, None)),
            StaticPostFilterDecision::UsePushdown(
                StaticPostFilterDecisionReason::CacheableProjectedPredicate
            )
        );
    }

    #[test]
    fn static_decision_keeps_pushdown_for_empty_filter() {
        let builder = fixed_prefix_builder(["a"], ["b"]);
        let filter = RowFilter::new(vec![]);

        assert_eq!(
            builder.static_post_filter_decision(&filter, 0, RowBudget::new(None, None)),
            StaticPostFilterDecision::UsePushdown(
                StaticPostFilterDecisionReason::UnsupportedProjection
            )
        );
    }

    #[test]
    fn static_decision_keeps_pushdown_when_limit_is_present() {
        let builder = fixed_prefix_builder(["c"], ["b"]);
        let filter = row_filter(&builder, ["c"]);

        assert_eq!(
            builder.static_post_filter_decision(&filter, 0, RowBudget::new(None, Some(10))),
            StaticPostFilterDecision::UsePushdown(
                StaticPostFilterDecisionReason::UnsupportedContext
            )
        );
    }

    fn fixed_prefix_builder<const P: usize, const O: usize>(
        predicate_columns: [&str; P],
        output_columns: [&str; O],
    ) -> RowGroupReaderBuilder {
        fixed_prefix_builder_with_cache(predicate_columns, output_columns, 0)
    }

    fn fixed_prefix_builder_with_cache<const P: usize, const O: usize>(
        predicate_columns: [&str; P],
        output_columns: [&str; O],
        max_predicate_cache_size: usize,
    ) -> RowGroupReaderBuilder {
        let metadata = fixed_prefix_metadata();
        let schema_descr = metadata.file_metadata().schema_descr_ptr();
        let projection = ProjectionMask::columns(&schema_descr, output_columns);
        let filter = row_filter_for_schema(&schema_descr, predicate_columns);

        RowGroupReaderBuilder::new(
            100,
            projection,
            metadata,
            None,
            Some(filter),
            ArrowReaderMetrics::disabled(),
            max_predicate_cache_size,
            PushBuffers::default(),
            RowSelectionPolicy::Auto { threshold: 32 },
        )
    }

    fn row_filter<const N: usize>(
        builder: &RowGroupReaderBuilder,
        columns: [&str; N],
    ) -> RowFilter {
        let schema_descr = builder.metadata.file_metadata().schema_descr_ptr();
        row_filter_for_schema(&schema_descr, columns)
    }

    fn row_filter_for_schema<const N: usize>(
        schema_descr: &crate::schema::types::SchemaDescPtr,
        columns: [&str; N],
    ) -> RowFilter {
        RowFilter::new(
            columns
                .into_iter()
                .map(|column| {
                    let projection = ProjectionMask::columns(schema_descr, [column]);
                    Box::new(ArrowPredicateFn::new(projection, |batch| {
                        Ok(arrow_array::BooleanArray::from(vec![
                            true;
                            batch.num_rows()
                        ]))
                    })) as Box<dyn crate::arrow::arrow_reader::ArrowPredicate>
                })
                .collect(),
        )
    }

    fn fixed_prefix_metadata() -> Arc<ParquetMetaData> {
        let a: ArrayRef = Arc::new(Int64Array::from_iter_values(0..200));
        let b: ArrayRef = Arc::new(Int64Array::from_iter_values(200..400));
        let c: ArrayRef = Arc::new(StringViewArray::from_iter_values(
            (0..200).map(|idx| format!("string_{idx}")),
        ));
        let batch = RecordBatch::try_from_iter(vec![("a", a), ("b", b), ("c", c)]).unwrap();

        let writer_options = WriterProperties::builder()
            .set_max_row_group_row_count(Some(100))
            .build();
        let mut parquet_data = Vec::new();
        let mut writer =
            ArrowWriter::try_new(&mut parquet_data, batch.schema(), Some(writer_options)).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();

        let data = Bytes::from(parquet_data);
        let mut reader = ParquetMetaDataReader::new();
        reader.try_parse(&data).unwrap();
        Arc::new(reader.finish().unwrap())
    }
}
