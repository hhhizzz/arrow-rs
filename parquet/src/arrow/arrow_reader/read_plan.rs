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

//! [`ReadPlan`] and [`ReadPlanBuilder`] for determining which rows to read
//! from a Parquet file

use crate::arrow::array_reader::ArrayReader;
use crate::arrow::arrow_reader::selection::RowSelectionPolicy;
use crate::arrow::arrow_reader::selection::RowSelectionStrategy;
use crate::arrow::arrow_reader::{
    ArrowPredicate, ParquetRecordBatchReader, RowSelection, RowSelectionCursor, RowSelector,
};
use crate::errors::{ParquetError, Result};
use arrow_array::{Array, BooleanArray};
use arrow_buffer::{BooleanBuffer, BooleanBufferBuilder};
use arrow_select::filter::{SlicesIterator, prep_null_mask_filter};
use std::collections::VecDeque;

/// Options for [`ReadPlanBuilder::with_predicate_options`].
pub struct PredicateOptions<'a> {
    array_reader: Box<dyn ArrayReader>,
    predicate: &'a mut dyn ArrowPredicate,
    limit: Option<usize>,
    total_rows: usize,
}

impl<'a> PredicateOptions<'a> {
    /// Create options for evaluating `predicate` against rows produced by
    /// `array_reader`.
    ///
    /// By default there is no match-count limit; the predicate is evaluated
    /// over every row the reader yields. Use [`Self::with_limit`] to enable
    /// early termination.
    pub fn new(array_reader: Box<dyn ArrayReader>, predicate: &'a mut dyn ArrowPredicate) -> Self {
        Self {
            array_reader,
            predicate,
            limit: None,
            total_rows: 0,
        }
    }

    /// Stop scanning `array_reader` once `limit` matches have accumulated.
    ///
    /// Performance optimization for `LIMIT` / TopK: when the cumulative
    /// `true_count` reaches `limit`, the current filter batch is truncated
    /// at the `limit`-th match and remaining batches are never decoded.
    ///
    /// `limit` counts predicate matches, not output rows — callers applying
    /// an offset must pass `offset + limit`.
    ///
    /// `total_rows` is the row count `array_reader` would yield if iterated
    /// to completion. It is used to pad un-evaluated trailing rows as "not
    /// selected" so the returned [`RowSelection`] covers the full row group.
    ///
    /// Only valid for the *last* predicate in a filter chain: intermediate
    /// predicates' match counts do not map 1:1 to output rows.
    pub fn with_limit(mut self, limit: usize, total_rows: usize) -> Self {
        self.limit = Some(limit);
        self.total_rows = total_rows;
        self
    }
}

#[derive(Clone, Debug)]
struct PredicateMask {
    filters: Vec<BooleanArray>,
    selected_rows: usize,
    total_rows: usize,
}

impl PredicateMask {
    fn new(filters: Vec<BooleanArray>) -> Self {
        let selected_rows = filters.iter().map(|filter| filter.true_count()).sum();
        let total_rows = filters.iter().map(|filter| filter.len()).sum();
        Self {
            filters,
            selected_rows,
            total_rows,
        }
    }

    fn selects_any(&self) -> bool {
        self.selected_rows != 0
    }

    fn row_count(&self) -> usize {
        self.selected_rows
    }

    fn into_selection(self) -> RowSelection {
        RowSelection::from_filters(&self.filters)
    }

    fn into_filters(self) -> Vec<BooleanArray> {
        self.filters
    }

    fn fast_auto_strategy(&self, threshold: usize) -> Option<RowSelectionStrategy> {
        let skipped_rows = self.total_rows.saturating_sub(self.selected_rows);
        let max_selector_count = self
            .selected_rows
            .min(skipped_rows)
            .saturating_mul(2)
            .saturating_add(1);

        if self.total_rows >= max_selector_count.saturating_mul(threshold) {
            Some(RowSelectionStrategy::Selectors)
        } else {
            None
        }
    }

    fn resolve_auto_strategy(&self, threshold: usize) -> RowSelectionStrategy {
        if let Some(strategy) = self.fast_auto_strategy(threshold) {
            return strategy;
        }

        let selector_count = predicate_filter_selector_count(&self.filters, self.total_rows);
        if selector_count == 0 {
            return RowSelectionStrategy::Mask;
        }
        if self.total_rows < selector_count.saturating_mul(threshold) {
            RowSelectionStrategy::Mask
        } else {
            RowSelectionStrategy::Selectors
        }
    }
}

#[derive(Clone, Debug, Default)]
enum SelectionState {
    #[default]
    All,
    Selection(RowSelection),
    PredicateMask(Box<PredicateMask>),
}

impl SelectionState {
    fn selection(&self) -> Option<&RowSelection> {
        match self {
            Self::Selection(selection) => Some(selection),
            Self::All | Self::PredicateMask(_) => None,
        }
    }

    fn is_all(&self) -> bool {
        matches!(self, Self::All)
    }
}

fn predicate_filter_selector_count(filters: &[BooleanArray], total_rows: usize) -> usize {
    let mut next_offset = 0;
    let mut last_end = 0;
    let mut selector_count = 0;
    let mut has_selected_range = false;

    for filter in filters {
        let offset = next_offset;
        next_offset += filter.len();
        assert_eq!(filter.null_count(), 0);

        for (start, end) in SlicesIterator::new(filter) {
            let start = start + offset;
            let end = end + offset;

            if start == last_end {
                if !has_selected_range {
                    selector_count += 1;
                    has_selected_range = true;
                }
            } else {
                selector_count += 2;
                has_selected_range = true;
            }
            last_end = end;
        }
    }

    if last_end != total_rows {
        selector_count += 1;
    }

    selector_count
}

/// A builder for [`ReadPlan`]
#[derive(Clone, Debug)]
pub struct ReadPlanBuilder {
    batch_size: usize,
    /// Which rows to select. Includes the result of all filters applied so far.
    selection: SelectionState,
    /// Policy to use when materializing the row selection
    row_selection_policy: RowSelectionPolicy,
}

impl ReadPlanBuilder {
    /// Create a `ReadPlanBuilder` with the given batch size
    pub fn new(batch_size: usize) -> Self {
        Self {
            batch_size,
            selection: SelectionState::All,
            row_selection_policy: RowSelectionPolicy::default(),
        }
    }

    /// Set the current selection to the given value
    pub fn with_selection(mut self, selection: Option<RowSelection>) -> Self {
        self.selection = selection
            .map(SelectionState::Selection)
            .unwrap_or(SelectionState::All);
        self
    }

    /// Configure the policy to use when materialising the [`RowSelection`]
    ///
    /// Defaults to [`RowSelectionPolicy::Auto`]
    pub fn with_row_selection_policy(mut self, policy: RowSelectionPolicy) -> Self {
        self.row_selection_policy = policy;
        self
    }

    /// Returns the current row selection policy
    pub fn row_selection_policy(&self) -> &RowSelectionPolicy {
        &self.row_selection_policy
    }

    /// Returns the current selection, if any
    pub fn selection(&self) -> Option<&RowSelection> {
        self.selection.selection()
    }

    /// Specifies the number of rows in the row group, before filtering is applied.
    ///
    /// Returns a [`LimitedReadPlanBuilder`] that can apply
    /// offset and limit.
    ///
    /// Call [`LimitedReadPlanBuilder::build_limited`] to apply the limits to this
    /// selection.
    pub(crate) fn limited(self, row_count: usize) -> LimitedReadPlanBuilder {
        LimitedReadPlanBuilder::new(self, row_count)
    }

    /// Returns true if the current plan selects any rows
    pub fn selects_any(&self) -> bool {
        match &self.selection {
            SelectionState::Selection(selection) => selection.selects_any(),
            SelectionState::PredicateMask(predicate_mask) => predicate_mask.selects_any(),
            SelectionState::All => true,
        }
    }

    /// Returns the number of rows selected, or `None` if all rows are selected.
    pub fn num_rows_selected(&self) -> Option<usize> {
        match &self.selection {
            SelectionState::Selection(selection) => Some(selection.row_count()),
            SelectionState::PredicateMask(predicate_mask) => Some(predicate_mask.row_count()),
            SelectionState::All => None,
        }
    }

    /// Returns the [`RowSelectionStrategy`] for this plan.
    ///
    /// Guarantees to return either `Selectors` or `Mask`, never `Auto`.
    pub(crate) fn resolve_selection_strategy(&self) -> RowSelectionStrategy {
        match self.row_selection_policy {
            RowSelectionPolicy::Selectors => RowSelectionStrategy::Selectors,
            RowSelectionPolicy::Mask => RowSelectionStrategy::Mask,
            RowSelectionPolicy::Auto { threshold, .. } => {
                let selection = match &self.selection {
                    SelectionState::PredicateMask(_) => return RowSelectionStrategy::Mask,
                    SelectionState::Selection(selection) => selection,
                    SelectionState::All => return RowSelectionStrategy::Selectors,
                };

                // total_rows: total number of rows selected / skipped
                // effective_count: number of non-empty selectors
                let (total_rows, effective_count) =
                    selection.iter().fold((0usize, 0usize), |(rows, count), s| {
                        if s.row_count > 0 {
                            (rows + s.row_count, count + 1)
                        } else {
                            (rows, count)
                        }
                    });

                if effective_count == 0 {
                    return RowSelectionStrategy::Mask;
                }

                if total_rows < effective_count.saturating_mul(threshold) {
                    RowSelectionStrategy::Mask
                } else {
                    RowSelectionStrategy::Selectors
                }
            }
        }
    }

    fn materialize_predicate_mask(&mut self) {
        let selection = std::mem::take(&mut self.selection);
        self.selection = match selection {
            SelectionState::PredicateMask(predicate_mask) => {
                SelectionState::Selection((*predicate_mask).into_selection())
            }
            selection => selection,
        };
    }

    fn direct_predicate_mask_enabled(&self, predicate_mask: &PredicateMask) -> bool {
        if std::env::var_os("PARQUET_DISABLE_DIRECT_PREDICATE_MASK").is_some() {
            return false;
        }

        match self.row_selection_policy {
            RowSelectionPolicy::Mask => true,
            RowSelectionPolicy::Auto { threshold, .. } => {
                predicate_mask.resolve_auto_strategy(threshold) == RowSelectionStrategy::Mask
            }
            RowSelectionPolicy::Selectors => false,
        }
    }

    /// Evaluates an [`ArrowPredicate`], updating this plan's `selection`
    ///
    /// If the current `selection` is `Some`, the resulting [`RowSelection`]
    /// will be the conjunction of the existing selection and the rows selected
    /// by `predicate`.
    ///
    /// Note: pre-existing selections may come from evaluating a previous predicate
    /// or if the [`ParquetRecordBatchReader`] specified an explicit
    /// [`RowSelection`] in addition to one or more predicates.
    pub fn with_predicate(
        self,
        array_reader: Box<dyn ArrayReader>,
        predicate: &mut dyn ArrowPredicate,
    ) -> Result<Self> {
        self.with_predicate_options(PredicateOptions::new(array_reader, predicate))
    }

    /// Evaluates an [`ArrowPredicate`] with the given [`PredicateOptions`],
    /// updating this plan's `selection`.
    ///
    /// Like [`Self::with_predicate`], but allows additional options such as a
    /// match-count limit for early termination (see
    /// [`PredicateOptions::with_limit`]).
    pub fn with_predicate_options(mut self, options: PredicateOptions<'_>) -> Result<Self> {
        // Chained predicates still need the existing RowSelection::and_then
        // semantics. Keep this experiment focused on the common single-predicate
        // mask path.
        self.materialize_predicate_mask();

        let PredicateOptions {
            array_reader,
            predicate,
            limit,
            total_rows,
        } = options;

        // Target length for the concatenated filter output:
        // - Prior selection ⇒ the reader yields that many rows; `and_then`
        //   below requires the filter output to match.
        // - No prior selection ⇒ the reader yields `total_rows`. We only
        //   need to pad when `limit` may short-circuit the loop; otherwise
        //   iteration naturally exhausts.
        let expected_rows = match self.selection() {
            Some(s) => Some(s.row_count()),
            None => limit.map(|_| total_rows),
        };

        let reader = ParquetRecordBatchReader::new(array_reader, self.clone().build());
        let mut filters = vec![];
        let mut processed_rows: usize = 0;
        let mut matched_rows: usize = 0;
        for maybe_batch in reader {
            let maybe_batch = maybe_batch?;
            let input_rows = maybe_batch.num_rows();
            let filter = predicate.evaluate(maybe_batch)?;
            // Since user supplied predicate, check error here to catch bugs quickly
            if filter.len() != input_rows {
                return Err(arrow_err!(
                    "ArrowPredicate predicate returned {} rows, expected {input_rows}",
                    filter.len()
                ));
            }
            let filter = match filter.null_count() {
                0 => filter,
                _ => prep_null_mask_filter(&filter),
            };

            processed_rows += input_rows;

            match limit {
                Some(limit) if matched_rows + filter.true_count() >= limit => {
                    let needed = limit - matched_rows;
                    let truncated = truncate_filter_after_n_trues(filter, needed);
                    filters.push(truncated);
                    break;
                }
                _ => {
                    matched_rows += filter.true_count();
                    filters.push(filter);
                }
            }
        }

        // Pad the tail so the filters cover `expected_rows` total. This keeps
        // the invariant that the resulting `RowSelection` spans every row the
        // reader would have produced — rows past the early break are marked
        // "not selected". When no limit is set the loop always exhausts and
        // no padding is needed.
        if let Some(expected) = expected_rows {
            if processed_rows < expected {
                let pad_len = expected - processed_rows;
                filters.push(BooleanArray::new(BooleanBuffer::new_unset(pad_len), None));
            }
        }

        // If the predicate selected all rows and there is no prior selection,
        // skip creating a RowSelection entirely — this avoids the allocation
        // and keeps selection as None which enables coalesced page fetches.
        let all_selected = filters.iter().all(|f| f.true_count() == f.len());
        if all_selected && self.selection.is_all() {
            return Ok(self);
        }

        let predicate_mask = PredicateMask::new(filters);
        let can_keep_predicate_mask = self.selection.is_all()
            && limit.is_none()
            && self.direct_predicate_mask_enabled(&predicate_mask);
        if can_keep_predicate_mask {
            self.selection = SelectionState::PredicateMask(Box::new(predicate_mask));
            return Ok(self);
        }

        let raw = predicate_mask.into_selection();
        self.selection = match std::mem::take(&mut self.selection) {
            SelectionState::Selection(selection) => {
                SelectionState::Selection(selection.and_then(&raw))
            }
            SelectionState::All => SelectionState::Selection(raw),
            SelectionState::PredicateMask(_) => {
                unreachable!("predicate masks are materialized above")
            }
        };
        Ok(self)
    }

    /// Create a final `ReadPlan` the read plan for the scan
    pub fn build(mut self) -> ReadPlan {
        // If selection is empty, truncate
        if !self.selects_any() {
            self.selection = SelectionState::Selection(RowSelection::from(vec![]));
        }

        // Preferred strategy must not be Auto
        let selection_strategy = self.resolve_selection_strategy();

        let Self {
            batch_size,
            selection,
            row_selection_policy: _,
        } = self;

        let row_selection_cursor = match selection {
            SelectionState::Selection(s) => {
                let s = s.trim();
                let selectors: Vec<RowSelector> = s.into();
                match selection_strategy {
                    RowSelectionStrategy::Mask => {
                        RowSelectionCursor::new_mask_from_selectors(selectors)
                    }
                    RowSelectionStrategy::Selectors => RowSelectionCursor::new_selectors(selectors),
                }
            }
            SelectionState::PredicateMask(predicate_mask) => match selection_strategy {
                RowSelectionStrategy::Mask => {
                    RowSelectionCursor::new_mask_from_filters((*predicate_mask).into_filters())
                }
                RowSelectionStrategy::Selectors => {
                    let selectors: Vec<RowSelector> =
                        (*predicate_mask).into_selection().trim().into();
                    RowSelectionCursor::new_selectors(selectors)
                }
            },
            SelectionState::All => RowSelectionCursor::new_all(),
        };

        ReadPlan {
            batch_size,
            row_selection_cursor,
        }
    }
}

/// Builder for [`ReadPlan`] that applies a limit and offset to the read plan
///
/// See [`ReadPlanBuilder::limited`] to create this builder.
pub(crate) struct LimitedReadPlanBuilder {
    /// The underlying builder
    inner: ReadPlanBuilder,
    /// Total number of rows in the row group before the selection, limit or
    /// offset are applied
    row_count: usize,
    /// The offset to apply, if any
    offset: Option<usize>,
    /// The limit to apply, if any
    limit: Option<usize>,
}

impl LimitedReadPlanBuilder {
    /// Create a new `LimitedReadPlanBuilder` from the existing builder and number of rows
    fn new(inner: ReadPlanBuilder, row_count: usize) -> Self {
        Self {
            inner,
            row_count,
            offset: None,
            limit: None,
        }
    }

    /// Set the offset to apply to the read plan
    pub(crate) fn with_offset(mut self, offset: Option<usize>) -> Self {
        self.offset = offset;
        self
    }

    /// Set the limit to apply to the read plan
    pub(crate) fn with_limit(mut self, limit: Option<usize>) -> Self {
        self.limit = limit;
        self
    }

    /// Apply offset and limit, updating the selection on the underlying builder
    /// and returning it.
    pub(crate) fn build_limited(self) -> ReadPlanBuilder {
        let Self {
            mut inner,
            row_count,
            offset,
            limit,
        } = self;

        // Offset and limit are expressed against the materialized selection.
        if offset.is_some() || limit.is_some() {
            inner.materialize_predicate_mask();
        }

        // If the selection is empty, truncate
        if !inner.selects_any() {
            inner.selection = SelectionState::Selection(RowSelection::from(vec![]));
        }

        // If an offset is defined, apply it to the `selection`
        if let Some(offset) = offset {
            let selection = match std::mem::take(&mut inner.selection) {
                SelectionState::Selection(selection) => Some(selection),
                SelectionState::All => None,
                SelectionState::PredicateMask(_) => {
                    unreachable!("predicate masks are materialized above")
                }
            };
            inner.selection = SelectionState::Selection(match row_count.checked_sub(offset) {
                None => RowSelection::from(vec![]),
                Some(remaining) => selection
                    .map(|selection| selection.offset(offset))
                    .unwrap_or_else(|| {
                        RowSelection::from(vec![
                            RowSelector::skip(offset),
                            RowSelector::select(remaining),
                        ])
                    }),
            });
        }

        // If a limit is defined, apply it to the final `selection`
        if let Some(limit) = limit {
            let selection = match std::mem::take(&mut inner.selection) {
                SelectionState::Selection(selection) => Some(selection),
                SelectionState::All => None,
                SelectionState::PredicateMask(_) => {
                    unreachable!("predicate masks are materialized above")
                }
            };
            inner.selection = SelectionState::Selection(
                selection
                    .map(|selection| selection.limit(limit))
                    .unwrap_or_else(|| {
                        RowSelection::from(vec![RowSelector::select(limit.min(row_count))])
                    }),
            );
        }

        inner
    }
}

/// Produce a new `BooleanArray` of the same length as `filter` in which only
/// the first `n` `true` positions from `filter` remain `true`; any `true`
/// positions beyond the first `n` are replaced with `false`.
///
/// `filter` must not contain nulls (callers apply [`prep_null_mask_filter`]
/// first). If `filter` has at most `n` `true` values, a clone is returned.
fn truncate_filter_after_n_trues(filter: BooleanArray, n: usize) -> BooleanArray {
    if filter.true_count() <= n {
        return filter;
    }
    let len = filter.len();
    if n == 0 {
        return BooleanArray::new(BooleanBuffer::new_unset(len), None);
    }
    // `set_indices` scans 64 bits at a time via `trailing_zeros`, so locating
    // the `n`-th set bit is cheaper than visiting every bit. Everything up to
    // and including that position is copied verbatim; the rest is zeroed.
    let values = filter.values();
    let last_kept = values
        .set_indices()
        .nth(n - 1)
        .expect("n - 1 < true_count, checked above");

    let mut builder = BooleanBufferBuilder::new(len);
    builder.append_buffer(&values.slice(0, last_kept + 1));
    builder.append_n(len - last_kept - 1, false);
    BooleanArray::new(builder.finish(), None)
}

/// A plan reading specific rows from a Parquet Row Group.
///
/// See [`ReadPlanBuilder`] to create `ReadPlan`s
#[derive(Debug)]
pub struct ReadPlan {
    /// The number of rows to read in each batch
    batch_size: usize,
    /// Row ranges to be selected from the data source
    row_selection_cursor: RowSelectionCursor,
}

impl ReadPlan {
    /// Returns a mutable reference to the selection selectors, if any
    #[deprecated(since = "57.1.0", note = "Use `row_selection_cursor_mut` instead")]
    pub fn selection_mut(&mut self) -> Option<&mut VecDeque<RowSelector>> {
        if let RowSelectionCursor::Selectors(selectors_cursor) = &mut self.row_selection_cursor {
            Some(selectors_cursor.selectors_mut())
        } else {
            None
        }
    }

    /// Returns a mutable reference to the row selection cursor
    pub fn row_selection_cursor_mut(&mut self) -> &mut RowSelectionCursor {
        &mut self.row_selection_cursor
    }

    /// Return the number of rows to read in each output batch
    #[inline(always)]
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn builder_with_selection(selection: RowSelection) -> ReadPlanBuilder {
        ReadPlanBuilder::new(1024).with_selection(Some(selection))
    }

    #[test]
    fn preferred_selection_strategy_prefers_mask_by_default() {
        let selection = RowSelection::from(vec![RowSelector::select(8)]);
        let builder = builder_with_selection(selection);
        assert_eq!(
            builder.resolve_selection_strategy(),
            RowSelectionStrategy::Mask
        );
    }

    #[test]
    fn preferred_selection_strategy_prefers_selectors_when_threshold_small() {
        let selection = RowSelection::from(vec![RowSelector::select(8)]);
        let builder = builder_with_selection(selection)
            .with_row_selection_policy(RowSelectionPolicy::Auto { threshold: 1 });
        assert_eq!(
            builder.resolve_selection_strategy(),
            RowSelectionStrategy::Selectors
        );
    }

    #[test]
    fn truncate_filter_after_n_trues_keeps_first_n_matches() {
        let f = BooleanArray::from(vec![true, false, true, true, false, true, true]);
        // true positions: 0, 2, 3, 5, 6
        let t = truncate_filter_after_n_trues(f.clone(), 3);
        assert_eq!(t.len(), f.len());
        assert_eq!(t.true_count(), 3);
        let out: Vec<bool> = (0..t.len()).map(|i| t.value(i)).collect();
        assert_eq!(
            out,
            vec![true, false, true, true, false, false, false],
            "first three trues should survive, the rest become false"
        );
    }

    #[test]
    fn truncate_filter_after_n_trues_passes_through_when_already_small_enough() {
        let f = BooleanArray::from(vec![true, false, true, false]);
        let t = truncate_filter_after_n_trues(f.clone(), 5);
        assert_eq!(t.len(), f.len());
        assert_eq!(t.true_count(), 2);
    }

    #[test]
    fn truncate_filter_after_n_trues_zero_returns_all_false() {
        let f = BooleanArray::from(vec![true, true, true]);
        let t = truncate_filter_after_n_trues(f, 0);
        assert_eq!(t.len(), 3);
        assert_eq!(t.true_count(), 0);
    }

    #[test]
    fn with_predicate_options_limit_pads_tail_when_no_prior_selection() {
        use crate::arrow::ProjectionMask;
        use crate::arrow::array_reader::StructArrayReader;
        use crate::arrow::array_reader::test_util::make_int32_page_reader;
        use crate::arrow::arrow_reader::ArrowPredicateFn;
        use arrow_schema::{DataType as ArrowType, Field, Fields};

        // 100 rows, all match the predicate. Limit stops the loop after 10
        // matches — but the resulting RowSelection must still describe the
        // full 100-row row group (90 trailing rows as "not selected"), not
        // only the 10 rows we happened to evaluate before breaking.
        const TOTAL_ROWS: usize = 100;
        const LIMIT: usize = 10;

        let data: Vec<i32> = (0..TOTAL_ROWS as i32).collect();
        let levels = vec![0; TOTAL_ROWS];
        let leaf = make_int32_page_reader(&data, &levels, &levels, 0, 0);
        let struct_type = ArrowType::Struct(Fields::from(vec![Field::new(
            "c0",
            ArrowType::Int32,
            false,
        )]));
        let struct_reader = StructArrayReader::new(struct_type, vec![leaf], 0, 0, false);

        let mut predicate = ArrowPredicateFn::new(ProjectionMask::all(), |batch| {
            Ok(BooleanArray::from(vec![true; batch.num_rows()]))
        });

        let builder = ReadPlanBuilder::new(16)
            .with_predicate_options(
                PredicateOptions::new(Box::new(struct_reader), &mut predicate)
                    .with_limit(LIMIT, TOTAL_ROWS),
            )
            .unwrap();

        let selection = builder
            .selection()
            .expect("limit-driven early break must produce a selection");

        // `row_count` counts selected rows — must equal the limit.
        assert_eq!(selection.row_count(), LIMIT);

        // Total rows covered (selects + skips) must equal the full row group
        // so downstream offset/limit math stays in absolute-row space.
        let total: usize = selection.iter().map(|s| s.row_count).sum();
        assert_eq!(
            total, TOTAL_ROWS,
            "selection must span the full row group, not only the prefix evaluated before the limit"
        );
    }

    #[test]
    fn forced_mask_predicate_keeps_filter_without_row_selection() {
        use crate::arrow::ProjectionMask;
        use crate::arrow::array_reader::StructArrayReader;
        use crate::arrow::array_reader::test_util::make_int32_page_reader;
        use crate::arrow::arrow_reader::ArrowPredicateFn;
        use arrow_schema::{DataType as ArrowType, Field, Fields};

        const TOTAL_ROWS: usize = 16;

        let data: Vec<i32> = (0..TOTAL_ROWS as i32).collect();
        let levels = vec![0; TOTAL_ROWS];
        let leaf = make_int32_page_reader(&data, &levels, &levels, 0, 0);
        let struct_type = ArrowType::Struct(Fields::from(vec![Field::new(
            "c0",
            ArrowType::Int32,
            false,
        )]));
        let struct_reader = StructArrayReader::new(struct_type, vec![leaf], 0, 0, false);

        let mut predicate = ArrowPredicateFn::new(ProjectionMask::all(), |batch| {
            Ok(BooleanArray::from(
                (0..batch.num_rows())
                    .map(|idx| idx % 2 == 0)
                    .collect::<Vec<_>>(),
            ))
        });

        let builder = ReadPlanBuilder::new(8)
            .with_row_selection_policy(RowSelectionPolicy::Mask)
            .with_predicate(Box::new(struct_reader), &mut predicate)
            .unwrap();

        assert!(
            builder.selection().is_none(),
            "forced mask path should keep predicate filters as masks without materializing RowSelection"
        );
        assert_eq!(builder.num_rows_selected(), Some(TOTAL_ROWS / 2));

        let mut plan = builder.build();
        assert!(matches!(
            plan.row_selection_cursor_mut(),
            RowSelectionCursor::Mask(_)
        ));
    }

    #[test]
    fn auto_mask_friendly_predicate_keeps_filter_without_row_selection() {
        use crate::arrow::ProjectionMask;
        use crate::arrow::array_reader::StructArrayReader;
        use crate::arrow::array_reader::test_util::make_int32_page_reader;
        use crate::arrow::arrow_reader::ArrowPredicateFn;
        use arrow_schema::{DataType as ArrowType, Field, Fields};

        const TOTAL_ROWS: usize = 64;

        let data: Vec<i32> = (0..TOTAL_ROWS as i32).collect();
        let levels = vec![0; TOTAL_ROWS];
        let leaf = make_int32_page_reader(&data, &levels, &levels, 0, 0);
        let struct_type = ArrowType::Struct(Fields::from(vec![Field::new(
            "c0",
            ArrowType::Int32,
            false,
        )]));
        let struct_reader = StructArrayReader::new(struct_type, vec![leaf], 0, 0, false);

        let mut predicate = ArrowPredicateFn::new(ProjectionMask::all(), |batch| {
            Ok(BooleanArray::from(
                (0..batch.num_rows())
                    .map(|idx| idx % 2 == 0)
                    .collect::<Vec<_>>(),
            ))
        });

        let builder = ReadPlanBuilder::new(8)
            .with_predicate(Box::new(struct_reader), &mut predicate)
            .unwrap();

        assert!(
            builder.selection().is_none(),
            "Auto should keep mask-friendly predicate filters as masks without materializing RowSelection"
        );
        assert_eq!(builder.num_rows_selected(), Some(TOTAL_ROWS / 2));

        let mut plan = builder.build();
        assert!(matches!(
            plan.row_selection_cursor_mut(),
            RowSelectionCursor::Mask(_)
        ));
    }

    #[test]
    fn predicate_mask_fast_auto_strategy_rules_out_sparse_masks() {
        let filter = BooleanArray::from(
            (0..256)
                .map(|idx| idx == 17 || idx == 211)
                .collect::<Vec<_>>(),
        );
        let predicate_mask = PredicateMask::new(vec![filter]);

        assert_eq!(
            predicate_mask.fast_auto_strategy(32),
            Some(RowSelectionStrategy::Selectors)
        );
    }

    #[test]
    fn predicate_mask_fast_auto_strategy_defers_fragmented_masks() {
        let filter = BooleanArray::from((0..128).map(|idx| idx % 2 == 0).collect::<Vec<_>>());
        let predicate_mask = PredicateMask::new(vec![filter]);

        assert_eq!(predicate_mask.fast_auto_strategy(32), None);
    }
}
