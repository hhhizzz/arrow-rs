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

//! Experimental flat-projection execution with one cursor per output column.
//!
//! The logical [`RowSelection`] remains shared. Its native RLE form is compiled
//! once into selected-row batches, and an optional shared bitmap is retained
//! only when at least one projected column needs decode-then-filter.
//! Unsupported projections never enter this module.

use super::selection::{RowSelectionStrategy, boolean_mask_from_selectors};
use super::{
    ReadPlanBuilder, RowSelection, RowSelectionPolicy, RowSelector, SelectionDecodeCounts,
    consume_record_batch, counted_read_records, counted_skip_records,
};
use crate::arrow::ProjectionMask;
use crate::arrow::array_reader::{ArrayReader, ArrayReaderBuilder, RowGroups};
use crate::arrow::arrow_reader::metrics::{ArrowReaderMetrics, Pc1cAttributionSite};
use crate::arrow::schema::{ParquetField, ParquetFieldType};
use crate::basic::Encoding;
use crate::errors::{ParquetError, Result};
use arrow_array::{ArrayRef, BooleanArray, RecordBatch};
use arrow_buffer::BooleanBuffer;
use arrow_schema::{DataType as ArrowType, FieldRef, Schema, SchemaRef};
use arrow_select::filter::FilterBuilder;
use std::ops::Range;
use std::sync::Arc;

const PURE_DICTIONARY_THRESHOLD: usize = 4;
const DEFAULT_COLUMN_THRESHOLD: usize = 32;

struct ColumnReader {
    reader: Box<dyn ArrayReader>,
    strategy: RowSelectionStrategy,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
struct PhysicalSpan {
    /// Rows between the previous batch's span and this batch's first selected row.
    gap_skip: usize,
    /// Absolute physical-row offset of this batch's first selected row.
    span_start: usize,
    /// Physical rows from the first through the last selected row, inclusive of gaps.
    span_rows: usize,
    /// Logical rows selected into this output batch.
    selected: usize,
}

#[derive(Debug, Clone, Eq, PartialEq)]
struct NativeBatchPlan {
    instruction_range: Range<usize>,
    span: PhysicalSpan,
}

#[derive(Debug)]
struct NativeSelectionPlan {
    instructions: Vec<RowSelector>,
    batches: Vec<NativeBatchPlan>,
    mask: Option<BooleanBuffer>,
    next_batch: usize,
}

/// A record-batch reader for the narrow PC-1 flat-output experiment.
pub(super) enum PerColumnDecision {
    FallbackAuto,
    FallbackForced(RowSelectionStrategy),
    Engaged(PerColumnReader),
}

pub(super) struct PerColumnReader {
    columns: Vec<ColumnReader>,
    schema: SchemaRef,
    batch_size: usize,
    metrics: ArrowReaderMetrics,
    selection_plan: NativeSelectionPlan,
}

impl PerColumnReader {
    /// Build the experimental reader when every hard scope condition holds.
    /// Unsupported shapes and uniform Auto32 decisions return
    /// [`PerColumnDecision::FallbackAuto`]. A uniform dictionary override
    /// returns [`PerColumnDecision::FallbackForced`]. Only a genuine strategy
    /// disagreement constructs the per-column reader.
    pub(super) fn try_new(
        row_groups: &dyn RowGroups,
        metrics: &ArrowReaderMetrics,
        batch_size: usize,
        fields: Option<&ParquetField>,
        projection: &ProjectionMask,
        plan_builder: &ReadPlanBuilder,
    ) -> Result<PerColumnDecision> {
        if !matches!(
            plan_builder.row_selection_policy(),
            RowSelectionPolicy::PerColumn
        ) || batch_size == 0
        {
            return Ok(PerColumnDecision::FallbackAuto);
        }

        let Some(selection) = plan_builder.selection() else {
            // The all-selected fast path must remain exactly the current path.
            return Ok(PerColumnDecision::FallbackAuto);
        };
        if !selection.selects_any() || selection.skipped_row_count() == 0 {
            return Ok(PerColumnDecision::FallbackAuto);
        }

        let Some(fields) = fields else {
            return Ok(PerColumnDecision::FallbackAuto);
        };
        let Some(column_indices) = projected_flat_columns(fields, projection) else {
            return Ok(PerColumnDecision::FallbackAuto);
        };
        if column_indices.is_empty() {
            return Ok(PerColumnDecision::FallbackAuto);
        }

        // Loaded-page ranges alter legal decoder boundaries. Until the native
        // compiler accepts those boundaries directly, keep every constrained
        // plan on the existing Auto32 cursor implementation.
        if plan_builder.has_loaded_row_ranges() {
            return Ok(PerColumnDecision::FallbackAuto);
        }

        let mut row_group_iter = row_groups.row_groups();
        let Some(row_group) = row_group_iter.next() else {
            return Ok(PerColumnDecision::FallbackAuto);
        };
        if row_group_iter.next().is_some() {
            // Per-column thresholds are row-group local. The push decoder
            // naturally constructs one reader per row group; the synchronous
            // multi-row-group path falls back until it has the same boundary.
            return Ok(PerColumnDecision::FallbackAuto);
        }

        let strategies = column_indices
            .iter()
            .map(|&column_idx| {
                let threshold = if is_pure_dictionary(row_group, column_idx) {
                    PURE_DICTIONARY_THRESHOLD
                } else {
                    DEFAULT_COLUMN_THRESHOLD
                };
                selection.auto_selection_strategy(threshold)
            })
            .collect::<Vec<_>>();

        let auto32 = selection.auto_selection_strategy(DEFAULT_COLUMN_THRESHOLD);
        if let Some(strategy) = uniform_strategy(&strategies) {
            return Ok(if strategy == auto32 {
                PerColumnDecision::FallbackAuto
            } else {
                PerColumnDecision::FallbackForced(strategy)
            });
        }

        let schema_descr = row_groups.metadata().file_metadata().schema_descr();
        let reader_build_started = metrics.start_timing();
        let mut columns = Vec::with_capacity(column_indices.len());
        let mut output_fields: Vec<FieldRef> = Vec::with_capacity(column_indices.len());
        for (column_idx, strategy) in column_indices.into_iter().zip(strategies) {
            let column_projection = ProjectionMask::leaves(schema_descr, [column_idx]);
            let reader = ArrayReaderBuilder::new(row_groups, metrics)
                .with_batch_size(batch_size)
                .with_parquet_metadata(row_groups.metadata())
                .build_array_reader(Some(fields), &column_projection)?;
            let one_field = match reader.get_data_type() {
                ArrowType::Struct(fields) if fields.len() == 1 => fields[0].clone(),
                ArrowType::Struct(fields) => {
                    return Err(general_err!(
                        "PerColumn reader for leaf {column_idx} produced {} fields",
                        fields.len()
                    ));
                }
                data_type => {
                    return Err(general_err!(
                        "PerColumn reader for leaf {column_idx} produced non-struct type {data_type}"
                    ));
                }
            };
            output_fields.push(one_field);
            columns.push(ColumnReader { reader, strategy });
        }
        metrics.record_pc1c_attribution(Pc1cAttributionSite::ReaderBuild, reader_build_started);

        let plan_started = metrics.start_timing();
        let selection_plan = NativeSelectionPlan::try_new(
            selection.clone(),
            batch_size,
            columns
                .iter()
                .any(|column| column.strategy == RowSelectionStrategy::Mask),
            metrics,
        )?;
        metrics.record_pc1c_attribution(Pc1cAttributionSite::Window, plan_started);

        let reader = Self {
            columns,
            schema: Arc::new(Schema::new(output_fields)),
            batch_size,
            metrics: metrics.clone(),
            selection_plan,
        };
        Ok(PerColumnDecision::Engaged(reader))
    }

    pub(super) fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    pub(super) fn batch_size(&self) -> usize {
        self.batch_size
    }

    pub(super) fn next_batch(&mut self) -> Result<Option<RecordBatch>> {
        let Some(batch_plan) = self
            .selection_plan
            .batches
            .get(self.selection_plan.next_batch)
        else {
            return Ok(None);
        };
        self.selection_plan.next_batch += 1;

        let selected_rows = batch_plan.span.selected;
        let instructions = &self.selection_plan.instructions[batch_plan.instruction_range.clone()];
        let mask = self
            .selection_plan
            .mask
            .as_ref()
            .map(|mask| mask.slice(batch_plan.span.span_start, batch_plan.span.span_rows));

        let mut arrays = Vec::with_capacity(self.columns.len());
        for column in &mut self.columns {
            let array = match column.strategy {
                RowSelectionStrategy::Selectors => read_selectors_column(
                    column.reader.as_mut(),
                    instructions,
                    batch_plan.span,
                    &self.metrics,
                )?,
                RowSelectionStrategy::Mask => {
                    mask.as_ref().ok_or_else(|| {
                        general_err!("Internal Error: PerColumn mask column has no shared mask")
                    })?;
                    read_mask_column(column.reader.as_mut(), batch_plan.span, &self.metrics)?
                }
            };
            let expected_rows = match column.strategy {
                RowSelectionStrategy::Selectors => selected_rows,
                RowSelectionStrategy::Mask => batch_plan.span.span_rows,
            };
            if array.len() != expected_rows {
                return Err(general_err!(
                    "PerColumn decoded length mismatch: expected {expected_rows}, got {}",
                    array.len()
                ));
            }
            arrays.push(array);
        }

        if let Some(mask) = mask {
            let filter_started = self.metrics.start_timing();
            let filter = BooleanArray::from(mask);
            let predicate = FilterBuilder::new(&filter).optimize().build();
            if predicate.count() != selected_rows {
                return Err(general_err!(
                    "Internal Error: PerColumn shared filter selects {} rows, expected {selected_rows}",
                    predicate.count()
                ));
            }
            for (column, array) in self.columns.iter().zip(&mut arrays) {
                if column.strategy == RowSelectionStrategy::Mask {
                    *array = predicate.filter(array.as_ref())?;
                }
            }
            self.metrics
                .record_pc1c_attribution(Pc1cAttributionSite::Filter, filter_started);
            self.metrics.record_filter_record_batch(filter_started);
        }

        if let Some(array) = arrays.iter().find(|array| array.len() != selected_rows) {
            return Err(general_err!(
                "PerColumn output length mismatch: expected {selected_rows}, got {}",
                array.len()
            ));
        }

        let assembly_started = self.metrics.start_timing();
        let batch = RecordBatch::try_new(Arc::clone(&self.schema), arrays)?;
        self.metrics
            .record_pc1c_attribution(Pc1cAttributionSite::BatchAssembly, assembly_started);
        if batch.num_rows() != selected_rows {
            return Err(general_err!(
                "PerColumn RecordBatch row mismatch: expected {selected_rows}, got {}",
                batch.num_rows()
            ));
        }
        Ok(Some(batch))
    }
}

impl NativeSelectionPlan {
    fn try_new(
        selection: RowSelection,
        batch_size: usize,
        needs_mask: bool,
        metrics: &ArrowReaderMetrics,
    ) -> Result<Self> {
        if batch_size == 0 {
            return Err(general_err!(
                "Internal Error: PerColumn native plan requires a non-zero batch size"
            ));
        }

        let selection = selection.trim();
        if !selection.selects_any() {
            return Err(general_err!(
                "Internal Error: PerColumn native plan requires selected rows"
            ));
        }
        let source = selection.iter().copied().collect::<Vec<_>>();
        let mask = if needs_mask {
            Some(match selection.as_mask() {
                Some(mask) => mask.clone(),
                None => {
                    let started = metrics.start_general_timing();
                    let mask = boolean_mask_from_selectors(&source);
                    metrics.record_selectors_to_mask(started);
                    mask
                }
            })
        } else {
            None
        };

        let mut instructions = Vec::with_capacity(source.len());
        let mut batches = Vec::new();
        let mut physical_position = 0usize;
        let mut pending_gap_skip = 0usize;
        let mut batch_instruction_start = 0usize;
        let mut batch_gap_skip = 0usize;
        let mut batch_span_start = 0usize;
        let mut batch_span_rows = 0usize;
        let mut batch_selected = 0usize;

        for selector in source {
            if selector.row_count == 0 {
                continue;
            }

            if selector.skip {
                physical_position = checked_add(physical_position, selector.row_count)?;
                if batch_selected == 0 {
                    pending_gap_skip = checked_add(pending_gap_skip, selector.row_count)?;
                } else {
                    append_instruction(&mut instructions, batch_instruction_start, selector)?;
                    batch_span_rows = checked_add(batch_span_rows, selector.row_count)?;
                }
                continue;
            }

            let mut remaining = selector.row_count;
            while remaining != 0 {
                if batch_selected == 0 {
                    batch_instruction_start = instructions.len();
                    batch_gap_skip = pending_gap_skip;
                    pending_gap_skip = 0;
                    batch_span_start = physical_position;
                    batch_span_rows = 0;
                }

                let take = remaining.min(batch_size - batch_selected);
                append_instruction(
                    &mut instructions,
                    batch_instruction_start,
                    RowSelector::select(take),
                )?;
                physical_position = checked_add(physical_position, take)?;
                batch_span_rows = checked_add(batch_span_rows, take)?;
                batch_selected += take;
                remaining -= take;

                if batch_selected == batch_size {
                    push_batch(
                        &mut batches,
                        batch_instruction_start..instructions.len(),
                        PhysicalSpan {
                            gap_skip: batch_gap_skip,
                            span_start: batch_span_start,
                            span_rows: batch_span_rows,
                            selected: batch_selected,
                        },
                    )?;
                    batch_selected = 0;
                }
            }
        }

        if batch_selected != 0 {
            push_batch(
                &mut batches,
                batch_instruction_start..instructions.len(),
                PhysicalSpan {
                    gap_skip: batch_gap_skip,
                    span_start: batch_span_start,
                    span_rows: batch_span_rows,
                    selected: batch_selected,
                },
            )?;
        }

        if batches.is_empty() || pending_gap_skip != 0 {
            return Err(general_err!(
                "Internal Error: PerColumn native plan produced an invalid empty or trailing-gap plan"
            ));
        }
        if let Some(mask) = &mask {
            if mask.len() != physical_position {
                return Err(general_err!(
                    "Internal Error: PerColumn shared mask has {} rows, expected {physical_position}",
                    mask.len()
                ));
            }
        }

        Ok(Self {
            instructions,
            batches,
            mask,
            next_batch: 0,
        })
    }
}

fn checked_add(left: usize, right: usize) -> Result<usize> {
    left.checked_add(right)
        .ok_or_else(|| general_err!("Internal Error: PerColumn row count overflow"))
}

fn append_instruction(
    instructions: &mut Vec<RowSelector>,
    batch_start: usize,
    selector: RowSelector,
) -> Result<()> {
    if instructions.len() > batch_start {
        let last = instructions
            .last_mut()
            .expect("non-empty batch instructions");
        if last.skip == selector.skip {
            last.row_count = checked_add(last.row_count, selector.row_count)?;
            return Ok(());
        }
    }
    instructions.push(selector);
    Ok(())
}

fn push_batch(
    batches: &mut Vec<NativeBatchPlan>,
    instruction_range: Range<usize>,
    span: PhysicalSpan,
) -> Result<()> {
    if span.selected == 0 || span.span_rows == 0 || instruction_range.is_empty() {
        return Err(general_err!(
            "Internal Error: PerColumn native plan produced an empty batch"
        ));
    }
    batches.push(NativeBatchPlan {
        instruction_range,
        span,
    });
    Ok(())
}

fn uniform_strategy(strategies: &[RowSelectionStrategy]) -> Option<RowSelectionStrategy> {
    let first = *strategies.first()?;
    strategies
        .iter()
        .all(|strategy| *strategy == first)
        .then_some(first)
}

fn projected_flat_columns(
    fields: &ParquetField,
    projection: &ProjectionMask,
) -> Option<Vec<usize>> {
    let ParquetFieldType::Group { children } = &fields.field_type else {
        return None;
    };
    let mut result = Vec::new();
    for child in children {
        if !field_is_projected(child, projection) {
            continue;
        }
        match &child.field_type {
            ParquetFieldType::Primitive { col_idx, .. } => result.push(*col_idx),
            // Nested and virtual output are explicitly outside PC-1.
            ParquetFieldType::Group { .. } | ParquetFieldType::Virtual(_) => return None,
        }
    }
    Some(result)
}

fn field_is_projected(field: &ParquetField, projection: &ProjectionMask) -> bool {
    match &field.field_type {
        ParquetFieldType::Primitive { col_idx, .. } => projection.leaf_included(*col_idx),
        ParquetFieldType::Group { children } => children
            .iter()
            .any(|child| field_is_projected(child, projection)),
        // Virtual fields have no physical leaf in ProjectionMask. If one was
        // requested into FieldLevels, conservatively force fallback.
        ParquetFieldType::Virtual(_) => true,
    }
}

fn is_pure_dictionary(
    row_group: &crate::file::metadata::RowGroupMetaData,
    column_idx: usize,
) -> bool {
    let column = row_group.column(column_idx);
    column.dictionary_page_offset().is_some()
        && column.page_encoding_stats_mask().is_some_and(|mask| {
            mask.is_only(Encoding::PLAIN_DICTIONARY) || mask.is_only(Encoding::RLE_DICTIONARY)
        })
}

fn exact_skip(
    reader: &mut dyn ArrayReader,
    counts: &mut SelectionDecodeCounts,
    rows: usize,
) -> Result<()> {
    if rows == 0 {
        return Ok(());
    }
    let skipped = counted_skip_records(reader, counts, rows)?;
    if skipped != rows {
        return Err(general_err!(
            "PerColumn failed to skip rows: expected {rows}, got {skipped}"
        ));
    }
    Ok(())
}

fn exact_read(
    reader: &mut dyn ArrayReader,
    counts: &mut SelectionDecodeCounts,
    rows: usize,
) -> Result<()> {
    if rows == 0 {
        return Ok(());
    }
    let read = counted_read_records(reader, counts, rows)?;
    if read != rows {
        return Err(general_err!(
            "PerColumn failed to read rows: expected {rows}, got {read}"
        ));
    }
    Ok(())
}

fn one_column_array(batch: RecordBatch) -> Result<ArrayRef> {
    if batch.num_columns() != 1 {
        return Err(general_err!(
            "Internal Error: PerColumn child produced {} columns",
            batch.num_columns()
        ));
    }
    Ok(Arc::clone(batch.column(0)))
}

fn read_selectors_column(
    reader: &mut dyn ArrayReader,
    instructions: &[RowSelector],
    span: PhysicalSpan,
    metrics: &ArrowReaderMetrics,
) -> Result<ArrayRef> {
    let started = metrics.start_timing();
    let mut counts = SelectionDecodeCounts::default();
    exact_skip(reader, &mut counts, span.gap_skip)?;
    for selector in instructions {
        if selector.skip {
            exact_skip(reader, &mut counts, selector.row_count)?;
        } else {
            exact_read(reader, &mut counts, selector.row_count)?;
        }
    }
    metrics.record_pc1c_attribution(Pc1cAttributionSite::Dispatch, started);
    counts.record(metrics, started);
    let consume_started = metrics.start_timing();
    let batch = consume_record_batch(reader, metrics, false)?;
    metrics.record_pc1c_attribution(Pc1cAttributionSite::Consume, consume_started);
    one_column_array(batch)
}

fn read_mask_column(
    reader: &mut dyn ArrayReader,
    span: PhysicalSpan,
    metrics: &ArrowReaderMetrics,
) -> Result<ArrayRef> {
    let started = metrics.start_timing();
    let mut counts = SelectionDecodeCounts::default();
    exact_skip(reader, &mut counts, span.gap_skip)?;
    exact_read(reader, &mut counts, span.span_rows)?;
    metrics.record_pc1c_attribution(Pc1cAttributionSite::Dispatch, started);
    counts.record(metrics, started);
    let consume_started = metrics.start_timing();
    let batch = consume_record_batch(reader, metrics, false)?;
    metrics.record_pc1c_attribution(Pc1cAttributionSite::Consume, consume_started);
    one_column_array(batch)
}

#[cfg(test)]
mod tests {
    use super::{
        ArrowReaderMetrics, NativeSelectionPlan, PhysicalSpan, RowSelection, RowSelectionStrategy,
        RowSelector, uniform_strategy,
    };
    use arrow_buffer::BooleanBuffer;

    fn batch_instructions(plan: &NativeSelectionPlan, batch: usize) -> &[RowSelector] {
        let range = plan.batches[batch].instruction_range.clone();
        &plan.instructions[range]
    }

    fn native_plan(
        selection: RowSelection,
        batch_size: usize,
        needs_mask: bool,
    ) -> super::Result<NativeSelectionPlan> {
        NativeSelectionPlan::try_new(
            selection,
            batch_size,
            needs_mask,
            &ArrowReaderMetrics::disabled(),
        )
    }

    #[test]
    fn uniform_strategy_requires_one_shared_choice() {
        assert_eq!(uniform_strategy(&[]), None);
        assert_eq!(
            uniform_strategy(&[RowSelectionStrategy::Selectors]),
            Some(RowSelectionStrategy::Selectors)
        );
        assert_eq!(
            uniform_strategy(&[RowSelectionStrategy::Mask, RowSelectionStrategy::Mask,]),
            Some(RowSelectionStrategy::Mask)
        );
        assert_eq!(
            uniform_strategy(&[RowSelectionStrategy::Mask, RowSelectionStrategy::Selectors,]),
            None
        );
    }

    #[test]
    fn native_plan_splits_long_select_and_short_last_batch() {
        let selection = RowSelection::from(vec![
            RowSelector::skip(2),
            RowSelector::select(10),
            RowSelector::skip(3),
        ]);
        let plan = native_plan(selection, 4, true).unwrap();

        assert_eq!(plan.batches.len(), 3);
        assert_eq!(batch_instructions(&plan, 0), &[RowSelector::select(4)]);
        assert_eq!(batch_instructions(&plan, 1), &[RowSelector::select(4)]);
        assert_eq!(batch_instructions(&plan, 2), &[RowSelector::select(2)]);
        assert_eq!(
            plan.batches
                .iter()
                .map(|batch| batch.span)
                .collect::<Vec<_>>(),
            vec![
                PhysicalSpan {
                    gap_skip: 2,
                    span_start: 2,
                    span_rows: 4,
                    selected: 4,
                },
                PhysicalSpan {
                    gap_skip: 0,
                    span_start: 6,
                    span_rows: 4,
                    selected: 4,
                },
                PhysicalSpan {
                    gap_skip: 0,
                    span_start: 10,
                    span_rows: 2,
                    selected: 2,
                },
            ]
        );
        let mask = plan.mask.unwrap();
        assert_eq!(mask.len(), 12);
        assert_eq!(mask.slice(2, 4).count_set_bits(), 4);
        assert_eq!(mask.slice(6, 4).count_set_bits(), 4);
        assert_eq!(mask.slice(10, 2).count_set_bits(), 2);
    }

    #[test]
    fn native_plan_preserves_internal_skip_and_boundary_gap() {
        let selection = RowSelection::from(vec![
            RowSelector::skip(3),
            RowSelector::select(2),
            RowSelector::skip(5),
            RowSelector::select(6),
            RowSelector::skip(7),
            RowSelector::select(1),
            RowSelector::skip(4),
        ]);
        let plan = native_plan(selection, 4, false).unwrap();

        assert!(plan.mask.is_none());
        assert_eq!(plan.batches.len(), 3);
        assert_eq!(
            batch_instructions(&plan, 0),
            &[
                RowSelector::select(2),
                RowSelector::skip(5),
                RowSelector::select(2),
            ]
        );
        assert_eq!(
            plan.batches[0].span,
            PhysicalSpan {
                gap_skip: 3,
                span_start: 3,
                span_rows: 9,
                selected: 4,
            }
        );
        assert_eq!(batch_instructions(&plan, 1), &[RowSelector::select(4)]);
        assert_eq!(
            plan.batches[1].span,
            PhysicalSpan {
                gap_skip: 0,
                span_start: 12,
                span_rows: 4,
                selected: 4,
            }
        );
        assert_eq!(batch_instructions(&plan, 2), &[RowSelector::select(1)]);
        assert_eq!(
            plan.batches[2].span,
            PhysicalSpan {
                gap_skip: 7,
                span_start: 23,
                span_rows: 1,
                selected: 1,
            }
        );
    }

    #[test]
    fn native_plan_accepts_mask_backing_without_changing_boundaries() {
        let selection = RowSelection::from_boolean_buffer(BooleanBuffer::from(vec![
            false, true, true, false, true, false,
        ]));
        let plan = native_plan(selection, 2, true).unwrap();

        assert_eq!(plan.batches.len(), 2);
        assert_eq!(batch_instructions(&plan, 0), &[RowSelector::select(2)]);
        assert_eq!(
            plan.batches[0].span,
            PhysicalSpan {
                gap_skip: 1,
                span_start: 1,
                span_rows: 2,
                selected: 2,
            }
        );
        assert_eq!(batch_instructions(&plan, 1), &[RowSelector::select(1)]);
        assert_eq!(
            plan.batches[1].span,
            PhysicalSpan {
                gap_skip: 1,
                span_start: 4,
                span_rows: 1,
                selected: 1,
            }
        );
        assert_eq!(plan.mask.unwrap().len(), 5);
    }

    #[test]
    fn native_plan_rejects_zero_batch_or_empty_selection() {
        let selected = RowSelection::from(vec![RowSelector::select(1)]);
        assert!(native_plan(selected, 0, false).is_err());
        assert!(native_plan(RowSelection::from(vec![]), 4, false).is_err());
        assert!(native_plan(RowSelection::from(vec![RowSelector::skip(4)]), 4, false).is_err());
    }
}
