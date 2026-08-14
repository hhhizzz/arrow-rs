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
//! The logical [`RowSelection`] remains shared. A mask cursor cuts it into
//! physical-row windows that are safe for every projected column, and each
//! column independently consumes a window with either selector calls or
//! decode-then-filter. Unsupported projections never enter this module.

use super::selection::RowSelectionStrategy;
use super::{
    FilterMaskAccumulator, MaskRunIter, ReadPlan, ReadPlanBuilder, RowSelectionCursor,
    RowSelectionPolicy, SelectionDecodeCounts, consume_record_batch, counted_read_records,
    counted_skip_records,
};
use crate::arrow::ProjectionMask;
use crate::arrow::array_reader::{ArrayReader, ArrayReaderBuilder, RowGroups};
use crate::arrow::arrow_reader::metrics::ArrowReaderMetrics;
use crate::arrow::schema::{ParquetField, ParquetFieldType};
use crate::basic::Encoding;
use crate::errors::{ParquetError, Result};
use arrow_array::{ArrayRef, BooleanArray, RecordBatch};
use arrow_buffer::BooleanBuffer;
use arrow_schema::{DataType as ArrowType, FieldRef, Schema, SchemaRef};
use arrow_select::filter::filter_record_batch;
use std::sync::Arc;

const PURE_DICTIONARY_THRESHOLD: usize = 4;
const DEFAULT_COLUMN_THRESHOLD: usize = 16;

struct ColumnReader {
    reader: Box<dyn ArrayReader>,
    strategy: RowSelectionStrategy,
}

struct WindowChunk {
    initial_skip: usize,
    mask: BooleanBuffer,
}

/// A record-batch reader for the narrow PC-1 flat-output experiment.
pub(super) struct PerColumnReader {
    columns: Vec<ColumnReader>,
    schema: SchemaRef,
    mask_plan: ReadPlan,
}

impl PerColumnReader {
    /// Build the experimental reader when every hard scope condition holds.
    /// Returning `None` means the caller must build the existing Auto32 path.
    pub(super) fn try_new(
        row_groups: &dyn RowGroups,
        metrics: &ArrowReaderMetrics,
        batch_size: usize,
        fields: Option<&ParquetField>,
        projection: &ProjectionMask,
        plan_builder: &ReadPlanBuilder,
    ) -> Result<Option<Self>> {
        if !matches!(
            plan_builder.row_selection_policy(),
            RowSelectionPolicy::PerColumn
        ) {
            return Ok(None);
        }

        let Some(selection) = plan_builder.selection() else {
            // The all-selected fast path must remain exactly the current path.
            return Ok(None);
        };
        if !selection.selects_any() || selection.skipped_row_count() == 0 {
            return Ok(None);
        }

        let Some(fields) = fields else {
            return Ok(None);
        };
        let Some(column_indices) = projected_flat_columns(fields, projection) else {
            return Ok(None);
        };
        if column_indices.is_empty() {
            return Ok(None);
        }

        let mut row_group_iter = row_groups.row_groups();
        let Some(row_group) = row_group_iter.next() else {
            return Ok(None);
        };
        if row_group_iter.next().is_some() {
            // Per-column thresholds are row-group local. The push decoder
            // naturally constructs one reader per row group; the synchronous
            // multi-row-group path falls back until it has the same boundary.
            return Ok(None);
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

        let schema_descr = row_groups.metadata().file_metadata().schema_descr();
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

        // One shared Mask cursor defines physical windows. Individual columns
        // never mutate this cursor and therefore cannot drift in logical row
        // alignment.
        let mask_plan = plan_builder
            .clone()
            .with_row_selection_policy(RowSelectionPolicy::Mask)
            .build();

        Ok(Some(Self {
            columns,
            schema: Arc::new(Schema::new(output_fields)),
            mask_plan,
        }))
    }

    pub(super) fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    pub(super) fn batch_size(&self) -> usize {
        self.mask_plan.batch_size()
    }

    pub(super) fn next_batch(&mut self) -> Result<Option<RecordBatch>> {
        let batch_size = self.mask_plan.batch_size();
        if batch_size == 0 {
            return Ok(None);
        }
        let windows = next_windows(&mut self.mask_plan, batch_size)?;
        if windows.is_empty() {
            return Ok(None);
        }
        let selected_rows = windows
            .iter()
            .map(|window| window.mask.count_set_bits())
            .sum::<usize>();
        if selected_rows == 0 {
            return Err(general_err!(
                "Internal Error: PerColumn window contains no selected rows"
            ));
        }

        let metrics = self.mask_plan.metrics().clone();
        let mut arrays = Vec::with_capacity(self.columns.len());
        for column in &mut self.columns {
            let array = match column.strategy {
                RowSelectionStrategy::Selectors => {
                    read_selectors_column(column.reader.as_mut(), &windows, &metrics)?
                }
                RowSelectionStrategy::Mask => {
                    read_mask_column(column.reader.as_mut(), &windows, &metrics)?
                }
            };
            if array.len() != selected_rows {
                return Err(general_err!(
                    "PerColumn output length mismatch: expected {selected_rows}, got {}",
                    array.len()
                ));
            }
            arrays.push(array);
        }

        let batch = RecordBatch::try_new(Arc::clone(&self.schema), arrays)?;
        if batch.num_rows() != selected_rows {
            return Err(general_err!(
                "PerColumn RecordBatch row mismatch: expected {selected_rows}, got {}",
                batch.num_rows()
            ));
        }
        Ok(Some(batch))
    }
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

fn next_windows(plan: &mut ReadPlan, batch_size: usize) -> Result<Vec<WindowChunk>> {
    let cursor = match plan.row_selection_cursor_mut() {
        RowSelectionCursor::Mask(cursor) => cursor,
        RowSelectionCursor::All | RowSelectionCursor::Selectors(_) => {
            return Err(general_err!(
                "Internal Error: PerColumn shared cursor is not mask-backed"
            ));
        }
    };
    let mut windows = Vec::new();
    let mut selected_rows = 0usize;
    while selected_rows < batch_size && !cursor.is_empty() {
        let chunk = cursor.next_chunk(batch_size - selected_rows)?;
        let mask = cursor.mask_values_for(&chunk)?.values().clone();
        selected_rows += chunk.selected_rows;
        windows.push(WindowChunk {
            initial_skip: chunk.initial_skip,
            mask,
        });
    }
    Ok(windows)
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
    windows: &[WindowChunk],
    metrics: &ArrowReaderMetrics,
) -> Result<ArrayRef> {
    let started = metrics.start_timing();
    let mut counts = SelectionDecodeCounts::default();
    for window in windows {
        exact_skip(reader, &mut counts, window.initial_skip)?;
        for selector in MaskRunIter::new(&window.mask) {
            if selector.skip {
                exact_skip(reader, &mut counts, selector.row_count)?;
            } else {
                exact_read(reader, &mut counts, selector.row_count)?;
            }
        }
    }
    counts.record(metrics, started);
    one_column_array(consume_record_batch(reader, metrics)?)
}

fn read_mask_column(
    reader: &mut dyn ArrayReader,
    windows: &[WindowChunk],
    metrics: &ArrowReaderMetrics,
) -> Result<ArrayRef> {
    let started = metrics.start_timing();
    let mut counts = SelectionDecodeCounts::default();
    let mut filter_mask = FilterMaskAccumulator::default();
    for window in windows {
        exact_skip(reader, &mut counts, window.initial_skip)?;
        exact_read(reader, &mut counts, window.mask.len())?;
        filter_mask.append(window.mask.clone());
    }
    counts.record(metrics, started);
    let filter_mask = filter_mask.finish().ok_or_else(|| {
        general_err!("Internal Error: PerColumn mask column has no filter values")
    })?;
    let batch = consume_record_batch(reader, metrics)?;
    let filter_started = metrics.start_timing();
    let filtered = filter_record_batch(&batch, &BooleanArray::from(filter_mask))?;
    metrics.record_filter_record_batch(filter_started);
    one_column_array(filtered)
}
