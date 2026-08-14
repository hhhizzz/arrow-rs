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

//! Experimental flat-projection execution with one cursor per selection strategy.
//!
//! The logical [`RowSelection`] remains shared. Its native RLE form is compiled
//! once into selected-row batches. Native execution groups projected columns by
//! selector or mask strategy, drives each group once, and scatters the resulting
//! arrays back into projection order. The bench-only legacy replay deliberately
//! retains one reader per output column. Unsupported projections never enter
//! this module.

use super::selection::{RowSelectionStrategy, boolean_mask_from_selector_iter};
#[cfg(feature = "test_common")]
use super::{FilterMaskAccumulator, MaskRunIter, ReadPlan, RowSelectionCursor};
use super::{
    ReadPlanBuilder, RowSelection, RowSelectionPolicy, RowSelector, SelectionDecodeCounts,
    consume_record_batch, counted_read_records, counted_skip_records,
};
use crate::arrow::ProjectionMask;
use crate::arrow::array_reader::{ArrayReader, ArrayReaderBuilder, RowGroups};
use crate::arrow::arrow_reader::metrics::{
    ArrowReaderMetrics, Pc1cAttributionSite, PerColumnDecisionKind,
};
use crate::arrow::schema::{ParquetField, ParquetFieldType};
use crate::basic::Encoding;
use crate::errors::{ParquetError, Result};
use arrow_array::{ArrayRef, BooleanArray, RecordBatch};
use arrow_buffer::BooleanBuffer;
#[cfg(feature = "test_common")]
use arrow_schema::FieldRef;
use arrow_schema::{DataType as ArrowType, Schema, SchemaRef};
use arrow_select::filter::FilterBuilder;
#[cfg(feature = "test_common")]
use arrow_select::filter::filter_record_batch;
use std::ops::Range;
use std::sync::Arc;

const PURE_DICTIONARY_THRESHOLD: usize = 4;
const DEFAULT_COLUMN_THRESHOLD: usize = 32;
const LEGACY_COLUMN_THRESHOLD: usize = 16;

#[cfg(feature = "test_common")]
struct ColumnReader {
    reader: Box<dyn ArrayReader>,
    strategy: RowSelectionStrategy,
}

struct NativeColumnGroupReader {
    reader: Box<dyn ArrayReader>,
    strategy: RowSelectionStrategy,
    output_indices: Vec<usize>,
}

#[derive(Debug, Clone, Eq, PartialEq)]
struct NativeColumnGroupPlan {
    strategy: RowSelectionStrategy,
    column_indices: Vec<usize>,
    output_indices: Vec<usize>,
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

#[cfg(feature = "test_common")]
struct LegacyWindowChunk {
    initial_skip: usize,
    mask: BooleanBuffer,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PerColumnMode {
    Product,
    #[cfg(feature = "test_common")]
    Legacy,
    #[cfg(feature = "test_common")]
    ForcedThin,
    #[cfg(feature = "test_common")]
    R16,
}

impl PerColumnMode {
    fn from_policy(policy: RowSelectionPolicy) -> Option<Self> {
        match policy {
            RowSelectionPolicy::PerColumn => Some(Self::Product),
            #[cfg(feature = "test_common")]
            RowSelectionPolicy::PerColumnLegacy => Some(Self::Legacy),
            #[cfg(feature = "test_common")]
            RowSelectionPolicy::PerColumnForcedThin => Some(Self::ForcedThin),
            #[cfg(feature = "test_common")]
            RowSelectionPolicy::PerColumnR16 => Some(Self::R16),
            RowSelectionPolicy::Selectors
            | RowSelectionPolicy::Mask
            | RowSelectionPolicy::Auto { .. } => None,
        }
    }

    const fn collapses_uniform(self) -> bool {
        match self {
            Self::Product => true,
            #[cfg(feature = "test_common")]
            Self::R16 => true,
            #[cfg(feature = "test_common")]
            Self::Legacy | Self::ForcedThin => false,
        }
    }

    const fn other_threshold(self) -> usize {
        match self {
            Self::Product => DEFAULT_COLUMN_THRESHOLD,
            #[cfg(feature = "test_common")]
            Self::ForcedThin => DEFAULT_COLUMN_THRESHOLD,
            #[cfg(feature = "test_common")]
            Self::Legacy | Self::R16 => LEGACY_COLUMN_THRESHOLD,
        }
    }

    const fn is_legacy(self) -> bool {
        match self {
            Self::Product => false,
            #[cfg(feature = "test_common")]
            Self::Legacy => true,
            #[cfg(feature = "test_common")]
            Self::ForcedThin | Self::R16 => false,
        }
    }

    /// Product preserves the exact Auto32 path when every projected column is
    /// plain, without resolving a selection strategy. R16 cannot use this
    /// shortcut because its plain-column threshold is deliberately 16.
    const fn falls_back_when_all_plain(self) -> bool {
        match self {
            Self::Product => true,
            #[cfg(feature = "test_common")]
            Self::Legacy | Self::ForcedThin | Self::R16 => false,
        }
    }
}

/// Lazily resolves the only three thresholds understood by the per-column
/// policy. A scan of the shared row selection is paid at most once per
/// threshold, independent of the projected column count.
struct SelectionStrategyCache<'a> {
    selection: &'a RowSelection,
    threshold4: Option<RowSelectionStrategy>,
    threshold16: Option<RowSelectionStrategy>,
    threshold32: Option<RowSelectionStrategy>,
    #[cfg(test)]
    evaluations: usize,
}

impl<'a> SelectionStrategyCache<'a> {
    fn new(selection: &'a RowSelection) -> Self {
        Self {
            selection,
            threshold4: None,
            threshold16: None,
            threshold32: None,
            #[cfg(test)]
            evaluations: 0,
        }
    }

    fn resolve(&mut self, threshold: usize) -> RowSelectionStrategy {
        let cached = match threshold {
            PURE_DICTIONARY_THRESHOLD => self.threshold4,
            LEGACY_COLUMN_THRESHOLD => self.threshold16,
            DEFAULT_COLUMN_THRESHOLD => self.threshold32,
            _ => unreachable!("unsupported PerColumn threshold {threshold}"),
        };
        if let Some(strategy) = cached {
            return strategy;
        }

        let strategy = self.selection.auto_selection_strategy(threshold);
        match threshold {
            PURE_DICTIONARY_THRESHOLD => self.threshold4 = Some(strategy),
            LEGACY_COLUMN_THRESHOLD => self.threshold16 = Some(strategy),
            DEFAULT_COLUMN_THRESHOLD => self.threshold32 = Some(strategy),
            _ => unreachable!("unsupported PerColumn threshold {threshold}"),
        }
        #[cfg(test)]
        {
            self.evaluations += 1;
        }
        strategy
    }

    #[cfg(test)]
    fn evaluations(&self) -> usize {
        self.evaluations
    }
}

fn cached_product_strategies(
    selection: &RowSelection,
    mode: PerColumnMode,
    pure_dictionary_columns: &[bool],
) -> (Vec<RowSelectionStrategy>, Option<RowSelectionStrategy>) {
    debug_assert!(mode.collapses_uniform());
    let mut cache = SelectionStrategyCache::new(selection);
    let strategies = pure_dictionary_columns
        .iter()
        .map(|&is_dictionary| {
            let threshold = if is_dictionary {
                PURE_DICTIONARY_THRESHOLD
            } else {
                mode.other_threshold()
            };
            cache.resolve(threshold)
        })
        .collect::<Vec<_>>();
    let auto32 = cache.resolve(DEFAULT_COLUMN_THRESHOLD);
    (strategies, Some(auto32))
}

enum PerColumnExecution {
    Native {
        groups: Vec<NativeColumnGroupReader>,
        selection_plan: NativeSelectionPlan,
    },
    #[cfg(feature = "test_common")]
    Legacy {
        columns: Vec<ColumnReader>,
        mask_plan: ReadPlan,
    },
}

/// A record-batch reader for the narrow PC-1 flat-output experiment.
pub(super) enum PerColumnDecision {
    FallbackAuto,
    FallbackForced(RowSelectionStrategy),
    Engaged(PerColumnReader),
}

pub(super) struct PerColumnReader {
    schema: SchemaRef,
    batch_size: usize,
    metrics: ArrowReaderMetrics,
    execution: PerColumnExecution,
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
        let decision = Self::try_new_inner(
            row_groups,
            metrics,
            batch_size,
            fields,
            projection,
            plan_builder,
        )?;
        metrics.record_per_column_decision(match &decision {
            PerColumnDecision::FallbackAuto => PerColumnDecisionKind::FallbackAuto,
            PerColumnDecision::FallbackForced(_) => PerColumnDecisionKind::FallbackForced,
            PerColumnDecision::Engaged(_) => PerColumnDecisionKind::Engaged,
        });
        Ok(decision)
    }

    fn try_new_inner(
        row_groups: &dyn RowGroups,
        metrics: &ArrowReaderMetrics,
        batch_size: usize,
        fields: Option<&ParquetField>,
        projection: &ProjectionMask,
        plan_builder: &ReadPlanBuilder,
    ) -> Result<PerColumnDecision> {
        let Some(mode) = PerColumnMode::from_policy(*plan_builder.row_selection_policy()) else {
            return Ok(PerColumnDecision::FallbackAuto);
        };
        if batch_size == 0 {
            return Ok(PerColumnDecision::FallbackAuto);
        }

        let Some(selection) = plan_builder.selection() else {
            // The all-selected fast path must remain exactly the current path.
            return Ok(PerColumnDecision::FallbackAuto);
        };
        if !mode.falls_back_when_all_plain()
            && (!selection.selects_any() || selection.skipped_row_count() == 0)
        {
            // Diagnostic and Legacy modes preserve their original selection
            // shape check. Only Product may defer it to the all-plain path.
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
            if mode.falls_back_when_all_plain()
                && (!selection.selects_any() || selection.skipped_row_count() == 0)
            {
                // Preserve the original counter contract for trivial
                // selections: they fallback before loaded ranges matter.
                return Ok(PerColumnDecision::FallbackAuto);
            }
            metrics.record_per_column_loaded_row_ranges_fallback();
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

        if mode.falls_back_when_all_plain()
            && !column_indices
                .iter()
                .any(|&column_idx| is_pure_dictionary(row_group, column_idx))
        {
            // This metadata-only Product shortcut deliberately precedes the
            // O(runs) selection-shape checks. Explicit all-selected still
            // returns the same FallbackAuto decision, and try_new records the
            // same single fallback_auto counter for every return path.
            return Ok(PerColumnDecision::FallbackAuto);
        }
        if mode.falls_back_when_all_plain()
            && (!selection.selects_any() || selection.skipped_row_count() == 0)
        {
            return Ok(PerColumnDecision::FallbackAuto);
        }

        let legacy_reader_build_started =
            mode.is_legacy().then(|| metrics.start_timing()).flatten();
        let pure_dictionary_columns = match mode {
            PerColumnMode::Product => column_indices
                .iter()
                .map(|&column_idx| is_pure_dictionary(row_group, column_idx))
                .collect::<Vec<_>>(),
            #[cfg(feature = "test_common")]
            PerColumnMode::R16 => column_indices
                .iter()
                .map(|&column_idx| is_pure_dictionary(row_group, column_idx))
                .collect::<Vec<_>>(),
            #[cfg(feature = "test_common")]
            PerColumnMode::Legacy | PerColumnMode::ForcedThin => Vec::new(),
        };

        let (strategies, auto32_for_collapse) = match mode {
            PerColumnMode::Product => {
                cached_product_strategies(selection, mode, &pure_dictionary_columns)
            }
            #[cfg(feature = "test_common")]
            PerColumnMode::R16 => {
                cached_product_strategies(selection, mode, &pure_dictionary_columns)
            }
            #[cfg(feature = "test_common")]
            PerColumnMode::ForcedThin => {
                let auto32 = selection.auto_selection_strategy(DEFAULT_COLUMN_THRESHOLD);
                (vec![auto32; column_indices.len()], None)
            }
            #[cfg(feature = "test_common")]
            PerColumnMode::Legacy => {
                // Keep the unused Auto32 scan that the historical bolt-on
                // paid before resolving each column independently.
                let _auto32 = selection.auto_selection_strategy(DEFAULT_COLUMN_THRESHOLD);
                (
                    column_indices
                        .iter()
                        .map(|&column_idx| {
                            let threshold = if is_pure_dictionary(row_group, column_idx) {
                                PURE_DICTIONARY_THRESHOLD
                            } else {
                                LEGACY_COLUMN_THRESHOLD
                            };
                            // Preserve the historical replay exactly: it paid this
                            // selection scan independently for every output column.
                            selection.auto_selection_strategy(threshold)
                        })
                        .collect::<Vec<_>>(),
                    None,
                )
            }
        };

        if mode.collapses_uniform()
            && let Some(strategy) = uniform_strategy(&strategies)
        {
            let auto32 =
                auto32_for_collapse.expect("product-like modes resolve Auto32 exactly once");
            return Ok(if strategy == auto32 {
                PerColumnDecision::FallbackAuto
            } else {
                PerColumnDecision::FallbackForced(strategy)
            });
        }

        let schema_descr = row_groups.metadata().file_metadata().schema_descr();
        let (execution, output_fields) = match mode {
            #[cfg(feature = "test_common")]
            PerColumnMode::Legacy => {
                let mut columns = Vec::with_capacity(column_indices.len());
                let mut output_fields: Vec<FieldRef> = Vec::with_capacity(column_indices.len());
                for (column_idx, strategy) in column_indices
                    .iter()
                    .copied()
                    .zip(strategies.iter().copied())
                {
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
                let plan = plan_builder
                    .clone()
                    .with_row_selection_policy(RowSelectionPolicy::Mask)
                    .build();
                metrics.record_pc1c_attribution(
                    Pc1cAttributionSite::ReaderBuild,
                    legacy_reader_build_started,
                );
                (
                    PerColumnExecution::Legacy {
                        columns,
                        mask_plan: plan,
                    },
                    output_fields,
                )
            }
            _ => {
                let reader_build_started = metrics.start_timing();
                let group_plans = native_column_group_plans(&column_indices, &strategies)?;
                let needs_selectors = group_plans
                    .iter()
                    .any(|group| group.strategy == RowSelectionStrategy::Selectors);
                let needs_mask = group_plans
                    .iter()
                    .any(|group| group.strategy == RowSelectionStrategy::Mask);
                let mut output_fields = vec![None; column_indices.len()];
                let mut groups = Vec::with_capacity(group_plans.len());
                for group in group_plans {
                    let group_projection =
                        ProjectionMask::leaves(schema_descr, group.column_indices.iter().copied());
                    let reader = ArrayReaderBuilder::new(row_groups, metrics)
                        .with_batch_size(batch_size)
                        .with_parquet_metadata(row_groups.metadata())
                        .build_array_reader(Some(fields), &group_projection)?;
                    let group_fields = match reader.get_data_type() {
                        ArrowType::Struct(fields) if fields.len() == group.output_indices.len() => {
                            fields
                        }
                        ArrowType::Struct(fields) => {
                            return Err(general_err!(
                                "PerColumn {:?} group produced {} fields, expected {}",
                                group.strategy,
                                fields.len(),
                                group.output_indices.len()
                            ));
                        }
                        data_type => {
                            return Err(general_err!(
                                "PerColumn {:?} group produced non-struct type {data_type}",
                                group.strategy
                            ));
                        }
                    };
                    for (&output_index, field) in
                        group.output_indices.iter().zip(group_fields.iter())
                    {
                        if output_fields[output_index].replace(field.clone()).is_some() {
                            return Err(general_err!(
                                "Internal Error: PerColumn output {output_index} assigned twice"
                            ));
                        }
                    }
                    groups.push(NativeColumnGroupReader {
                        reader,
                        strategy: group.strategy,
                        output_indices: group.output_indices,
                    });
                }
                let output_fields = output_fields
                    .into_iter()
                    .enumerate()
                    .map(|(output_index, field)| {
                        field.ok_or_else(|| {
                            general_err!(
                                "Internal Error: PerColumn output {output_index} was not assigned"
                            )
                        })
                    })
                    .collect::<Result<Vec<_>>>()?;
                metrics.record_pc1c_attribution(
                    Pc1cAttributionSite::ReaderBuild,
                    reader_build_started,
                );
                let plan_started = metrics.start_timing();
                let selection_plan = NativeSelectionPlan::try_new(
                    selection,
                    batch_size,
                    needs_selectors,
                    needs_mask,
                    metrics,
                )?;
                metrics.record_pc1c_attribution(Pc1cAttributionSite::Window, plan_started);
                (
                    PerColumnExecution::Native {
                        groups,
                        selection_plan,
                    },
                    output_fields,
                )
            }
        };

        let reader = Self {
            schema: Arc::new(Schema::new(output_fields)),
            batch_size,
            metrics: metrics.clone(),
            execution,
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
        let Self {
            schema,
            metrics,
            execution,
            ..
        } = self;
        match execution {
            PerColumnExecution::Native {
                groups,
                selection_plan,
            } => next_native_batch(groups, schema, metrics, selection_plan),
            #[cfg(feature = "test_common")]
            PerColumnExecution::Legacy { columns, mask_plan } => {
                next_legacy_batch(columns, schema, metrics, mask_plan)
            }
        }
    }
}

fn next_native_batch(
    groups: &mut [NativeColumnGroupReader],
    schema: &SchemaRef,
    metrics: &ArrowReaderMetrics,
    selection_plan: &mut NativeSelectionPlan,
) -> Result<Option<RecordBatch>> {
    let Some(batch_plan) = selection_plan.batches.get(selection_plan.next_batch) else {
        return Ok(None);
    };
    selection_plan.next_batch += 1;

    let selected_rows = batch_plan.span.selected;
    let instructions = &selection_plan.instructions[batch_plan.instruction_range.clone()];
    let mask = selection_plan
        .mask
        .as_ref()
        .map(|mask| mask.slice(batch_plan.span.span_start, batch_plan.span.span_rows));

    let mut arrays = vec![None; schema.fields().len()];
    for group in groups.iter_mut() {
        let batch = match group.strategy {
            RowSelectionStrategy::Selectors => read_selectors_group(
                group.reader.as_mut(),
                instructions,
                batch_plan.span,
                metrics,
            )?,
            RowSelectionStrategy::Mask => {
                mask.as_ref().ok_or_else(|| {
                    general_err!("Internal Error: PerColumn mask group has no shared mask")
                })?;
                read_mask_group(group.reader.as_mut(), batch_plan.span, metrics)?
            }
        };
        let expected_rows = match group.strategy {
            RowSelectionStrategy::Selectors => selected_rows,
            RowSelectionStrategy::Mask => batch_plan.span.span_rows,
        };
        if batch.num_rows() != expected_rows {
            return Err(general_err!(
                "PerColumn {:?} group decoded length mismatch: expected {expected_rows}, got {}",
                group.strategy,
                batch.num_rows()
            ));
        }
        let batch = match group.strategy {
            RowSelectionStrategy::Selectors => batch,
            RowSelectionStrategy::Mask => {
                let mask = mask.as_ref().expect("mask group validated shared mask");
                let filter_started = metrics.start_timing();
                let filter = BooleanArray::from(mask.clone());
                let predicate = FilterBuilder::new(&filter).optimize().build();
                if predicate.count() != selected_rows {
                    return Err(general_err!(
                        "Internal Error: PerColumn shared filter selects {} rows, expected {selected_rows}",
                        predicate.count()
                    ));
                }
                let filtered = predicate.filter_record_batch(&batch)?;
                metrics.record_pc1c_attribution(Pc1cAttributionSite::Filter, filter_started);
                metrics.record_filter_record_batch(filter_started);
                filtered
            }
        };
        if batch.num_rows() != selected_rows {
            return Err(general_err!(
                "PerColumn {:?} group output length mismatch: expected {selected_rows}, got {}",
                group.strategy,
                batch.num_rows()
            ));
        }
        scatter_group_arrays(&mut arrays, &group.output_indices, batch.columns())?;
    }

    let arrays = arrays
        .into_iter()
        .enumerate()
        .map(|(output_index, array)| {
            array.ok_or_else(|| {
                general_err!("Internal Error: PerColumn output {output_index} was not produced")
            })
        })
        .collect::<Result<Vec<_>>>()?;
    finish_batch(schema, metrics, arrays, selected_rows)
}

#[cfg(feature = "test_common")]
fn next_legacy_batch(
    columns: &mut [ColumnReader],
    schema: &SchemaRef,
    metrics: &ArrowReaderMetrics,
    mask_plan: &mut ReadPlan,
) -> Result<Option<RecordBatch>> {
    let batch_size = mask_plan.batch_size();
    if batch_size == 0 {
        return Ok(None);
    }
    let window_started = metrics.start_timing();
    let windows = next_legacy_windows(mask_plan, batch_size)?;
    metrics.record_pc1c_attribution(Pc1cAttributionSite::Window, window_started);
    if windows.is_empty() {
        return Ok(None);
    }
    let selected_rows = windows
        .iter()
        .map(|window| window.mask.count_set_bits())
        .sum::<usize>();
    if selected_rows == 0 {
        return Err(general_err!(
            "Internal Error: legacy PerColumn window contains no selected rows"
        ));
    }

    let mut arrays = Vec::with_capacity(columns.len());
    for column in columns {
        let array = match column.strategy {
            RowSelectionStrategy::Selectors => {
                read_legacy_selectors_column(column.reader.as_mut(), &windows, metrics)?
            }
            RowSelectionStrategy::Mask => {
                read_legacy_mask_column(column.reader.as_mut(), &windows, metrics)?
            }
        };
        if array.len() != selected_rows {
            return Err(general_err!(
                "Legacy PerColumn output length mismatch: expected {selected_rows}, got {}",
                array.len()
            ));
        }
        arrays.push(array);
    }
    finish_legacy_batch(schema, metrics, arrays, selected_rows)
}

#[cfg(feature = "test_common")]
fn finish_legacy_batch(
    schema: &SchemaRef,
    metrics: &ArrowReaderMetrics,
    arrays: Vec<ArrayRef>,
    selected_rows: usize,
) -> Result<Option<RecordBatch>> {
    let assembly_started = metrics.start_timing();
    let batch = RecordBatch::try_new(Arc::clone(schema), arrays)?;
    metrics.record_pc1c_attribution(Pc1cAttributionSite::BatchAssembly, assembly_started);
    if batch.num_rows() != selected_rows {
        return Err(general_err!(
            "Legacy PerColumn RecordBatch row mismatch: expected {selected_rows}, got {}",
            batch.num_rows()
        ));
    }
    Ok(Some(batch))
}

fn finish_batch(
    schema: &SchemaRef,
    metrics: &ArrowReaderMetrics,
    arrays: Vec<ArrayRef>,
    selected_rows: usize,
) -> Result<Option<RecordBatch>> {
    if let Some(array) = arrays.iter().find(|array| array.len() != selected_rows) {
        return Err(general_err!(
            "PerColumn output length mismatch: expected {selected_rows}, got {}",
            array.len()
        ));
    }
    let assembly_started = metrics.start_timing();
    let batch = RecordBatch::try_new(Arc::clone(schema), arrays)?;
    metrics.record_pc1c_attribution(Pc1cAttributionSite::BatchAssembly, assembly_started);
    if batch.num_rows() != selected_rows {
        return Err(general_err!(
            "PerColumn RecordBatch row mismatch: expected {selected_rows}, got {}",
            batch.num_rows()
        ));
    }
    Ok(Some(batch))
}

impl NativeSelectionPlan {
    fn try_new(
        selection: &RowSelection,
        batch_size: usize,
        needs_selectors: bool,
        needs_mask: bool,
        metrics: &ArrowReaderMetrics,
    ) -> Result<Self> {
        if batch_size == 0 {
            return Err(general_err!(
                "Internal Error: PerColumn native plan requires a non-zero batch size"
            ));
        }
        if !needs_selectors && !needs_mask {
            return Err(general_err!(
                "Internal Error: PerColumn native plan requires a selection strategy"
            ));
        }
        let mut source = selection.iter().copied().peekable();
        let mut instructions = if needs_selectors {
            Vec::with_capacity(source.size_hint().0)
        } else {
            Vec::new()
        };
        let mut batches = Vec::new();
        let mut physical_position = 0usize;
        let mut pending_gap_skip = 0usize;
        let mut batch_instruction_start = 0usize;
        let mut batch_gap_skip = 0usize;
        let mut batch_span_start = 0usize;
        let mut batch_span_rows = 0usize;
        let mut batch_selected = 0usize;

        while let Some(selector) = source.next() {
            // RowSelection::trim only removes this final trailing skip. Keep
            // the original borrowed selection intact and ignore it in place.
            if selector.skip && source.peek().is_none() {
                break;
            }
            if selector.row_count == 0 {
                continue;
            }

            if selector.skip {
                physical_position = checked_add(physical_position, selector.row_count)?;
                if batch_selected == 0 {
                    pending_gap_skip = checked_add(pending_gap_skip, selector.row_count)?;
                } else {
                    if needs_selectors {
                        append_instruction(&mut instructions, batch_instruction_start, selector)?;
                    }
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
                if needs_selectors {
                    append_instruction(
                        &mut instructions,
                        batch_instruction_start,
                        RowSelector::select(take),
                    )?;
                }
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
                        needs_selectors,
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
                needs_selectors,
            )?;
        }

        if batches.is_empty() || pending_gap_skip != 0 {
            return Err(general_err!(
                "Internal Error: PerColumn native plan produced an invalid empty or trailing-gap plan"
            ));
        }

        let mask = if needs_mask {
            Some(match selection.as_mask() {
                Some(mask) => {
                    if physical_position > mask.len() {
                        return Err(general_err!(
                            "Internal Error: PerColumn shared mask has {} rows, expected at least {physical_position}",
                            mask.len()
                        ));
                    }
                    mask.slice(0, physical_position)
                }
                None => {
                    let started = metrics.start_general_timing();
                    let mut selectors = selection.iter().copied().peekable();
                    let selectors_without_trailing_skip = std::iter::from_fn(move || {
                        let selector = selectors.next()?;
                        (!(selector.skip && selectors.peek().is_none())).then_some(selector)
                    });
                    let mask = boolean_mask_from_selector_iter(
                        physical_position,
                        selectors_without_trailing_skip,
                    )?;
                    metrics.record_selectors_to_mask(started);
                    mask
                }
            })
        } else {
            None
        };

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
    needs_selectors: bool,
) -> Result<()> {
    if span.selected == 0
        || span.span_rows == 0
        || (needs_selectors && instruction_range.is_empty())
    {
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

fn native_column_group_plans(
    column_indices: &[usize],
    strategies: &[RowSelectionStrategy],
) -> Result<Vec<NativeColumnGroupPlan>> {
    if column_indices.len() != strategies.len() {
        return Err(general_err!(
            "Internal Error: PerColumn column/strategy cardinality mismatch: {} != {}",
            column_indices.len(),
            strategies.len()
        ));
    }

    let mut groups: Vec<NativeColumnGroupPlan> = Vec::with_capacity(2);
    for (output_index, (&column_idx, &strategy)) in
        column_indices.iter().zip(strategies).enumerate()
    {
        if let Some(group) = groups.iter_mut().find(|group| group.strategy == strategy) {
            group.column_indices.push(column_idx);
            group.output_indices.push(output_index);
        } else {
            groups.push(NativeColumnGroupPlan {
                strategy,
                column_indices: vec![column_idx],
                output_indices: vec![output_index],
            });
        }
    }
    if groups.len() > 2 {
        return Err(general_err!(
            "Internal Error: PerColumn produced more than two strategy groups"
        ));
    }
    Ok(groups)
}

fn scatter_group_arrays(
    output: &mut [Option<ArrayRef>],
    output_indices: &[usize],
    arrays: &[ArrayRef],
) -> Result<()> {
    if output_indices.len() != arrays.len() {
        return Err(general_err!(
            "Internal Error: PerColumn group output cardinality mismatch: {} != {}",
            output_indices.len(),
            arrays.len()
        ));
    }
    for (&output_index, array) in output_indices.iter().zip(arrays) {
        let slot = output.get_mut(output_index).ok_or_else(|| {
            general_err!("Internal Error: PerColumn output index {output_index} is out of bounds")
        })?;
        if slot.replace(Arc::clone(array)).is_some() {
            return Err(general_err!(
                "Internal Error: PerColumn output {output_index} was produced twice"
            ));
        }
    }
    Ok(())
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

#[cfg(feature = "test_common")]
fn next_legacy_windows(plan: &mut ReadPlan, batch_size: usize) -> Result<Vec<LegacyWindowChunk>> {
    let cursor = match plan.row_selection_cursor_mut() {
        RowSelectionCursor::Mask(cursor) => cursor,
        RowSelectionCursor::All | RowSelectionCursor::Selectors(_) => {
            return Err(general_err!(
                "Internal Error: legacy PerColumn shared cursor is not mask-backed"
            ));
        }
    };
    let mut windows = Vec::new();
    let mut selected_rows = 0usize;
    while selected_rows < batch_size && !cursor.is_empty() {
        let chunk = cursor.next_chunk(batch_size - selected_rows)?;
        let mask = cursor.mask_values_for(&chunk)?.values().clone();
        selected_rows += chunk.selected_rows;
        windows.push(LegacyWindowChunk {
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

#[cfg(feature = "test_common")]
fn one_column_array(batch: RecordBatch) -> Result<ArrayRef> {
    if batch.num_columns() != 1 {
        return Err(general_err!(
            "Internal Error: PerColumn child produced {} columns",
            batch.num_columns()
        ));
    }
    Ok(Arc::clone(batch.column(0)))
}

fn read_selectors_group(
    reader: &mut dyn ArrayReader,
    instructions: &[RowSelector],
    span: PhysicalSpan,
    metrics: &ArrowReaderMetrics,
) -> Result<RecordBatch> {
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
    Ok(batch)
}

fn read_mask_group(
    reader: &mut dyn ArrayReader,
    span: PhysicalSpan,
    metrics: &ArrowReaderMetrics,
) -> Result<RecordBatch> {
    let started = metrics.start_timing();
    let mut counts = SelectionDecodeCounts::default();
    exact_skip(reader, &mut counts, span.gap_skip)?;
    exact_read(reader, &mut counts, span.span_rows)?;
    metrics.record_pc1c_attribution(Pc1cAttributionSite::Dispatch, started);
    counts.record(metrics, started);
    let consume_started = metrics.start_timing();
    let batch = consume_record_batch(reader, metrics, false)?;
    metrics.record_pc1c_attribution(Pc1cAttributionSite::Consume, consume_started);
    Ok(batch)
}

#[cfg(feature = "test_common")]
fn read_legacy_selectors_column(
    reader: &mut dyn ArrayReader,
    windows: &[LegacyWindowChunk],
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
    metrics.record_pc1c_attribution(Pc1cAttributionSite::Dispatch, started);
    counts.record(metrics, started);
    let consume_started = metrics.start_timing();
    let batch = consume_record_batch(reader, metrics, false)?;
    metrics.record_pc1c_attribution(Pc1cAttributionSite::Consume, consume_started);
    one_column_array(batch)
}

#[cfg(feature = "test_common")]
fn read_legacy_mask_column(
    reader: &mut dyn ArrayReader,
    windows: &[LegacyWindowChunk],
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
    metrics.record_pc1c_attribution(Pc1cAttributionSite::Dispatch, started);
    counts.record(metrics, started);
    let filter_mask = filter_mask.finish().ok_or_else(|| {
        general_err!("Internal Error: legacy PerColumn mask column has no filter values")
    })?;
    let consume_started = metrics.start_timing();
    let batch = consume_record_batch(reader, metrics, false)?;
    metrics.record_pc1c_attribution(Pc1cAttributionSite::Consume, consume_started);
    let filter_started = metrics.start_timing();
    let filtered = filter_record_batch(&batch, &BooleanArray::from(filter_mask))?;
    metrics.record_pc1c_attribution(Pc1cAttributionSite::Filter, filter_started);
    metrics.record_filter_record_batch(filter_started);
    one_column_array(filtered)
}

#[cfg(test)]
mod tests {
    use super::{
        ArrowReaderMetrics, NativeColumnGroupPlan, NativeSelectionPlan, PerColumnMode,
        PhysicalSpan, RowSelection, RowSelectionStrategy, RowSelector, SelectionStrategyCache,
        native_column_group_plans, scatter_group_arrays, uniform_strategy,
    };
    use arrow_array::{ArrayRef, Int32Array};
    use arrow_buffer::BooleanBuffer;
    use std::sync::Arc;

    fn batch_instructions(plan: &NativeSelectionPlan, batch: usize) -> &[RowSelector] {
        let range = plan.batches[batch].instruction_range.clone();
        &plan.instructions[range]
    }

    fn native_plan(
        selection: RowSelection,
        batch_size: usize,
        needs_selectors: bool,
        needs_mask: bool,
    ) -> super::Result<NativeSelectionPlan> {
        NativeSelectionPlan::try_new(
            &selection,
            batch_size,
            needs_selectors,
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
    fn strategy_cache_scans_each_threshold_once() {
        let selection = RowSelection::from(vec![
            RowSelector::skip(1),
            RowSelector::select(1),
            RowSelector::skip(1),
            RowSelector::select(5),
        ]);
        let mut cache = SelectionStrategyCache::new(&selection);

        assert_eq!(cache.evaluations(), 0);
        let auto32 = cache.resolve(32);
        assert_eq!(cache.resolve(32), auto32);
        assert_eq!(cache.evaluations(), 1);
        let dictionary4 = cache.resolve(4);
        assert_eq!(cache.resolve(4), dictionary4);
        assert_eq!(cache.evaluations(), 2);
        let diagnostic16 = cache.resolve(16);
        assert_eq!(cache.resolve(16), diagnostic16);
        assert_eq!(cache.evaluations(), 3);
    }

    #[test]
    fn product_all_plain_falls_back_without_strategy_resolution() {
        assert!(PerColumnMode::Product.falls_back_when_all_plain());
        let explicit_all_selected = RowSelection::from(vec![RowSelector::select(8)]);
        assert!(explicit_all_selected.selects_any());
        assert_eq!(explicit_all_selected.skipped_row_count(), 0);
        #[cfg(feature = "test_common")]
        {
            assert!(!PerColumnMode::R16.falls_back_when_all_plain());
            assert!(!PerColumnMode::ForcedThin.falls_back_when_all_plain());
            assert!(!PerColumnMode::Legacy.falls_back_when_all_plain());
        }
    }

    #[test]
    fn native_groups_have_one_reader_per_strategy() {
        let one_group = native_column_group_plans(
            &[2, 5, 9],
            &[
                RowSelectionStrategy::Selectors,
                RowSelectionStrategy::Selectors,
                RowSelectionStrategy::Selectors,
            ],
        )
        .unwrap();
        assert_eq!(
            one_group,
            vec![NativeColumnGroupPlan {
                strategy: RowSelectionStrategy::Selectors,
                column_indices: vec![2, 5, 9],
                output_indices: vec![0, 1, 2],
            }]
        );

        let two_groups = native_column_group_plans(
            &[2, 5, 9],
            &[
                RowSelectionStrategy::Mask,
                RowSelectionStrategy::Selectors,
                RowSelectionStrategy::Mask,
            ],
        )
        .unwrap();
        assert_eq!(
            two_groups,
            vec![
                NativeColumnGroupPlan {
                    strategy: RowSelectionStrategy::Mask,
                    column_indices: vec![2, 9],
                    output_indices: vec![0, 2],
                },
                NativeColumnGroupPlan {
                    strategy: RowSelectionStrategy::Selectors,
                    column_indices: vec![5],
                    output_indices: vec![1],
                },
            ]
        );
    }

    #[test]
    fn grouped_arrays_scatter_back_to_projection_order() {
        let mask_arrays: Vec<ArrayRef> = vec![
            Arc::new(Int32Array::from(vec![20])),
            Arc::new(Int32Array::from(vec![40])),
        ];
        let selector_arrays: Vec<ArrayRef> = vec![
            Arc::new(Int32Array::from(vec![10])),
            Arc::new(Int32Array::from(vec![30])),
        ];
        let mut output = vec![None; 4];

        scatter_group_arrays(&mut output, &[1, 3], &mask_arrays).unwrap();
        scatter_group_arrays(&mut output, &[0, 2], &selector_arrays).unwrap();
        let values = output
            .into_iter()
            .map(|array| {
                array
                    .unwrap()
                    .as_any()
                    .downcast_ref::<Int32Array>()
                    .unwrap()
                    .value(0)
            })
            .collect::<Vec<_>>();
        assert_eq!(values, vec![10, 20, 30, 40]);
    }

    #[test]
    fn native_plan_splits_long_select_and_short_last_batch() {
        let selection = RowSelection::from(vec![
            RowSelector::skip(2),
            RowSelector::select(10),
            RowSelector::skip(3),
        ]);
        let plan = native_plan(selection, 4, true, true).unwrap();

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
        let plan = native_plan(selection, 4, true, false).unwrap();

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
        let plan = native_plan(selection, 2, true, true).unwrap();

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
    fn native_plan_mask_only_omits_instructions_but_keeps_batch_spans() {
        let selection = RowSelection::from(vec![
            RowSelector::skip(3),
            RowSelector::select(2),
            RowSelector::skip(5),
            RowSelector::select(6),
            RowSelector::skip(7),
            RowSelector::select(1),
            RowSelector::skip(4),
        ]);
        let plan = native_plan(selection, 4, false, true).unwrap();

        assert!(plan.instructions.is_empty());
        assert!(
            plan.batches
                .iter()
                .all(|batch| batch.instruction_range.is_empty())
        );
        assert_eq!(
            plan.batches
                .iter()
                .map(|batch| batch.span)
                .collect::<Vec<_>>(),
            vec![
                PhysicalSpan {
                    gap_skip: 3,
                    span_start: 3,
                    span_rows: 9,
                    selected: 4,
                },
                PhysicalSpan {
                    gap_skip: 0,
                    span_start: 12,
                    span_rows: 4,
                    selected: 4,
                },
                PhysicalSpan {
                    gap_skip: 7,
                    span_start: 23,
                    span_rows: 1,
                    selected: 1,
                },
            ]
        );
        let mask = plan.mask.unwrap();
        assert_eq!(mask.len(), 24);
        assert_eq!(mask.count_set_bits(), 9);
    }

    #[test]
    fn native_plan_borrows_selection_and_ignores_trailing_skip() {
        let original = vec![
            RowSelector::skip(2),
            RowSelector::select(3),
            RowSelector::skip(9),
        ];
        let selection = RowSelection::from(original.clone());
        let plan = NativeSelectionPlan::try_new(
            &selection,
            4,
            true,
            false,
            &ArrowReaderMetrics::disabled(),
        )
        .unwrap();

        assert_eq!(selection.iter().copied().collect::<Vec<_>>(), original);
        assert_eq!(batch_instructions(&plan, 0), &[RowSelector::select(3)]);
        assert_eq!(
            plan.batches[0].span,
            PhysicalSpan {
                gap_skip: 2,
                span_start: 2,
                span_rows: 3,
                selected: 3,
            }
        );
    }

    #[test]
    fn native_plan_mixed_keeps_selector_instructions_and_trimmed_mask() {
        let selection = RowSelection::from(vec![
            RowSelector::skip(1),
            RowSelector::select(2),
            RowSelector::skip(2),
            RowSelector::select(1),
            RowSelector::skip(5),
        ]);
        let plan = native_plan(selection, 4, true, true).unwrap();

        assert_eq!(
            batch_instructions(&plan, 0),
            &[
                RowSelector::select(2),
                RowSelector::skip(2),
                RowSelector::select(1),
            ]
        );
        let mask = plan.mask.unwrap();
        assert_eq!(mask.len(), 6);
        assert_eq!(
            mask.iter().collect::<Vec<_>>(),
            vec![false, true, true, false, false, true]
        );
    }

    #[test]
    fn native_plan_rejects_zero_batch_or_empty_selection() {
        let selected = RowSelection::from(vec![RowSelector::select(1)]);
        assert!(native_plan(selected.clone(), 0, true, false).is_err());
        assert!(native_plan(selected, 4, false, false).is_err());
        assert!(native_plan(RowSelection::from(vec![]), 4, true, false).is_err());
        assert!(
            native_plan(
                RowSelection::from(vec![RowSelector::skip(4)]),
                4,
                true,
                false,
            )
            .is_err()
        );
    }
}
