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

//! Benchmark for evaluating row filters and projections on a Parquet file.
//!
//! # Background:
//!
//! As described in [Efficient Filter Pushdown in Parquet], evaluating
//! pushdown filters is a two-step process:
//!
//! 1. Build a filter mask by decoding and evaluating filter functions on
//!    the filter column(s).
//!
//! 2. Decode the rows that match the filter mask from the projected columns.
//!
//! The performance depends on factors such as the number of rows selected,
//! the clustering of results (which affects the efficiency of the filter mask),
//! and whether the same column is used for both filtering and projection.
//!
//! This benchmark helps measure the performance of these operations.
//!
//! [Efficient Filter Pushdown in Parquet]: https://datafusion.apache.org/blog/2025/03/21/parquet-pushdown/
//!
//! The benchmark creates an in-memory Parquet file with 500K rows and four root
//! columns:
//! - `int64`: random integers with an injected point-lookup value.
//! - `float64`: random floating-point values used for sparse and dense filters.
//! - `utf8View`: ClickBench-like empty and non-empty string runs.
//! - `ts`: sequential timestamps used for clustered filters.
//!
//! The benchmark groups cover a few distinct reader-level questions:
//! - `arrow_reader_row_filter`: baseline filter/projection combinations.
//! - `arrow_reader_row_filter/row_selection_policy`: async row-filter pushdown
//!   with `Auto`, forced `Selectors`, and forced `Mask`.
//! - `arrow_reader_row_filter/manual_post_filter_diagnostic`: manual
//!   full-decode-then-filter execution for the same workloads. This is a
//!   diagnostic oracle, not a [`RowSelectionPolicy`] strategy or a complete
//!   query-engine post-scan pipeline.
//! - `arrow_reader_row_filter/predicate_order`: sequential [`RowFilter`]
//!   predicate execution under the three row-selection policies.

mod arrow_reader_common;

use arrow::array::{BooleanArray, Float64Array, Int64Array, TimestampMillisecondArray};
use arrow::compute::and;
use arrow::compute::kernels::cmp::{eq, gt, lt, lt_eq, neq};
use arrow::record_batch::RecordBatch;
use arrow_array::StringViewArray;
use arrow_reader_common::{
    COLUMN_NAMES, InMemoryReader, post_filter_projected_num_rows, projection_names,
    read_projection_for_post_filter, write_parquet_file,
};
use bytes::Bytes;
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use futures::StreamExt;
use parquet::arrow::arrow_reader::{
    ArrowPredicateFn, ParquetRecordBatchReaderBuilder, RowFilter, RowSelectionPolicy,
};
use parquet::arrow::{ParquetRecordBatchStreamBuilder, ProjectionMask};
use std::sync::Arc;

/// ProjectionCase defines the projection mode for the benchmark:
/// either projecting all columns or excluding the column that is used for filtering.
#[derive(Clone, Copy)]
enum ProjectionCase {
    AllColumns,
    ExcludeFilterColumn,
}

impl std::fmt::Display for ProjectionCase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ProjectionCase::AllColumns => write!(f, "all_columns"),
            ProjectionCase::ExcludeFilterColumn => write!(f, "exclude_filter_column"),
        }
    }
}

#[derive(Clone, Copy)]
enum PushdownPolicyCase {
    Auto,
    Selectors,
    Mask,
}

impl std::fmt::Display for PushdownPolicyCase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Auto => write!(f, "auto"),
            Self::Selectors => write!(f, "selectors"),
            Self::Mask => write!(f, "mask"),
        }
    }
}

impl PushdownPolicyCase {
    fn row_selection_policy(self) -> RowSelectionPolicy {
        match self {
            Self::Auto => RowSelectionPolicy::default(),
            Self::Selectors => RowSelectionPolicy::Selectors,
            Self::Mask => RowSelectionPolicy::Mask,
        }
    }
}

const PUSHDOWN_POLICIES: &[PushdownPolicyCase] = &[
    PushdownPolicyCase::Auto,
    PushdownPolicyCase::Selectors,
    PushdownPolicyCase::Mask,
];

/// FilterType encapsulates the different filter comparisons.
/// The variants correspond to the different filter patterns.
#[derive(Clone, Copy, Debug)]
enum FilterType {
    /// point lookup: selects a single row in 500K.
    /// ```text
    /// ┌───────────────┐    ┌───────────────┐
    /// │               │    │               │
    /// │               │    │      ...      │
    /// │               │    │               │
    /// │               │    │               │
    /// │      ...      │    │▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒│
    /// │               │    │               │
    /// │               │    │      ...      │
    /// │               │    │               │
    /// │               │    │               │
    /// └───────────────┘    └───────────────┘
    /// ```
    PointLookup,
    /// selective (1%) unclustered filter: approx 5K selected rows in 500K.
    /// ```text
    /// ┌───────────────┐    ┌───────────────┐
    /// │      ...      │    │               │
    /// │               │    │               │
    /// │▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒│    │               │
    /// │               │    │      ...      │
    /// │               │    │               │
    /// │               │    │               │
    /// │      ...      │    │               │
    /// │               │    │▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒│
    /// │               │    │               │
    /// └───────────────┘    └───────────────┘
    /// ```
    SelectiveUnclustered,
    /// moderately selective (10%) clustered filter: 50 selected runs of 1K
    /// rows each in 500K.
    /// ```text
    /// ┌───────────────┐    ┌───────────────┐
    /// │               │    │               │
    /// │               │    │               │
    /// │      ...      │    │      ...      │
    /// │               │    │               │
    /// │               │    │               │
    /// │               │    │               │
    /// │               │    │               │
    /// │▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒│    │▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒│
    /// │▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒│    │▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒│
    /// └───────────────┘    └───────────────┘
    /// ```
    ModeratelySelectiveClustered,
    /// moderately selective (~9%) unclustered filter: approx 45K selected
    /// rows in 500K.
    /// ```text
    /// ┌───────────────┐    ┌───────────────┐
    /// │      ...      │    │               │
    /// │               │    │               │
    /// │▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒│    │▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒│
    /// │               │    │               │
    /// │               │    │               │
    /// │               │    │      ...      │
    /// │      ...      │    │               │
    /// │               │    │               │
    /// │               │    │▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒│
    /// └───────────────┘    └───────────────┘
    /// ```
    ModeratelySelectiveUnclustered,
    /// unselective (99%) unclustered filter: approx 495K selected rows in
    /// 500K.
    /// ```text
    /// ┌───────────────┐    ┌───────────────┐
    /// │▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒│    │▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒│
    /// │▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒│    │▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒│
    /// │▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒│    │▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒│
    /// │▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒│    │▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒│
    /// │▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒│    │▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒│
    /// │▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒│    │               │
    /// │▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒│    │▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒│
    /// │▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒│    │▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒│
    /// │▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒│    │▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒│
    /// └───────────────┘    └───────────────┘
    /// ```
    UnselectiveUnclustered,
    /// unselective (90%) clustered filter: 50 selected runs of 9K rows each
    /// in 500K.
    /// ```text
    /// ┌───────────────┐    ┌───────────────┐
    /// │▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒│    │▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒│
    /// │▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒│    │▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒│
    /// │▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒│    │▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒│
    /// │▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒│    │▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒│
    /// │▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒│    │▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒│
    /// │▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒│    │▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒│
    /// │▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒│    │▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒│
    /// │▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒│    │      ...      │
    /// │               │    │               │
    /// └───────────────┘    └───────────────┘
    /// ```
    UnselectiveClustered,
    /// composite sparse filter: `SelectiveUnclustered` AND
    /// `ModeratelySelectiveClustered`, approx 0.1% selected rows in 500K.
    /// ```text
    /// ┌───────────────┐    ┌───────────────┐
    /// │               │    │               │
    /// │               │    │      ...      │
    /// │               │    │               │
    /// │               │    │               │
    /// │      ...      │    │               │
    /// │               │    │               │
    /// │               │    │▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒│
    /// │               │    │               │
    /// │▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒│    │               │
    /// └───────────────┘    └───────────────┘
    /// ```
    Composite,
    /// `utf8View <> ''` modeling [ClickBench] [Q21-Q27] with fragmented
    /// empty and non-empty string runs.
    /// ```text
    /// ┌───────────────┐    ┌───────────────┐
    /// │               │    │               │
    /// │      ...      │    │      ...      │
    /// │▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒│    │               │
    /// │▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒│    │               │
    /// │               │    │▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒│
    /// │               │    │▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒│
    /// │      ...      │    │      ...      │
    /// │               │    │               │
    /// │▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒│    │               │
    /// └───────────────┘    └───────────────┘
    /// ```
    ///
    /// [ClickBench]: https://github.com/ClickHouse/ClickBench
    /// [Q21-Q27]: https://github.com/apache/datafusion/blob/b7177234e65cbbb2dcc04c252f6acd80bb026362/benchmarks/queries/clickbench/queries.sql#L22-L28
    Utf8ViewNonEmpty,
}

impl std::fmt::Display for FilterType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            FilterType::PointLookup => "int64 == 9999",
            FilterType::SelectiveUnclustered => "float64 > 99.0",
            FilterType::ModeratelySelectiveClustered => "ts >= 9000",
            FilterType::ModeratelySelectiveUnclustered => "int64 > 90",
            FilterType::UnselectiveUnclustered => "float64 <= 99.0",
            FilterType::UnselectiveClustered => "ts < 9000",
            FilterType::Composite => "float64 > 99.0 AND ts >= 9000",
            FilterType::Utf8ViewNonEmpty => "utf8View <> ''",
        };
        write!(f, "{s}")
    }
}

impl FilterType {
    /// Applies the specified filter on the given RecordBatch and returns a BooleanArray mask.
    fn filter_batch(&self, batch: &RecordBatch) -> arrow::error::Result<BooleanArray> {
        match self {
            // Point Lookup on int64 column
            FilterType::PointLookup => {
                let array = batch.column(batch.schema().index_of("int64")?);
                let scalar = Int64Array::new_scalar(9999);
                eq(array, &scalar)
            }
            // Selective Unclustered on float64 column: float64 > 99.0
            FilterType::SelectiveUnclustered => {
                let array = batch.column(batch.schema().index_of("float64")?);
                let scalar = Float64Array::new_scalar(99.0);
                gt(array, &scalar)
            }
            // Moderately Selective Clustered on ts column: ts >= 9000 (implemented as > 8999)
            FilterType::ModeratelySelectiveClustered => {
                let array = batch.column(batch.schema().index_of("ts")?);
                gt(array, &TimestampMillisecondArray::new_scalar(8999))
            }
            // Moderately Selective Unclustered on int64 column: int64 > 90
            FilterType::ModeratelySelectiveUnclustered => {
                let array = batch.column(batch.schema().index_of("int64")?);
                let scalar = Int64Array::new_scalar(90);
                gt(array, &scalar)
            }
            // Unselective Unclustered on float64 column: NOT (float64 > 99.0)
            FilterType::UnselectiveUnclustered => {
                let array = batch.column(batch.schema().index_of("float64")?);
                lt_eq(array, &Float64Array::new_scalar(99.0))
            }
            // Unselective Clustered on ts column: ts < 9000
            FilterType::UnselectiveClustered => {
                let array = batch.column(batch.schema().index_of("ts")?);
                lt(array, &TimestampMillisecondArray::new_scalar(9000))
            }
            // Composite filter: logical AND of (float64 > 99.0) and (ts >= 9000)
            FilterType::Composite => {
                let mask1 = FilterType::SelectiveUnclustered.filter_batch(batch)?;
                let mask2 = FilterType::ModeratelySelectiveClustered.filter_batch(batch)?;
                and(&mask1, &mask2)
            }
            // Utf8ViewNonEmpty: selects rows where the utf8View column is not an empty string.
            FilterType::Utf8ViewNonEmpty => {
                let array = batch.column(batch.schema().index_of("utf8View")?);
                let scalar = StringViewArray::new_scalar("");
                neq(array, &scalar)
            }
        }
    }

    /// Return the indexes in the batch's schema that are used for filtering.
    fn filter_projection(&self) -> &'static [usize] {
        match self {
            FilterType::PointLookup => &[0],
            FilterType::SelectiveUnclustered => &[1],
            FilterType::ModeratelySelectiveClustered => &[3],
            FilterType::ModeratelySelectiveUnclustered => &[0],
            FilterType::UnselectiveUnclustered => &[1],
            FilterType::UnselectiveClustered => &[3],
            FilterType::Composite => &[1, 3], // Use float64 column and ts column as representative for composite
            FilterType::Utf8ViewNonEmpty => &[2],
        }
    }
}

/// Benchmark filters and projections by reading the Parquet file.
/// This benchmark iterates over all individual filter types and two projection cases.
/// It measures the time to read and filter the Parquet file according to each scenario.
fn benchmark_filters_and_projections(c: &mut Criterion) {
    // make the parquet file in memory that can be shared
    let parquet_file = Bytes::from(write_parquet_file());
    let filter_types = vec![
        FilterType::PointLookup,
        FilterType::SelectiveUnclustered,
        FilterType::ModeratelySelectiveClustered,
        FilterType::ModeratelySelectiveUnclustered,
        FilterType::UnselectiveUnclustered,
        FilterType::UnselectiveClustered,
        FilterType::Utf8ViewNonEmpty,
        FilterType::Composite,
    ];
    let projection_cases = vec![
        ProjectionCase::AllColumns,
        ProjectionCase::ExcludeFilterColumn,
    ];

    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();

    let mut group = c.benchmark_group("arrow_reader_row_filter");

    for filter_type in filter_types {
        for proj_case in &projection_cases {
            let filter_col = filter_type.filter_projection().to_vec();
            let output_projection = output_projection_for(filter_type, proj_case);

            let reader = InMemoryReader::try_new(&parquet_file).unwrap();
            let metadata = Arc::clone(reader.metadata());

            let schema_descr = metadata.file_metadata().schema_descr();
            let projection_mask = ProjectionMask::roots(schema_descr, output_projection.clone());
            let pred_mask = ProjectionMask::roots(schema_descr, filter_col.clone());

            let benchmark_name = format!("{filter_type}/{proj_case}",);

            // run the benchmark for the async reader
            let bench_id = BenchmarkId::new(benchmark_name.clone(), "async");
            let rt_captured = rt.handle().clone();
            group.bench_function(bench_id, |b| {
                b.iter(|| {
                    let reader = reader.clone();
                    let pred_mask = pred_mask.clone();
                    let projection_mask = projection_mask.clone();
                    // row filters are not clone, so must make it each iter
                    let filter = ArrowPredicateFn::new(pred_mask, move |batch: RecordBatch| {
                        Ok(filter_type.filter_batch(&batch).unwrap())
                    });
                    let row_filter = RowFilter::new(vec![Box::new(filter)]);

                    rt_captured.block_on(async {
                        benchmark_async_reader(reader, projection_mask, row_filter).await;
                    })
                });
            });

            // run the benchmark for the sync reader
            let bench_id = BenchmarkId::new(benchmark_name, "sync");
            group.bench_function(bench_id, |b| {
                b.iter(|| {
                    let reader = reader.clone();
                    let pred_mask = pred_mask.clone();
                    let projection_mask = projection_mask.clone();
                    // row filters are not clone, so must make it each iter
                    let filter = ArrowPredicateFn::new(pred_mask, move |batch: RecordBatch| {
                        Ok(filter_type.filter_batch(&batch).unwrap())
                    });
                    let row_filter = RowFilter::new(vec![Box::new(filter)]);

                    benchmark_sync_reader(reader, projection_mask, row_filter)
                });
            });
        }
    }
}

#[derive(Clone, Copy)]
struct AsyncRowFilterCase {
    name: &'static str,
    filter_type: FilterType,
    projection_case: ProjectionCase,
}

const ASYNC_ROW_FILTER_CASES: &[AsyncRowFilterCase] = &[
    AsyncRowFilterCase {
        name: "selective_unclustered/all_columns",
        filter_type: FilterType::SelectiveUnclustered,
        projection_case: ProjectionCase::AllColumns,
    },
    AsyncRowFilterCase {
        name: "selective_unclustered/exclude_filter_column",
        filter_type: FilterType::SelectiveUnclustered,
        projection_case: ProjectionCase::ExcludeFilterColumn,
    },
    AsyncRowFilterCase {
        name: "utf8view_non_empty/all_columns",
        filter_type: FilterType::Utf8ViewNonEmpty,
        projection_case: ProjectionCase::AllColumns,
    },
    AsyncRowFilterCase {
        name: "utf8view_non_empty/exclude_filter_column",
        filter_type: FilterType::Utf8ViewNonEmpty,
        projection_case: ProjectionCase::ExcludeFilterColumn,
    },
];

struct AsyncRowFilterInput {
    reader: InMemoryReader,
    output_projection: ProjectionMask,
    read_projection: ProjectionMask,
    predicate_projection: ProjectionMask,
    output_column_names: Vec<&'static str>,
}

fn prepare_async_row_filter_input(
    parquet_file: &Bytes,
    case: AsyncRowFilterCase,
) -> AsyncRowFilterInput {
    let reader = InMemoryReader::try_new(parquet_file).unwrap();
    let metadata = Arc::clone(reader.metadata());
    let schema_descr = metadata.file_metadata().schema_descr();
    let output_projection = output_projection_for(case.filter_type, &case.projection_case);
    let read_projection =
        read_projection_for_post_filter(&output_projection, case.filter_type.filter_projection());
    let output_column_names = projection_names(&output_projection);

    AsyncRowFilterInput {
        reader,
        output_projection: ProjectionMask::roots(schema_descr, output_projection),
        read_projection: ProjectionMask::roots(schema_descr, read_projection),
        predicate_projection: ProjectionMask::roots(
            schema_descr,
            case.filter_type.filter_projection().iter().copied(),
        ),
        output_column_names,
    }
}

/// Exercise reader-level `Auto`, forced `Selectors`, and forced `Mask` using
/// selection shapes produced by real [`RowFilter`] predicates.
fn benchmark_async_row_selection_policy(c: &mut Criterion) {
    let parquet_file = Bytes::from(write_parquet_file());

    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();

    let mut group = c.benchmark_group("arrow_reader_row_filter/row_selection_policy");

    for &case in ASYNC_ROW_FILTER_CASES {
        let input = prepare_async_row_filter_input(&parquet_file, case);
        for &policy_case in PUSHDOWN_POLICIES {
            let bench_id = BenchmarkId::new(case.name, policy_case.to_string());
            let rt_captured = rt.handle().clone();

            group.bench_function(bench_id, |b| {
                b.iter(|| {
                    let reader = input.reader.clone();
                    let predicate_projection = input.predicate_projection.clone();
                    let output_projection = input.output_projection.clone();

                    rt_captured.block_on(benchmark_async_reader_with_policy(
                        reader,
                        output_projection,
                        row_filter_for(case.filter_type, predicate_projection),
                        policy_case.row_selection_policy(),
                    ))
                });
            });
        }
    }
}

/// Manually decode predicate and output columns, evaluate the predicate with
/// Arrow kernels, and filter the output projection. This is a diagnostic
/// oracle for filter placement, not a `RowSelectionPolicy` strategy or a
/// complete query-engine post-scan pipeline.
fn benchmark_manual_post_filter_diagnostic(c: &mut Criterion) {
    let parquet_file = Bytes::from(write_parquet_file());
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();
    let mut group = c.benchmark_group("arrow_reader_row_filter/manual_post_filter_diagnostic");

    for &case in ASYNC_ROW_FILTER_CASES {
        let input = prepare_async_row_filter_input(&parquet_file, case);
        let rt_captured = rt.handle().clone();

        group.bench_function(case.name, |b| {
            b.iter(|| {
                rt_captured.block_on(benchmark_async_reader_post_filter(
                    input.reader.clone(),
                    input.read_projection.clone(),
                    input.output_column_names.clone(),
                    case.filter_type,
                ))
            });
        });
    }
}

/// Isolate sequential [`RowFilter`] predicate ordering.
///
/// The existing `Composite` filter evaluates both predicates inside one
/// [`ArrowPredicateFn`]. This focus case uses two chained predicates so the
/// reader can prune rows after the cheap fixed-width predicate before deciding
/// whether to decode the variable-width predicate column.
fn benchmark_async_predicate_order_focus(c: &mut Criterion) {
    let parquet_file = Bytes::from(write_parquet_file());
    let predicate_orders = [
        PredicateOrder::FixedThenVarWidth,
        PredicateOrder::VarWidthThenFixed,
    ];

    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();

    let mut group = c.benchmark_group("arrow_reader_row_filter/predicate_order");

    let reader = InMemoryReader::try_new(&parquet_file).unwrap();
    let metadata = Arc::clone(reader.metadata());
    let schema_descr = metadata.file_metadata().schema_descr();
    let projection_mask = ProjectionMask::roots(schema_descr, [1]);
    let fixed_pred_mask = ProjectionMask::roots(schema_descr, [0]);
    let varwidth_pred_mask = ProjectionMask::roots(schema_descr, [2]);

    for predicate_order in predicate_orders {
        for &policy_case in PUSHDOWN_POLICIES {
            let bench_id = BenchmarkId::new(
                format!("{predicate_order}/float64_only"),
                policy_case.to_string(),
            );
            let rt_captured = rt.handle().clone();

            group.bench_function(bench_id, |b| {
                b.iter(|| {
                    let reader = reader.clone();
                    let projection_mask = projection_mask.clone();
                    let fixed_pred_mask = fixed_pred_mask.clone();
                    let varwidth_pred_mask = varwidth_pred_mask.clone();

                    rt_captured.block_on(async {
                        benchmark_async_reader_with_policy(
                            reader,
                            projection_mask,
                            chained_row_filter_for(
                                predicate_order,
                                fixed_pred_mask,
                                varwidth_pred_mask,
                            ),
                            policy_case.row_selection_policy(),
                        )
                        .await
                    });
                });
            });
        }
    }
}

fn output_projection_for(filter_type: FilterType, projection_case: &ProjectionCase) -> Vec<usize> {
    let filter_columns = filter_type.filter_projection();
    match projection_case {
        ProjectionCase::AllColumns | ProjectionCase::ExcludeFilterColumn => COLUMN_NAMES
            .iter()
            .enumerate()
            .map(|(idx, _)| idx)
            .filter(move |idx| {
                matches!(projection_case, ProjectionCase::AllColumns)
                    || !filter_columns.contains(idx)
            })
            .collect(),
    }
}

fn row_filter_for(filter_type: FilterType, pred_mask: ProjectionMask) -> RowFilter {
    let filter = ArrowPredicateFn::new(pred_mask, move |batch| filter_type.filter_batch(&batch));
    RowFilter::new(vec![Box::new(filter)])
}

#[derive(Clone, Copy)]
enum PredicateOrder {
    FixedThenVarWidth,
    VarWidthThenFixed,
}

impl std::fmt::Display for PredicateOrder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::FixedThenVarWidth => write!(f, "fixed_then_varwidth"),
            Self::VarWidthThenFixed => write!(f, "varwidth_then_fixed"),
        }
    }
}

fn chained_row_filter_for(
    predicate_order: PredicateOrder,
    fixed_pred_mask: ProjectionMask,
    varwidth_pred_mask: ProjectionMask,
) -> RowFilter {
    let int64_filter = ArrowPredicateFn::new(fixed_pred_mask, move |batch: RecordBatch| {
        let int64 = batch.column(batch.schema().index_of("int64")?);
        eq(int64, &Int64Array::new_scalar(9999))
    });
    let utf8_filter = ArrowPredicateFn::new(varwidth_pred_mask, move |batch: RecordBatch| {
        let utf8 = batch.column(batch.schema().index_of("utf8View")?);
        neq(utf8, &StringViewArray::new_scalar(""))
    });

    match predicate_order {
        PredicateOrder::FixedThenVarWidth => {
            RowFilter::new(vec![Box::new(int64_filter), Box::new(utf8_filter)])
        }
        PredicateOrder::VarWidthThenFixed => {
            RowFilter::new(vec![Box::new(utf8_filter), Box::new(int64_filter)])
        }
    }
}

/// Use async API
async fn benchmark_async_reader(
    reader: InMemoryReader,
    projection_mask: ProjectionMask,
    row_filter: RowFilter,
) {
    let mut stream = ParquetRecordBatchStreamBuilder::new(reader)
        .await
        .unwrap()
        .with_batch_size(8192)
        .with_projection(projection_mask)
        .with_row_filter(row_filter)
        .build()
        .unwrap();
    while let Some(b) = stream.next().await {
        b.unwrap(); // consume the batches, no buffering
    }
}

async fn benchmark_async_reader_with_policy(
    reader: InMemoryReader,
    projection_mask: ProjectionMask,
    row_filter: RowFilter,
    row_selection_policy: RowSelectionPolicy,
) {
    let mut stream = ParquetRecordBatchStreamBuilder::new(reader)
        .await
        .unwrap()
        .with_batch_size(8192)
        .with_projection(projection_mask)
        .with_row_filter(row_filter)
        .with_row_selection_policy(row_selection_policy)
        .build()
        .unwrap();
    while let Some(b) = stream.next().await {
        b.unwrap(); // consume the batches, no buffering
    }
}

async fn benchmark_async_reader_post_filter(
    reader: InMemoryReader,
    read_projection: ProjectionMask,
    output_column_names: Vec<&'static str>,
    filter_type: FilterType,
) {
    let mut stream = ParquetRecordBatchStreamBuilder::new(reader)
        .await
        .unwrap()
        .with_batch_size(8192)
        .with_projection(read_projection)
        .build()
        .unwrap();

    while let Some(b) = stream.next().await {
        let batch = b.unwrap();
        let filter = filter_type.filter_batch(&batch).unwrap();
        let output_rows =
            post_filter_projected_num_rows(&batch, &filter, &output_column_names).unwrap();
        std::hint::black_box(output_rows);
    }
}

/// Like [`benchmark_async_reader`] but also threads `with_limit(limit)` into
/// the stream builder. Used by the `LIMIT` benchmark below.
async fn benchmark_async_reader_with_limit(
    reader: InMemoryReader,
    projection_mask: ProjectionMask,
    row_filter: RowFilter,
    limit: usize,
) {
    let mut stream = ParquetRecordBatchStreamBuilder::new(reader)
        .await
        .unwrap()
        .with_batch_size(8192)
        .with_projection(projection_mask)
        .with_row_filter(row_filter)
        .with_limit(limit)
        .build()
        .unwrap();
    while let Some(b) = stream.next().await {
        b.unwrap(); // consume the batches, no buffering
    }
}

/// Use sync API
fn benchmark_sync_reader(
    reader: InMemoryReader,
    projection_mask: ProjectionMask,
    row_filter: RowFilter,
) {
    let stream = ParquetRecordBatchReaderBuilder::try_new(reader.into_inner())
        .unwrap()
        .with_batch_size(8192)
        .with_projection(projection_mask)
        .with_row_filter(row_filter)
        .build()
        .unwrap();
    for b in stream {
        b.unwrap(); // consume the batches, no buffering
    }
}

/// Benchmark filters with `LIMIT` short-circuit (`with_limit(N)`)
///
/// `PointLookup` is excluded because the filter has only 1 match in the
/// whole file; `LIMIT 10` is not binding.
fn benchmark_filters_with_limit(c: &mut Criterion) {
    const LIMIT: usize = 10;

    let parquet_file = Bytes::from(write_parquet_file());
    let filter_types = vec![
        FilterType::SelectiveUnclustered,
        FilterType::ModeratelySelectiveClustered,
        FilterType::ModeratelySelectiveUnclustered,
        FilterType::UnselectiveUnclustered,
        FilterType::UnselectiveClustered,
        FilterType::Utf8ViewNonEmpty,
        FilterType::Composite,
    ];
    let projection_cases = vec![
        ProjectionCase::AllColumns,
        ProjectionCase::ExcludeFilterColumn,
    ];

    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();

    let mut group = c.benchmark_group("arrow_reader_row_filter_limit");

    for filter_type in filter_types {
        for proj_case in &projection_cases {
            let filter_col = filter_type.filter_projection().to_vec();
            let output_projection = output_projection_for(filter_type, proj_case);

            let reader = InMemoryReader::try_new(&parquet_file).unwrap();
            let metadata = Arc::clone(reader.metadata());
            let schema_descr = metadata.file_metadata().schema_descr();
            let projection_mask = ProjectionMask::roots(schema_descr, output_projection);
            let pred_mask = ProjectionMask::roots(schema_descr, filter_col);

            let benchmark_name = format!("{filter_type}/{proj_case}/limit{LIMIT}");

            // async variant
            let bench_id = BenchmarkId::new(benchmark_name.clone(), "async");
            let rt_handle = rt.handle().clone();
            let pred_mask_async = pred_mask.clone();
            let projection_mask_async = projection_mask.clone();
            let reader_async = reader.clone();
            group.bench_function(bench_id, |b| {
                b.iter(|| {
                    let reader = reader_async.clone();
                    let pred_mask = pred_mask_async.clone();
                    let projection_mask = projection_mask_async.clone();
                    // RowFilter and ArrowPredicateFn are not Clone — fresh each iter.
                    let predicate = ArrowPredicateFn::new(pred_mask, move |batch: RecordBatch| {
                        Ok(filter_type.filter_batch(&batch).unwrap())
                    });
                    let row_filter = RowFilter::new(vec![Box::new(predicate)]);
                    rt_handle.block_on(benchmark_async_reader_with_limit(
                        reader,
                        projection_mask,
                        row_filter,
                        LIMIT,
                    ));
                });
            });
        }
    }
}

criterion_group!(
    benches,
    benchmark_filters_and_projections,
    benchmark_async_row_selection_policy,
    benchmark_manual_post_filter_diagnostic,
    benchmark_async_predicate_order_focus,
    benchmark_filters_with_limit,
);
criterion_main!(benches);
