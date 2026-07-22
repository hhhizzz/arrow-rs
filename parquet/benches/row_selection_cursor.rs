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

use std::hint;
use std::sync::{Arc, OnceLock};

use arrow_array::builder::StringViewBuilder;
use arrow_array::cast::AsArray;
use arrow_array::types::Int32Type;
use arrow_array::{ArrayRef, BooleanArray, Float64Array, Int32Array, RecordBatch, StringViewArray};
use arrow_schema::{DataType, Field, Schema};
use arrow_select::filter::filter_record_batch;
use bytes::Bytes;
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use parquet::arrow::arrow_reader::{
    ArrowPredicateFn, ParquetRecordBatchReader, ParquetRecordBatchReaderBuilder, RowFilter,
    RowSelection, RowSelectionPolicy, RowSelector,
};
use parquet::arrow::{ArrowWriter, ProjectionMask};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

const TOTAL_ROWS: usize = 1 << 20;
const BATCH_SIZE: usize = 1 << 10;
const BASE_SEED: u64 = 0xA55AA55A;
const AVG_SELECTOR_LENGTHS: &[usize] = &[4, 8, 12, 16, 20, 24, 28, 32, 36, 40];
const COLUMN_WIDTHS: &[usize] = &[2, 4, 8, 16, 32];
const UTF8VIEW_LENS: &[usize] = &[4, 8, 16, 32, 64, 128, 256];
const BENCH_MODES: &[BenchMode] = &[BenchMode::ReadSelector, BenchMode::ReadMask];
const PAYLOAD_COLUMNS: usize = 8;

#[derive(Clone, Copy)]
struct RunPair {
    skip: usize,
    select: usize,
}

impl RunPair {
    const fn new(skip: usize, select: usize) -> Self {
        Self { skip, select }
    }

    const fn len(self) -> usize {
        self.skip + self.select
    }
}

#[derive(Clone, Copy)]
enum FilterShape {
    Regular(RunPair),
    Bursty(&'static [RunPair]),
}

impl FilterShape {
    fn is_selected(self, row: usize) -> bool {
        match self {
            Self::Regular(run) => row % run.len() >= run.skip,
            Self::Bursty(runs) => {
                let cycle_len: usize = runs.iter().map(|run| run.len()).sum();
                let mut offset = row % cycle_len;

                for run in runs {
                    if offset < run.skip {
                        return false;
                    }
                    offset -= run.skip;

                    if offset < run.select {
                        return true;
                    }
                    offset -= run.select;
                }

                unreachable!("offset must fall within a bursty run")
            }
        }
    }
}

#[derive(Clone, Copy)]
struct FilterShapeCase {
    id: &'static str,
    shape: FilterShape,
    expected_selected_rows: usize,
    expected_run_count: usize,
}

const BURSTY_RUNS: &[RunPair] = &[
    RunPair::new(1, 1),
    RunPair::new(1, 1),
    RunPair::new(1, 1),
    RunPair::new(125, 125),
];

const FILTER_SHAPE_CASES: &[FilterShapeCase] = &[
    FilterShapeCase {
        id: "regular-sel0156-run0032",
        shape: FilterShape::Regular(RunPair::new(63, 1)),
        expected_selected_rows: TOTAL_ROWS / 64,
        expected_run_count: TOTAL_ROWS / 32,
    },
    FilterShapeCase {
        id: "regular-sel1250-run0032",
        shape: FilterShape::Regular(RunPair::new(56, 8)),
        expected_selected_rows: TOTAL_ROWS / 8,
        expected_run_count: TOTAL_ROWS / 32,
    },
    FilterShapeCase {
        id: "regular-sel5000-run0032",
        shape: FilterShape::Regular(RunPair::new(32, 32)),
        expected_selected_rows: TOTAL_ROWS / 2,
        expected_run_count: TOTAL_ROWS / 32,
    },
    FilterShapeCase {
        id: "regular-sel8750-run0032",
        shape: FilterShape::Regular(RunPair::new(8, 56)),
        expected_selected_rows: TOTAL_ROWS * 7 / 8,
        expected_run_count: TOTAL_ROWS / 32,
    },
    FilterShapeCase {
        id: "regular-sel9844-run0032",
        shape: FilterShape::Regular(RunPair::new(1, 63)),
        expected_selected_rows: TOTAL_ROWS * 63 / 64,
        expected_run_count: TOTAL_ROWS / 32,
    },
    FilterShapeCase {
        id: "regular-sel5000-run0001",
        shape: FilterShape::Regular(RunPair::new(1, 1)),
        expected_selected_rows: TOTAL_ROWS / 2,
        expected_run_count: TOTAL_ROWS,
    },
    FilterShapeCase {
        id: "regular-sel5000-run0008",
        shape: FilterShape::Regular(RunPair::new(8, 8)),
        expected_selected_rows: TOTAL_ROWS / 2,
        expected_run_count: TOTAL_ROWS / 8,
    },
    FilterShapeCase {
        id: "regular-sel5000-run0128",
        shape: FilterShape::Regular(RunPair::new(128, 128)),
        expected_selected_rows: TOTAL_ROWS / 2,
        expected_run_count: TOTAL_ROWS / 128,
    },
    FilterShapeCase {
        id: "regular-sel5000-run1024",
        shape: FilterShape::Regular(RunPair::new(1024, 1024)),
        expected_selected_rows: TOTAL_ROWS / 2,
        expected_run_count: TOTAL_ROWS / 1024,
    },
    FilterShapeCase {
        id: "bursty-sel5000-run0032",
        shape: FilterShape::Bursty(BURSTY_RUNS),
        expected_selected_rows: TOTAL_ROWS / 2,
        expected_run_count: TOTAL_ROWS / 32,
    },
];

#[derive(Clone, Copy)]
enum FilterReadMode {
    Auto,
    Fallback,
}

impl FilterReadMode {
    fn label(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Fallback => "fallback",
        }
    }
}

const FALLBACK_PAYLOAD_COLUMNS: [usize; PAYLOAD_COLUMNS] = [1, 2, 3, 4, 5, 6, 7, 8];

fn payload_leaf_indices() -> std::ops::Range<usize> {
    FILTER_SHAPE_CASES.len()..FILTER_SHAPE_CASES.len() + PAYLOAD_COLUMNS
}

#[derive(Clone, Copy)]
enum ExistingFixture {
    Batch(fn(usize) -> RecordBatch),
    Int32Columns(usize),
    Utf8ViewLen(usize),
}

impl ExistingFixture {
    fn build(self) -> Bytes {
        match self {
            Self::Batch(build_batch) => build_parquet_data(TOTAL_ROWS, build_batch),
            Self::Int32Columns(column_count) => {
                write_parquet_batch(build_int32_columns_batch(TOTAL_ROWS, column_count))
            }
            Self::Utf8ViewLen(len) => {
                write_parquet_batch(build_utf8view_batch_with_len(TOTAL_ROWS, len))
            }
        }
    }
}

struct DataProfile {
    name: &'static str,
    build_batch: fn(usize) -> RecordBatch,
}

const DATA_PROFILES: &[DataProfile] = &[
    DataProfile {
        name: "int32",
        build_batch: build_int32_batch,
    },
    DataProfile {
        name: "float64",
        build_batch: build_float64_batch,
    },
    DataProfile {
        name: "utf8view",
        build_batch: build_utf8view_batch,
    },
];

fn bench_forced_policy_sweep(c: &mut Criterion) {
    let scenarios = [
        /* uniform50 (50% selected, constant run lengths, starts with skip)
        ```text
        ┌───────────────┐
        │               │  skip
        │               │
        │▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒│  select
        │▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒│
        │               │  skip
        │               │
        │▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒│  select
        │▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒│
        │      ...      │
        └───────────────┘
        ``` */
        Scenario {
            name: "uniform50",
            select_ratio: 0.5,
            start_with_select: false,
            distribution: RunDistribution::Constant,
        },
        /* spread50 (50% selected, large jitter in run lengths, starts with skip)
        ```text
        ┌───────────────┐
        │               │  skip (long)
        │               │
        │▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒│  select (short)
        │               │  skip (short)
        │▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒│  select (long)
        │▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒│
        │               │  skip (medium)
        │▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒│  select (medium)
        │      ...      │
        └───────────────┘
        ``` */
        Scenario {
            name: "spread50",
            select_ratio: 0.5,
            start_with_select: false,
            distribution: RunDistribution::Uniform { spread: 0.9 },
        },
        /* sparse20 (20% selected, bimodal: occasional long runs, starts with skip)
        ```text
        ┌───────────────┐
        │               │  skip (long)
        │               │
        │               │
        │▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒│  select (short)
        │               │  skip (long)
        │               │
        │▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒│  select (occasional long)
        │▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒│
        │      ...      │
        └───────────────┘
        ``` */
        Scenario {
            name: "sparse20",
            select_ratio: 0.2,
            start_with_select: false,
            distribution: RunDistribution::Bimodal {
                long_factor: 6.0,
                long_prob: 0.1,
            },
        },
        /* dense80 (80% selected, bimodal: occasional long runs, starts with select)
        ```text
        ┌───────────────┐
        │▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒│  select (long)
        │▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒│
        │               │  skip (short)
        │▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒│  select (long)
        │▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒│
        │               │  skip (very short)
        │▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒│  select (long)
        │▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒│
        │      ...      │
        └───────────────┘
        ``` */
        Scenario {
            name: "dense80",
            select_ratio: 0.8,
            start_with_select: true,
            distribution: RunDistribution::Bimodal {
                long_factor: 4.0,
                long_prob: 0.05,
            },
        },
    ];

    let base_parquet = Arc::new(OnceLock::new());
    let base_scenario = &scenarios[0];

    for (idx, scenario) in scenarios.iter().enumerate() {
        // The first scenario is a special case for backwards compatibility with
        // existing benchmark result formats.
        let suite = if idx == 0 { "len" } else { "scenario" };
        bench_over_lengths(
            c,
            suite,
            scenario.name,
            Arc::clone(&base_parquet),
            ExistingFixture::Batch(build_int32_batch),
            scenario,
            BASE_SEED ^ ((idx as u64) << 16),
        );
    }

    for (profile_idx, profile) in DATA_PROFILES.iter().enumerate() {
        bench_over_lengths(
            c,
            "dtype",
            profile.name,
            Arc::new(OnceLock::new()),
            ExistingFixture::Batch(profile.build_batch),
            base_scenario,
            BASE_SEED ^ ((profile_idx as u64) << 24),
        );
    }

    for (offset, &column_count) in COLUMN_WIDTHS.iter().enumerate() {
        let variant_label = format!("C{:02}", column_count);
        bench_over_lengths(
            c,
            "columns",
            &variant_label,
            Arc::new(OnceLock::new()),
            ExistingFixture::Int32Columns(column_count),
            base_scenario,
            BASE_SEED ^ ((offset as u64) << 32),
        );
    }

    for (offset, &len) in UTF8VIEW_LENS.iter().enumerate() {
        let variant_label = format!("utf8view-L{:03}", len);
        bench_over_lengths(
            c,
            "utf8view-len",
            &variant_label,
            Arc::new(OnceLock::new()),
            ExistingFixture::Utf8ViewLen(len),
            base_scenario,
            BASE_SEED ^ ((offset as u64) << 40),
        );
    }
}

fn bench_over_lengths(
    c: &mut Criterion,
    suite: &str,
    variant: &str,
    parquet_data: Arc<OnceLock<Bytes>>,
    fixture: ExistingFixture,
    scenario: &Scenario,
    seed_base: u64,
) {
    for (offset, &avg_len) in AVG_SELECTOR_LENGTHS.iter().enumerate() {
        let selectors =
            generate_selectors(avg_len, TOTAL_ROWS, scenario, seed_base + offset as u64);
        let stats = SelectorStats::new(&selectors);
        let selection = RowSelection::from(selectors);
        let suffix = format!(
            "{}-{}-{}-L{:02}-avg{:.1}-sel{:02}",
            suite,
            scenario.name,
            variant,
            avg_len,
            stats.average_selector_len,
            (stats.select_ratio * 100.0).round() as u32
        );

        for &mode in BENCH_MODES {
            let parquet_data = Arc::clone(&parquet_data);
            let selection = selection.clone();
            let benchmark_id = format!("{}/{}", mode.label(), suffix);
            c.bench_function(&benchmark_id, move |b| {
                let parquet_data = parquet_data.get_or_init(|| fixture.build());
                b.iter(|| {
                    let total = run_read(parquet_data, &selection, mode.policy());
                    hint::black_box(total);
                });
            });
        }
    }
}

fn bench_filter_shapes(c: &mut Criterion) {
    let parquet_data = Arc::new(OnceLock::new());
    let mut group = c.benchmark_group("row_selection_cursor/filter_shapes");
    group.throughput(Throughput::Elements(TOTAL_ROWS as u64));

    for (case_index, case) in FILTER_SHAPE_CASES.iter().enumerate() {
        for mode in [FilterReadMode::Auto, FilterReadMode::Fallback] {
            let parquet_data = Arc::clone(&parquet_data);
            group.bench_function(BenchmarkId::new(mode.label(), case.id), move |b| {
                let parquet_data = parquet_data.get_or_init(build_filter_shape_parquet);
                validate_filter_shape(parquet_data, case_index, mode);

                b.iter(|| {
                    let rows = run_filter_shape(parquet_data, case_index, mode, |batch| {
                        hint::black_box(batch);
                    });
                    hint::black_box(rows);
                });
            });
        }
    }

    group.finish();
}

criterion_group!(benches, bench_forced_policy_sweep, bench_filter_shapes);
criterion_main!(benches);

fn run_read(parquet_data: &Bytes, selection: &RowSelection, policy: RowSelectionPolicy) -> usize {
    let reader = ParquetRecordBatchReaderBuilder::try_new(parquet_data.clone())
        .unwrap()
        .with_batch_size(BATCH_SIZE)
        .with_row_selection(selection.clone())
        .with_row_selection_policy(policy)
        .build()
        .unwrap();

    let mut total_rows = 0usize;
    for batch in reader {
        let batch = batch.unwrap();
        total_rows += batch.num_rows();
    }
    total_rows
}

fn build_parquet_data(total_rows: usize, build_batch: fn(usize) -> RecordBatch) -> Bytes {
    let batch = build_batch(total_rows);
    write_parquet_batch(batch)
}

fn assert_filter_shape_fixture_contract(batch: &RecordBatch) {
    assert_eq!(
        batch.num_columns(),
        FILTER_SHAPE_CASES.len() + PAYLOAD_COLUMNS
    );
    assert_eq!(batch.num_rows(), TOTAL_ROWS);
}

struct FilterShapeStats {
    selected_rows: usize,
    run_count: usize,
}

impl FilterShapeStats {
    fn new(filter: &BooleanArray) -> Self {
        let mut selected_rows = 0;
        let mut run_count = 0;
        let mut previous = None;

        for row in 0..filter.len() {
            let selected = filter.value(row);
            selected_rows += selected as usize;
            if previous != Some(selected) {
                run_count += 1;
                previous = Some(selected);
            }
        }

        Self {
            selected_rows,
            run_count,
        }
    }
}

fn build_filter_shape_batch() -> RecordBatch {
    let mut fields = Vec::with_capacity(FILTER_SHAPE_CASES.len() + PAYLOAD_COLUMNS);
    let mut columns = Vec::with_capacity(fields.capacity());

    for case in FILTER_SHAPE_CASES {
        let filter = BooleanArray::from(
            (0..TOTAL_ROWS)
                .map(|row| case.shape.is_selected(row))
                .collect::<Vec<_>>(),
        );
        let stats = FilterShapeStats::new(&filter);
        assert_eq!(
            stats.selected_rows, case.expected_selected_rows,
            "{}",
            case.id
        );
        assert_eq!(stats.run_count, case.expected_run_count, "{}", case.id);

        fields.push(Field::new(
            format!("filter_{}", case.id),
            DataType::Boolean,
            false,
        ));
        columns.push(Arc::new(filter) as ArrayRef);
    }

    for column_idx in 0..PAYLOAD_COLUMNS {
        let values =
            Int32Array::from_iter_values((0..TOTAL_ROWS).map(|row| row as i32 + column_idx as i32));
        fields.push(Field::new(
            format!("payload_{column_idx}"),
            DataType::Int32,
            false,
        ));
        columns.push(Arc::new(values) as ArrayRef);
    }

    RecordBatch::try_new(Arc::new(Schema::new(fields)), columns).unwrap()
}

fn build_filter_shape_parquet() -> Bytes {
    let batch = build_filter_shape_batch();
    assert_filter_shape_fixture_contract(&batch);
    write_parquet_batch(batch)
}

fn run_auto_filter_shape<F>(parquet_data: &Bytes, case_index: usize, mut consume: F) -> usize
where
    F: FnMut(RecordBatch),
{
    let builder = ParquetRecordBatchReaderBuilder::try_new(parquet_data.clone()).unwrap();
    let predicate_projection = ProjectionMask::leaves(builder.parquet_schema(), [case_index]);
    let payload_projection =
        ProjectionMask::leaves(builder.parquet_schema(), payload_leaf_indices());
    let predicate = ArrowPredicateFn::new(predicate_projection, |batch: RecordBatch| {
        Ok(batch.column(0).as_boolean().clone())
    });
    let reader = builder
        .with_batch_size(BATCH_SIZE)
        .with_projection(payload_projection)
        .with_row_filter(RowFilter::new(vec![Box::new(predicate)]))
        .build()
        .unwrap();

    consume_reader(reader, &mut consume)
}

fn run_fallback_filter_shape<F>(parquet_data: &Bytes, case_index: usize, mut consume: F) -> usize
where
    F: FnMut(RecordBatch),
{
    let builder = ParquetRecordBatchReaderBuilder::try_new(parquet_data.clone()).unwrap();
    let projection = ProjectionMask::leaves(
        builder.parquet_schema(),
        std::iter::once(case_index).chain(payload_leaf_indices()),
    );
    let reader = builder
        .with_batch_size(BATCH_SIZE)
        .with_projection(projection)
        .build()
        .unwrap();

    let mut total_rows = 0;
    for batch in reader {
        let batch = batch.unwrap();
        let filter = batch.column(0).as_boolean().clone();
        let payload = batch.project(&FALLBACK_PAYLOAD_COLUMNS).unwrap();
        let filtered = filter_record_batch(&payload, &filter).unwrap();
        total_rows += filtered.num_rows();
        consume(filtered);
    }
    total_rows
}

fn consume_reader<F>(reader: ParquetRecordBatchReader, consume: &mut F) -> usize
where
    F: FnMut(RecordBatch),
{
    let mut total_rows = 0;
    for batch in reader {
        let batch = batch.unwrap();
        total_rows += batch.num_rows();
        consume(batch);
    }
    total_rows
}

fn run_filter_shape<F>(
    parquet_data: &Bytes,
    case_index: usize,
    mode: FilterReadMode,
    consume: F,
) -> usize
where
    F: FnMut(RecordBatch),
{
    match mode {
        FilterReadMode::Auto => run_auto_filter_shape(parquet_data, case_index, consume),
        FilterReadMode::Fallback => run_fallback_filter_shape(parquet_data, case_index, consume),
    }
}

fn validate_filter_shape(parquet_data: &Bytes, case_index: usize, mode: FilterReadMode) {
    let case = FILTER_SHAPE_CASES[case_index];
    let mut expected_rows = (0..TOTAL_ROWS).filter(|&row| case.shape.is_selected(row));

    let actual_rows = run_filter_shape(parquet_data, case_index, mode, |batch| {
        assert_eq!(batch.num_columns(), PAYLOAD_COLUMNS);
        for row_index in 0..batch.num_rows() {
            let expected_row = expected_rows.next().expect("unexpected output row");
            for payload_index in 0..PAYLOAD_COLUMNS {
                let values = batch.column(payload_index).as_primitive::<Int32Type>();
                assert_eq!(
                    values.value(row_index),
                    expected_row as i32 + payload_index as i32,
                    "{} payload column {payload_index}",
                    case.id
                );
            }
        }
    });

    assert_eq!(actual_rows, case.expected_selected_rows, "{}", case.id);
    assert!(
        expected_rows.next().is_none(),
        "{} omitted output rows",
        case.id
    );
}

fn build_single_column_batch(data_type: DataType, array: ArrayRef) -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![Field::new("value", data_type, false)]));
    RecordBatch::try_new(schema, vec![array]).unwrap()
}

fn build_int32_batch(total_rows: usize) -> RecordBatch {
    let values = Int32Array::from_iter_values((0..total_rows).map(|v| v as i32));
    build_single_column_batch(DataType::Int32, Arc::new(values) as ArrayRef)
}

fn build_float64_batch(total_rows: usize) -> RecordBatch {
    let values = Float64Array::from_iter_values((0..total_rows).map(|v| v as f64));
    build_single_column_batch(DataType::Float64, Arc::new(values) as ArrayRef)
}

fn build_utf8view_batch(total_rows: usize) -> RecordBatch {
    let mut builder = StringViewBuilder::new();
    // Mix short and long values.
    for i in 0..total_rows {
        match i % 5 {
            0 => builder.append_value("alpha"),
            1 => builder.append_value("beta"),
            2 => builder.append_value("gamma"),
            3 => builder.append_value("delta"),
            _ => builder.append_value("a longer utf8 string payload to test view storage"),
        }
    }
    let values: StringViewArray = builder.finish();
    build_single_column_batch(DataType::Utf8View, Arc::new(values) as ArrayRef)
}

fn build_utf8view_batch_with_len(total_rows: usize, len: usize) -> RecordBatch {
    let mut builder = StringViewBuilder::new();
    let value: String = "a".repeat(len);
    for _ in 0..total_rows {
        builder.append_value(&value);
    }
    let values: StringViewArray = builder.finish();
    build_single_column_batch(DataType::Utf8View, Arc::new(values) as ArrayRef)
}

fn build_int32_columns_batch(total_rows: usize, num_columns: usize) -> RecordBatch {
    let base_values: ArrayRef = Arc::new(Int32Array::from_iter_values(
        (0..total_rows).map(|v| v as i32),
    ));
    let mut fields = Vec::with_capacity(num_columns);
    let mut columns = Vec::with_capacity(num_columns);
    for idx in 0..num_columns {
        fields.push(Field::new(format!("value{}", idx), DataType::Int32, false));
        columns.push(base_values.clone());
    }
    let schema = Arc::new(Schema::new(fields));
    RecordBatch::try_new(schema, columns).unwrap()
}

fn write_parquet_batch(batch: RecordBatch) -> Bytes {
    let schema = batch.schema();
    let mut writer = ArrowWriter::try_new(Vec::new(), schema.clone(), None).unwrap();
    writer.write(&batch).unwrap();
    let buffer = writer.into_inner().unwrap();
    Bytes::from(buffer)
}

#[derive(Clone)]
struct Scenario {
    name: &'static str,
    select_ratio: f64,
    start_with_select: bool,
    distribution: RunDistribution,
}

#[derive(Clone)]
enum RunDistribution {
    Constant,
    Uniform { spread: f64 },
    Bimodal { long_factor: f64, long_prob: f64 },
}

fn generate_selectors(
    avg_selector_len: usize,
    total_rows: usize,
    scenario: &Scenario,
    seed: u64,
) -> Vec<RowSelector> {
    assert!(
        (0.0..=1.0).contains(&scenario.select_ratio),
        "select_ratio must be in [0, 1]"
    );

    let mut select_mean = scenario.select_ratio * 2.0 * avg_selector_len as f64;
    let mut skip_mean = (1.0 - scenario.select_ratio) * 2.0 * avg_selector_len as f64;

    select_mean = select_mean.max(1.0);
    skip_mean = skip_mean.max(1.0);

    let sum = select_mean + skip_mean;
    // Rebalance the sampled select/skip run lengths so their sum matches the requested
    // average selector length while respecting the configured selectivity ratio.
    let scale = if sum == 0.0 {
        1.0
    } else {
        (2.0 * avg_selector_len as f64) / sum
    };
    select_mean *= scale;
    skip_mean *= scale;

    let mut rng = StdRng::seed_from_u64(seed ^ (avg_selector_len as u64).wrapping_mul(0x9E3779B1));
    let mut selectors = Vec::with_capacity(total_rows / avg_selector_len.max(1));
    let mut remaining = total_rows;
    let mut is_select = scenario.start_with_select;

    while remaining > 0 {
        let mean = if is_select { select_mean } else { skip_mean };
        let len = sample_length(mean, &scenario.distribution, &mut rng).max(1);
        let len = len.min(remaining);
        selectors.push(if is_select {
            RowSelector::select(len)
        } else {
            RowSelector::skip(len)
        });
        remaining -= len;
        if remaining == 0 {
            break;
        }
        is_select = !is_select;
    }

    let selection: RowSelection = selectors.into();
    selection.into()
}

fn sample_length(mean: f64, distribution: &RunDistribution, rng: &mut StdRng) -> usize {
    match distribution {
        RunDistribution::Constant => mean.round().max(1.0) as usize,
        RunDistribution::Uniform { spread } => {
            let spread = spread.clamp(0.0, 0.99);
            let lower = (mean * (1.0 - spread)).max(1.0);
            let upper = (mean * (1.0 + spread)).max(lower + f64::EPSILON);
            if (upper - lower) < 1.0 {
                lower.round().max(1.0) as usize
            } else {
                let low = lower.floor() as usize;
                let high = upper.ceil() as usize;
                rng.random_range(low..=high).max(1)
            }
        }
        RunDistribution::Bimodal {
            long_factor,
            long_prob,
        } => {
            let long_prob = long_prob.clamp(0.0, 0.5);
            let short_prob = 1.0 - long_prob;
            let short_factor = if short_prob == 0.0 {
                1.0 / long_factor.max(f64::EPSILON)
            } else {
                (1.0 - long_prob * long_factor).max(0.0) / short_prob
            };
            let use_long = rng.random_bool(long_prob);
            let factor = if use_long {
                *long_factor
            } else {
                short_factor.max(0.1)
            };
            (mean * factor).round().max(1.0) as usize
        }
    }
}

#[derive(Clone, Copy)]
enum BenchMode {
    ReadSelector,
    ReadMask,
}

impl BenchMode {
    fn label(self) -> &'static str {
        match self {
            BenchMode::ReadSelector => "read_selector",
            BenchMode::ReadMask => "read_mask",
        }
    }

    fn policy(self) -> RowSelectionPolicy {
        match self {
            BenchMode::ReadSelector => RowSelectionPolicy::Selectors,
            BenchMode::ReadMask => RowSelectionPolicy::Mask,
        }
    }
}

struct SelectorStats {
    average_selector_len: f64,
    select_ratio: f64,
}

impl SelectorStats {
    fn new(selectors: &[RowSelector]) -> Self {
        if selectors.is_empty() {
            return Self {
                average_selector_len: 0.0,
                select_ratio: 0.0,
            };
        }

        let total_rows: usize = selectors.iter().map(|s| s.row_count).sum();
        let selected_rows: usize = selectors
            .iter()
            .filter(|s| !s.skip)
            .map(|s| s.row_count)
            .sum();

        Self {
            average_selector_len: total_rows as f64 / selectors.len() as f64,
            select_ratio: if total_rows == 0 {
                0.0
            } else {
                selected_rows as f64 / total_rows as f64
            },
        }
    }
}
