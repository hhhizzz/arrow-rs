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

//! Focused benchmark for predicate-result `RowSelection` construction.
//!
//! Boolean inputs, batch splits, expected selected counts, and expected run
//! counts are prepared and checked before Criterion starts timing.

use arrow_array::{Array, BooleanArray};
use arrow_buffer::{BooleanBuffer, BooleanBufferBuilder};
use criterion::{
    BenchmarkGroup, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main,
};
use parquet::arrow::arrow_reader::{
    ReadPlanBuilder, RowSelection, RowSelectionCursor, RowSelectionPolicy,
};
use std::hint::black_box;
use std::time::Duration;

const MAIN_ROWS: usize = 4_194_304;
const SCALE_ROWS: &[usize] = &[65_536, 1_048_576];
const FILTER_BATCH_SIZE: usize = 8_192;
const AUTO_THRESHOLD: usize = 32;

const Q25_PATTERN: &[bool] = &[
    true, false, false, false, false, false, false, true, false, false, false, false, false, false,
    true, false, false, false, false, false,
];

const MAIN_CASES: &[CaseSpec] = &[
    CaseSpec::new("q25_like_15_isolated", Shape::Pattern(Q25_PATTERN)),
    CaseSpec::new(
        "fragmented_50_run1",
        Shape::Runs {
            selected: 1,
            skipped: 1,
        },
    ),
    CaseSpec::new(
        "boundary_mask_run31",
        Shape::Runs {
            selected: 31,
            skipped: 31,
        },
    ),
    CaseSpec::new(
        "boundary_selectors_run32",
        Shape::Runs {
            selected: 32,
            skipped: 32,
        },
    ),
    CaseSpec::new(
        "clustered_50_run128",
        Shape::Runs {
            selected: 128,
            skipped: 128,
        },
    ),
    CaseSpec::new(
        "sparse_1_56_run32",
        Shape::Runs {
            selected: 32,
            skipped: 2_016,
        },
    ),
    CaseSpec::new("all_selected", Shape::All(true)),
    CaseSpec::new("all_skipped", Shape::All(false)),
];

fn benchmark(c: &mut Criterion) {
    let main_cases: Vec<_> = MAIN_CASES
        .iter()
        .map(|spec| CaseInput::new(*spec, MAIN_ROWS))
        .collect();
    let scale_cases: Vec<_> = SCALE_ROWS
        .iter()
        .flat_map(|rows| {
            MAIN_CASES[..2]
                .iter()
                .map(|spec| CaseInput::new(*spec, *rows))
        })
        .collect();

    let mut construct = c.benchmark_group("row_selection_construction/construct");
    for case in main_cases.iter().chain(scale_cases.iter()) {
        register_construct_case(&mut construct, case);
    }
    construct.finish();

    let mut lower = c.benchmark_group("row_selection_construction/lower");
    for case in &main_cases {
        register_lower_case(&mut lower, case);
    }
    lower.finish();
}

fn register_construct_case(
    group: &mut BenchmarkGroup<'_, criterion::measurement::WallTime>,
    case: &CaseInput,
) {
    group.throughput(Throughput::Elements(case.rows as u64));
    let label = case.benchmark_label();

    group.bench_with_input(
        BenchmarkId::new(&label, "current_selectors"),
        case,
        |b, case| {
            b.iter(|| {
                let selection = RowSelection::from_filters(black_box(&case.filters));
                black_box(selection);
            })
        },
    );

    group.bench_with_input(BenchmarkId::new(&label, "direct_mask"), case, |b, case| {
        b.iter(|| {
            let selection = direct_mask_selection(black_box(&case.filters));
            black_box(selection);
        })
    });

    // At construction time V1 and V2 are intentionally the same operation.
    // Their lowering policies differ in the second benchmark group.
    group.bench_with_input(BenchmarkId::new(&label, "mask_first"), case, |b, case| {
        b.iter(|| {
            let selection = direct_mask_selection(black_box(&case.filters));
            black_box(selection);
        })
    });

    group.bench_with_input(BenchmarkId::new(&label, "capped_auto"), case, |b, case| {
        b.iter(|| {
            let selection =
                RowSelection::from_filters_auto(black_box(&case.filters), AUTO_THRESHOLD);
            black_box(selection);
        })
    });
}

fn register_lower_case(
    group: &mut BenchmarkGroup<'_, criterion::measurement::WallTime>,
    case: &CaseInput,
) {
    group.throughput(Throughput::Elements(case.rows as u64));
    let label = case.benchmark_label();

    group.bench_with_input(BenchmarkId::new(&label, "current_auto"), case, |b, case| {
        b.iter(|| {
            let selection = RowSelection::from_filters(black_box(&case.filters));
            let plan = ReadPlanBuilder::new(FILTER_BATCH_SIZE)
                .with_selection(Some(selection))
                .with_row_selection_policy(RowSelectionPolicy::default())
                .build();
            black_box(plan);
        })
    });

    group.bench_with_input(BenchmarkId::new(&label, "direct_mask"), case, |b, case| {
        b.iter(|| {
            let selection = direct_mask_selection(black_box(&case.filters));
            let plan = ReadPlanBuilder::new(FILTER_BATCH_SIZE)
                .with_selection(Some(selection))
                .with_row_selection_policy(RowSelectionPolicy::Mask)
                .build();
            black_box(plan);
        })
    });

    group.bench_with_input(
        BenchmarkId::new(&label, "mask_first_auto"),
        case,
        |b, case| {
            b.iter(|| {
                let selection = direct_mask_selection(black_box(&case.filters));
                let plan = ReadPlanBuilder::new(FILTER_BATCH_SIZE)
                    .with_selection(Some(selection))
                    .with_row_selection_policy(RowSelectionPolicy::default())
                    .build();
                black_box(plan);
            })
        },
    );

    group.bench_with_input(BenchmarkId::new(&label, "capped_auto"), case, |b, case| {
        b.iter(|| {
            let selection =
                RowSelection::from_filters_auto(black_box(&case.filters), AUTO_THRESHOLD);
            let plan = ReadPlanBuilder::new(FILTER_BATCH_SIZE)
                .with_selection(Some(selection))
                .with_row_selection_policy(RowSelectionPolicy::default())
                .build();
            black_box(plan);
        })
    });
}

fn direct_mask_selection(filters: &[BooleanArray]) -> RowSelection {
    RowSelection::from_boolean_buffer(filters_to_boolean_buffer(filters))
}

fn filters_to_boolean_buffer(filters: &[BooleanArray]) -> BooleanBuffer {
    let total_rows = filters.iter().map(|filter| filter.len()).sum();
    let mut builder = BooleanBufferBuilder::new(total_rows);
    for filter in filters {
        assert_eq!(filter.null_count(), 0);
        builder.append_buffer(filter.values());
    }
    builder.finish()
}

#[derive(Clone, Copy)]
struct CaseSpec {
    name: &'static str,
    shape: Shape,
}

impl CaseSpec {
    const fn new(name: &'static str, shape: Shape) -> Self {
        Self { name, shape }
    }
}

#[derive(Clone, Copy)]
enum Shape {
    Pattern(&'static [bool]),
    Runs { selected: usize, skipped: usize },
    All(bool),
}

struct CaseInput {
    spec: CaseSpec,
    rows: usize,
    filters: Vec<BooleanArray>,
}

impl CaseInput {
    fn new(spec: CaseSpec, rows: usize) -> Self {
        let mask = build_mask(spec.shape, rows);
        let selected_rows = mask.count_set_bits();
        let run_count = count_runs(&mask);
        let expected_strategy = if run_count == 0 || rows < run_count.saturating_mul(AUTO_THRESHOLD)
        {
            ExpectedStrategy::Mask
        } else {
            ExpectedStrategy::Selectors
        };
        let filters = split_filters(&mask);
        assert_eq!(filters.len(), rows.div_ceil(FILTER_BATCH_SIZE));

        let current = RowSelection::from_filters(&filters);
        let candidate = RowSelection::from_filters_auto(&filters, AUTO_THRESHOLD);
        assert_eq!(current.total_row_count(), rows, "{} length", spec.name);
        assert_eq!(current.row_count(), selected_rows, "{} selected", spec.name);
        assert_eq!(current.iter().count(), run_count, "{} runs", spec.name);
        assert_eq!(current, candidate, "{} candidate selection", spec.name);
        assert_eq!(
            candidate.as_mask().is_some(),
            expected_strategy == ExpectedStrategy::Mask,
            "{} candidate backing",
            spec.name
        );

        let plan_strategy = resolved_plan_strategy(current.clone());
        let expected_plan_strategy = if selected_rows == 0 {
            // ReadPlan canonicalizes an empty selection before resolving Auto.
            ExpectedStrategy::Mask
        } else {
            expected_strategy
        };
        assert_eq!(
            plan_strategy, expected_plan_strategy,
            "{} current Auto plan strategy",
            spec.name
        );
        assert_eq!(
            resolved_plan_strategy(candidate),
            expected_plan_strategy,
            "{} candidate Auto plan strategy",
            spec.name
        );

        Self {
            spec,
            rows,
            filters,
        }
    }

    fn benchmark_label(&self) -> String {
        if self.rows == MAIN_ROWS {
            self.spec.name.to_string()
        } else {
            format!("scale/{}/rows_{}", self.spec.name, self.rows)
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ExpectedStrategy {
    Mask,
    Selectors,
}

fn resolved_plan_strategy(selection: RowSelection) -> ExpectedStrategy {
    let mut plan = ReadPlanBuilder::new(FILTER_BATCH_SIZE)
        .with_selection(Some(selection))
        .with_row_selection_policy(RowSelectionPolicy::default())
        .build();
    match plan.row_selection_cursor_mut() {
        RowSelectionCursor::Mask(_) => ExpectedStrategy::Mask,
        RowSelectionCursor::Selectors(_) => ExpectedStrategy::Selectors,
        RowSelectionCursor::All => panic!("explicit selection unexpectedly lowered to All"),
    }
}

fn build_mask(shape: Shape, rows: usize) -> BooleanBuffer {
    match shape {
        Shape::Pattern(pattern) => {
            BooleanBuffer::from_iter((0..rows).map(|row| pattern[row % pattern.len()]))
        }
        Shape::Runs { selected, skipped } => {
            let period = selected + skipped;
            BooleanBuffer::from_iter((0..rows).map(|row| row % period < selected))
        }
        Shape::All(value) => {
            if value {
                BooleanBuffer::new_set(rows)
            } else {
                BooleanBuffer::new_unset(rows)
            }
        }
    }
}

fn split_filters(mask: &BooleanBuffer) -> Vec<BooleanArray> {
    (0..mask.len())
        .step_by(FILTER_BATCH_SIZE)
        .map(|offset| {
            let len = FILTER_BATCH_SIZE.min(mask.len() - offset);
            BooleanArray::new(mask.slice(offset, len), None)
        })
        .collect()
}

fn count_runs(mask: &BooleanBuffer) -> usize {
    if mask.is_empty() {
        return 0;
    }

    let mut runs = 1usize;
    let mut previous = mask.value(0);
    for row in 1..mask.len() {
        let value = mask.value(row);
        if value != previous {
            runs += 1;
            previous = value;
        }
    }
    runs
}

fn criterion_config() -> Criterion {
    Criterion::default()
        .sample_size(30)
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(3))
        .noise_threshold(0.02)
}

criterion_group! {
    name = benches;
    config = criterion_config();
    targets = benchmark
}
criterion_main!(benches);
