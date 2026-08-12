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

//! P-series differential mechanism decomposition.

use std::collections::BTreeSet;
use std::env;
use std::fs::{self, File};
use std::hint;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use parquet::arrow::arrow_reader::metrics::{ArrowReaderDecompositionMetrics, ArrowReaderMetrics};
use serde_json::json;
use sha2::{Digest, Sha256};
use tokio::runtime::Runtime;

use super::fixture::{
    ORACLE_CONTEXTS, ORACLE_ROW_GROUPS, OracleContext, OracleFixture, build_oracle_fixture,
};
use super::model::ROWS_PER_GROUP;
use super::runner::{
    OracleArm, OracleSelectionSource, run_oracle_row_group, run_oracle_row_group_with_metrics,
};
use super::shapes::{OracleShape, OracleShapeSummary, assert_oracle_shape_contracts};

const CSV_SCHEMA_VERSION: &str = "arrow-row-selection-decomposition-v1";
const MANIFEST_SCHEMA_VERSION: &str = "arrow-row-selection-decomposition-manifest-v1";
const DEFAULT_SAMPLES: usize = 12;
const WARMUPS_PER_CONDITION: usize = 2;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Phase {
    P0,
    P1,
}

impl Phase {
    fn label(self) -> &'static str {
        match self {
            Self::P0 => "p0",
            Self::P1 => "p1",
        }
    }
}

#[derive(Debug)]
struct Options {
    phase: Phase,
    samples: usize,
    csv: PathBuf,
    manifest: PathBuf,
    emit_artifacts: bool,
    list: bool,
}

#[derive(Clone, Debug)]
struct CellSpec {
    id: String,
    context_id: &'static str,
    shape: OracleShape,
}

#[derive(Clone, Copy, Debug)]
struct Condition {
    arm: OracleArm,
    enabled: bool,
}

#[derive(Clone, Copy, Debug)]
struct Sample {
    wall_ns: u64,
    metrics: Option<ArrowReaderDecompositionMetrics>,
}

#[derive(Debug)]
struct Measurement {
    cell_id: String,
    context: OracleContext,
    fixture_sha256: String,
    row_group_index: usize,
    shape_name: String,
    shape: OracleShapeSummary,
    arm: OracleArm,
    enabled: bool,
    samples: Vec<Sample>,
}

pub(crate) fn main() {
    if let Err(error) = try_main() {
        eprintln!("row-selection decomposition failed: {error}");
        std::process::exit(2);
    }
}

fn try_main() -> Result<(), String> {
    assert_oracle_shape_contracts();
    let options = parse_options()?;
    let cells = phase_cells(options.phase);
    if options.list {
        for cell in &cells {
            println!(
                "{}\t{}\t{}",
                options.phase.label(),
                cell.context_id,
                cell.id
            );
        }
        eprintln!(
            "listed {} {} decomposition cells",
            cells.len(),
            options.phase.label()
        );
        return Ok(());
    }

    let started_unix_ns = unix_nanos();
    let started = Instant::now();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|error| format!("cannot build Tokio runtime: {error}"))?;
    let mut measurements = Vec::new();
    for context_id in cells
        .iter()
        .map(|cell| cell.context_id)
        .collect::<BTreeSet<_>>()
    {
        let context = context_by_id(context_id)?;
        let fixture = build_oracle_fixture(context, None)
            .map_err(|error| format!("cannot build decomposition context {context_id}: {error}"))?;
        for cell in cells.iter().filter(|cell| cell.context_id == context_id) {
            eprintln!("decomposing {}", cell.id);
            for row_group_index in 0..ORACLE_ROW_GROUPS {
                measure_row_group(
                    &runtime,
                    &options,
                    &fixture,
                    cell,
                    row_group_index,
                    &mut measurements,
                )?;
            }
        }
    }

    write_csv(&options.csv, options.phase, &measurements)?;
    let csv_sha256 = file_sha256(&options.csv)?;
    let p0_gate = (options.phase == Phase::P0).then(|| p0_gate(&measurements));
    let manifest = json!({
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "phase": options.phase.label(),
        "source_commit": command_output(
            "git",
            &["-C", env!("CARGO_MANIFEST_DIR"), "rev-parse", "HEAD"],
        ),
        "samples_per_condition": options.samples,
        "warmups_per_condition": WARMUPS_PER_CONDITION,
        "logical_cells": cells.iter().map(|cell| &cell.id).collect::<Vec<_>>(),
        "row_group_units": cells.len() * ORACLE_ROW_GROUPS,
        "measurement_rows": measurements.len(),
        "csv_sha256": format!("sha256:{csv_sha256}"),
        "started_unix_ns": started_unix_ns,
        "completed_unix_ns": unix_nanos(),
        "elapsed_ns": started.elapsed().as_nanos() as u64,
        "p0_gate": p0_gate.as_ref().map(|gate| json!({
            "pass": gate.pass,
            "max_overhead_fraction": gate.max_overhead_fraction,
            "min_completeness_fraction": gate.min_completeness_fraction,
            "evaluated_cell_arms": gate.evaluated_cell_arms,
            "aggregation": "sum_row_groups_by_sample_then_median_per_cell_arm",
            "overhead_limit": 0.02,
            "completeness_minimum": 0.90,
        })),
    });
    write_json(&options.manifest, &manifest)?;

    println!(
        "DFEXP_SELECTION_DECOMPOSITION_PHASE={}",
        options.phase.label()
    );
    println!("DFEXP_SELECTION_DECOMPOSITION_CELLS={}", cells.len());
    println!("DFEXP_SELECTION_DECOMPOSITION_ROWS={}", measurements.len());
    if let Some(gate) = p0_gate {
        println!(
            "DFEXP_SELECTION_DECOMPOSITION_P0_GATE={}",
            if gate.pass { "PASS" } else { "FAIL" }
        );
        println!(
            "DFEXP_SELECTION_DECOMPOSITION_P0_MAX_OVERHEAD={:.9}",
            gate.max_overhead_fraction
        );
        println!(
            "DFEXP_SELECTION_DECOMPOSITION_P0_MIN_COMPLETENESS={:.9}",
            gate.min_completeness_fraction
        );
    }
    if options.emit_artifacts {
        emit_artifact("CSV", &options.csv)?;
        emit_artifact("MANIFEST", &options.manifest)?;
    }
    Ok(())
}

fn parse_options() -> Result<Options, String> {
    let mut phase = None;
    let mut samples = DEFAULT_SAMPLES;
    let mut csv = None;
    let mut manifest = None;
    let mut emit_artifacts = false;
    let mut list = false;
    let mut args = env::args().skip(1);
    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--selection-oracle" | "--bench" => {}
            "--decompose" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--decompose requires p0 or p1".to_string())?;
                phase = Some(match value.as_str() {
                    "p0" => Phase::P0,
                    "p1" => Phase::P1,
                    _ => return Err(format!("unsupported decomposition phase {value:?}")),
                });
            }
            "--samples" => {
                samples = args
                    .next()
                    .ok_or_else(|| "--samples requires a value".to_string())?
                    .parse()
                    .map_err(|_| "--samples must be an integer".to_string())?;
            }
            "--csv" => {
                csv = Some(PathBuf::from(
                    args.next()
                        .ok_or_else(|| "--csv requires a path".to_string())?,
                ));
            }
            "--manifest" => {
                manifest = Some(PathBuf::from(
                    args.next()
                        .ok_or_else(|| "--manifest requires a path".to_string())?,
                ));
            }
            "--emit-artifacts" => emit_artifacts = true,
            "--list" => list = true,
            "--help" | "-h" => {
                println!(
                    "row_selector --selection-oracle --decompose <p0|p1> \
                     [--samples EVEN] [--list] [--emit-artifacts]"
                );
                std::process::exit(0);
            }
            _ => return Err(format!("unsupported decomposition argument {argument:?}")),
        }
    }
    let phase = phase.ok_or_else(|| "decomposition mode requires --decompose p0|p1".to_string())?;
    if !(2..=100).contains(&samples) || !samples.is_multiple_of(2) {
        return Err("--samples must be an even integer in 2..=100".to_string());
    }
    let csv = csv.unwrap_or_else(|| {
        default_artifact_path(&format!("selection-decomposition-{}.csv", phase.label()))
    });
    let manifest = manifest.unwrap_or_else(|| {
        default_artifact_path(&format!(
            "selection-decomposition-{}-manifest.json",
            phase.label()
        ))
    });
    if csv == manifest {
        return Err("decomposition CSV and manifest paths must differ".to_string());
    }
    Ok(Options {
        phase,
        samples,
        csv,
        manifest,
        emit_artifacts,
        list,
    })
}

fn phase_cells(phase: Phase) -> Vec<CellSpec> {
    let make = |context_id, shape: OracleShape| CellSpec {
        id: format!("P/{}/{context_id}/{}", phase.label(), shape.name),
        context_id,
        shape,
    };
    match phase {
        Phase::P0 => vec![
            make("C0", OracleShape::l_sweep(32)),
            make("C4", OracleShape::selectivity(2, 64)),
        ],
        Phase::P1 => vec![
            make("C8", OracleShape::selectivity(2, 8)),
            make("C8", OracleShape::selectivity(2, 64)),
            make("C4", OracleShape::l_sweep(64)),
            make("C4", OracleShape::selectivity(2, 512)),
            make("C0", OracleShape::selectivity(2, 512)),
            make("C4", OracleShape::selectivity(2, 64)),
            make("C13", OracleShape::selectivity(2, 64)),
            make("C4", OracleShape::selectivity(2, 8)),
            make("C13", OracleShape::selectivity(2, 8)),
            make("C5", OracleShape::l_sweep(8)),
            make("C0", OracleShape::l_sweep(32)),
        ],
    }
}

fn measure_row_group(
    runtime: &Runtime,
    options: &Options,
    fixture: &OracleFixture,
    cell: &CellSpec,
    row_group_index: usize,
    output: &mut Vec<Measurement>,
) -> Result<(), String> {
    let expected = cell.shape.summary().selected_rows;
    let selection = || Some(cell.shape.selection_for_row_group());
    let selectors_check = runtime.block_on(run_oracle_row_group(
        fixture,
        row_group_index,
        selection(),
        OracleSelectionSource::External,
        OracleArm::Selectors,
        true,
    ));
    let mask_check = runtime.block_on(run_oracle_row_group(
        fixture,
        row_group_index,
        selection(),
        OracleSelectionSource::External,
        OracleArm::Mask,
        true,
    ));
    if selectors_check.row_count != expected
        || mask_check.row_count != expected
        || selectors_check.content != mask_check.content
    {
        return Err(format!(
            "{}/rg{} forced-arm correctness mismatch",
            cell.id, row_group_index
        ));
    }

    let conditions: Vec<Condition> = match options.phase {
        Phase::P0 => vec![
            Condition {
                arm: OracleArm::Selectors,
                enabled: false,
            },
            Condition {
                arm: OracleArm::Selectors,
                enabled: true,
            },
            Condition {
                arm: OracleArm::Mask,
                enabled: false,
            },
            Condition {
                arm: OracleArm::Mask,
                enabled: true,
            },
        ],
        Phase::P1 => vec![
            Condition {
                arm: OracleArm::Selectors,
                enabled: true,
            },
            Condition {
                arm: OracleArm::Mask,
                enabled: true,
            },
        ],
    };

    for _ in 0..WARMUPS_PER_CONDITION {
        for condition in &conditions {
            let sample = run_sample(runtime, fixture, cell, row_group_index, *condition);
            if sample.0 != expected {
                return Err(format!(
                    "{}/rg{} warmup row mismatch",
                    cell.id, row_group_index
                ));
            }
        }
    }

    let mut samples = vec![Vec::with_capacity(options.samples); conditions.len()];
    let schedule: Vec<usize> = match options.phase {
        Phase::P0 => vec![1, 3, 2, 0, 0, 2, 3, 1],
        Phase::P1 => vec![0, 1, 1, 0],
    };
    while samples
        .iter()
        .any(|samples| samples.len() < options.samples)
    {
        for idx in &schedule {
            if samples[*idx].len() >= options.samples {
                continue;
            }
            let (rows, sample) =
                run_sample(runtime, fixture, cell, row_group_index, conditions[*idx]);
            if rows != expected {
                return Err(format!(
                    "{}/rg{} timed row mismatch",
                    cell.id, row_group_index
                ));
            }
            hint::black_box(rows);
            samples[*idx].push(sample);
        }
    }

    let fixture_sha256 = fixture.bytes_sha256();
    for (condition, samples) in conditions.into_iter().zip(samples) {
        validate_counter_stability(&cell.id, row_group_index, condition, &samples)?;
        output.push(Measurement {
            cell_id: cell.id.clone(),
            context: fixture.context(),
            fixture_sha256: fixture_sha256.clone(),
            row_group_index,
            shape_name: cell.shape.name.clone(),
            shape: cell.shape.summary(),
            arm: condition.arm,
            enabled: condition.enabled,
            samples,
        });
    }
    Ok(())
}

fn run_sample(
    runtime: &Runtime,
    fixture: &OracleFixture,
    cell: &CellSpec,
    row_group_index: usize,
    condition: Condition,
) -> (usize, Sample) {
    let metrics = if condition.enabled {
        ArrowReaderMetrics::enabled()
    } else {
        ArrowReaderMetrics::disabled()
    };
    let started = Instant::now();
    let result = runtime.block_on(run_oracle_row_group_with_metrics(
        fixture,
        row_group_index,
        Some(cell.shape.selection_for_row_group()),
        OracleSelectionSource::External,
        condition.arm,
        false,
        metrics.clone(),
    ));
    let wall_ns = started.elapsed().as_nanos().min(u128::from(u64::MAX)) as u64;
    (
        result.row_count,
        Sample {
            wall_ns,
            metrics: metrics.decomposition(),
        },
    )
}

fn validate_counter_stability(
    cell: &str,
    row_group_index: usize,
    condition: Condition,
    samples: &[Sample],
) -> Result<(), String> {
    if !condition.enabled {
        if samples.iter().any(|sample| sample.metrics.is_some()) {
            return Err(format!(
                "{cell}/rg{row_group_index} disabled metrics produced a snapshot"
            ));
        }
        return Ok(());
    }
    let first = samples[0]
        .metrics
        .ok_or_else(|| format!("{cell}/rg{row_group_index} missing enabled metrics"))?;
    let key = |metric: ArrowReaderDecompositionMetrics| {
        (
            metric.skip_records_calls,
            metric.skip_records_rows,
            metric.read_records_calls,
            metric.read_records_rows,
            metric.page_decompression_pages,
            metric.page_decompression_bytes,
            metric.filter_record_batch_calls,
            metric.selectors_to_mask_calls,
            metric.consume_batch_calls,
        )
    };
    if samples
        .iter()
        .any(|sample| sample.metrics.map(key) != Some(key(first)))
    {
        return Err(format!(
            "{cell}/rg{row_group_index}/{:?} decomposition counters are not deterministic",
            condition.arm
        ));
    }
    Ok(())
}

#[derive(Clone, Copy, Debug)]
struct P0Gate {
    pass: bool,
    max_overhead_fraction: f64,
    min_completeness_fraction: f64,
    evaluated_cell_arms: usize,
}

fn p0_gate(measurements: &[Measurement]) -> P0Gate {
    let mut max_overhead_fraction = f64::NEG_INFINITY;
    let mut min_completeness_fraction = f64::INFINITY;
    let cell_arms = measurements
        .iter()
        .filter(|row| row.enabled)
        .map(|row| (row.cell_id.as_str(), row.arm))
        .collect::<BTreeSet<_>>();
    for (cell_id, arm) in &cell_arms {
        let enabled = aggregate_samples(measurements, cell_id, *arm, true);
        let disabled = aggregate_samples(measurements, cell_id, *arm, false);
        let overhead = median(&enabled.iter().map(|sample| sample.0).collect::<Vec<_>>()) as f64
            / median(&disabled.iter().map(|sample| sample.0).collect::<Vec<_>>()) as f64
            - 1.0;
        max_overhead_fraction = max_overhead_fraction.max(overhead);
        min_completeness_fraction = min_completeness_fraction.min(median_f64(
            &enabled
                .iter()
                .map(|(wall_ns, named_ns)| *named_ns as f64 / (*wall_ns).max(1) as f64)
                .collect::<Vec<_>>(),
        ));
    }
    P0Gate {
        pass: max_overhead_fraction <= 0.02 && min_completeness_fraction >= 0.90,
        max_overhead_fraction,
        min_completeness_fraction,
        evaluated_cell_arms: cell_arms.len(),
    }
}

fn aggregate_samples(
    measurements: &[Measurement],
    cell_id: &str,
    arm: OracleArm,
    enabled: bool,
) -> Vec<(u64, u64)> {
    let rows = measurements
        .iter()
        .filter(|row| row.cell_id == cell_id && row.arm == arm && row.enabled == enabled)
        .collect::<Vec<_>>();
    assert_eq!(rows.len(), ORACLE_ROW_GROUPS);
    let sample_count = rows[0].samples.len();
    assert!(rows.iter().all(|row| row.samples.len() == sample_count));
    (0..sample_count)
        .map(|sample_idx| {
            rows.iter().fold((0u64, 0u64), |(wall_ns, named_ns), row| {
                let sample = &row.samples[sample_idx];
                (
                    wall_ns.saturating_add(sample.wall_ns),
                    named_ns.saturating_add(
                        sample
                            .metrics
                            .map(|_| named_exclusive_sum(sample))
                            .unwrap_or(0),
                    ),
                )
            })
        })
        .collect()
}

fn named_exclusive_sum(sample: &Sample) -> u64 {
    let metric = sample.metrics.unwrap();
    metric
        .skip_records_ns
        .saturating_add(metric.read_records_ns)
        .saturating_add(metric.filter_record_batch_ns)
        .saturating_add(metric.selectors_to_mask_ns)
        .saturating_add(metric.consume_batch_ns)
}

fn skip_read_exclusive(sample: &Sample) -> u64 {
    let metric = sample.metrics.unwrap();
    metric
        .skip_records_ns
        .saturating_add(metric.read_records_ns)
        .saturating_sub(metric.page_decompression_ns)
}

fn completeness(sample: &Sample) -> f64 {
    named_exclusive_sum(sample) as f64 / sample.wall_ns.max(1) as f64
}

fn write_csv(path: &Path, phase: Phase, rows: &[Measurement]) -> Result<(), String> {
    let file = File::create(path).map_err(|error| {
        format!(
            "cannot create decomposition CSV {}: {error}",
            path.display()
        )
    })?;
    let mut writer = BufWriter::new(file);
    let columns = [
        "schema_version",
        "phase",
        "cell_id",
        "context_id",
        "fixture_sha256",
        "row_group_index",
        "dtype",
        "payload_columns",
        "encoding",
        "compression",
        "batch_size",
        "rows_per_group",
        "shape_name",
        "selected_rows",
        "selected_fraction",
        "avg_run_len",
        "run_count",
        "arm",
        "instrumentation",
        "sample_count",
        "wall_samples_ns",
        "median_wall_ns",
        "skip_records_ns_samples",
        "median_skip_records_ns",
        "skip_records_calls",
        "skip_records_rows",
        "read_records_ns_samples",
        "median_read_records_ns",
        "read_records_calls",
        "read_records_rows",
        "page_decompression_ns_samples",
        "median_page_decompression_ns",
        "page_decompression_pages",
        "page_decompression_bytes",
        "filter_record_batch_ns_samples",
        "median_filter_record_batch_ns",
        "filter_record_batch_calls",
        "selectors_to_mask_ns_samples",
        "median_selectors_to_mask_ns",
        "selectors_to_mask_calls",
        "consume_batch_ns_samples",
        "median_consume_batch_ns",
        "consume_batch_calls",
        "skip_read_exclusive_ns_samples",
        "median_skip_read_exclusive_ns",
        "named_exclusive_sum_ns_samples",
        "median_named_exclusive_sum_ns",
        "completeness_samples",
        "median_completeness",
    ];
    writeln!(writer, "{}", columns.join(","))
        .map_err(|error| format!("cannot write decomposition CSV header: {error}"))?;
    for row in rows {
        let metrics = row
            .samples
            .iter()
            .filter_map(|sample| sample.metrics)
            .collect::<Vec<_>>();
        let first = metrics.first().copied();
        let metric_samples = |get: fn(ArrowReaderDecompositionMetrics) -> u64| {
            join_u64(&metrics.iter().copied().map(get).collect::<Vec<_>>())
        };
        let metric_median = |get: fn(ArrowReaderDecompositionMetrics) -> u64| {
            optional_u64(
                (!metrics.is_empty())
                    .then(|| median(&metrics.iter().copied().map(get).collect::<Vec<_>>())),
            )
        };
        let enabled_samples = row.enabled.then(|| {
            row.samples
                .iter()
                .map(skip_read_exclusive)
                .collect::<Vec<_>>()
        });
        let named_samples = row.enabled.then(|| {
            row.samples
                .iter()
                .map(named_exclusive_sum)
                .collect::<Vec<_>>()
        });
        let completeness_samples = row
            .enabled
            .then(|| row.samples.iter().map(completeness).collect::<Vec<_>>());
        let fields = vec![
            CSV_SCHEMA_VERSION.to_string(),
            phase.label().to_string(),
            row.cell_id.clone(),
            row.context.id.to_string(),
            row.fixture_sha256.clone(),
            row.row_group_index.to_string(),
            row.context.payload.dtype().to_string(),
            row.context.payload_columns.to_string(),
            row.context.encoding().to_string(),
            row.context.compression.label().to_string(),
            row.context.batch_size.to_string(),
            ROWS_PER_GROUP.to_string(),
            row.shape_name.clone(),
            row.shape.selected_rows.to_string(),
            format!("{:.9}", row.shape.selected_fraction),
            format!("{:.9}", row.shape.avg_run_len),
            row.shape.run_count.to_string(),
            row.arm.label().to_string(),
            if row.enabled { "enabled" } else { "disabled" }.to_string(),
            row.samples.len().to_string(),
            join_u64(
                &row.samples
                    .iter()
                    .map(|sample| sample.wall_ns)
                    .collect::<Vec<_>>(),
            ),
            median_wall(&row.samples).to_string(),
            metric_samples(|metric| metric.skip_records_ns),
            metric_median(|metric| metric.skip_records_ns),
            optional_usize(first.map(|metric| metric.skip_records_calls)),
            optional_usize(first.map(|metric| metric.skip_records_rows)),
            metric_samples(|metric| metric.read_records_ns),
            metric_median(|metric| metric.read_records_ns),
            optional_usize(first.map(|metric| metric.read_records_calls)),
            optional_usize(first.map(|metric| metric.read_records_rows)),
            metric_samples(|metric| metric.page_decompression_ns),
            metric_median(|metric| metric.page_decompression_ns),
            optional_usize(first.map(|metric| metric.page_decompression_pages)),
            optional_usize(first.map(|metric| metric.page_decompression_bytes)),
            metric_samples(|metric| metric.filter_record_batch_ns),
            metric_median(|metric| metric.filter_record_batch_ns),
            optional_usize(first.map(|metric| metric.filter_record_batch_calls)),
            metric_samples(|metric| metric.selectors_to_mask_ns),
            metric_median(|metric| metric.selectors_to_mask_ns),
            optional_usize(first.map(|metric| metric.selectors_to_mask_calls)),
            metric_samples(|metric| metric.consume_batch_ns),
            metric_median(|metric| metric.consume_batch_ns),
            optional_usize(first.map(|metric| metric.consume_batch_calls)),
            enabled_samples
                .as_ref()
                .map(|values| join_u64(values))
                .unwrap_or_default(),
            enabled_samples
                .as_ref()
                .map(|values| median(values).to_string())
                .unwrap_or_default(),
            named_samples
                .as_ref()
                .map(|values| join_u64(values))
                .unwrap_or_default(),
            named_samples
                .as_ref()
                .map(|values| median(values).to_string())
                .unwrap_or_default(),
            completeness_samples
                .as_ref()
                .map(|values| join_f64(values))
                .unwrap_or_default(),
            completeness_samples
                .as_ref()
                .map(|values| format!("{:.9}", median_f64(values)))
                .unwrap_or_default(),
        ];
        writeln!(
            writer,
            "{}",
            fields
                .iter()
                .map(|field| escape_csv(field))
                .collect::<Vec<_>>()
                .join(",")
        )
        .map_err(|error| format!("cannot write decomposition CSV row: {error}"))?;
    }
    Ok(())
}

fn context_by_id(id: &str) -> Result<OracleContext, String> {
    ORACLE_CONTEXTS
        .iter()
        .copied()
        .find(|context| context.id == id)
        .ok_or_else(|| format!("unknown oracle context {id}"))
}

fn median_wall(samples: &[Sample]) -> u64 {
    median(
        &samples
            .iter()
            .map(|sample| sample.wall_ns)
            .collect::<Vec<_>>(),
    )
}

fn median(values: &[u64]) -> u64 {
    assert!(!values.is_empty());
    let mut values = values.to_vec();
    values.sort_unstable();
    let midpoint = values.len() / 2;
    if values.len().is_multiple_of(2) {
        values[midpoint - 1] / 2
            + values[midpoint] / 2
            + (values[midpoint - 1] % 2 + values[midpoint] % 2) / 2
    } else {
        values[midpoint]
    }
}

fn median_f64(values: &[f64]) -> f64 {
    assert!(!values.is_empty());
    let mut values = values.to_vec();
    values.sort_by(f64::total_cmp);
    let midpoint = values.len() / 2;
    if values.len().is_multiple_of(2) {
        (values[midpoint - 1] + values[midpoint]) / 2.0
    } else {
        values[midpoint]
    }
}

fn default_artifact_path(filename: &str) -> PathBuf {
    env::var_os("CARGO_TARGET_DIR")
        .map(PathBuf::from)
        .map(|target_dir| target_dir.join(filename))
        .unwrap_or_else(|| PathBuf::from(filename))
}

fn write_json(path: &Path, value: &serde_json::Value) -> Result<(), String> {
    let file = File::create(path)
        .map_err(|error| format!("cannot create JSON {}: {error}", path.display()))?;
    serde_json::to_writer_pretty(file, value)
        .map_err(|error| format!("cannot write JSON {}: {error}", path.display()))?;
    let mut file = fs::OpenOptions::new()
        .append(true)
        .open(path)
        .map_err(|error| format!("cannot reopen JSON {}: {error}", path.display()))?;
    writeln!(file).map_err(|error| format!("cannot finish JSON {}: {error}", path.display()))
}

fn emit_artifact(kind: &str, path: &Path) -> Result<(), String> {
    let raw = fs::read_to_string(path)
        .map_err(|error| format!("cannot read {} for log embedding: {error}", path.display()))?;
    println!("DFEXP_SELECTION_DECOMPOSITION_{kind}_BEGIN");
    print!("{raw}");
    if !raw.ends_with('\n') {
        println!();
    }
    println!("DFEXP_SELECTION_DECOMPOSITION_{kind}_END");
    Ok(())
}

fn file_sha256(path: &Path) -> Result<String, String> {
    let bytes =
        fs::read(path).map_err(|error| format!("cannot hash {}: {error}", path.display()))?;
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    Ok(hex_digest(&hasher.finalize()))
}

fn command_output(command: &str, arguments: &[&str]) -> String {
    Command::new(command)
        .args(arguments)
        .output()
        .ok()
        .filter(|output| output.status.success())
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .unwrap_or_default()
        .trim()
        .to_string()
}

fn unix_nanos() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos() as u64
}

fn join_u64(values: &[u64]) -> String {
    values
        .iter()
        .map(u64::to_string)
        .collect::<Vec<_>>()
        .join("|")
}

fn join_f64(values: &[f64]) -> String {
    values
        .iter()
        .map(|value| format!("{value:.9}"))
        .collect::<Vec<_>>()
        .join("|")
}

fn optional_u64(value: Option<u64>) -> String {
    value.map(|value| value.to_string()).unwrap_or_default()
}

fn optional_usize(value: Option<usize>) -> String {
    value.map(|value| value.to_string()).unwrap_or_default()
}

fn escape_csv(value: &str) -> String {
    if value.contains(',') || value.contains('"') || value.contains('\n') || value.contains('\r') {
        format!("\"{}\"", value.replace('"', "\"\""))
    } else {
        value.to_string()
    }
}

fn hex_digest(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        output.push(HEX[(byte >> 4) as usize] as char);
        output.push(HEX[(byte & 0x0f) as usize] as char);
    }
    output
}
