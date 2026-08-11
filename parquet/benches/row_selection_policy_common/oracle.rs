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

use std::collections::{BTreeMap, BTreeSet, hash_map::DefaultHasher};
use std::env;
use std::fs::{self, File};
use std::hash::{Hash, Hasher};
use std::hint;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use regex::Regex;
use serde_json::json;
use tokio::runtime::Runtime;

use super::fixture::{
    ORACLE_CONTEXTS, ORACLE_PAGE_ROWS, ORACLE_ROW_GROUPS, OracleContext, OracleFixture,
    build_oracle_fixture,
};
use super::model::ROWS_PER_GROUP;
use super::runner::{OracleArm, OracleRunResult, OracleSelectionSource, run_oracle};
use super::shapes::{
    ORACLE_L_SWEEP, ORACLE_SELECTIVITY_L, ORACLE_SELECTIVITY_PERCENT, OracleShape,
    OracleShapeSummary, assert_oracle_shape_contracts,
};

const CSV_SCHEMA_VERSION: &str = "arrow-row-selection-oracle-v1";
const MANIFEST_SCHEMA_VERSION: &str = "arrow-row-selection-oracle-manifest-v1";
const DEFAULT_SAMPLES: usize = 12;
const QUICK_SAMPLES: usize = 4;
const WARMUPS_PER_ARM: usize = 2;
const SELECTIVITY_CONTEXTS: &[&str] = &["C0", "C3", "C4", "C5", "C8", "C11"];
const PAGE_INDEX_CONTEXTS: &[&str] = &["C0", "C3", "C4", "C8"];
const BATCH_CONTEXTS: &[&str] = &["C0", "C4"];
const PREDICATE_CONTEXTS: &[&str] = &["C0", "C4"];
const BATCH_SENSITIVITY_L: &[usize] = &[8, 32, 128];

#[derive(Debug)]
struct Options {
    list: bool,
    quick: bool,
    samples: usize,
    filter: Option<Regex>,
    filter_text: Option<String>,
    csv: PathBuf,
    manifest: PathBuf,
    emit_artifacts: bool,
}

#[derive(Clone, Debug)]
struct ArmMeasurement {
    arm: OracleArm,
    samples_ns: Vec<u64>,
    sample_started_unix_ns: Vec<u64>,
    median_ns: u64,
    mad_ns: u64,
    rows_out: usize,
    checksum: u64,
}

#[derive(Clone, Debug)]
struct PairMeasurement {
    selectors: ArmMeasurement,
    mask: ArmMeasurement,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Winner {
    Selectors,
    Mask,
    Tie,
}

impl PairMeasurement {
    fn winner(&self) -> Winner {
        let difference = self.selectors.median_ns.abs_diff(self.mask.median_ns) as f64;
        let pooled_mad =
            (((self.selectors.mad_ns as f64).powi(2) + (self.mask.mad_ns as f64).powi(2)) / 2.0)
                .sqrt();
        let median_scale = (self.selectors.median_ns as f64 + self.mask.median_ns as f64) / 2.0;
        let tie_limit = (3.0 * pooled_mad).max(0.01 * median_scale);
        if difference < tie_limit {
            Winner::Tie
        } else if self.selectors.median_ns < self.mask.median_ns {
            Winner::Selectors
        } else {
            Winner::Mask
        }
    }
}

#[derive(Clone, Debug)]
struct CsvRow {
    group: String,
    cell_id: String,
    context: OracleContext,
    shape_name: String,
    nominal_skip: Option<usize>,
    nominal_select: Option<usize>,
    summary: OracleShapeSummary,
    source: OracleSelectionSource,
    measurement: ArmMeasurement,
    reused_samples: bool,
    auto_choice: String,
}

pub(crate) fn main() {
    if let Err(error) = try_main() {
        eprintln!("row-selection oracle failed: {error}");
        std::process::exit(2);
    }
}

fn try_main() -> Result<(), String> {
    assert_oracle_shape_contracts();
    let options = parse_options()?;
    if options.list {
        list_cells(&options);
        return Ok(());
    }

    let started_unix_ns = unix_nanos();
    let started = Instant::now();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|error| format!("cannot build Tokio runtime: {error}"))?;
    let mut rows = Vec::new();

    let contexts = ORACLE_CONTEXTS
        .iter()
        .copied()
        .filter(|context| !options.quick || matches!(context.id, "C0" | "C3"));
    for context in contexts {
        run_context(&runtime, &options, context, &mut rows)?;
    }
    if rows.is_empty() {
        return Err("cell filter selected no benchmark cells".to_string());
    }

    let completed_unix_ns = unix_nanos();
    write_csv(&options.csv, &rows)?;
    write_manifest(
        &options,
        &rows,
        started_unix_ns,
        completed_unix_ns,
        started.elapsed().as_nanos() as u64,
    )?;

    let cells = rows
        .iter()
        .map(|row| row.cell_id.as_str())
        .collect::<BTreeSet<_>>()
        .len();
    eprintln!(
        "selection oracle complete: {cells} cells, {} arm rows, csv={}, manifest={}",
        rows.len(),
        options.csv.display(),
        options.manifest.display()
    );
    println!("DFEXP_SELECTION_ORACLE_CELLS={cells}");
    println!("DFEXP_SELECTION_ORACLE_ROWS={}", rows.len());
    if options.emit_artifacts {
        emit_artifact("CSV", &options.csv)?;
        emit_artifact("MANIFEST", &options.manifest)?;
    }
    Ok(())
}

fn parse_options() -> Result<Options, String> {
    let mut list = false;
    let mut quick = false;
    let mut samples = DEFAULT_SAMPLES;
    let mut samples_explicit = false;
    let mut filter_text = None;
    let mut csv = PathBuf::from("selection-oracle.csv");
    let mut manifest = PathBuf::from("selection-oracle-manifest.json");
    let mut emit_artifacts = false;
    let mut args = env::args().skip(1);
    while let Some(argument) = args.next() {
        match argument.as_str() {
            // The existing row_selector target uses this one flag as a
            // temporary dfexp transport adapter. The standalone target does
            // not need it, so the oracle parser simply consumes it.
            "--selection-oracle" => {}
            "--list" => list = true,
            "--quick" => quick = true,
            "--samples" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--samples requires a value".to_string())?;
                samples = value
                    .parse::<usize>()
                    .map_err(|_| "--samples must be an integer".to_string())?;
                samples_explicit = true;
            }
            "--filter" => {
                filter_text = Some(
                    args.next()
                        .ok_or_else(|| "--filter requires a regular expression".to_string())?,
                );
            }
            "--csv" => {
                csv = PathBuf::from(
                    args.next()
                        .ok_or_else(|| "--csv requires a path".to_string())?,
                );
            }
            "--manifest" => {
                manifest = PathBuf::from(
                    args.next()
                        .ok_or_else(|| "--manifest requires a path".to_string())?,
                );
            }
            "--emit-artifacts" => emit_artifacts = true,
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            _ => return Err(format!("unsupported argument {argument:?}")),
        }
    }
    if quick && !samples_explicit {
        samples = QUICK_SAMPLES;
    }
    if !(2..=100).contains(&samples) || !samples.is_multiple_of(2) {
        return Err("--samples must be an even integer in 2..=100".to_string());
    }
    if csv == manifest {
        return Err("--csv and --manifest must name different paths".to_string());
    }
    let filter = filter_text
        .as_deref()
        .map(Regex::new)
        .transpose()
        .map_err(|error| format!("invalid --filter regex: {error}"))?;
    Ok(Options {
        list,
        quick,
        samples,
        filter,
        filter_text,
        csv,
        manifest,
        emit_artifacts,
    })
}

fn print_help() {
    println!(
        "arrow_reader_row_selection_oracle \
         [--list] [--quick] [--filter REGEX] [--samples EVEN] \
         [--csv PATH] [--manifest PATH] [--emit-artifacts]"
    );
}

fn list_cells(options: &Options) {
    let contexts = ORACLE_CONTEXTS
        .iter()
        .copied()
        .filter(|context| !options.quick || matches!(context.id, "C0" | "C3"));
    let mut count = 0usize;
    for context in contexts {
        for run_len in ORACLE_L_SWEEP {
            count += list_cell(
                options,
                cell_id("S-A", context.id, &format!("f50_l{run_len}")),
            );
        }
        if options.quick {
            continue;
        }
        count += list_cell(options, format!("S-B/{}/adaptive<=3", context.id));
        if includes(SELECTIVITY_CONTEXTS, context.id) {
            for percent in ORACLE_SELECTIVITY_PERCENT {
                for run_len in ORACLE_SELECTIVITY_L {
                    count += list_cell(
                        options,
                        cell_id("S-C", context.id, &format!("f{percent:02}_l{run_len}")),
                    );
                }
            }
            for shape in special_shapes() {
                count += list_cell(options, cell_id("S-D", context.id, &shape.name));
            }
        }
        if includes(PAGE_INDEX_CONTEXTS, context.id) {
            for percent in [2, 10] {
                for run_len in ORACLE_SELECTIVITY_L {
                    count += list_cell(
                        options,
                        cell_id("S-E", context.id, &format!("f{percent:02}_l{run_len}")),
                    );
                }
            }
        }
        if includes(BATCH_CONTEXTS, context.id) {
            for run_len in BATCH_SENSITIVITY_L {
                count += list_cell(
                    options,
                    cell_id("S-F", context.id, &format!("f50_l{run_len}")),
                );
            }
        }
        if includes(PREDICATE_CONTEXTS, context.id) {
            for shape in predicate_shapes() {
                count += list_cell(options, cell_id("S-G", context.id, &shape.name));
            }
        }
        for run_len in auto_validation_lengths(context.id) {
            count += list_cell(
                options,
                cell_id("S-H", context.id, &format!("f50_l{run_len}")),
            );
        }
    }
    eprintln!("listed {count} fixed cells plus measured adaptive cells");
}

fn list_cell(options: &Options, cell: String) -> usize {
    if matches_filter(options, &cell) {
        println!("{cell}");
        1
    } else {
        0
    }
}

fn run_context(
    runtime: &Runtime,
    options: &Options,
    context: OracleContext,
    rows: &mut Vec<CsvRow>,
) -> Result<(), String> {
    let adaptive_requested = !options.quick && adaptive_requested(options, context.id);
    let normal_needed = normal_fixture_needed(options, context, adaptive_requested);
    let mut sweep = BTreeMap::<usize, PairMeasurement>::new();

    if normal_needed {
        eprintln!("building {} fixture", context.id);
        let fixture = build_oracle_fixture(context, None)
            .map_err(|error| format!("cannot build {} fixture: {error}", context.id))?;

        for run_len in ORACLE_L_SWEEP {
            let shape = OracleShape::l_sweep(*run_len);
            let primary_cell = cell_id("S-A", context.id, &shape.name);
            let h_cell = cell_id("S-H", context.id, &shape.name);
            let h_requested = !options.quick
                && auto_validation_lengths(context.id).contains(run_len)
                && matches_filter(options, &h_cell);
            let primary_requested = matches_filter(options, &primary_cell);
            if !primary_requested && !h_requested && !adaptive_requested {
                continue;
            }
            let pair = measure_pair(
                runtime,
                options.samples,
                &fixture,
                &shape,
                OracleSelectionSource::External,
                &primary_cell,
            );
            if primary_requested {
                append_pair_rows(rows, "S-A", primary_cell, context, &shape, &pair, false);
            }
            if h_requested {
                append_pair_rows(rows, "S-H", h_cell.clone(), context, &shape, &pair, true);
                let auto = measure_single(
                    runtime,
                    options.samples,
                    &fixture,
                    Some(&shape),
                    OracleSelectionSource::External,
                    OracleArm::Auto,
                    pair.selectors.rows_out,
                    pair.selectors.checksum,
                    &h_cell,
                );
                rows.push(csv_row(
                    "S-H",
                    h_cell,
                    context,
                    &shape,
                    OracleSelectionSource::External,
                    auto,
                    false,
                ));
            }
            sweep.insert(*run_len, pair);
        }

        if adaptive_requested {
            run_adaptive(runtime, options, context, &fixture, &mut sweep, rows);
        }

        if !options.quick && includes(SELECTIVITY_CONTEXTS, context.id) {
            for shape in selectivity_shapes() {
                run_pair_if_selected(runtime, options, context, &fixture, "S-C", &shape, rows);
            }
            for shape in special_shapes() {
                let cell = cell_id("S-D", context.id, &shape.name);
                if !matches_filter(options, &cell) {
                    continue;
                }
                if shape.name == "all_selected" {
                    let measurement = measure_single(
                        runtime,
                        options.samples,
                        &fixture,
                        None,
                        OracleSelectionSource::None,
                        OracleArm::NoSelection,
                        ORACLE_ROW_GROUPS * ROWS_PER_GROUP,
                        0,
                        &cell,
                    );
                    rows.push(csv_row(
                        "S-D",
                        cell,
                        context,
                        &shape,
                        OracleSelectionSource::None,
                        measurement,
                        false,
                    ));
                } else {
                    let pair = measure_pair(
                        runtime,
                        options.samples,
                        &fixture,
                        &shape,
                        OracleSelectionSource::External,
                        &cell,
                    );
                    append_pair_rows(rows, "S-D", cell, context, &shape, &pair, false);
                }
            }
        }

        if !options.quick && includes(BATCH_CONTEXTS, context.id) {
            let batch_fixture = fixture.with_batch_size(1_024);
            for run_len in BATCH_SENSITIVITY_L {
                let shape = OracleShape::l_sweep(*run_len);
                run_pair_if_selected(
                    runtime,
                    options,
                    batch_fixture.context(),
                    &batch_fixture,
                    "S-F",
                    &shape,
                    rows,
                );
            }
        }
    }

    if !options.quick && includes(PAGE_INDEX_CONTEXTS, context.id) {
        let page_shapes = page_index_shapes();
        if page_shapes
            .iter()
            .any(|shape| matches_filter(options, &cell_id("S-E", context.id, &shape.name)))
        {
            let page_context = context.with_page_index();
            eprintln!("building {} page-index fixture", context.id);
            let fixture = build_oracle_fixture(page_context, None).map_err(|error| {
                format!("cannot build {} page-index fixture: {error}", context.id)
            })?;
            for shape in page_shapes {
                run_pair_if_selected(
                    runtime,
                    options,
                    page_context,
                    &fixture,
                    "S-E",
                    &shape,
                    rows,
                );
            }
        }
    }

    if !options.quick && includes(PREDICATE_CONTEXTS, context.id) {
        for shape in predicate_shapes() {
            let cell = cell_id("S-G", context.id, &shape.name);
            if !matches_filter(options, &cell) {
                continue;
            }
            eprintln!("building {cell} predicate fixture");
            let predicate = shape.predicate_values();
            let fixture = build_oracle_fixture(context, Some(&predicate))
                .map_err(|error| format!("cannot build {cell} predicate fixture: {error}"))?;
            let pair = measure_pair(
                runtime,
                options.samples,
                &fixture,
                &shape,
                OracleSelectionSource::Predicate,
                &cell,
            );
            append_pair_rows(rows, "S-G", cell, context, &shape, &pair, false);
        }
    }
    Ok(())
}

fn normal_fixture_needed(
    options: &Options,
    context: OracleContext,
    adaptive_requested: bool,
) -> bool {
    if adaptive_requested {
        return true;
    }
    if ORACLE_L_SWEEP.iter().any(|run_len| {
        matches_filter(
            options,
            &cell_id("S-A", context.id, &format!("f50_l{run_len}")),
        ) || (!options.quick
            && auto_validation_lengths(context.id).contains(run_len)
            && matches_filter(
                options,
                &cell_id("S-H", context.id, &format!("f50_l{run_len}")),
            ))
    }) {
        return true;
    }
    if options.quick {
        return false;
    }
    (includes(SELECTIVITY_CONTEXTS, context.id)
        && selectivity_shapes()
            .into_iter()
            .chain(special_shapes())
            .any(|shape| {
                matches_filter(options, &cell_id("S-C", context.id, &shape.name))
                    || matches_filter(options, &cell_id("S-D", context.id, &shape.name))
            }))
        || (includes(BATCH_CONTEXTS, context.id)
            && BATCH_SENSITIVITY_L.iter().any(|run_len| {
                matches_filter(
                    options,
                    &cell_id("S-F", context.id, &format!("f50_l{run_len}")),
                )
            }))
}

fn run_pair_if_selected(
    runtime: &Runtime,
    options: &Options,
    context: OracleContext,
    fixture: &OracleFixture,
    group: &str,
    shape: &OracleShape,
    rows: &mut Vec<CsvRow>,
) {
    let cell = cell_id(group, context.id, &shape.name);
    if !matches_filter(options, &cell) {
        return;
    }
    let pair = measure_pair(
        runtime,
        options.samples,
        fixture,
        shape,
        OracleSelectionSource::External,
        &cell,
    );
    append_pair_rows(rows, group, cell, context, shape, &pair, false);
}

fn run_adaptive(
    runtime: &Runtime,
    options: &Options,
    context: OracleContext,
    fixture: &OracleFixture,
    points: &mut BTreeMap<usize, PairMeasurement>,
    rows: &mut Vec<CsvRow>,
) {
    for _ in 0..3 {
        let Some((low, high)) = refinement_interval(points) else {
            break;
        };
        let mid = geometric_midpoint(low, high);
        if mid <= low || mid >= high || points.contains_key(&mid) {
            break;
        }
        let shape = OracleShape::l_sweep(mid);
        let cell = cell_id("S-B", context.id, &shape.name);
        let pair = measure_pair(
            runtime,
            options.samples,
            fixture,
            &shape,
            OracleSelectionSource::External,
            &cell,
        );
        if matches_filter(options, &cell) {
            append_pair_rows(rows, "S-B", cell, context, &shape, &pair, false);
        }
        let tie = pair.winner() == Winner::Tie;
        points.insert(mid, pair);
        if tie {
            break;
        }
    }
}

fn refinement_interval(points: &BTreeMap<usize, PairMeasurement>) -> Option<(usize, usize)> {
    if points.values().any(|point| point.winner() == Winner::Tie) {
        return None;
    }
    points
        .iter()
        .zip(points.iter().skip(1))
        .filter_map(|((low, low_result), (high, high_result))| {
            (low_result.winner() != high_result.winner()).then_some((
                ((*low, *high)),
                ((geometric_midpoint(*low, *high) as f64 / 32.0).ln().abs()),
            ))
        })
        .min_by(|left, right| left.1.total_cmp(&right.1))
        .map(|(interval, _)| interval)
}

fn geometric_midpoint(low: usize, high: usize) -> usize {
    if high <= low + 1 {
        return low;
    }
    ((low as f64 * high as f64).sqrt().round() as usize).clamp(low + 1, high - 1)
}

fn measure_pair(
    runtime: &Runtime,
    samples: usize,
    fixture: &OracleFixture,
    shape: &OracleShape,
    source: OracleSelectionSource,
    cell: &str,
) -> PairMeasurement {
    eprintln!("measuring {cell}");
    let selection = (source == OracleSelectionSource::External).then(|| shape.selection());
    let selectors_check = runtime.block_on(run_oracle(
        fixture,
        selection.clone(),
        source,
        OracleArm::Selectors,
        true,
    ));
    let mask_check = runtime.block_on(run_oracle(
        fixture,
        selection.clone(),
        source,
        OracleArm::Mask,
        true,
    ));
    assert_equivalent(
        cell,
        shape.total_selected_rows(),
        selectors_check,
        mask_check,
    );

    for _ in 0..WARMUPS_PER_ARM {
        let selectors = runtime.block_on(run_oracle(
            fixture,
            selection.clone(),
            source,
            OracleArm::Selectors,
            false,
        ));
        let mask = runtime.block_on(run_oracle(
            fixture,
            selection.clone(),
            source,
            OracleArm::Mask,
            false,
        ));
        assert_eq!(selectors.row_count, selectors_check.row_count, "{cell}");
        assert_eq!(mask.row_count, mask_check.row_count, "{cell}");
    }

    let mut selector_samples = Vec::with_capacity(samples);
    let mut selector_timestamps = Vec::with_capacity(samples);
    let mut mask_samples = Vec::with_capacity(samples);
    let mut mask_timestamps = Vec::with_capacity(samples);
    while selector_samples.len() < samples || mask_samples.len() < samples {
        for arm in [
            OracleArm::Selectors,
            OracleArm::Mask,
            OracleArm::Mask,
            OracleArm::Selectors,
        ] {
            let (target, timestamps) = match arm {
                OracleArm::Selectors if selector_samples.len() < samples => {
                    (&mut selector_samples, &mut selector_timestamps)
                }
                OracleArm::Mask if mask_samples.len() < samples => {
                    (&mut mask_samples, &mut mask_timestamps)
                }
                _ => continue,
            };
            let (elapsed, result, timestamp) =
                time_arm(runtime, fixture, selection.as_ref(), source, arm);
            assert_eq!(result.row_count, selectors_check.row_count, "{cell}");
            target.push(elapsed);
            timestamps.push(timestamp);
        }
    }

    PairMeasurement {
        selectors: arm_measurement(
            OracleArm::Selectors,
            selector_samples,
            selector_timestamps,
            selectors_check,
        ),
        mask: arm_measurement(OracleArm::Mask, mask_samples, mask_timestamps, mask_check),
    }
}

#[allow(clippy::too_many_arguments)]
fn measure_single(
    runtime: &Runtime,
    samples: usize,
    fixture: &OracleFixture,
    shape: Option<&OracleShape>,
    source: OracleSelectionSource,
    arm: OracleArm,
    expected_rows: usize,
    expected_checksum: u64,
    cell: &str,
) -> ArmMeasurement {
    eprintln!("measuring {cell}/{}", arm.label());
    let selection = match (source, shape) {
        (OracleSelectionSource::External, Some(shape)) => Some(shape.selection()),
        (OracleSelectionSource::External, None) => panic!("external source needs a shape"),
        _ => None,
    };
    let check = runtime.block_on(run_oracle(fixture, selection.clone(), source, arm, true));
    assert_eq!(check.row_count, expected_rows, "{cell}");
    if source != OracleSelectionSource::None {
        assert_eq!(check.checksum, expected_checksum, "{cell}");
    }
    for _ in 0..WARMUPS_PER_ARM {
        let result = runtime.block_on(run_oracle(fixture, selection.clone(), source, arm, false));
        assert_eq!(result.row_count, expected_rows, "{cell}");
    }
    let mut values = Vec::with_capacity(samples);
    let mut timestamps = Vec::with_capacity(samples);
    for _ in 0..samples {
        let (elapsed, result, timestamp) =
            time_arm(runtime, fixture, selection.as_ref(), source, arm);
        assert_eq!(result.row_count, expected_rows, "{cell}");
        values.push(elapsed);
        timestamps.push(timestamp);
    }
    arm_measurement(arm, values, timestamps, check)
}

fn time_arm(
    runtime: &Runtime,
    fixture: &OracleFixture,
    selection: Option<&parquet::arrow::arrow_reader::RowSelection>,
    source: OracleSelectionSource,
    arm: OracleArm,
) -> (u64, OracleRunResult, u64) {
    // RowSelection::clone is a deep copy for selector-backed selections. Keep
    // that caller-side input materialisation outside the execution timer so
    // fragmented shapes do not gain a large, policy-independent penalty.
    let selection = selection.cloned();
    let timestamp = unix_nanos();
    let started = Instant::now();
    let result = runtime.block_on(run_oracle(fixture, selection, source, arm, false));
    let elapsed = started.elapsed().as_nanos() as u64;
    hint::black_box(result.row_count);
    (elapsed, result, timestamp)
}

fn arm_measurement(
    arm: OracleArm,
    samples_ns: Vec<u64>,
    sample_started_unix_ns: Vec<u64>,
    check: OracleRunResult,
) -> ArmMeasurement {
    let median_ns = median(&samples_ns);
    let deviations = samples_ns
        .iter()
        .map(|sample| sample.abs_diff(median_ns))
        .collect::<Vec<_>>();
    ArmMeasurement {
        arm,
        samples_ns,
        sample_started_unix_ns,
        median_ns,
        mad_ns: median(&deviations),
        rows_out: check.row_count,
        checksum: check.checksum,
    }
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

fn assert_equivalent(
    cell: &str,
    expected_rows: usize,
    selectors: OracleRunResult,
    mask: OracleRunResult,
) {
    assert_eq!(selectors.row_count, expected_rows, "{cell}/selectors");
    assert_eq!(mask.row_count, expected_rows, "{cell}/mask");
    assert_eq!(
        selectors, mask,
        "{cell}: forced arms returned different data"
    );
}

fn append_pair_rows(
    rows: &mut Vec<CsvRow>,
    group: &str,
    cell: String,
    context: OracleContext,
    shape: &OracleShape,
    pair: &PairMeasurement,
    reused_samples: bool,
) {
    let mut selector_row = csv_row(
        group,
        cell.clone(),
        context,
        shape,
        OracleSelectionSource::External,
        pair.selectors.clone(),
        reused_samples,
    );
    let mut mask_row = csv_row(
        group,
        cell,
        context,
        shape,
        OracleSelectionSource::External,
        pair.mask.clone(),
        reused_samples,
    );
    if group == "S-G" {
        selector_row.source = OracleSelectionSource::Predicate;
        mask_row.source = OracleSelectionSource::Predicate;
    }
    rows.push(selector_row);
    rows.push(mask_row);
}

#[allow(clippy::too_many_arguments)]
fn csv_row(
    group: &str,
    cell_id: String,
    context: OracleContext,
    shape: &OracleShape,
    source: OracleSelectionSource,
    measurement: ArmMeasurement,
    reused_samples: bool,
) -> CsvRow {
    CsvRow {
        group: group.to_string(),
        cell_id,
        context,
        shape_name: shape.name.clone(),
        nominal_skip: shape.nominal_skip,
        nominal_select: shape.nominal_select,
        summary: shape.summary(),
        source,
        measurement,
        reused_samples,
        auto_choice: shape.auto_choice().label().to_string(),
    }
}

fn selectivity_shapes() -> Vec<OracleShape> {
    ORACLE_SELECTIVITY_PERCENT
        .iter()
        .flat_map(|percent| {
            ORACLE_SELECTIVITY_L
                .iter()
                .map(move |run_len| OracleShape::selectivity(*percent, *run_len))
        })
        .collect()
}

fn page_index_shapes() -> Vec<OracleShape> {
    [2, 10]
        .into_iter()
        .flat_map(|percent| {
            ORACLE_SELECTIVITY_L
                .iter()
                .map(move |run_len| OracleShape::selectivity(percent, *run_len))
        })
        .collect()
}

fn special_shapes() -> Vec<OracleShape> {
    vec![
        OracleShape::bursty(30),
        OracleShape::bursty(70),
        OracleShape::sparse_cluster(),
        OracleShape::dense(),
        OracleShape::all_selected(),
    ]
}

fn predicate_shapes() -> Vec<OracleShape> {
    vec![
        OracleShape::l_sweep(8),
        OracleShape::l_sweep(32),
        OracleShape::l_sweep(128),
        OracleShape::selectivity(2, 64),
        OracleShape::selectivity(98, 64),
        OracleShape::bursty(70),
    ]
}

fn auto_validation_lengths(context_id: &str) -> Vec<usize> {
    let index = context_id
        .strip_prefix('C')
        .unwrap()
        .parse::<usize>()
        .unwrap();
    let mut lengths = vec![if index.is_multiple_of(2) { 16 } else { 64 }];
    if matches!(context_id, "C0" | "C3" | "C4" | "C5" | "C8" | "C11") {
        lengths.push(if index.is_multiple_of(2) { 64 } else { 16 });
    }
    lengths.sort_unstable();
    lengths
}

fn adaptive_requested(options: &Options, context_id: &str) -> bool {
    if options.filter.is_none() {
        return true;
    }
    (2..2_048).any(|run_len| {
        !ORACLE_L_SWEEP.contains(&run_len)
            && matches_filter(
                options,
                &cell_id("S-B", context_id, &format!("f50_l{run_len}")),
            )
    })
}

fn includes(values: &[&str], value: &str) -> bool {
    values.contains(&value)
}

fn cell_id(group: &str, context_id: &str, shape_name: &str) -> String {
    format!("{group}/{context_id}/{shape_name}")
}

fn matches_filter(options: &Options, cell: &str) -> bool {
    options
        .filter
        .as_ref()
        .is_none_or(|filter| filter.is_match(cell))
}

fn write_csv(path: &Path, rows: &[CsvRow]) -> Result<(), String> {
    let file = File::create(path)
        .map_err(|error| format!("cannot create CSV {}: {error}", path.display()))?;
    let mut writer = BufWriter::new(file);
    writeln!(
        writer,
        "schema_version,group,cell_id,context_id,dtype,payload_columns,encoding,compression,page_index,batch_size,rows_per_group,row_groups,shape_name,skip_rows,select_rows,selected_fraction,avg_run_len,run_count,long_skip_share_1024,long_skip_share_4096,long_skip_share_page,selection_source,arm,sample_count,samples_ns,sample_started_unix_ns,median_ns,mad_ns,rows_out,checksum,auto_choice,reused_samples"
    )
    .map_err(|error| format!("cannot write CSV header: {error}"))?;
    for row in rows {
        let fields = vec![
            CSV_SCHEMA_VERSION.to_string(),
            row.group.clone(),
            row.cell_id.clone(),
            row.context.id.to_string(),
            row.context.payload.dtype().to_string(),
            row.context.payload_columns.to_string(),
            row.context.encoding().to_string(),
            row.context.compression.label().to_string(),
            row.context.page_index.to_string(),
            row.context.batch_size.to_string(),
            ROWS_PER_GROUP.to_string(),
            ORACLE_ROW_GROUPS.to_string(),
            row.shape_name.clone(),
            row.nominal_skip
                .map(|value| value.to_string())
                .unwrap_or_default(),
            row.nominal_select
                .map(|value| value.to_string())
                .unwrap_or_default(),
            format!("{:.9}", row.summary.selected_fraction),
            format!("{:.9}", row.summary.avg_run_len),
            row.summary.run_count.to_string(),
            format!("{:.9}", row.summary.long_skip_share_1024),
            format!("{:.9}", row.summary.long_skip_share_4096),
            row.context
                .page_index
                .then(|| format!("{:.9}", row.summary.long_skip_share_4096))
                .unwrap_or_default(),
            row.source.label().to_string(),
            row.measurement.arm.label().to_string(),
            row.measurement.samples_ns.len().to_string(),
            join_numbers(&row.measurement.samples_ns),
            join_numbers(&row.measurement.sample_started_unix_ns),
            row.measurement.median_ns.to_string(),
            row.measurement.mad_ns.to_string(),
            row.measurement.rows_out.to_string(),
            format!("{:016x}", row.measurement.checksum),
            row.auto_choice.clone(),
            row.reused_samples.to_string(),
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
        .map_err(|error| format!("cannot write CSV row: {error}"))?;
    }
    writer
        .flush()
        .map_err(|error| format!("cannot flush CSV {}: {error}", path.display()))
}

fn write_manifest(
    options: &Options,
    rows: &[CsvRow],
    started_unix_ns: u64,
    completed_unix_ns: u64,
    elapsed_ns: u64,
) -> Result<(), String> {
    let rustc = command_output("rustc", &["-vV"]);
    let git_sha = command_output(
        "git",
        &["-C", env!("CARGO_MANIFEST_DIR"), "rev-parse", "HEAD"],
    );
    let cpu_model = cpu_model();
    let os = command_output("uname", &["-a"]);
    let hostname = fs::read_to_string("/etc/hostname")
        .unwrap_or_default()
        .trim()
        .to_string();
    let mut hasher = DefaultHasher::new();
    // Pod hostnames include the Run ID and change across otherwise identical
    // repetitions. Record them, but exclude them from this stable fingerprint.
    (&rustc, &cpu_model, &os).hash(&mut hasher);
    let cells = rows
        .iter()
        .map(|row| row.cell_id.as_str())
        .collect::<BTreeSet<_>>()
        .len();
    let adaptive_run_lengths = rows
        .iter()
        .filter(|row| row.group == "S-B" && row.measurement.arm == OracleArm::Selectors)
        .filter_map(|row| row.nominal_select)
        .collect::<Vec<_>>();
    let manifest = json!({
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "benchmark": "arrow_reader_row_selection_oracle",
        "csv_schema_version": CSV_SCHEMA_VERSION,
        "git_sha": git_sha,
        "rustc": rustc,
        "cpu_model": cpu_model,
        "hostname": hostname,
        "os": os,
        "environment_fingerprint": format!("{:016x}", hasher.finish()),
        "started_unix_ns": started_unix_ns,
        "completed_unix_ns": completed_unix_ns,
        "elapsed_ns": elapsed_ns,
        "quick": options.quick,
        "samples_per_arm": options.samples,
        "warmups_per_arm": WARMUPS_PER_ARM,
        "filter": options.filter_text.as_deref(),
        "cell_count": cells,
        "arm_row_count": rows.len(),
        "adaptive_run_lengths": adaptive_run_lengths,
        "fixture": {
            "row_groups": ORACLE_ROW_GROUPS,
            "rows_per_group": ROWS_PER_GROUP,
            "default_batch_size": 8192,
            "page_row_limit": ORACLE_PAGE_ROWS,
            "in_memory": true,
            "metadata_preparsed": true
        },
        "timing_protocol": {
            "forced_arm_order": "selectors,mask,mask,selectors repeated",
            "statistic": "median",
            "dispersion": "median_absolute_deviation",
            "tie_zone": "abs(delta) < max(3*sqrt((mad_s^2+mad_m^2)/2), 1%*mean(medians))",
            "clock": "std::time::Instant",
            "sample_start_clock": "unix_epoch_nanoseconds"
        },
        "auto_rule": "mask iff rows_per_group < effective_run_count * 32",
        "checksum": "row_count plus position-mixed payload0 XOR",
        "classification": "non-formal"
    });
    let file = File::create(&options.manifest).map_err(|error| {
        format!(
            "cannot create manifest {}: {error}",
            options.manifest.display()
        )
    })?;
    serde_json::to_writer_pretty(file, &manifest)
        .map_err(|error| format!("cannot write manifest: {error}"))?;
    let mut file = fs::OpenOptions::new()
        .append(true)
        .open(&options.manifest)
        .map_err(|error| format!("cannot reopen manifest: {error}"))?;
    writeln!(file).map_err(|error| format!("cannot finish manifest: {error}"))
}

fn emit_artifact(kind: &str, path: &Path) -> Result<(), String> {
    let raw = fs::read_to_string(path)
        .map_err(|error| format!("cannot read {} for log embedding: {error}", path.display()))?;
    println!("DFEXP_SELECTION_ORACLE_{kind}_BEGIN");
    print!("{raw}");
    if !raw.ends_with('\n') {
        println!();
    }
    println!("DFEXP_SELECTION_ORACLE_{kind}_END");
    Ok(())
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

fn cpu_model() -> String {
    fs::read_to_string("/proc/cpuinfo")
        .ok()
        .and_then(|contents| {
            contents.lines().find_map(|line| {
                line.split_once(':')
                    .filter(|(key, _)| key.trim() == "model name")
                    .map(|(_, value)| value.trim().to_string())
            })
        })
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| command_output("sysctl", &["-n", "machdep.cpu.brand_string"]))
}

fn unix_nanos() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos() as u64
}

fn join_numbers(values: &[u64]) -> String {
    values
        .iter()
        .map(u64::to_string)
        .collect::<Vec<_>>()
        .join("|")
}

fn escape_csv(value: &str) -> String {
    if value.contains(',') || value.contains('"') || value.contains('\n') || value.contains('\r') {
        format!("\"{}\"", value.replace('"', "\"\""))
    } else {
        value.to_string()
    }
}
