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

//! PC-1 four-arm evaluation for experimental per-column row selection.

use std::collections::{BTreeMap, BTreeSet, hash_map::DefaultHasher};
use std::env;
use std::fs::{self, File};
use std::hash::{Hash, Hasher};
use std::hint;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use futures::StreamExt;
use parquet::arrow::arrow_reader::{RowSelection, RowSelectionPolicy};
use parquet::arrow::{ParquetRecordBatchStreamBuilder, ProjectionMask};
use regex::Regex;
use serde_json::json;
use sha2::{Digest, Sha256};
use tokio::runtime::Runtime;

use super::fixture::{
    ORACLE_CONTEXTS, ORACLE_PAGE_ROWS, ORACLE_ROW_GROUPS, OracleContext, OracleFixture,
    OraclePayload, PC_MIXED_CONTEXTS, TT_CONTEXTS, build_oracle_fixture,
    build_oracle_fixture_with_dimensions,
};
use super::model::{BATCH_SIZE, ROWS_PER_GROUP};
use super::runner::{ProjectedContentDigest, ProjectedContentDigester};
use super::shapes::{OracleShape, OracleShapeSummary};

const CSV_SCHEMA_VERSION: &str = "arrow-row-selection-pc1-v1";
const MANIFEST_SCHEMA_VERSION: &str = "arrow-row-selection-pc1-manifest-v1";
const PC1D_CSV_SCHEMA_VERSION: &str = "arrow-row-selection-pc1d-rg16-v1";
const PC1D_MANIFEST_SCHEMA_VERSION: &str = "arrow-row-selection-pc1d-rg16-manifest-v1";
const PC1D_ROW_GROUPS: usize = 16;
const DEFAULT_SAMPLES: usize = 12;
const WARMUPS_PER_ARM: usize = 2;
const PURE_DICTIONARY_THRESHOLD: usize = 4;
const DEFAULT_COLUMN_THRESHOLD: usize = 16;

#[derive(Debug)]
struct Options {
    list: bool,
    phase_d_rg16: bool,
    samples: usize,
    filter: Option<Regex>,
    filter_text: Option<String>,
    csv: PathBuf,
    manifest: PathBuf,
    emit_artifacts: bool,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum PcArm {
    Auto32,
    Selectors,
    Mask,
    PerColumn,
}

impl PcArm {
    const ALL: [Self; 4] = [Self::Auto32, Self::Selectors, Self::Mask, Self::PerColumn];
    const FAST: [Self; 2] = [Self::Auto32, Self::PerColumn];

    const fn label(self) -> &'static str {
        match self {
            Self::Auto32 => "auto32",
            Self::Selectors => "selectors",
            Self::Mask => "mask",
            Self::PerColumn => "percolumn",
        }
    }

    const fn policy(self) -> RowSelectionPolicy {
        match self {
            Self::Auto32 => RowSelectionPolicy::Auto { threshold: 32 },
            Self::Selectors => RowSelectionPolicy::Selectors,
            Self::Mask => RowSelectionPolicy::Mask,
            Self::PerColumn => RowSelectionPolicy::PerColumn,
        }
    }

    const fn index(self) -> usize {
        match self {
            Self::Auto32 => 0,
            Self::Selectors => 1,
            Self::Mask => 2,
            Self::PerColumn => 3,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct PcRunResult {
    row_count: usize,
    content: Option<ProjectedContentDigest>,
}

#[derive(Clone, Debug)]
struct ArmMeasurement {
    arm: PcArm,
    samples_ns: Vec<u64>,
    sample_started_unix_ns: Vec<u64>,
    median_ns: u64,
    mad_ns: u64,
    rows_out: usize,
    content: ProjectedContentDigest,
}

#[derive(Clone, Debug)]
struct CsvRow {
    group: &'static str,
    cell_id: String,
    context: OracleContext,
    context_role: &'static str,
    shape_name: String,
    nominal_skip: Option<usize>,
    nominal_select: Option<usize>,
    summary: OracleShapeSummary,
    selection_source: &'static str,
    row_groups: usize,
    measurement: ArmMeasurement,
}

pub(crate) fn main() {
    if let Err(error) = try_main() {
        eprintln!("PC-1 row-selection evaluation failed: {error}");
        std::process::exit(2);
    }
}

fn try_main() -> Result<(), String> {
    let options = parse_options()?;
    let contexts = pc_contexts();
    let shapes = pc_shapes();
    assert_eq!(
        contexts
            .iter()
            .filter(|(_, role)| *role == "homogeneous")
            .count(),
        44
    );
    assert_eq!(
        contexts
            .iter()
            .filter(|(_, role)| matches!(*role, "mixed" | "holdout"))
            .count(),
        6
    );
    assert_eq!(shapes.len(), 8);

    if options.phase_d_rg16 {
        return run_phase_d_rg16(&options, &contexts, &shapes);
    }

    if options.list {
        list_cells(&options, &contexts, &shapes);
        return Ok(());
    }

    let started_unix_ns = unix_nanos();
    let started = Instant::now();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|error| format!("cannot build Tokio runtime: {error}"))?;
    let mut rows = Vec::new();
    let mut fixture_sha256 = BTreeMap::new();

    for (context, role) in contexts {
        let selected_shapes = shapes
            .iter()
            .filter(|shape| matches_filter(&options, &cell_id(context.id, &shape.name)))
            .collect::<Vec<_>>();
        let fast_requested = context.id == "PC-M2"
            && [
                "PC-FAST/PC-M2/all_selected_4rg",
                "PC-FAST/PC-M2/all_selected_1rg",
            ]
            .into_iter()
            .any(|cell| matches_filter(&options, cell));
        if selected_shapes.is_empty() && !fast_requested {
            continue;
        }

        eprintln!("building PC-1 {} fixture", context.id);
        let fixture = build_oracle_fixture(context, None)
            .map_err(|error| format!("cannot build PC-1 {} fixture: {error}", context.id))?;
        fixture_sha256.insert(context.id.to_string(), fixture.bytes_sha256());

        for shape in selected_shapes {
            let cell = cell_id(context.id, &shape.name);
            eprintln!("measuring {cell}");
            rows.extend(measure_four_arm_cell(
                &runtime,
                &fixture,
                role,
                shape,
                &cell,
                options.samples,
            )?);
        }

        if fast_requested {
            rows.extend(measure_fast_controls(
                &runtime,
                &options,
                &fixture,
                role,
                options.samples,
            )?);
        }
    }

    if rows.is_empty() {
        return Err("cell filter selected no PC-1 benchmark cells".to_string());
    }

    let completed_unix_ns = unix_nanos();
    write_csv(&options.csv, &rows, CSV_SCHEMA_VERSION)?;
    write_manifest(
        &options,
        &rows,
        &fixture_sha256,
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
        "PC-1 evaluation complete: {cells} cells, {} arm rows, csv={}, manifest={}",
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

fn run_phase_d_rg16(
    options: &Options,
    contexts: &[(OracleContext, &'static str)],
    shapes: &[OracleShape],
) -> Result<(), String> {
    if options.list {
        let mut count = 0usize;
        for (context, role) in contexts {
            if *role != "homogeneous" {
                continue;
            }
            for shape in shapes {
                let cell = phase_d_cell_id(context.id, &shape.name);
                if matches_filter(options, &cell) {
                    println!("{cell}");
                    count += 1;
                }
            }
        }
        eprintln!("listed {count} PC-1d RG16 cells");
        return Ok(());
    }

    let started_unix_ns = unix_nanos();
    let started = Instant::now();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|error| format!("cannot build Tokio runtime: {error}"))?;
    let mut rows = Vec::new();
    let mut fixture_sha256 = BTreeMap::new();

    for (context, role) in contexts {
        if *role != "homogeneous" {
            continue;
        }
        let selected_shapes = shapes
            .iter()
            .filter(|shape| matches_filter(options, &phase_d_cell_id(context.id, &shape.name)))
            .collect::<Vec<_>>();
        if selected_shapes.is_empty() {
            continue;
        }
        eprintln!("building PC-1d RG16 {} fixture", context.id);
        let fixture = build_oracle_fixture_with_dimensions(
            *context,
            None,
            PC1D_ROW_GROUPS,
            ROWS_PER_GROUP,
        )
        .map_err(|error| format!("cannot build PC-1d RG16 {} fixture: {error}", context.id))?;
        fixture_sha256.insert(context.id.to_string(), fixture.bytes_sha256());
        for shape in selected_shapes {
            let cell = phase_d_cell_id(context.id, &shape.name);
            eprintln!("measuring {cell}");
            rows.extend(measure_phase_d_cell(
                &runtime,
                &fixture,
                shape,
                &cell,
                options.samples,
            )?);
        }
    }

    if rows.is_empty() {
        return Err("cell filter selected no PC-1d RG16 cells".to_string());
    }
    let completed_unix_ns = unix_nanos();
    write_csv(&options.csv, &rows, PC1D_CSV_SCHEMA_VERSION)?;
    write_phase_d_manifest(
        options,
        &rows,
        &fixture_sha256,
        started_unix_ns,
        completed_unix_ns,
        started.elapsed().as_nanos() as u64,
    )?;
    let cells = rows
        .iter()
        .map(|row| row.cell_id.as_str())
        .collect::<BTreeSet<_>>()
        .len();
    println!("DFEXP_PC1D_RG16_CELLS={cells}");
    println!("DFEXP_PC1D_RG16_ROWS={}", rows.len());
    if options.emit_artifacts {
        emit_phase_d_artifact("CSV", &options.csv)?;
        emit_phase_d_artifact("MANIFEST", &options.manifest)?;
    }
    Ok(())
}

fn parse_options() -> Result<Options, String> {
    let mut list = false;
    let mut phase_d_rg16 = false;
    let mut samples = DEFAULT_SAMPLES;
    let mut filter_text = None;
    let mut csv = default_artifact_path("pc1.csv");
    let mut manifest = default_artifact_path("pc1-manifest.json");
    let mut emit_artifacts = false;
    let mut args = env::args().skip(1);
    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--selection-oracle" | "--pc-series" | "--bench" => {}
            "--pc1d-rg16" => phase_d_rg16 = true,
            "--list" => list = true,
            "--samples" => {
                samples = args
                    .next()
                    .ok_or_else(|| "--samples requires a value".to_string())?
                    .parse::<usize>()
                    .map_err(|_| "--samples must be an integer".to_string())?;
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
                println!(
                    "arrow_reader_row_selection_oracle --pc-series \
                     [--pc1d-rg16] \
                     [--list] [--filter REGEX] [--samples EVEN] \
                     [--csv PATH] [--manifest PATH] [--emit-artifacts]"
                );
                std::process::exit(0);
            }
            _ => return Err(format!("unsupported argument {argument:?}")),
        }
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
    if phase_d_rg16 {
        if csv == default_artifact_path("pc1.csv") {
            csv = default_artifact_path("pc1d-rg16.csv");
        }
        if manifest == default_artifact_path("pc1-manifest.json") {
            manifest = default_artifact_path("pc1d-rg16-manifest.json");
        }
    }
    Ok(Options {
        list,
        phase_d_rg16,
        samples,
        filter,
        filter_text,
        csv,
        manifest,
        emit_artifacts,
    })
}

fn pc_contexts() -> Vec<(OracleContext, &'static str)> {
    ORACLE_CONTEXTS
        .iter()
        .copied()
        .map(|context| (context, "homogeneous"))
        .chain(
            TT_CONTEXTS
                .iter()
                .copied()
                .filter(|context| !context.id.starts_with("TT-H-"))
                .map(|context| (context, "homogeneous")),
        )
        .chain(PC_MIXED_CONTEXTS.iter().copied().map(|context| {
            let role = if context.id == "PC-M6" {
                "holdout"
            } else {
                "mixed"
            };
            (context, role)
        }))
        .collect()
}

fn pc_shapes() -> Vec<OracleShape> {
    [4, 16, 64, 256, 1_024]
        .into_iter()
        .map(OracleShape::l_sweep)
        .chain([
            OracleShape::selectivity(2, 64),
            OracleShape::selectivity(98, 64),
            OracleShape::pc_bursty03_l4(),
        ])
        .collect()
}

fn list_cells(
    options: &Options,
    contexts: &[(OracleContext, &'static str)],
    shapes: &[OracleShape],
) {
    let mut count = 0usize;
    for (context, _) in contexts {
        for shape in shapes {
            let cell = cell_id(context.id, &shape.name);
            if matches_filter(options, &cell) {
                println!("{cell}");
                count += 1;
            }
        }
    }
    for cell in [
        "PC-FAST/PC-M2/all_selected_4rg",
        "PC-FAST/PC-M2/all_selected_1rg",
    ] {
        if matches_filter(options, cell) {
            println!("{cell}");
            count += 1;
        }
    }
    eprintln!("listed {count} PC-1 cells");
}

fn cell_id(context: &str, shape: &str) -> String {
    format!("PC-1/{context}/{shape}")
}

fn phase_d_cell_id(context: &str, shape: &str) -> String {
    format!("PC-1D-RG16/{context}/{shape}")
}

fn matches_filter(options: &Options, cell: &str) -> bool {
    options
        .filter
        .as_ref()
        .is_none_or(|filter| filter.is_match(cell))
}

async fn run_pc(
    fixture: &OracleFixture,
    selection: Option<RowSelection>,
    row_groups: Option<&[usize]>,
    arm: PcArm,
    attribution: bool,
) -> Result<PcRunResult, String> {
    let context = fixture.context();
    let projection = ProjectionMask::roots(fixture.schema_descr(), 0..context.payload_columns);
    let mut builder = ParquetRecordBatchStreamBuilder::new(fixture.reader())
        .await
        .map_err(|error| format!("cannot build PC-1 stream metadata: {error}"))?
        .with_batch_size(context.batch_size)
        .with_projection(projection)
        .with_row_selection_policy(arm.policy());
    if let Some(row_groups) = row_groups {
        builder = builder.with_row_groups(row_groups.to_vec());
    }
    if let Some(selection) = selection {
        builder = builder.with_row_selection(selection);
    }
    let mut stream = builder
        .build()
        .map_err(|error| format!("cannot build PC-1 stream: {error}"))?;
    let mut row_count = 0usize;
    let mut digester = attribution.then(ProjectedContentDigester::default);
    while let Some(batch) = stream.next().await {
        let batch = batch.map_err(|error| format!("PC-1 stream failed: {error}"))?;
        if let Some(digester) = &mut digester {
            digester.update(&batch);
        }
        row_count += batch.num_rows();
    }
    Ok(PcRunResult {
        row_count,
        content: digester.map(ProjectedContentDigester::finish),
    })
}

fn measure_phase_d_cell(
    runtime: &Runtime,
    fixture: &OracleFixture,
    shape: &OracleShape,
    cell: &str,
    samples: usize,
) -> Result<Vec<CsvRow>, String> {
    let selection = shape.selection_for_row_groups(PC1D_ROW_GROUPS);
    let expected_rows = shape.total_selected_rows_for_row_groups(PC1D_ROW_GROUPS);
    let checks = check_arms(
        runtime,
        fixture,
        Some(&selection),
        None,
        &PcArm::FAST,
        expected_rows,
        cell,
    )?;
    warm_arms(
        runtime,
        fixture,
        Some(&selection),
        None,
        &PcArm::FAST,
        expected_rows,
        cell,
    )?;

    const ORDER: [PcArm; 4] = [
        PcArm::Auto32,
        PcArm::PerColumn,
        PcArm::PerColumn,
        PcArm::Auto32,
    ];
    let mut values: [Vec<u64>; 2] = std::array::from_fn(|_| Vec::with_capacity(samples));
    let mut timestamps: [Vec<u64>; 2] = std::array::from_fn(|_| Vec::with_capacity(samples));
    while values.iter().any(|values| values.len() < samples) {
        for arm in ORDER {
            let index = usize::from(arm == PcArm::PerColumn);
            if values[index].len() == samples {
                continue;
            }
            let (elapsed, result, timestamp) =
                time_arm(runtime, fixture, Some(&selection), None, arm)?;
            if result.row_count != expected_rows {
                return Err(format!(
                    "{cell}/{} timed row mismatch: expected {expected_rows}, got {}",
                    arm.label(),
                    result.row_count
                ));
            }
            values[index].push(elapsed);
            timestamps[index].push(timestamp);
        }
    }

    let context = fixture.context();
    Ok(PcArm::FAST
        .into_iter()
        .enumerate()
        .map(|(index, arm)| CsvRow {
            group: "PC1D-HOM",
            cell_id: cell.to_string(),
            context,
            context_role: "homogeneous",
            shape_name: shape.name.clone(),
            nominal_skip: shape.nominal_skip,
            nominal_select: shape.nominal_select,
            summary: shape.summary(),
            selection_source: "external",
            row_groups: PC1D_ROW_GROUPS,
            measurement: measurement(
                arm,
                std::mem::take(&mut values[index]),
                std::mem::take(&mut timestamps[index]),
                &checks[index],
            ),
        })
        .collect())
}

fn measure_four_arm_cell(
    runtime: &Runtime,
    fixture: &OracleFixture,
    context_role: &'static str,
    shape: &OracleShape,
    cell: &str,
    samples: usize,
) -> Result<Vec<CsvRow>, String> {
    let selection = shape.selection();
    let expected_rows = shape.total_selected_rows();
    let checks = check_arms(
        runtime,
        fixture,
        Some(&selection),
        None,
        &PcArm::ALL,
        expected_rows,
        cell,
    )?;

    warm_arms(
        runtime,
        fixture,
        Some(&selection),
        None,
        &PcArm::ALL,
        expected_rows,
        cell,
    )?;

    // Four-arm mirrored ABBA generalisation: each forward order is followed
    // by its exact reverse, repeated until every arm has 12 samples.
    const ORDER: [PcArm; 8] = [
        PcArm::Auto32,
        PcArm::Selectors,
        PcArm::Mask,
        PcArm::PerColumn,
        PcArm::PerColumn,
        PcArm::Mask,
        PcArm::Selectors,
        PcArm::Auto32,
    ];
    let mut values: [Vec<u64>; 4] = std::array::from_fn(|_| Vec::with_capacity(samples));
    let mut timestamps: [Vec<u64>; 4] = std::array::from_fn(|_| Vec::with_capacity(samples));
    while values.iter().any(|values| values.len() < samples) {
        for arm in ORDER {
            let index = arm.index();
            if values[index].len() == samples {
                continue;
            }
            let (elapsed, result, timestamp) =
                time_arm(runtime, fixture, Some(&selection), None, arm)?;
            if result.row_count != expected_rows {
                return Err(format!(
                    "{cell}/{} timed row mismatch: expected {expected_rows}, got {}",
                    arm.label(),
                    result.row_count
                ));
            }
            values[index].push(elapsed);
            timestamps[index].push(timestamp);
        }
    }

    let context = fixture.context();
    Ok(PcArm::ALL
        .into_iter()
        .map(|arm| {
            let index = arm.index();
            CsvRow {
                group: if context_role == "homogeneous" {
                    "PC-HOM"
                } else if context_role == "holdout" {
                    "PC-HOLDOUT"
                } else {
                    "PC-MIX"
                },
                cell_id: cell.to_string(),
                context,
                context_role,
                shape_name: shape.name.clone(),
                nominal_skip: shape.nominal_skip,
                nominal_select: shape.nominal_select,
                summary: shape.summary(),
                selection_source: "external",
                row_groups: ORACLE_ROW_GROUPS,
                measurement: measurement(
                    arm,
                    std::mem::take(&mut values[index]),
                    std::mem::take(&mut timestamps[index]),
                    &checks[index],
                ),
            }
        })
        .collect())
}

fn measure_fast_controls(
    runtime: &Runtime,
    options: &Options,
    fixture: &OracleFixture,
    context_role: &'static str,
    samples: usize,
) -> Result<Vec<CsvRow>, String> {
    let shape = OracleShape::all_selected();
    let mut rows = Vec::new();
    for (cell, row_groups) in [
        ("PC-FAST/PC-M2/all_selected_4rg", None),
        ("PC-FAST/PC-M2/all_selected_1rg", Some(vec![0usize])),
    ] {
        if !matches_filter(options, cell) {
            continue;
        }
        eprintln!("measuring {cell}");
        let expected_rows = row_groups
            .as_ref()
            .map_or(ORACLE_ROW_GROUPS * ROWS_PER_GROUP, |groups| {
                groups.len() * ROWS_PER_GROUP
            });
        let row_group_slice = row_groups.as_deref();
        let checks = check_arms(
            runtime,
            fixture,
            None,
            row_group_slice,
            &PcArm::FAST,
            expected_rows,
            cell,
        )?;
        warm_arms(
            runtime,
            fixture,
            None,
            row_group_slice,
            &PcArm::FAST,
            expected_rows,
            cell,
        )?;

        let mut values: [Vec<u64>; 2] = std::array::from_fn(|_| Vec::with_capacity(samples));
        let mut timestamps: [Vec<u64>; 2] = std::array::from_fn(|_| Vec::with_capacity(samples));
        const ORDER: [PcArm; 4] = [
            PcArm::Auto32,
            PcArm::PerColumn,
            PcArm::PerColumn,
            PcArm::Auto32,
        ];
        while values.iter().any(|values| values.len() < samples) {
            for arm in ORDER {
                let index = usize::from(arm == PcArm::PerColumn);
                if values[index].len() == samples {
                    continue;
                }
                let (elapsed, result, timestamp) =
                    time_arm(runtime, fixture, None, row_group_slice, arm)?;
                if result.row_count != expected_rows {
                    return Err(format!(
                        "{cell}/{} timed row mismatch: expected {expected_rows}, got {}",
                        arm.label(),
                        result.row_count
                    ));
                }
                values[index].push(elapsed);
                timestamps[index].push(timestamp);
            }
        }

        let context = fixture.context();
        for (index, arm) in PcArm::FAST.into_iter().enumerate() {
            rows.push(CsvRow {
                group: "PC-FAST",
                cell_id: cell.to_string(),
                context,
                context_role,
                shape_name: cell.rsplit('/').next().unwrap().to_string(),
                nominal_skip: None,
                nominal_select: None,
                summary: shape.summary(),
                selection_source: "none",
                row_groups: row_groups.as_ref().map_or(ORACLE_ROW_GROUPS, Vec::len),
                measurement: measurement(
                    arm,
                    std::mem::take(&mut values[index]),
                    std::mem::take(&mut timestamps[index]),
                    &checks[index],
                ),
            });
        }
    }
    Ok(rows)
}

fn check_arms(
    runtime: &Runtime,
    fixture: &OracleFixture,
    selection: Option<&RowSelection>,
    row_groups: Option<&[usize]>,
    arms: &[PcArm],
    expected_rows: usize,
    cell: &str,
) -> Result<Vec<PcRunResult>, String> {
    let mut checks = Vec::with_capacity(arms.len());
    for &arm in arms {
        let result =
            runtime.block_on(run_pc(fixture, selection.cloned(), row_groups, arm, true))?;
        if result.row_count != expected_rows {
            return Err(format!(
                "{cell}/{} correctness row mismatch: expected {expected_rows}, got {}",
                arm.label(),
                result.row_count
            ));
        }
        if result.content.is_none() {
            return Err(format!("{cell}/{} has no content digest", arm.label()));
        }
        if let Some(first) = checks.first()
            && first != &result
        {
            return Err(format!(
                "{cell}: {} differs from {} in row count or full projected content",
                arm.label(),
                arms[0].label()
            ));
        }
        checks.push(result);
    }
    Ok(checks)
}

fn warm_arms(
    runtime: &Runtime,
    fixture: &OracleFixture,
    selection: Option<&RowSelection>,
    row_groups: Option<&[usize]>,
    arms: &[PcArm],
    expected_rows: usize,
    cell: &str,
) -> Result<(), String> {
    for _ in 0..WARMUPS_PER_ARM {
        for &arm in arms {
            let result =
                runtime.block_on(run_pc(fixture, selection.cloned(), row_groups, arm, false))?;
            if result.row_count != expected_rows {
                return Err(format!(
                    "{cell}/{} warmup row mismatch: expected {expected_rows}, got {}",
                    arm.label(),
                    result.row_count
                ));
            }
        }
    }
    Ok(())
}

fn time_arm(
    runtime: &Runtime,
    fixture: &OracleFixture,
    selection: Option<&RowSelection>,
    row_groups: Option<&[usize]>,
    arm: PcArm,
) -> Result<(u64, PcRunResult, u64), String> {
    // Deep selection materialisation stays outside the timed scan, matching
    // the established oracle protocol.
    let selection = selection.cloned();
    let timestamp = unix_nanos();
    let started = Instant::now();
    let result = runtime.block_on(run_pc(fixture, selection, row_groups, arm, false))?;
    let elapsed = started.elapsed().as_nanos() as u64;
    hint::black_box(result.row_count);
    Ok((elapsed, result, timestamp))
}

fn measurement(
    arm: PcArm,
    samples_ns: Vec<u64>,
    sample_started_unix_ns: Vec<u64>,
    check: &PcRunResult,
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
        content: check.content.clone().unwrap(),
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

fn write_csv(path: &Path, rows: &[CsvRow], schema_version: &str) -> Result<(), String> {
    let file = File::create(path)
        .map_err(|error| format!("cannot create PC-1 CSV {}: {error}", path.display()))?;
    let mut writer = BufWriter::new(file);
    writeln!(
        writer,
        "schema_version,group,cell_id,context_id,context_role,output_layout,payload_spec,payload_columns,encoding,compression,page_index,batch_size,rows_per_group,row_groups,shape_name,skip_rows,select_rows,selected_fraction,avg_run_len,run_count,long_skip_share_1024,long_skip_share_4096,selection_source,arm,column_strategies,sample_count,samples_ns,sample_started_unix_ns,median_ns,mad_ns,rows_out,schema_sha256,leaf_sha256"
    )
    .map_err(|error| format!("cannot write PC-1 CSV header: {error}"))?;
    for row in rows {
        let measurement = &row.measurement;
        let fields = vec![
            schema_version.to_string(),
            row.group.to_string(),
            row.cell_id.clone(),
            row.context.id.to_string(),
            row.context_role.to_string(),
            row.context.output_layout().to_string(),
            payload_spec(row.context),
            row.context.payload_columns.to_string(),
            row.context.encoding(),
            row.context.compression.label().to_string(),
            row.context.page_index.to_string(),
            row.context.batch_size.to_string(),
            ROWS_PER_GROUP.to_string(),
            row.row_groups.to_string(),
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
            row.selection_source.to_string(),
            measurement.arm.label().to_string(),
            column_strategies(row),
            measurement.samples_ns.len().to_string(),
            join_numbers(&measurement.samples_ns),
            join_numbers(&measurement.sample_started_unix_ns),
            measurement.median_ns.to_string(),
            measurement.mad_ns.to_string(),
            measurement.rows_out.to_string(),
            measurement.content.schema_sha256.clone(),
            measurement.content.leaf_sha256.join("|"),
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
        .map_err(|error| format!("cannot write PC-1 CSV row: {error}"))?;
    }
    writer
        .flush()
        .map_err(|error| format!("cannot flush PC-1 CSV {}: {error}", path.display()))
}

fn payload_spec(context: OracleContext) -> String {
    (0..context.payload_columns)
        .map(|column_idx| payload_label(context.payload_at(column_idx)))
        .collect::<Vec<_>>()
        .join("|")
}

fn payload_label(payload: OraclePayload) -> String {
    match payload {
        OraclePayload::Utf8Dictionary1k => "dict-c1024-w5".to_string(),
        OraclePayload::Utf8Dictionary {
            cardinality,
            value_width,
            fallback_plain_percent,
        } => fallback_plain_percent.map_or_else(
            || format!("dict-c{cardinality}-w{value_width}"),
            |percent| format!("dict-fallback-p{percent}-w{value_width}"),
        ),
        _ => payload.dtype().to_string(),
    }
}

fn column_strategies(row: &CsvRow) -> String {
    if row.selection_source == "none" {
        return if row.measurement.arm == PcArm::PerColumn {
            "fallback-auto32".to_string()
        } else {
            "all-selected-fast-path".to_string()
        };
    }
    let fixed = match row.measurement.arm {
        PcArm::Selectors => Some("selectors"),
        PcArm::Mask => Some("mask"),
        PcArm::Auto32 => Some(if row.summary.avg_run_len < 32.0 {
            "mask"
        } else {
            "selectors"
        }),
        PcArm::PerColumn => None,
    };
    (0..row.context.payload_columns)
        .map(|column_idx| {
            let strategy = fixed.unwrap_or_else(|| {
                let threshold = if pure_dictionary(row.context.payload_at(column_idx)) {
                    PURE_DICTIONARY_THRESHOLD
                } else {
                    DEFAULT_COLUMN_THRESHOLD
                };
                if row.summary.avg_run_len < threshold as f64 {
                    "mask"
                } else {
                    "selectors"
                }
            });
            format!("{column_idx}:{strategy}")
        })
        .collect::<Vec<_>>()
        .join("|")
}

const fn pure_dictionary(payload: OraclePayload) -> bool {
    matches!(
        payload,
        OraclePayload::Utf8Dictionary1k
            | OraclePayload::Utf8Dictionary {
                fallback_plain_percent: None,
                ..
            }
    )
}

fn write_phase_d_manifest(
    options: &Options,
    rows: &[CsvRow],
    fixture_sha256: &BTreeMap<String, String>,
    started_unix_ns: u64,
    completed_unix_ns: u64,
    elapsed_ns: u64,
) -> Result<(), String> {
    let rustc = command_output("rustc", &["-vV"]);
    let git_sha = command_output(
        "git",
        &["-C", env!("CARGO_MANIFEST_DIR"), "rev-parse", "HEAD"],
    );
    let git_status = command_output(
        "git",
        &[
            "-C",
            env!("CARGO_MANIFEST_DIR"),
            "status",
            "--short",
            "--untracked-files=no",
        ],
    );
    let cpu_model = cpu_model();
    let os = command_output("uname", &["-srmv"]);
    let hostname = fs::read_to_string("/etc/hostname")
        .unwrap_or_default()
        .trim()
        .to_string();
    let mut environment_hasher = DefaultHasher::new();
    (&rustc, &cpu_model, &os).hash(&mut environment_hasher);
    let context_ids = rows
        .iter()
        .map(|row| row.context.id)
        .collect::<BTreeSet<_>>();
    let cells = rows
        .iter()
        .map(|row| row.cell_id.as_str())
        .collect::<BTreeSet<_>>();
    let manifest = json!({
        "schema_version": PC1D_MANIFEST_SCHEMA_VERSION,
        "benchmark": "arrow_reader_row_selection_pc1d_rg16",
        "csv_schema_version": PC1D_CSV_SCHEMA_VERSION,
        "git_sha": git_sha,
        "git_status_porcelain": git_status,
        "rustc": rustc,
        "cpu_model": cpu_model,
        "hostname": hostname,
        "os": os,
        "environment_fingerprint": format!("{:016x}", environment_hasher.finish()),
        "started_unix_ns": started_unix_ns,
        "completed_unix_ns": completed_unix_ns,
        "elapsed_ns": elapsed_ns,
        "samples_per_arm": options.samples,
        "warmups_per_arm": WARMUPS_PER_ARM,
        "filter": options.filter_text.as_deref(),
        "cell_count": cells.len(),
        "arm_row_count": rows.len(),
        "measured_context_ids": context_ids,
        "declared_matrix": {
            "homogeneous_contexts": 44,
            "shapes": [
                "f50_l4", "f50_l16", "f50_l64", "f50_l256", "f50_l1024",
                "f02_l64", "f98_l64", "bursty03_l4"
            ],
            "arms": ["auto32", "percolumn"]
        },
        "fixture": {
            "row_groups": PC1D_ROW_GROUPS,
            "rows_per_group": ROWS_PER_GROUP,
            "default_batch_size": BATCH_SIZE,
            "page_row_limit": ORACLE_PAGE_ROWS,
            "in_memory": true,
            "metadata_preparsed": true,
            "sha256_by_context": fixture_sha256
        },
        "timing_protocol": {
            "order": "auto32,percolumn,percolumn,auto32 repeated",
            "statistic": "median",
            "dispersion": "median_absolute_deviation",
            "clock": "std::time::Instant",
            "sample_start_clock": "unix_epoch_nanoseconds",
            "selection_clone": "outside timed region"
        },
        "correctness": {
            "hard_gate": "row count, schema digest, and every projected leaf digest equal across both arms",
            "digest": "arrow-projected-leaf-content-v1"
        },
        "noharm_gate": {
            "requirement": "percolumn >= auto32 - max(2%, 3*MAD) per cell",
            "max_regression_percent": 2.0,
            "mad_multiplier": 3.0
        },
        "csv_sha256": sha256_path(&options.csv)?,
        "classification": "non-formal"
    });
    let file = File::create(&options.manifest).map_err(|error| {
        format!(
            "cannot create PC-1d manifest {}: {error}",
            options.manifest.display()
        )
    })?;
    serde_json::to_writer_pretty(file, &manifest)
        .map_err(|error| format!("cannot write PC-1d manifest: {error}"))?;
    let mut file = fs::OpenOptions::new()
        .append(true)
        .open(&options.manifest)
        .map_err(|error| format!("cannot reopen PC-1d manifest: {error}"))?;
    writeln!(file).map_err(|error| format!("cannot finish PC-1d manifest: {error}"))
}

fn write_manifest(
    options: &Options,
    rows: &[CsvRow],
    fixture_sha256: &BTreeMap<String, String>,
    started_unix_ns: u64,
    completed_unix_ns: u64,
    elapsed_ns: u64,
) -> Result<(), String> {
    let rustc = command_output("rustc", &["-vV"]);
    let git_sha = command_output(
        "git",
        &["-C", env!("CARGO_MANIFEST_DIR"), "rev-parse", "HEAD"],
    );
    let git_status = command_output(
        "git",
        &[
            "-C",
            env!("CARGO_MANIFEST_DIR"),
            "status",
            "--short",
            "--untracked-files=no",
        ],
    );
    let cpu_model = cpu_model();
    let os = command_output("uname", &["-srmv"]);
    let hostname = fs::read_to_string("/etc/hostname")
        .unwrap_or_default()
        .trim()
        .to_string();
    let mut environment_hasher = DefaultHasher::new();
    (&rustc, &cpu_model, &os).hash(&mut environment_hasher);
    let context_ids = rows
        .iter()
        .map(|row| row.context.id)
        .collect::<BTreeSet<_>>();
    let cells = rows
        .iter()
        .map(|row| row.cell_id.as_str())
        .collect::<BTreeSet<_>>();
    let manifest = json!({
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "benchmark": "arrow_reader_row_selection_pc1",
        "csv_schema_version": CSV_SCHEMA_VERSION,
        "git_sha": git_sha,
        "git_status_porcelain": git_status,
        "rustc": rustc,
        "cpu_model": cpu_model,
        "hostname": hostname,
        "os": os,
        "environment_fingerprint": format!("{:016x}", environment_hasher.finish()),
        "started_unix_ns": started_unix_ns,
        "completed_unix_ns": completed_unix_ns,
        "elapsed_ns": elapsed_ns,
        "samples_per_arm": options.samples,
        "warmups_per_arm": WARMUPS_PER_ARM,
        "filter": options.filter_text.as_deref(),
        "cell_count": cells.len(),
        "arm_row_count": rows.len(),
        "measured_context_ids": context_ids,
        "declared_matrix": {
            "homogeneous_contexts": 44,
            "mixed_contexts": 5,
            "holdout_contexts": 1,
            "shapes": [
                "f50_l4", "f50_l16", "f50_l64", "f50_l256", "f50_l1024",
                "f02_l64", "f98_l64", "bursty03_l4"
            ],
            "arms": ["auto32", "selectors", "mask", "percolumn"],
            "fast_controls": ["all_selected_4rg", "all_selected_1rg"]
        },
        "fixture": {
            "row_groups": ORACLE_ROW_GROUPS,
            "rows_per_group": ROWS_PER_GROUP,
            "default_batch_size": BATCH_SIZE,
            "page_row_limit": ORACLE_PAGE_ROWS,
            "in_memory": true,
            "metadata_preparsed": true,
            "sha256_by_context": fixture_sha256
        },
        "percolumn_contract": {
            "scope": "flat output projection only; predicates and unsupported shapes fall back to Auto32",
            "physical_window": "one shared MaskCursor window",
            "pure_dictionary_threshold": PURE_DICTIONARY_THRESHOLD,
            "other_column_threshold": DEFAULT_COLUMN_THRESHOLD,
            "choice": "mask iff average selector run length is strictly below the column threshold"
        },
        "timing_protocol": {
            "four_arm_order": "auto32,selectors,mask,percolumn,percolumn,mask,selectors,auto32 repeated",
            "fast_control_order": "auto32,percolumn,percolumn,auto32 repeated",
            "statistic": "median",
            "dispersion": "median_absolute_deviation",
            "clock": "std::time::Instant",
            "sample_start_clock": "unix_epoch_nanoseconds",
            "selection_clone": "outside timed region"
        },
        "correctness": {
            "hard_gate": "row count, schema digest, and every projected leaf digest equal across all arms",
            "digest": "arrow-projected-leaf-content-v1"
        },
        "csv_sha256": sha256_path(&options.csv)?,
        "classification": "non-formal"
    });
    let file = File::create(&options.manifest).map_err(|error| {
        format!(
            "cannot create PC-1 manifest {}: {error}",
            options.manifest.display()
        )
    })?;
    serde_json::to_writer_pretty(file, &manifest)
        .map_err(|error| format!("cannot write PC-1 manifest: {error}"))?;
    let mut file = fs::OpenOptions::new()
        .append(true)
        .open(&options.manifest)
        .map_err(|error| format!("cannot reopen PC-1 manifest: {error}"))?;
    writeln!(file).map_err(|error| format!("cannot finish PC-1 manifest: {error}"))
}

fn sha256_path(path: &Path) -> Result<String, String> {
    let bytes = fs::read(path)
        .map_err(|error| format!("cannot hash artifact {}: {error}", path.display()))?;
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    Ok(hex_digest(&hasher.finalize()))
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

fn emit_phase_d_artifact(kind: &str, path: &Path) -> Result<(), String> {
    let raw = fs::read_to_string(path).map_err(|error| {
        format!(
            "cannot read PC-1d {} for log embedding: {error}",
            path.display()
        )
    })?;
    println!("DFEXP_PC1D_RG16_{kind}_BEGIN");
    print!("{raw}");
    if !raw.ends_with('\n') {
        println!();
    }
    println!("DFEXP_PC1D_RG16_{kind}_END");
    Ok(())
}

fn default_artifact_path(filename: &str) -> PathBuf {
    env::var_os("CARGO_TARGET_DIR")
        .map(PathBuf::from)
        .map(|target_dir| target_dir.join(filename))
        .unwrap_or_else(|| PathBuf::from(filename))
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
