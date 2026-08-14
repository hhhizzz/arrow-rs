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

//! PC-1c/PC-2 physical-scale scan, attribution, and single-cell perf workload.
//!
//! The scale matrix changes row-group count independently from physical rows
//! per group.  The profile mode holds the established 4 x 65,536 fixture and
//! repeats exactly one arm of one preregistered cell so `dfexp plan-profile`
//! can collect enough samples without changing the scan path.
//! PC-2 reuses those controls with same-SHA legacy, forced-thin, and product
//! policies that are available only to `test_common` benchmark builds.

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
use parquet::arrow::arrow_reader::metrics::{ArrowReaderDecompositionMetrics, ArrowReaderMetrics};
use parquet::arrow::arrow_reader::{RowSelection, RowSelectionPolicy, RowSelector};
use parquet::arrow::{ParquetRecordBatchStreamBuilder, ProjectionMask};
use regex::Regex;
use serde_json::json;
use sha2::{Digest, Sha256};
use tokio::runtime::Runtime;

use super::fixture::{
    ORACLE_CONTEXTS, ORACLE_PAGE_ROWS, OracleContext, OracleFixture, OraclePayload,
    PC_MIXED_CONTEXTS, TT_CONTEXTS, build_oracle_fixture_with_dimensions,
};
use super::model::BATCH_SIZE;
use super::runner::{ProjectedContentDigest, ProjectedContentDigester};

const CSV_SCHEMA_VERSION: &str = "arrow-row-selection-pc1c-scale-v1";
const MANIFEST_SCHEMA_VERSION: &str = "arrow-row-selection-pc1c-scale-manifest-v1";
const ATTR_CSV_SCHEMA_VERSION: &str = "arrow-row-selection-pc1c-attribution-v1";
const ATTR_MANIFEST_SCHEMA_VERSION: &str = "arrow-row-selection-pc1c-attribution-manifest-v1";
const PC2_CSV_SCHEMA_VERSION: &str = "arrow-row-selection-pc2-scale-v1";
const PC2_MANIFEST_SCHEMA_VERSION: &str = "arrow-row-selection-pc2-scale-manifest-v1";
const PC2_ATTR_CSV_SCHEMA_VERSION: &str = "arrow-row-selection-pc2-attribution-v1";
const PC2_ATTR_MANIFEST_SCHEMA_VERSION: &str = "arrow-row-selection-pc2-attribution-manifest-v1";
const DEFAULT_SAMPLES: usize = 12;
const WARMUPS_PER_ARM: usize = 2;
const PROFILE_ROW_GROUPS: usize = 4;
const PROFILE_ROWS_PER_GROUP: usize = 65_536;
const PURE_DICTIONARY_THRESHOLD: usize = 4;
const DEFAULT_COLUMN_THRESHOLD: usize = 16;
const PC2_DEFAULT_COLUMN_THRESHOLD: usize = 32;
const SCALE_CONTEXT_IDS: &[&str] = &[
    "C0",
    "C6",
    "TT-D-C1024-W8-P8",
    "TT-V-U32-C8",
    "PC-M2",
    "PC-M6",
];
const SCALE_DIMENSIONS: &[(usize, usize)] = &[(1, 65_536), (4, 65_536), (16, 65_536), (4, 16_384)];
const SCALE_RUN_LENGTHS: &[usize] = &[64, 4];
const PC2_SCALE_DIMENSIONS: &[(usize, usize)] =
    &[(1, 16_384), (1, 65_536), (16, 16_384), (16, 65_536)];
const PC2_SCALE_RUN_LENGTHS: &[usize] = &[64];

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum Pc1cArm {
    Auto32,
    PerColumn,
    #[cfg(feature = "test_common")]
    Legacy,
    #[cfg(feature = "test_common")]
    Thin,
    #[cfg(feature = "test_common")]
    Product,
}

impl Pc1cArm {
    const ALL: [Self; 2] = [Self::Auto32, Self::PerColumn];
    #[cfg(feature = "test_common")]
    const PC2_SCALE: [Self; 3] = [Self::Auto32, Self::Legacy, Self::Thin];

    const fn label(self) -> &'static str {
        match self {
            Self::Auto32 => "auto32",
            Self::PerColumn => "percolumn",
            #[cfg(feature = "test_common")]
            Self::Legacy => "pc-bolton",
            #[cfg(feature = "test_common")]
            Self::Thin => "pc2-thin",
            #[cfg(feature = "test_common")]
            Self::Product => "pc2",
        }
    }

    const fn policy(self) -> RowSelectionPolicy {
        match self {
            Self::Auto32 => RowSelectionPolicy::Auto { threshold: 32 },
            Self::PerColumn => {
                #[cfg(feature = "test_common")]
                {
                    RowSelectionPolicy::PerColumnLegacy
                }
                #[cfg(not(feature = "test_common"))]
                {
                    RowSelectionPolicy::PerColumn
                }
            }
            #[cfg(feature = "test_common")]
            Self::Product => RowSelectionPolicy::PerColumn,
            #[cfg(feature = "test_common")]
            Self::Legacy => RowSelectionPolicy::PerColumnLegacy,
            #[cfg(feature = "test_common")]
            Self::Thin => RowSelectionPolicy::PerColumnForcedThin,
        }
    }

    fn parse(value: &str, pc2: bool) -> Result<Self, String> {
        match (pc2, value) {
            (false, "auto32") | (true, "auto32") => Ok(Self::Auto32),
            (false, "percolumn") => Ok(Self::PerColumn),
            #[cfg(feature = "test_common")]
            (true, "pc-bolton") => Ok(Self::Legacy),
            #[cfg(feature = "test_common")]
            (true, "pc2-thin") => Ok(Self::Thin),
            #[cfg(feature = "test_common")]
            (true, "pc2") => Ok(Self::Product),
            _ => Err(format!(
                "unsupported --arm {value:?}; expected {}",
                if pc2 {
                    "auto32, pc-bolton, pc2-thin, or pc2"
                } else {
                    "auto32 or percolumn"
                }
            )),
        }
    }
}

#[derive(Debug)]
struct ScaleOptions {
    pc2: bool,
    list: bool,
    samples: usize,
    filter: Option<Regex>,
    filter_text: Option<String>,
    csv: PathBuf,
    manifest: PathBuf,
    emit_artifacts: bool,
}

#[derive(Clone, Copy, Debug)]
struct ProfileCase {
    id: &'static str,
    context_id: &'static str,
    run_len: usize,
}

impl ProfileCase {
    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "c6-l64" => Ok(Self {
                id: "c6-l64",
                context_id: "C6",
                run_len: 64,
            }),
            "c0-l64" => Ok(Self {
                id: "c0-l64",
                context_id: "C0",
                run_len: 64,
            }),
            "c0-l4" => Ok(Self {
                id: "c0-l4",
                context_id: "C0",
                run_len: 4,
            }),
            _ => Err(format!(
                "unsupported --case {value:?}; expected c6-l64, c0-l64, or c0-l4"
            )),
        }
    }
}

#[derive(Debug)]
struct ProfileOptions {
    pc2: bool,
    case: ProfileCase,
    arm: Pc1cArm,
    iterations: usize,
}

#[derive(Debug)]
struct AttributionOptions {
    pc2: bool,
    samples: usize,
    csv: PathBuf,
    manifest: PathBuf,
    emit_artifacts: bool,
}

#[derive(Clone, Copy, Debug)]
struct AttributionCondition {
    arm: Pc1cArm,
    enabled: bool,
}

impl AttributionCondition {
    fn index(self, comparison_arm: Pc1cArm) -> usize {
        let arm_offset = if self.arm == Pc1cArm::Auto32 {
            0
        } else {
            assert_eq!(self.arm, comparison_arm);
            2
        };
        arm_offset + self.enabled as usize
    }
}

#[derive(Clone, Debug)]
struct AttributionSample {
    case: ProfileCase,
    context: OracleContext,
    run_len: usize,
    condition: AttributionCondition,
    sample_index: usize,
    started_unix_ns: u64,
    wall_ns: u64,
    metrics: Option<ArrowReaderDecompositionMetrics>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ScanResult {
    row_count: usize,
    content: Option<ProjectedContentDigest>,
}

#[derive(Clone, Debug)]
struct ArmMeasurement {
    arm: Pc1cArm,
    samples_ns: Vec<u64>,
    sample_started_unix_ns: Vec<u64>,
    median_ns: u64,
    mad_ns: u64,
    rows_out: usize,
    content: ProjectedContentDigest,
}

#[derive(Clone, Debug)]
struct ScaleShape {
    name: String,
    run_len: usize,
    rows_per_group: usize,
    row_groups: usize,
    selection: RowSelection,
    selected_rows_per_group: usize,
    runs_per_group: usize,
}

impl ScaleShape {
    fn f50(run_len: usize, row_groups: usize, rows_per_group: usize) -> Self {
        assert!(run_len > 0);
        assert_eq!(rows_per_group % (2 * run_len), 0);
        let mut one_group = Vec::with_capacity(rows_per_group / run_len);
        for _ in 0..rows_per_group / (2 * run_len) {
            one_group.push(RowSelector::skip(run_len));
            one_group.push(RowSelector::select(run_len));
        }
        let runs_per_group = one_group.len();
        let selection = RowSelection::from(
            (0..row_groups)
                .flat_map(|_| one_group.iter().copied())
                .collect::<Vec<_>>(),
        );
        Self {
            name: format!("f50_l{run_len}"),
            run_len,
            rows_per_group,
            row_groups,
            selection,
            selected_rows_per_group: rows_per_group / 2,
            runs_per_group,
        }
    }

    fn total_rows(&self) -> usize {
        self.row_groups * self.rows_per_group
    }

    fn selected_rows(&self) -> usize {
        self.row_groups * self.selected_rows_per_group
    }

    fn run_count(&self) -> usize {
        self.row_groups * self.runs_per_group
    }
}

#[derive(Clone, Debug)]
struct CsvRow {
    cell_id: String,
    context: OracleContext,
    context_role: &'static str,
    shape: ScaleShape,
    measurement: ArmMeasurement,
}

pub(crate) fn main() {
    let result = if env::args()
        .any(|argument| matches!(argument.as_str(), "--pc1c-profile" | "--pc2-profile"))
    {
        run_profile()
    } else if env::args().any(|argument| matches!(argument.as_str(), "--pc1c-attr" | "--pc2-attr"))
    {
        run_attribution()
    } else {
        run_scale()
    };
    if let Err(error) = result {
        eprintln!("PC-1c evaluation failed: {error}");
        std::process::exit(2);
    }
}

fn run_scale() -> Result<(), String> {
    let options = parse_scale_options()?;
    let dimensions = scale_dimensions(options.pc2);
    let run_lengths = scale_run_lengths(options.pc2);
    if options.list {
        let mut count = 0usize;
        for context_id in SCALE_CONTEXT_IDS {
            for &(row_groups, rows_per_group) in dimensions {
                for &run_len in run_lengths {
                    let cell =
                        scale_cell_id(options.pc2, context_id, row_groups, rows_per_group, run_len);
                    if matches_filter(&options, &cell) {
                        println!("{cell}");
                        count += 1;
                    }
                }
            }
        }
        eprintln!(
            "listed {count} {} scale cells",
            if options.pc2 { "PC-2" } else { "PC-1c" }
        );
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

    for context_id in SCALE_CONTEXT_IDS {
        let context = context_by_id(context_id)?;
        for &(row_groups, rows_per_group) in dimensions {
            let selected_run_lengths = run_lengths
                .iter()
                .copied()
                .filter(|run_len| {
                    matches_filter(
                        &options,
                        &scale_cell_id(
                            options.pc2,
                            context_id,
                            row_groups,
                            rows_per_group,
                            *run_len,
                        ),
                    )
                })
                .collect::<Vec<_>>();
            if selected_run_lengths.is_empty() {
                continue;
            }
            eprintln!(
                "building {} scale fixture {context_id}/rg{row_groups}-r{rows_per_group}",
                if options.pc2 { "PC-2" } else { "PC-1c" }
            );
            let fixture = build_oracle_fixture_with_dimensions(
                context,
                None,
                row_groups,
                rows_per_group,
            )
            .map_err(|error| {
                format!(
                    "cannot build scale fixture {context_id}/rg{row_groups}-r{rows_per_group}: {error}"
                )
            })?;
            fixture_sha256.insert(
                format!("{context_id}/rg{row_groups}-r{rows_per_group}"),
                fixture.bytes_sha256(),
            );
            for run_len in selected_run_lengths {
                let shape = ScaleShape::f50(run_len, row_groups, rows_per_group);
                let cell =
                    scale_cell_id(options.pc2, context_id, row_groups, rows_per_group, run_len);
                eprintln!("measuring {cell}");
                rows.extend(measure_scale_cell(
                    &runtime,
                    &fixture,
                    context_role(context_id, options.pc2),
                    shape,
                    cell,
                    options.samples,
                    scale_arms(options.pc2),
                )?);
            }
        }
    }
    if rows.is_empty() {
        return Err(format!(
            "cell filter selected no {} scale cells",
            if options.pc2 { "PC-2" } else { "PC-1c" }
        ));
    }

    let completed_unix_ns = unix_nanos();
    write_csv(&options.csv, &rows)?;
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
    let output_prefix = if options.pc2 {
        "DFEXP_PC2_SCALE"
    } else {
        "DFEXP_PC1C_SCALE"
    };
    println!("{output_prefix}_CELLS={cells}");
    println!("{output_prefix}_ROWS={}", rows.len());
    if options.emit_artifacts {
        emit_artifact(options.pc2, "CSV", &options.csv)?;
        emit_artifact(options.pc2, "MANIFEST", &options.manifest)?;
    }
    Ok(())
}

fn run_profile() -> Result<(), String> {
    let options = parse_profile_options()?;
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|error| format!("cannot build Tokio runtime: {error}"))?;
    let context = context_by_id(options.case.context_id)?;
    let fixture = build_oracle_fixture_with_dimensions(
        context,
        None,
        PROFILE_ROW_GROUPS,
        PROFILE_ROWS_PER_GROUP,
    )
    .map_err(|error| format!("cannot build PC-1c profile fixture: {error}"))?;
    let shape = ScaleShape::f50(
        options.case.run_len,
        PROFILE_ROW_GROUPS,
        PROFILE_ROWS_PER_GROUP,
    );
    let check_arms = if !options.pc2 {
        Pc1cArm::ALL.to_vec()
    } else if options.arm == Pc1cArm::Auto32 {
        vec![Pc1cArm::Auto32]
    } else {
        vec![Pc1cArm::Auto32, options.arm]
    };
    let checks = check_arms
        .into_iter()
        .map(|arm| runtime.block_on(run_scan(&fixture, shape.selection.clone(), arm, true)))
        .collect::<Result<Vec<_>, _>>()?;
    if checks.iter().any(|check| check != &checks[0])
        || checks[0].row_count != shape.selected_rows()
    {
        return Err(format!(
            "{} profile arms failed full-content correctness",
            if options.pc2 { "PC-2" } else { "PC-1c" }
        ));
    }
    for _ in 0..WARMUPS_PER_ARM {
        let result = runtime.block_on(run_scan(
            &fixture,
            shape.selection.clone(),
            options.arm,
            false,
        ))?;
        if result.row_count != shape.selected_rows() {
            return Err("profile warmup row count drifted".to_string());
        }
    }

    let started = Instant::now();
    for _ in 0..options.iterations {
        let result = runtime.block_on(run_scan(
            &fixture,
            shape.selection.clone(),
            options.arm,
            false,
        ))?;
        if result.row_count != shape.selected_rows() {
            return Err("profile timed row count drifted".to_string());
        }
        hint::black_box(result.row_count);
    }
    let output_prefix = if options.pc2 {
        "DFEXP_PC2_PROFILE"
    } else {
        "DFEXP_PC1C_PROFILE"
    };
    println!("{output_prefix}_CASE={}", options.case.id);
    println!("{output_prefix}_ARM={}", options.arm.label());
    println!("{output_prefix}_ITERATIONS={}", options.iterations);
    println!(
        "{output_prefix}_WORKLOAD_NS={}",
        started.elapsed().as_nanos()
    );
    Ok(())
}

fn run_attribution() -> Result<(), String> {
    const CASES: [&str; 3] = ["c6-l64", "c0-l64", "c0-l4"];
    let options = parse_attribution_options()?;
    let comparison_arm = if options.pc2 {
        #[cfg(feature = "test_common")]
        {
            Pc1cArm::Thin
        }
        #[cfg(not(feature = "test_common"))]
        {
            unreachable!("parse rejects PC-2 attribution without test_common")
        }
    } else {
        Pc1cArm::PerColumn
    };
    let order = [
        AttributionCondition {
            arm: Pc1cArm::Auto32,
            enabled: false,
        },
        AttributionCondition {
            arm: Pc1cArm::Auto32,
            enabled: true,
        },
        AttributionCondition {
            arm: comparison_arm,
            enabled: true,
        },
        AttributionCondition {
            arm: comparison_arm,
            enabled: false,
        },
        AttributionCondition {
            arm: comparison_arm,
            enabled: false,
        },
        AttributionCondition {
            arm: comparison_arm,
            enabled: true,
        },
        AttributionCondition {
            arm: Pc1cArm::Auto32,
            enabled: true,
        },
        AttributionCondition {
            arm: Pc1cArm::Auto32,
            enabled: false,
        },
    ];
    let started_unix_ns = unix_nanos();
    let started = Instant::now();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|error| format!("cannot build Tokio runtime: {error}"))?;
    let mut samples = Vec::new();
    let mut fixture_sha256 = BTreeMap::new();

    for case_id in CASES {
        let case = ProfileCase::parse(case_id)?;
        let context = context_by_id(case.context_id)?;
        let fixture = build_oracle_fixture_with_dimensions(
            context,
            None,
            PROFILE_ROW_GROUPS,
            PROFILE_ROWS_PER_GROUP,
        )
        .map_err(|error| format!("cannot build PC-1c attribution fixture {case_id}: {error}"))?;
        fixture_sha256.insert(case_id.to_string(), fixture.bytes_sha256());
        let shape = ScaleShape::f50(case.run_len, PROFILE_ROW_GROUPS, PROFILE_ROWS_PER_GROUP);

        let checks = [Pc1cArm::Auto32, comparison_arm]
            .into_iter()
            .map(|arm| runtime.block_on(run_scan(&fixture, shape.selection.clone(), arm, true)))
            .collect::<Result<Vec<_>, _>>()?;
        if checks[0] != checks[1] || checks[0].row_count != shape.selected_rows() {
            return Err(format!(
                "{} attribution {case_id} failed full-content correctness",
                if options.pc2 { "PC-2" } else { "PC-1c" }
            ));
        }

        for condition in order[..4].iter().copied() {
            for _ in 0..WARMUPS_PER_ARM {
                let metrics = if condition.enabled {
                    ArrowReaderMetrics::pc1c_attribution()
                } else {
                    ArrowReaderMetrics::disabled()
                };
                let (result, snapshot) = runtime.block_on(run_scan_with_metrics(
                    &fixture,
                    shape.selection.clone(),
                    condition.arm,
                    false,
                    metrics,
                ))?;
                if result.row_count != shape.selected_rows()
                    || snapshot.is_some() != condition.enabled
                {
                    return Err(format!(
                        "attribution {case_id}/{:?} warmup contract drifted",
                        condition
                    ));
                }
            }
        }

        let mut counts = [0usize; 4];
        while counts.iter().any(|count| *count < options.samples) {
            for condition in order {
                let index = condition.index(comparison_arm);
                if counts[index] == options.samples {
                    continue;
                }
                let metrics = if condition.enabled {
                    ArrowReaderMetrics::pc1c_attribution()
                } else {
                    ArrowReaderMetrics::disabled()
                };
                let timestamp = unix_nanos();
                let sample_started = Instant::now();
                let (result, snapshot) = runtime.block_on(run_scan_with_metrics(
                    &fixture,
                    shape.selection.clone(),
                    condition.arm,
                    false,
                    metrics,
                ))?;
                let wall_ns = sample_started.elapsed().as_nanos() as u64;
                if result.row_count != shape.selected_rows()
                    || snapshot.is_some() != condition.enabled
                {
                    return Err(format!(
                        "attribution {case_id}/{:?} sample contract drifted",
                        condition
                    ));
                }
                hint::black_box(result.row_count);
                samples.push(AttributionSample {
                    case,
                    context,
                    run_len: case.run_len,
                    condition,
                    sample_index: counts[index],
                    started_unix_ns: timestamp,
                    wall_ns,
                    metrics: snapshot,
                });
                counts[index] += 1;
            }
        }
    }

    write_attribution_csv(&options.csv, &samples, options.pc2)?;
    write_attribution_manifest(
        &options,
        &samples,
        &fixture_sha256,
        started_unix_ns,
        unix_nanos(),
        started.elapsed().as_nanos() as u64,
    )?;
    let output_prefix = if options.pc2 {
        "DFEXP_PC2_ATTR"
    } else {
        "DFEXP_PC1C_ATTR"
    };
    println!("{output_prefix}_CASES={}", CASES.len());
    println!("{output_prefix}_ROWS={}", samples.len());
    if options.emit_artifacts {
        emit_attribution_artifact(options.pc2, "CSV", &options.csv)?;
        emit_attribution_artifact(options.pc2, "MANIFEST", &options.manifest)?;
    }
    Ok(())
}

fn parse_scale_options() -> Result<ScaleOptions, String> {
    let pc2 = env::args().any(|argument| argument == "--pc2-scale");
    #[cfg(not(feature = "test_common"))]
    if pc2 {
        return Err("--pc2-scale requires the test_common feature".to_string());
    }
    let mut list = false;
    let mut samples = DEFAULT_SAMPLES;
    let mut filter_text = None;
    let mut csv = default_artifact_path(if pc2 {
        "pc2-scale.csv"
    } else {
        "pc1c-scale.csv"
    });
    let mut manifest = default_artifact_path(if pc2 {
        "pc2-scale-manifest.json"
    } else {
        "pc1c-scale-manifest.json"
    });
    let mut emit_artifacts = false;
    let mut args = env::args().skip(1);
    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--selection-oracle" | "--pc1c-scale" | "--pc2-scale" | "--bench" => {}
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
                    "row_selector --selection-oracle --{} \
                     [--list] [--filter REGEX] [--samples EVEN] \
                     [--csv PATH] [--manifest PATH] [--emit-artifacts]",
                    if pc2 { "pc2-scale" } else { "pc1c-scale" }
                );
                std::process::exit(0);
            }
            _ => return Err(format!("unsupported PC-1c scale argument {argument:?}")),
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
    Ok(ScaleOptions {
        pc2,
        list,
        samples,
        filter,
        filter_text,
        csv,
        manifest,
        emit_artifacts,
    })
}

fn parse_profile_options() -> Result<ProfileOptions, String> {
    let pc2 = env::args().any(|argument| argument == "--pc2-profile");
    #[cfg(not(feature = "test_common"))]
    if pc2 {
        return Err("--pc2-profile requires the test_common feature".to_string());
    }
    let mut case = None;
    let mut arm = None;
    let mut iterations = None;
    let mut args = env::args().skip(1);
    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--selection-oracle" | "--pc1c-profile" | "--pc2-profile" | "--bench" => {}
            "--case" => {
                case = Some(ProfileCase::parse(
                    &args
                        .next()
                        .ok_or_else(|| "--case requires a value".to_string())?,
                )?);
            }
            "--arm" => {
                arm = Some(Pc1cArm::parse(
                    &args
                        .next()
                        .ok_or_else(|| "--arm requires a value".to_string())?,
                    pc2,
                )?);
            }
            "--iterations" => {
                iterations = Some(
                    args.next()
                        .ok_or_else(|| "--iterations requires a value".to_string())?
                        .parse::<usize>()
                        .map_err(|_| "--iterations must be an integer".to_string())?,
                );
            }
            "--help" | "-h" => {
                println!(
                    "row_selector --selection-oracle --{} \
                     --case <c6-l64|c0-l64|c0-l4> \
                     --arm <{}> --iterations <1..=100000>",
                    if pc2 { "pc2-profile" } else { "pc1c-profile" },
                    if pc2 {
                        "auto32|pc-bolton|pc2-thin|pc2"
                    } else {
                        "auto32|percolumn"
                    }
                );
                std::process::exit(0);
            }
            _ => return Err(format!("unsupported PC-1c profile argument {argument:?}")),
        }
    }
    let iterations = iterations.ok_or_else(|| "--iterations is required".to_string())?;
    if !(1..=100_000).contains(&iterations) {
        return Err("--iterations must be in 1..=100000".to_string());
    }
    Ok(ProfileOptions {
        pc2,
        case: case.ok_or_else(|| "--case is required".to_string())?,
        arm: arm.ok_or_else(|| "--arm is required".to_string())?,
        iterations,
    })
}

fn parse_attribution_options() -> Result<AttributionOptions, String> {
    let pc2 = env::args().any(|argument| argument == "--pc2-attr");
    #[cfg(not(feature = "test_common"))]
    if pc2 {
        return Err("--pc2-attr requires the test_common feature".to_string());
    }
    let mut samples = DEFAULT_SAMPLES;
    let mut csv = default_artifact_path(if pc2 {
        "pc2-attribution.csv"
    } else {
        "pc1c-attribution.csv"
    });
    let mut manifest = default_artifact_path(if pc2 {
        "pc2-attribution-manifest.json"
    } else {
        "pc1c-attribution-manifest.json"
    });
    let mut emit_artifacts = false;
    let mut args = env::args().skip(1);
    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--selection-oracle" | "--pc1c-attr" | "--pc2-attr" | "--bench" => {}
            "--samples" => {
                samples = args
                    .next()
                    .ok_or_else(|| "--samples requires a value".to_string())?
                    .parse::<usize>()
                    .map_err(|_| "--samples must be an integer".to_string())?;
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
                    "row_selector --selection-oracle --{} \
                     [--samples EVEN] [--csv PATH] [--manifest PATH] \
                     [--emit-artifacts]",
                    if pc2 { "pc2-attr" } else { "pc1c-attr" }
                );
                std::process::exit(0);
            }
            _ => {
                return Err(format!(
                    "unsupported PC-1c attribution argument {argument:?}"
                ));
            }
        }
    }
    if !(4..=100).contains(&samples) || !samples.is_multiple_of(2) {
        return Err("--samples must be an even integer in 4..=100".to_string());
    }
    if csv == manifest {
        return Err("--csv and --manifest must name different paths".to_string());
    }
    Ok(AttributionOptions {
        pc2,
        samples,
        csv,
        manifest,
        emit_artifacts,
    })
}

fn context_by_id(id: &str) -> Result<OracleContext, String> {
    ORACLE_CONTEXTS
        .iter()
        .chain(TT_CONTEXTS)
        .chain(PC_MIXED_CONTEXTS)
        .copied()
        .find(|context| context.id == id)
        .ok_or_else(|| format!("unknown PC-1c context {id:?}"))
}

fn context_role(id: &str, pc2: bool) -> &'static str {
    match (id, pc2) {
        ("PC-M6", true) => "opened-mixed",
        ("PC-M2", _) => "mixed",
        ("PC-M6", false) => "holdout",
        _ => "homogeneous",
    }
}

const fn pc1c_other_threshold() -> usize {
    #[cfg(feature = "test_common")]
    {
        DEFAULT_COLUMN_THRESHOLD
    }
    #[cfg(not(feature = "test_common"))]
    {
        PC2_DEFAULT_COLUMN_THRESHOLD
    }
}

fn scale_dimensions(pc2: bool) -> &'static [(usize, usize)] {
    if pc2 {
        PC2_SCALE_DIMENSIONS
    } else {
        SCALE_DIMENSIONS
    }
}

fn scale_run_lengths(pc2: bool) -> &'static [usize] {
    if pc2 {
        PC2_SCALE_RUN_LENGTHS
    } else {
        SCALE_RUN_LENGTHS
    }
}

fn scale_arms(pc2: bool) -> &'static [Pc1cArm] {
    if pc2 {
        #[cfg(feature = "test_common")]
        {
            &Pc1cArm::PC2_SCALE
        }
        #[cfg(not(feature = "test_common"))]
        {
            unreachable!("parse rejects PC-2 scale without test_common")
        }
    } else {
        &Pc1cArm::ALL
    }
}

fn scale_cell_id(
    pc2: bool,
    context_id: &str,
    row_groups: usize,
    rows_per_group: usize,
    run_len: usize,
) -> String {
    format!(
        "{}-scale/{context_id}/rg{row_groups}-r{rows_per_group}/f50_l{run_len}",
        if pc2 { "PC-2" } else { "PC-1c" }
    )
}

fn matches_filter(options: &ScaleOptions, cell: &str) -> bool {
    options
        .filter
        .as_ref()
        .is_none_or(|filter| filter.is_match(cell))
}

fn measure_scale_cell(
    runtime: &Runtime,
    fixture: &OracleFixture,
    context_role: &'static str,
    shape: ScaleShape,
    cell_id: String,
    samples: usize,
    arms: &[Pc1cArm],
) -> Result<Vec<CsvRow>, String> {
    let checks = arms
        .iter()
        .copied()
        .map(|arm| {
            runtime
                .block_on(run_scan(fixture, shape.selection.clone(), arm, true))
                .map(|result| (arm, result))
        })
        .collect::<Result<BTreeMap<_, _>, _>>()?;
    let baseline = checks
        .get(&Pc1cArm::Auto32)
        .ok_or_else(|| format!("{cell_id} has no auto32 correctness arm"))?;
    if checks.values().any(|check| check != baseline) || baseline.row_count != shape.selected_rows()
    {
        return Err(format!("{cell_id} failed full-content correctness"));
    }

    for _ in 0..WARMUPS_PER_ARM {
        for &arm in arms {
            let result =
                runtime.block_on(run_scan(fixture, shape.selection.clone(), arm, false))?;
            if result.row_count != shape.selected_rows() {
                return Err(format!("{cell_id}/{} warmup row mismatch", arm.label()));
            }
        }
    }

    let order = arms
        .iter()
        .copied()
        .chain(arms.iter().rev().copied())
        .collect::<Vec<_>>();
    let mut values = arms
        .iter()
        .copied()
        .map(|arm| (arm, Vec::with_capacity(samples)))
        .collect::<BTreeMap<_, _>>();
    let mut timestamps = values.clone();
    while values.values().any(|values| values.len() < samples) {
        for &arm in &order {
            if values[&arm].len() == samples {
                continue;
            }
            let selection = shape.selection.clone();
            let timestamp = unix_nanos();
            let started = Instant::now();
            let result = runtime.block_on(run_scan(fixture, selection, arm, false))?;
            let elapsed = started.elapsed().as_nanos() as u64;
            if result.row_count != shape.selected_rows() {
                return Err(format!("{cell_id}/{} timed row mismatch", arm.label()));
            }
            hint::black_box(result.row_count);
            values.get_mut(&arm).unwrap().push(elapsed);
            timestamps.get_mut(&arm).unwrap().push(timestamp);
        }
    }

    let context = fixture.context();
    Ok(arms
        .iter()
        .copied()
        .map(|arm| {
            let samples_ns = values.remove(&arm).unwrap();
            let median_ns = median(&samples_ns);
            let deviations = samples_ns
                .iter()
                .map(|sample| sample.abs_diff(median_ns))
                .collect::<Vec<_>>();
            CsvRow {
                cell_id: cell_id.clone(),
                context,
                context_role,
                shape: shape.clone(),
                measurement: ArmMeasurement {
                    arm,
                    samples_ns,
                    sample_started_unix_ns: timestamps.remove(&arm).unwrap(),
                    median_ns,
                    mad_ns: median(&deviations),
                    rows_out: checks[&arm].row_count,
                    content: checks[&arm].content.clone().unwrap(),
                },
            }
        })
        .collect())
}

async fn run_scan(
    fixture: &OracleFixture,
    selection: RowSelection,
    arm: Pc1cArm,
    attribution: bool,
) -> Result<ScanResult, String> {
    run_scan_with_metrics(
        fixture,
        selection,
        arm,
        attribution,
        ArrowReaderMetrics::disabled(),
    )
    .await
    .map(|(result, _)| result)
}

async fn run_scan_with_metrics(
    fixture: &OracleFixture,
    selection: RowSelection,
    arm: Pc1cArm,
    attribution: bool,
    metrics: ArrowReaderMetrics,
) -> Result<(ScanResult, Option<ArrowReaderDecompositionMetrics>), String> {
    let context = fixture.context();
    let projection = ProjectionMask::roots(fixture.schema_descr(), 0..context.payload_columns);
    let mut stream = ParquetRecordBatchStreamBuilder::new(fixture.reader())
        .await
        .map_err(|error| format!("cannot build PC-1c stream metadata: {error}"))?
        .with_batch_size(context.batch_size)
        .with_projection(projection)
        .with_metrics(metrics.clone())
        .with_row_selection_policy(arm.policy())
        .with_row_selection(selection)
        .build()
        .map_err(|error| format!("cannot build PC-1c stream: {error}"))?;
    let mut row_count = 0usize;
    let mut digester = attribution.then(ProjectedContentDigester::default);
    while let Some(batch) = stream.next().await {
        let batch = batch.map_err(|error| format!("PC-1c stream failed: {error}"))?;
        if let Some(digester) = &mut digester {
            digester.update(&batch);
        }
        row_count += batch.num_rows();
    }
    Ok((
        ScanResult {
            row_count,
            content: digester.map(ProjectedContentDigester::finish),
        },
        metrics.decomposition(),
    ))
}

fn write_attribution_csv(
    path: &Path,
    samples: &[AttributionSample],
    pc2: bool,
) -> Result<(), String> {
    let file = File::create(path).map_err(|error| {
        format!(
            "cannot create PC-1c attribution CSV {}: {error}",
            path.display()
        )
    })?;
    let mut writer = BufWriter::new(file);
    let mut header = "schema_version,case_id,context_id,payload_columns,row_groups,rows_per_group,run_len,arm,instrumentation,sample_index,sample_started_unix_ns,wall_ns,b1_reader_build_ns,b1_reader_build_calls,b2_window_ns,b2_window_calls,b3_dispatch_ns,b3_dispatch_calls,b4_filter_ns,b4_filter_calls,b5_consume_ns,b5_consume_calls,b6_batch_assembly_ns,b6_batch_assembly_calls,skip_records_calls,read_records_calls,selection_decode_ns,page_decompression_ns".to_string();
    if pc2 {
        header.push_str(
            ",skip_records_rows,read_records_rows,selection_decode_calls,page_decompression_pages,page_decompression_bytes,filter_record_batch_ns,filter_record_batch_calls,selectors_to_mask_ns,selectors_to_mask_calls,consume_batch_ns,consume_batch_calls,per_column_fallback_auto,per_column_fallback_forced,per_column_engaged,per_column_loaded_row_ranges_fallback",
        );
    }
    writeln!(writer, "{header}")
        .map_err(|error| format!("cannot write PC-1c attribution CSV header: {error}"))?;
    for sample in samples {
        let metric = sample.metrics.as_ref();
        let optional =
            |value: Option<u64>| value.map(|value| value.to_string()).unwrap_or_default();
        let optional_count =
            |value: Option<usize>| value.map(|value| value.to_string()).unwrap_or_default();
        let mut fields = vec![
            if pc2 {
                PC2_ATTR_CSV_SCHEMA_VERSION
            } else {
                ATTR_CSV_SCHEMA_VERSION
            }
            .to_string(),
            sample.case.id.to_string(),
            sample.context.id.to_string(),
            sample.context.payload_columns.to_string(),
            PROFILE_ROW_GROUPS.to_string(),
            PROFILE_ROWS_PER_GROUP.to_string(),
            sample.run_len.to_string(),
            sample.condition.arm.label().to_string(),
            if sample.condition.enabled {
                "enabled"
            } else {
                "disabled"
            }
            .to_string(),
            sample.sample_index.to_string(),
            sample.started_unix_ns.to_string(),
            sample.wall_ns.to_string(),
            optional(metric.map(|metric| metric.pc1c_reader_build_ns)),
            optional_count(metric.map(|metric| metric.pc1c_reader_build_calls)),
            optional(metric.map(|metric| metric.pc1c_window_ns)),
            optional_count(metric.map(|metric| metric.pc1c_window_calls)),
            optional(metric.map(|metric| metric.pc1c_dispatch_ns)),
            optional_count(metric.map(|metric| metric.pc1c_dispatch_calls)),
            optional(metric.map(|metric| metric.pc1c_filter_ns)),
            optional_count(metric.map(|metric| metric.pc1c_filter_calls)),
            optional(metric.map(|metric| metric.pc1c_consume_ns)),
            optional_count(metric.map(|metric| metric.pc1c_consume_calls)),
            optional(metric.map(|metric| metric.pc1c_batch_assembly_ns)),
            optional_count(metric.map(|metric| metric.pc1c_batch_assembly_calls)),
            optional_count(metric.map(|metric| metric.skip_records_calls)),
            optional_count(metric.map(|metric| metric.read_records_calls)),
            optional(metric.map(|metric| metric.selection_decode_ns)),
            optional(metric.map(|metric| metric.page_decompression_ns)),
        ];
        if pc2 {
            fields.extend([
                optional_count(metric.map(|metric| metric.skip_records_rows)),
                optional_count(metric.map(|metric| metric.read_records_rows)),
                optional_count(metric.map(|metric| metric.selection_decode_calls)),
                optional_count(metric.map(|metric| metric.page_decompression_pages)),
                optional_count(metric.map(|metric| metric.page_decompression_bytes)),
                optional(metric.map(|metric| metric.filter_record_batch_ns)),
                optional_count(metric.map(|metric| metric.filter_record_batch_calls)),
                optional(metric.map(|metric| metric.selectors_to_mask_ns)),
                optional_count(metric.map(|metric| metric.selectors_to_mask_calls)),
                optional(metric.map(|metric| metric.consume_batch_ns)),
                optional_count(metric.map(|metric| metric.consume_batch_calls)),
                optional_count(metric.map(|metric| metric.per_column_decisions.fallback_auto)),
                optional_count(metric.map(|metric| metric.per_column_decisions.fallback_forced)),
                optional_count(metric.map(|metric| metric.per_column_decisions.engaged)),
                optional_count(
                    metric.map(|metric| metric.per_column_decisions.loaded_row_ranges_fallback),
                ),
            ]);
        }
        writeln!(
            writer,
            "{}",
            fields
                .iter()
                .map(|field| escape_csv(field))
                .collect::<Vec<_>>()
                .join(",")
        )
        .map_err(|error| format!("cannot write PC-1c attribution CSV row: {error}"))?;
    }
    writer.flush().map_err(|error| {
        format!(
            "cannot flush PC-1c attribution CSV {}: {error}",
            path.display()
        )
    })
}

#[expect(clippy::too_many_arguments)]
fn write_attribution_manifest(
    options: &AttributionOptions,
    samples: &[AttributionSample],
    fixture_sha256: &BTreeMap<String, String>,
    started_unix_ns: u64,
    completed_unix_ns: u64,
    elapsed_ns: u64,
) -> Result<(), String> {
    let schema_version = if options.pc2 {
        PC2_ATTR_MANIFEST_SCHEMA_VERSION
    } else {
        ATTR_MANIFEST_SCHEMA_VERSION
    };
    let csv_schema_version = if options.pc2 {
        PC2_ATTR_CSV_SCHEMA_VERSION
    } else {
        ATTR_CSV_SCHEMA_VERSION
    };
    let benchmark = if options.pc2 {
        "arrow_reader_row_selection_pc2_attribution"
    } else {
        "arrow_reader_row_selection_pc1c_attribution"
    };
    let arms = if options.pc2 {
        vec!["auto32", "pc2-thin"]
    } else {
        vec!["auto32", "percolumn"]
    };
    let timing_order = if options.pc2 {
        "auto32-disabled,auto32-enabled,pc2-thin-enabled,pc2-thin-disabled,pc2-thin-disabled,pc2-thin-enabled,auto32-enabled,auto32-disabled repeated"
    } else {
        "auto32-disabled,auto32-enabled,percolumn-enabled,percolumn-disabled,percolumn-disabled,percolumn-enabled,auto32-enabled,auto32-disabled repeated"
    };
    let mut manifest = json!({
        "schema_version": schema_version,
        "csv_schema_version": csv_schema_version,
        "benchmark": benchmark,
        "git_sha": command_output(
            "git",
            &["-C", env!("CARGO_MANIFEST_DIR"), "rev-parse", "HEAD"],
        ),
        "git_status_porcelain": command_output(
            "git",
            &[
                "-C",
                env!("CARGO_MANIFEST_DIR"),
                "status",
                "--short",
                "--untracked-files=no",
            ],
        ),
        "started_unix_ns": started_unix_ns,
        "completed_unix_ns": completed_unix_ns,
        "elapsed_ns": elapsed_ns,
        "samples_per_condition": options.samples,
        "warmups_per_condition": WARMUPS_PER_ARM,
        "row_count": samples.len(),
        "declared_matrix": {
            "cases": ["c6-l64", "c0-l64", "c0-l4"],
            "arms": arms,
            "instrumentation": ["disabled", "enabled"],
            "row_groups": PROFILE_ROW_GROUPS,
            "rows_per_group": PROFILE_ROWS_PER_GROUP,
        },
        "fixture_sha256_by_case": fixture_sha256,
        "timing_protocol": {
            "order": timing_order,
            "clock": "std::time::Instant",
            "statistic": "median",
            "instrumentation_scope": "fresh ArrowReaderMetrics per scan",
            "selector_calls": "counted but not individually timed",
        },
        "exclusive_boundaries": {
            "B1": "row-group reader construction; one coarse span per RG on the selected standard or per-column path",
            "B2": if options.pc2 {
                "shared native instruction/span/mask precompilation; standard cursor work remains in B3 so B2+B3 is cross-arm comparable"
            } else {
                "shared per-column next_windows materialisation; standard cursor work remains in B3 so B2+B3 is cross-arm comparable"
            },
            "B3": "coarse standard-batch or per-column skip/read driver including invoked decode work; excludes per-column B2, B4, B5, and B6",
            "B4": if options.pc2 {
                "one shared optimized filter-plan build per output batch plus application to every Mask column"
            } else {
                "filter_record_batch application"
            },
            "B5": "consume intermediate column or final standard batch; standard final conversion lives here",
            "B6": "extra final RecordBatch assembly unique to the per-column path",
        },
        "gates": {
            "instrumentation_overhead_max_fraction": 0.02,
            "named_boundary_excess_completeness_min_fraction": 0.85,
            "main_cases": ["c6-l64", "c0-l64"],
            "sampling_instrumentation_top3_order_must_match": true,
        },
        "correctness": {
            "hard_gate": if options.pc2 {
                "row count, schema digest, and every projected leaf digest equal between auto32 and pc2-thin before timing"
            } else {
                "row count, schema digest, and every projected leaf digest equal across both arms before timing"
            },
            "digest": "arrow-projected-leaf-content-v1",
        },
        "csv_sha256": format!("sha256:{}", sha256_path(&options.csv)?),
        "classification": "non-formal",
    });
    if options.pc2 {
        let root = manifest.as_object_mut().unwrap();
        let gates = root.get_mut("gates").unwrap().as_object_mut().unwrap();
        gates.insert("pc2_b2_max_ms".to_string(), json!(0.03));
        gates.insert(
            "pc2_b4_filter_calls".to_string(),
            json!("one shared filter-plan build per output batch, never per mask column"),
        );
        root.insert(
            "pc2_observability".to_string(),
            json!({
                "decision_counters": [
                    "fallback_auto",
                    "fallback_forced",
                    "engaged",
                    "loaded_row_ranges_fallback"
                ],
                "generic_filter_counters": [
                    "filter_record_batch_ns",
                    "filter_record_batch_calls",
                    "selectors_to_mask_ns",
                    "selectors_to_mask_calls"
                ],
                "timed_pair": "fresh disabled and enabled metrics instances per condition",
            }),
        );
    }
    let file = File::create(&options.manifest).map_err(|error| {
        format!(
            "cannot create PC-1c attribution manifest {}: {error}",
            options.manifest.display()
        )
    })?;
    serde_json::to_writer_pretty(file, &manifest)
        .map_err(|error| format!("cannot write PC-1c attribution manifest: {error}"))?;
    let mut file = fs::OpenOptions::new()
        .append(true)
        .open(&options.manifest)
        .map_err(|error| format!("cannot reopen PC-1c attribution manifest: {error}"))?;
    writeln!(file).map_err(|error| format!("cannot finish PC-1c attribution manifest: {error}"))
}

fn write_csv(path: &Path, rows: &[CsvRow]) -> Result<(), String> {
    let file = File::create(path)
        .map_err(|error| format!("cannot create PC-1c CSV {}: {error}", path.display()))?;
    let mut writer = BufWriter::new(file);
    writeln!(
        writer,
        "schema_version,group,cell_id,context_id,context_role,output_layout,payload_spec,payload_columns,encoding,compression,page_index,batch_size,rows_per_group,row_groups,total_rows,shape_name,skip_rows,select_rows,selected_fraction,avg_run_len,runs_per_group,run_count,selection_source,arm,column_strategies,sample_count,samples_ns,sample_started_unix_ns,median_ns,mad_ns,rows_out,schema_sha256,leaf_sha256"
    )
    .map_err(|error| format!("cannot write PC-1c CSV header: {error}"))?;
    for row in rows {
        let measurement = &row.measurement;
        let pc2 = row.cell_id.starts_with("PC-2-scale/");
        let fields = vec![
            if pc2 {
                PC2_CSV_SCHEMA_VERSION
            } else {
                CSV_SCHEMA_VERSION
            }
            .to_string(),
            if pc2 { "PC2-SCALE" } else { "PC1C-SCALE" }.to_string(),
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
            row.shape.rows_per_group.to_string(),
            row.shape.row_groups.to_string(),
            row.shape.total_rows().to_string(),
            row.shape.name.clone(),
            (row.shape.total_rows() - row.shape.selected_rows()).to_string(),
            row.shape.selected_rows().to_string(),
            "0.500000000".to_string(),
            format!("{:.9}", row.shape.run_len as f64),
            row.shape.runs_per_group.to_string(),
            row.shape.run_count().to_string(),
            "external".to_string(),
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
        .map_err(|error| format!("cannot write PC-1c CSV row: {error}"))?;
    }
    writer
        .flush()
        .map_err(|error| format!("cannot flush PC-1c CSV {}: {error}", path.display()))
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
    (0..row.context.payload_columns)
        .map(|column_idx| {
            let threshold = match row.measurement.arm {
                Pc1cArm::Auto32 => 32,
                Pc1cArm::PerColumn => {
                    if pure_dictionary(row.context.payload_at(column_idx)) {
                        PURE_DICTIONARY_THRESHOLD
                    } else {
                        pc1c_other_threshold()
                    }
                }
                #[cfg(feature = "test_common")]
                Pc1cArm::Legacy => {
                    if pure_dictionary(row.context.payload_at(column_idx)) {
                        PURE_DICTIONARY_THRESHOLD
                    } else {
                        DEFAULT_COLUMN_THRESHOLD
                    }
                }
                #[cfg(feature = "test_common")]
                Pc1cArm::Thin => PC2_DEFAULT_COLUMN_THRESHOLD,
                #[cfg(feature = "test_common")]
                Pc1cArm::Product => {
                    if pure_dictionary(row.context.payload_at(column_idx)) {
                        PURE_DICTIONARY_THRESHOLD
                    } else {
                        PC2_DEFAULT_COLUMN_THRESHOLD
                    }
                }
            };
            let strategy = if row.shape.run_len < threshold {
                "mask"
            } else {
                "selectors"
            };
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

fn write_manifest(
    options: &ScaleOptions,
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
    let schema_version = if options.pc2 {
        PC2_MANIFEST_SCHEMA_VERSION
    } else {
        MANIFEST_SCHEMA_VERSION
    };
    let csv_schema_version = if options.pc2 {
        PC2_CSV_SCHEMA_VERSION
    } else {
        CSV_SCHEMA_VERSION
    };
    let benchmark = if options.pc2 {
        "arrow_reader_row_selection_pc2_scale"
    } else {
        "arrow_reader_row_selection_pc1c_scale"
    };
    let declared_arms = if options.pc2 {
        vec!["auto32", "pc-bolton", "pc2-thin"]
    } else {
        vec!["auto32", "percolumn"]
    };
    let declared_shapes = if options.pc2 {
        vec!["f50_l64"]
    } else {
        vec!["f50_l64", "f50_l4"]
    };
    let mut manifest = json!({
        "schema_version": schema_version,
        "benchmark": benchmark,
        "csv_schema_version": csv_schema_version,
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
            "contexts": SCALE_CONTEXT_IDS,
            "dimensions": scale_dimensions(options.pc2).iter().map(|(row_groups, rows_per_group)| json!({
                "row_groups": row_groups,
                "rows_per_group": rows_per_group,
            })).collect::<Vec<_>>(),
            "shapes": declared_shapes,
            "arms": declared_arms,
        },
        "fixture": {
            "default_batch_size": BATCH_SIZE,
            "page_row_limit": ORACLE_PAGE_ROWS,
            "in_memory": true,
            "metadata_preparsed": true,
            "sha256_by_context_and_dimension": fixture_sha256,
        },
        "preregistration": {
            "purpose": if options.pc2 {
                "same-SHA fit of per-scan, per-row-group, and per-physical-row excess for pc-bolton and pc2-thin"
            } else {
                "distinguish per-scan, per-row-group, and per-physical-row excess before attribution"
            },
            "mechanism_table_frozen_before_timing": true,
            "phase_d_primary_row_groups": 16,
            "phase_d_fixed_cost_limit_ms_per_rg": 0.05,
            "phase_d_noharm_floor_percent": 2.0,
            "phase_d_noharm_mad_multiplier": 3.0,
        },
        "percolumn_contract": {
            "scope": "flat output projection only; predicates and unsupported shapes fall back to Auto32",
            "physical_window": if options.pc2 {
                "one native precompiled instruction/span plan shared across projected columns"
            } else if cfg!(feature = "test_common") {
                "one shared legacy MaskCursor window"
            } else {
                "native PC-2 compatibility path; enable test_common to replay historical PC-1c"
            },
            "pure_dictionary_threshold": PURE_DICTIONARY_THRESHOLD,
            "other_column_threshold": if options.pc2 {
                PC2_DEFAULT_COLUMN_THRESHOLD
            } else {
                pc1c_other_threshold()
            },
            "choice": "mask iff average selector run length is strictly below the column threshold",
        },
        "timing_protocol": {
            "order": if options.pc2 {
                "auto32,pc-bolton,pc2-thin,pc2-thin,pc-bolton,auto32 repeated"
            } else {
                "auto32,percolumn,percolumn,auto32 repeated"
            },
            "statistic": "median",
            "dispersion": "median_absolute_deviation",
            "clock": "std::time::Instant",
            "sample_start_clock": "unix_epoch_nanoseconds",
            "selection_clone": "outside timed region",
        },
        "correctness": {
            "hard_gate": if options.pc2 {
                "row count, schema digest, and every projected leaf digest equal across all three same-SHA arms"
            } else {
                "row count, schema digest, and every projected leaf digest equal across both arms"
            },
            "digest": "arrow-projected-leaf-content-v1",
        },
        "csv_sha256": format!("sha256:{}", sha256_path(&options.csv)?),
        "classification": "non-formal",
    });
    if options.pc2 {
        manifest
            .get_mut("percolumn_contract")
            .unwrap()
            .as_object_mut()
            .unwrap()
            .insert(
                "same_sha_arm_binding".to_string(),
                json!({
                    "auto32": "standard Auto32",
                    "pc-bolton": "bench-only legacy coordinator with dictionary=4 and other=16",
                    "pc2-thin": "native PC-2 coordinator forced to Auto32 for every column"
                }),
            );
    }
    let file = File::create(&options.manifest).map_err(|error| {
        format!(
            "cannot create PC-1c manifest {}: {error}",
            options.manifest.display()
        )
    })?;
    serde_json::to_writer_pretty(file, &manifest)
        .map_err(|error| format!("cannot write PC-1c manifest: {error}"))?;
    let mut file = fs::OpenOptions::new()
        .append(true)
        .open(&options.manifest)
        .map_err(|error| format!("cannot reopen PC-1c manifest: {error}"))?;
    writeln!(file).map_err(|error| format!("cannot finish PC-1c manifest: {error}"))
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

fn emit_artifact(pc2: bool, kind: &str, path: &Path) -> Result<(), String> {
    let raw = fs::read_to_string(path)
        .map_err(|error| format!("cannot read {} for log embedding: {error}", path.display()))?;
    let prefix = if pc2 { "DFEXP_PC2" } else { "DFEXP_PC1C" };
    println!("{prefix}_SCALE_{kind}_BEGIN");
    print!("{raw}");
    if !raw.ends_with('\n') {
        println!();
    }
    println!("{prefix}_SCALE_{kind}_END");
    Ok(())
}

fn emit_attribution_artifact(pc2: bool, kind: &str, path: &Path) -> Result<(), String> {
    let raw = fs::read_to_string(path)
        .map_err(|error| format!("cannot read {} for log embedding: {error}", path.display()))?;
    let prefix = if pc2 { "DFEXP_PC2" } else { "DFEXP_PC1C" };
    println!("{prefix}_ATTR_{kind}_BEGIN");
    print!("{raw}");
    if !raw.ends_with('\n') {
        println!();
    }
    println!("{prefix}_ATTR_{kind}_END");
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
    if value.contains([',', '"', '\n', '\r']) {
        format!("\"{}\"", value.replace('"', "\"\""))
    } else {
        value.to_string()
    }
}
