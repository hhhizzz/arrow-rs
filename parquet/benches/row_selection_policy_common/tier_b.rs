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

//! Tier-B row-selection oracle.
//!
//! D0 is deliberately a small executable contract check. D1 is the fixed
//! training-only identifiability matrix. Both measure one row group at a time,
//! bind timing to metadata from that exact row group, hash every projected
//! logical leaf, and prove that page Required/Skip views use identical Parquet
//! bytes. Blind contexts are not constructed by either matrix.

use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs::{self, File};
use std::hint;
use std::io::{BufWriter, Write};
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use parquet::file::metadata::PageIndexPolicy;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use tokio::runtime::Runtime;

use super::fixture::{
    ORACLE_CONTEXTS, ORACLE_PAGE_ROWS, ORACLE_ROW_GROUPS, OracleCompression, OracleContext,
    OracleFixture, OraclePayload, build_oracle_fixture,
};
use super::model::ROWS_PER_GROUP;
use super::runner::{
    OracleArm, OracleRowGroupRunResult, OracleSelectionSource, ProjectedContentDigest,
    run_oracle_row_group,
};
use super::shapes::{OracleShape, OracleShapeSummary, assert_oracle_shape_contracts};

const CSV_SCHEMA_VERSION: &str = "arrow-row-selection-oracle-v2";
const CSV_SCHEMA_VERSION_TIER_C: &str = "arrow-row-selection-oracle-v3";
const CSV_SCHEMA_VERSION_TIER_D: &str = "arrow-row-selection-oracle-v4";
const MANIFEST_SCHEMA_VERSION: &str = "arrow-row-selection-oracle-manifest-v2";
const MANIFEST_SCHEMA_VERSION_TIER_C: &str = "arrow-row-selection-oracle-manifest-v3";
const MANIFEST_SCHEMA_VERSION_TIER_D: &str = "arrow-row-selection-oracle-manifest-v4";
const CANDIDATE_SCHEMA_VERSION: &str = "arrow-row-selection-candidate-library-v1";
const MATRIX_D0_SMOKE: &str = "tier-b-d0-smoke-v1";
const MATRIX_D1_DISCOVERY: &str = "tier-b-d1-discovery-v1";
const MATRIX_D1_REPLAY: &str = "tier-b-d1-replay-v1";
const MATRIX_TIER_C_C0_SMOKE: &str = "tier-c-c0-smoke-v1";
const MATRIX_TIER_C_C1: &str = "tier-c-c1-cost-surface-v1";
const MATRIX_TIER_D_D0_SMOKE: &str = "tier-d-d0-guard-smoke-v1";
const MATRIX_TIER_D_D1: &str = "tier-d-d1-boundary-v1";
const MATRIX_TIER_D_D2: &str = "tier-d-d2-transfer-v1";
const D1_DISCOVERY_CSV_SHA256: &str =
    "9724c82e063c562a326ea7c6b8bc3bd0aa478da60d99b54ffbfaf20f0d661ea9";
const D1_ADAPTIVE_UNION_SHA256: &str =
    "sha256:c6e93b62e0e827382cdd101b794ece8a96df29ce9635fa851b88a0f67cd03c70";
const DEFAULT_SAMPLES: usize = 4;
const WARMUPS_PER_ARM: usize = 2;

const TIER_C_CONTEXT_IDS: &[&str] = &[
    "C6", "C0", "C8", "C9", "C3", "C11", "T1", "C4", "T2", "T3", "C13", "T4", "T5", "C5", "T6",
];
const TIER_C_SMOKE_CONTEXT_IDS: &[&str] = &["C0", "C4", "C13"];
const TIER_C_TRANSITION_RUNS: &[usize] = &[2, 8, 32, 128, 512];
const TIER_C_GAP_ROWS: &[usize] = &[64, 512, 4_096, 16_384, 63_488];
const TIER_C_SELECTED_ROWS: &[usize] = &[1_024, 8_192, 32_768, 57_344, 64_512];
const TIER_D_D0_CONTEXT_IDS: &[&str] = &["C8", "C11", "C4", "C13"];
const TIER_D_D1_CONTEXT_IDS: &[&str] =
    &["C0", "C8", "C3", "C11", "T1", "C4", "T2", "C13", "T4", "T6"];
const TIER_D_D2_CONTEXT_IDS: &[&str] = &[
    "U1", "U2", "U3", "U4", "U5", "U6", "U7", "U8", "U9", "U10", "U11",
];
const TIER_D_INTERACTION_CONTEXT_IDS: &[&str] = &["C8", "C11", "C4", "C13", "T4", "T6"];
const TIER_D_TRANSITION_RUNS: &[usize] = &[128, 256, 384, 512, 640, 768, 1_024];
const TIER_D_SELECTED_ROWS: &[usize] = &[1_024, 8_192, 32_768, 57_344, 64_512];
const TIER_D_D2_TRANSITION_RUNS: &[usize] = &[384, 512, 640];
const TIER_D_FROZEN_TAU_T: usize = 766;
const TIER_D_FROZEN_TAU_Q: u64 = 32;
const TIER_D_FROZEN_TAU_F_NUMERATOR: usize = 1;
const TIER_D_FROZEN_TAU_F_DENOMINATOR: usize = 2;
const TIER_D_FROZEN_GUARD_SHA256: &str =
    "2f30b136911be9a113d80f6190d82128f7d5bcf6d2bdc0f1c55c1d2469538290";
const TIER_D_FROZEN_DECISION_SHA256: &str =
    "sha256:7615f4bf2b4b3eeba1c7595182a1f2760fc48a0daad08315f3f61ea691b3a3b1";
const TIER_D_D1_DISCOVERY_CSV_SHA256: &str =
    "c6646ed7c04671ab88797d74db0e1618d3a323b17fb68797ba1b26e3b69d633a";
const TIER_D_D1_REPLAY_CSV_SHA256: &str =
    "239d5a2a2c089e48b63d35234f2db3f684ecadffdafe11aa637be0f3c7be7281";
const TIER_D_D1_VALIDATION_SHA256: &str =
    "36186fe3000a1fb4fe75550d2fa781787a61309da88877d24082d5c9d2d7d858";

const CSV_COLUMNS: &[&str] = &[
    "schema_version",
    "group",
    "role",
    "cell_id",
    "context_id",
    "row_group_index",
    "dtype",
    "projection_signature",
    "output_layout",
    "payload_columns",
    "encoding",
    "compression",
    "metadata_policy",
    "page_layout_rows",
    "fixture_sha256",
    "batch_size",
    "rows_per_group",
    "shape_name",
    "represented_rows",
    "selected_rows",
    "skipped_rows",
    "selected_fraction",
    "avg_run_len",
    "run_count",
    "selected_run_count",
    "skipped_run_count",
    "first_selected_row",
    "last_selected_row_exclusive",
    "max_skip_run",
    "long_skip_rows_1024",
    "long_skip_count_1024",
    "long_skip_rows_4096",
    "long_skip_count_4096",
    "long_skip_share_1024",
    "long_skip_share_4096",
    "selection_source",
    "selection_backing",
    "projected_leaf_count",
    "physical_type_histogram",
    "encoding_sets",
    "dictionary_leaf_count",
    "compressed_bytes",
    "uncompressed_bytes",
    "per_leaf_compressed_bytes",
    "per_leaf_uncompressed_bytes",
    "per_leaf_num_values",
    "compression_ratio",
    "num_values_min",
    "num_values_max",
    "num_values_consistent",
    "arrow_output_width_proxy",
    "loaded_page_rows",
    "loaded_range_count",
    "per_leaf_page_rows",
    "per_leaf_page_first_rows",
    "per_leaf_predicted_fetched_bytes",
    "predicted_fetched_bytes",
    "base_mask_chunk_count",
    "base_mask_decoded_rows",
    "base_run_count",
    "mask_chunk_count",
    "mask_decoded_rows",
    "loaded_run_count",
    "arm",
    "sample_count",
    "samples_ns",
    "sample_started_unix_ns",
    "median_ns",
    "mad_ns",
    "rows_out",
    "schema_sha256",
    "leaf_sha256",
    "requested_range_count",
    "requested_ranges",
    "requested_bytes",
];

const TIER_C_EXTRA_COLUMNS: &[&str] = &[
    "leading_skip_present",
    "internal_skip_run_count",
    "internal_transition_count",
    "shape_invariant_sha256",
];

const PLAIN_GRID: &[usize] = &[4, 16, 64, 256, 1_024, 4_096];
const ZSTD_GRID: &[usize] = &[1, 4, 16, 64, 256, 1_024];
const DICTIONARY_GRID: &[usize] = &[1, 2, 4, 8, 16, 64];

const TRAINING_CONTEXTS: &[OracleContext] = &[
    OracleContext {
        id: "T1",
        payload: OraclePayload::Utf8View64,
        payload_columns: 1,
        column_payloads: None,
        compression: OracleCompression::Uncompressed,
        page_index: false,
        batch_size: 8_192,
    },
    OracleContext {
        id: "T2",
        payload: OraclePayload::Utf8View64,
        payload_columns: 32,
        column_payloads: None,
        compression: OracleCompression::Uncompressed,
        page_index: false,
        batch_size: 8_192,
    },
    OracleContext {
        id: "T3",
        payload: OraclePayload::Utf8View64,
        payload_columns: 1,
        column_payloads: None,
        compression: OracleCompression::Zstd,
        page_index: false,
        batch_size: 8_192,
    },
    OracleContext {
        id: "T4",
        payload: OraclePayload::Utf8View64,
        payload_columns: 32,
        column_payloads: None,
        compression: OracleCompression::Zstd,
        page_index: false,
        batch_size: 8_192,
    },
    OracleContext {
        id: "T5",
        payload: OraclePayload::Utf8Dictionary1k,
        payload_columns: 1,
        column_payloads: None,
        compression: OracleCompression::Uncompressed,
        page_index: false,
        batch_size: 8_192,
    },
    OracleContext {
        id: "T6",
        payload: OraclePayload::Utf8Dictionary1k,
        payload_columns: 32,
        column_payloads: None,
        compression: OracleCompression::Uncompressed,
        page_index: false,
        batch_size: 8_192,
    },
];

const TIER_D_D2_CONTEXTS: &[OracleContext] = &[
    OracleContext {
        id: "U1",
        payload: OraclePayload::Int32,
        payload_columns: 28,
        column_payloads: None,
        compression: OracleCompression::Uncompressed,
        page_index: false,
        batch_size: 8_192,
    },
    OracleContext {
        id: "U2",
        payload: OraclePayload::Int32,
        payload_columns: 36,
        column_payloads: None,
        compression: OracleCompression::Uncompressed,
        page_index: false,
        batch_size: 8_192,
    },
    OracleContext {
        id: "U3",
        payload: OraclePayload::Utf8View8,
        payload_columns: 40,
        column_payloads: None,
        compression: OracleCompression::Uncompressed,
        page_index: false,
        batch_size: 8_192,
    },
    OracleContext {
        id: "U4",
        payload: OraclePayload::Utf8View8,
        payload_columns: 48,
        column_payloads: None,
        compression: OracleCompression::Uncompressed,
        page_index: false,
        batch_size: 8_192,
    },
    OracleContext {
        id: "U5",
        payload: OraclePayload::Utf8View32,
        payload_columns: 12,
        column_payloads: None,
        compression: OracleCompression::Uncompressed,
        page_index: false,
        batch_size: 8_192,
    },
    OracleContext {
        id: "U6",
        payload: OraclePayload::Utf8View32,
        payload_columns: 16,
        column_payloads: None,
        compression: OracleCompression::Uncompressed,
        page_index: false,
        batch_size: 8_192,
    },
    OracleContext {
        id: "U7",
        payload: OraclePayload::Utf8View64,
        payload_columns: 7,
        column_payloads: None,
        compression: OracleCompression::Uncompressed,
        page_index: false,
        batch_size: 8_192,
    },
    OracleContext {
        id: "U8",
        payload: OraclePayload::Utf8View64,
        payload_columns: 9,
        column_payloads: None,
        compression: OracleCompression::Uncompressed,
        page_index: false,
        batch_size: 8_192,
    },
    OracleContext {
        id: "U9",
        payload: OraclePayload::Utf8View64,
        payload_columns: 14,
        column_payloads: None,
        compression: OracleCompression::Zstd,
        page_index: false,
        batch_size: 8_192,
    },
    OracleContext {
        id: "U10",
        payload: OraclePayload::Utf8View64,
        payload_columns: 16,
        column_payloads: None,
        compression: OracleCompression::Zstd,
        page_index: false,
        batch_size: 8_192,
    },
    OracleContext {
        id: "U11",
        payload: OraclePayload::Utf8Dictionary1k,
        payload_columns: 64,
        column_payloads: None,
        compression: OracleCompression::Uncompressed,
        page_index: false,
        batch_size: 8_192,
    },
];

const D0_MIXED_COLUMNS: &[OraclePayload] = &[
    OraclePayload::Int32,
    OraclePayload::Int32,
    OraclePayload::Int32,
    OraclePayload::Int32,
    OraclePayload::Utf8View64,
    OraclePayload::Utf8View64,
    OraclePayload::Utf8View64,
    OraclePayload::Utf8View64,
];

#[derive(Debug)]
struct Options {
    matrix: String,
    list: bool,
    role: Option<String>,
    samples: usize,
    csv: PathBuf,
    manifest: PathBuf,
    candidate_library: PathBuf,
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
    content: ProjectedContentDigest,
    requested_ranges: Vec<Range<u64>>,
    requested_bytes: u64,
}

#[derive(Clone, Debug)]
struct PairMeasurement {
    selectors: ArmMeasurement,
    mask: ArmMeasurement,
}

#[derive(Clone, Debug)]
struct MetadataSummary {
    projected_leaf_count: usize,
    projection_signature: String,
    physical_type_histogram: String,
    encoding_sets: String,
    dictionary_leaf_count: usize,
    compressed_bytes: u64,
    uncompressed_bytes: u64,
    per_leaf_compressed_bytes: String,
    per_leaf_uncompressed_bytes: String,
    per_leaf_num_values: String,
    compression_ratio: f64,
    num_values_min: i64,
    num_values_max: i64,
    num_values_consistent: bool,
    arrow_output_width_proxy: u64,
    loaded_page_rows: Option<usize>,
    loaded_range_count: Option<usize>,
    per_leaf_page_rows: Option<String>,
    per_leaf_page_first_rows: Option<String>,
    per_leaf_predicted_fetched_bytes: Option<String>,
    predicted_fetched_bytes: Option<u64>,
    base_mask_chunk_count: usize,
    base_mask_decoded_rows: usize,
    base_run_count: usize,
    mask_chunk_count: usize,
    mask_decoded_rows: usize,
    loaded_run_count: usize,
}

#[derive(Clone, Debug)]
struct CsvRow {
    group: &'static str,
    role: &'static str,
    cell_id: String,
    context: OracleContext,
    metadata_policy: &'static str,
    page_layout_rows: Option<usize>,
    fixture_sha256: String,
    row_group_index: usize,
    shape_name: String,
    shape_invariant_sha256: String,
    shape: OracleShapeSummary,
    source: OracleSelectionSource,
    backing: &'static str,
    metadata: MetadataSummary,
    measurement: ArmMeasurement,
}

pub(crate) fn main() {
    if let Err(error) = try_main() {
        eprintln!("row-selection matrix oracle failed: {error}");
        std::process::exit(2);
    }
}

fn try_main() -> Result<(), String> {
    assert_oracle_shape_contracts();
    validate_structural_expressibility()?;
    let options = parse_options()?;
    if !matches!(
        options.matrix.as_str(),
        MATRIX_D0_SMOKE
            | MATRIX_D1_DISCOVERY
            | MATRIX_D1_REPLAY
            | MATRIX_TIER_C_C0_SMOKE
            | MATRIX_TIER_C_C1
            | MATRIX_TIER_D_D0_SMOKE
            | MATRIX_TIER_D_D1
            | MATRIX_TIER_D_D2
    ) {
        return Err(format!(
            "unsupported --matrix {:?}; expected {MATRIX_D0_SMOKE}, {MATRIX_D1_DISCOVERY}, {MATRIX_D1_REPLAY}, {MATRIX_TIER_C_C0_SMOKE}, {MATRIX_TIER_C_C1}, {MATRIX_TIER_D_D0_SMOKE}, {MATRIX_TIER_D_D1}, or {MATRIX_TIER_D_D2}",
            options.matrix
        ));
    }
    if options.list {
        match options.matrix.as_str() {
            MATRIX_D0_SMOKE => list_d0_cells(),
            MATRIX_D1_DISCOVERY => list_d1_cells(options.role.as_deref(), false)?,
            MATRIX_D1_REPLAY => list_d1_cells(options.role.as_deref(), true)?,
            MATRIX_TIER_C_C0_SMOKE => list_tier_c_cells(options.role.as_deref(), true)?,
            MATRIX_TIER_C_C1 => list_tier_c_cells(options.role.as_deref(), false)?,
            MATRIX_TIER_D_D0_SMOKE => list_tier_d_cells(options.role.as_deref(), true)?,
            MATRIX_TIER_D_D1 => list_tier_d_cells(options.role.as_deref(), false)?,
            MATRIX_TIER_D_D2 => list_tier_d_d2_cells(options.role.as_deref())?,
            _ => unreachable!(),
        }
        return Ok(());
    }
    if let Some(role) = &options.role {
        return Err(format!("--role {role:?} is valid only with --list"));
    }

    let started_unix_ns = unix_nanos();
    let started = Instant::now();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|error| format!("cannot build Tokio runtime: {error}"))?;
    let mut rows = Vec::new();
    match options.matrix.as_str() {
        MATRIX_D0_SMOKE => run_d0(&runtime, &options, &mut rows)?,
        MATRIX_D1_DISCOVERY => run_d1_training(&runtime, &options, false, &mut rows)?,
        MATRIX_D1_REPLAY => run_d1_training(&runtime, &options, true, &mut rows)?,
        MATRIX_TIER_C_C0_SMOKE => run_tier_c(&runtime, &options, true, &mut rows)?,
        MATRIX_TIER_C_C1 => run_tier_c(&runtime, &options, false, &mut rows)?,
        MATRIX_TIER_D_D0_SMOKE => run_tier_d(&runtime, &options, true, &mut rows)?,
        MATRIX_TIER_D_D1 => run_tier_d(&runtime, &options, false, &mut rows)?,
        MATRIX_TIER_D_D2 => run_tier_d_d2(&runtime, &options, &mut rows)?,
        _ => unreachable!(),
    }
    if rows.is_empty() {
        return Err("matrix produced no timing rows".to_string());
    }

    let candidate_library = experiment_contract(&options.matrix);
    write_json(&options.candidate_library, &candidate_library)?;
    write_csv(&options, &rows)?;
    write_manifest(
        &options,
        &rows,
        started_unix_ns,
        unix_nanos(),
        started.elapsed().as_nanos() as u64,
        &candidate_library,
    )?;

    let cells = rows
        .iter()
        .map(|row| row.cell_id.as_str())
        .collect::<BTreeSet<_>>()
        .len();
    println!("DFEXP_SELECTION_ORACLE_V2_CELLS={cells}");
    println!(
        "DFEXP_SELECTION_ORACLE_V2_RG_UNITS={}",
        cells * ORACLE_ROW_GROUPS
    );
    println!("DFEXP_SELECTION_ORACLE_V2_ROWS={}", rows.len());
    if options.emit_artifacts {
        emit_artifact("CSV", &options.csv)?;
        emit_artifact("MANIFEST", &options.manifest)?;
        emit_artifact("CANDIDATE_LIBRARY", &options.candidate_library)?;
    }
    Ok(())
}

fn run_d0(runtime: &Runtime, options: &Options, rows: &mut Vec<CsvRow>) -> Result<(), String> {
    for context_id in ["C3", "C4", "C13"] {
        let context = context_by_id(context_id)?;
        let fixture = build_oracle_fixture(context, None)
            .map_err(|error| format!("cannot build {context_id}: {error}"))?;
        run_fixture_shape(
            &runtime,
            &options,
            &fixture,
            "D0-COST",
            "training-contract",
            "Skip",
            None,
            &OracleShape::l_sweep(64),
            rows,
        )?;
    }

    let mixed = OracleContext {
        id: "D0M",
        payload: OraclePayload::Int32,
        payload_columns: D0_MIXED_COLUMNS.len(),
        column_payloads: Some(D0_MIXED_COLUMNS),
        compression: OracleCompression::Uncompressed,
        page_index: false,
        batch_size: 8_192,
    };
    let fixture = build_oracle_fixture(mixed, None)
        .map_err(|error| format!("cannot build D0 mixed fixture: {error}"))?;
    run_fixture_shape(
        &runtime,
        &options,
        &fixture,
        "D0-CORRECTNESS",
        "training-contract",
        "Skip",
        None,
        &OracleShape::l_sweep(64),
        rows,
    )?;

    let page_context = context_by_id("C0")?.with_page_index();
    let page_required = build_oracle_fixture(page_context, None)
        .map_err(|error| format!("cannot build D0 page fixture: {error}"))?;
    let page_skip = page_required
        .with_page_index_policy(PageIndexPolicy::Skip)
        .map_err(|error| format!("cannot create same-Bytes Skip view: {error}"))?;
    if page_required.bytes_sha256() != page_skip.bytes_sha256() {
        return Err("same-Bytes page views have different fixture digests".to_string());
    }
    for (fixture, policy) in [(&page_required, "Required"), (&page_skip, "Skip")] {
        for shape in [OracleShape::l_sweep(64), OracleShape::page_matched_bursty()] {
            run_fixture_shape(
                &runtime,
                &options,
                fixture,
                "D0-PAGE",
                "training-contract",
                policy,
                Some(ORACLE_PAGE_ROWS),
                &shape,
                rows,
            )?;
        }
    }
    validate_page_exposure(&rows)?;
    Ok(())
}

fn parse_options() -> Result<Options, String> {
    let mut matrix = None;
    let mut list = false;
    let mut role = None;
    let mut samples = DEFAULT_SAMPLES;
    let mut csv = default_artifact_path("selection-oracle-v2.csv");
    let mut manifest = default_artifact_path("selection-oracle-v2-manifest.json");
    let mut candidate_library = default_artifact_path("candidate-library-v1.json");
    let mut emit_artifacts = false;
    let mut args = env::args().skip(1);
    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--selection-oracle" | "--bench" => {}
            "--matrix" => {
                matrix = Some(
                    args.next()
                        .ok_or_else(|| "--matrix requires a value".to_string())?,
                );
            }
            "--list" => list = true,
            "--role" => {
                role = Some(
                    args.next()
                        .ok_or_else(|| "--role requires a value".to_string())?,
                )
            }
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
                )
            }
            "--manifest" => {
                manifest = PathBuf::from(
                    args.next()
                        .ok_or_else(|| "--manifest requires a path".to_string())?,
                )
            }
            "--candidate-library" => {
                candidate_library = PathBuf::from(
                    args.next()
                        .ok_or_else(|| "--candidate-library requires a path".to_string())?,
                )
            }
            "--emit-artifacts" => emit_artifacts = true,
            "--help" | "-h" => {
                println!(
                    "row_selector --selection-oracle --matrix \
                     <{MATRIX_D0_SMOKE}|{MATRIX_D1_DISCOVERY}|{MATRIX_D1_REPLAY}|\
                     {MATRIX_TIER_C_C0_SMOKE}|{MATRIX_TIER_C_C1}|\
                     {MATRIX_TIER_D_D0_SMOKE}|{MATRIX_TIER_D_D1}|{MATRIX_TIER_D_D2}> \
                     [--list [--role training]] [--samples EVEN] [--emit-artifacts]"
                );
                std::process::exit(0);
            }
            _ => {
                return Err(format!(
                    "unsupported row-selection matrix argument {argument:?}"
                ));
            }
        }
    }
    if !(2..=100).contains(&samples) || !samples.is_multiple_of(2) {
        return Err("--samples must be an even integer in 2..=100".to_string());
    }
    let unique_paths = [&csv, &manifest, &candidate_library]
        .into_iter()
        .collect::<BTreeSet<_>>();
    if unique_paths.len() != 3 {
        return Err("CSV, manifest, and candidate-library paths must differ".to_string());
    }
    Ok(Options {
        matrix: matrix.ok_or_else(|| "row-selection matrix mode requires --matrix".to_string())?,
        list,
        role,
        samples,
        csv,
        manifest,
        candidate_library,
        emit_artifacts,
    })
}

fn list_d0_cells() {
    for context in ["C3", "C4", "C13", "D0M"] {
        println!("D0/{context}/f50_l64");
    }
    for policy in ["Required", "Skip"] {
        for shape in ["f50_l64", "page_matched_bursty_f50_l64"] {
            println!("D0/C0/page-{policy}/{shape}");
        }
    }
    eprintln!("listed 8 logical cells; each emits four row-group pairs");
}

fn training_contexts() -> Vec<OracleContext> {
    ORACLE_CONTEXTS
        .iter()
        .chain(TRAINING_CONTEXTS)
        .copied()
        .collect()
}

fn coarse_grid(context: OracleContext) -> &'static [usize] {
    if context.uses_dictionary() {
        DICTIONARY_GRID
    } else if context.compression == OracleCompression::Zstd {
        ZSTD_GRID
    } else {
        PLAIN_GRID
    }
}

fn adaptive_grid(context_id: &str) -> &'static [usize] {
    match context_id {
        "T1" => &[6, 8, 11],
        "T2" => &[128, 512],
        "T3" => &[6, 8, 11],
        "T4" => &[91, 128, 181],
        "T5" => &[3, 6],
        "T6" => &[5, 6, 7],
        _ => &[],
    }
}

fn boundary_shapes(context_id: &str) -> Vec<OracleShape> {
    let run_lengths: &[usize] = match context_id {
        "C4" => &[128, 256, 512],
        "C5" => &[2, 4, 8],
        _ => &[],
    };
    run_lengths
        .iter()
        .flat_map(|run_len| {
            [2, 50, 98]
                .into_iter()
                .map(move |percent| OracleShape::selectivity(percent, *run_len))
        })
        .collect()
}

fn cost_anchor_shapes() -> Vec<OracleShape> {
    let shapes = vec![
        OracleShape::all_selected(),
        OracleShape::leading_only(),
        OracleShape::internal_bookend(64),
        OracleShape::internal_bookend(4_096),
        OracleShape::internal_bookend(ROWS_PER_GROUP - 2),
        OracleShape::multi_gap64(),
    ];
    let selected = shapes
        .iter()
        .map(|shape| shape.summary().selected_rows)
        .collect::<Vec<_>>();
    assert_eq!(selected, [ROWS_PER_GROUP, 1, 2, 2, 2, 65]);
    let multi = shapes.last().unwrap().summary();
    assert_eq!(multi.selected_run_count, 65);
    assert_eq!(multi.skipped_run_count, 65);
    shapes
}

fn d1_cell_listing(replay: bool) -> Vec<(&'static str, String)> {
    let mut cells = Vec::new();
    for context in training_contexts() {
        for run_len in coarse_grid(context) {
            cells.push((
                "pair",
                format!(
                    "D1/X-A/{}/{}",
                    context.id,
                    OracleShape::l_sweep(*run_len).name
                ),
            ));
        }
        if replay {
            for run_len in adaptive_grid(context.id) {
                cells.push((
                    "pair",
                    format!("D1/X-A-adaptive/{}/f50_l{run_len}", context.id),
                ));
            }
        }
        for shape in boundary_shapes(context.id) {
            cells.push(("pair", format!("D1/X-B/{}/{}", context.id, shape.name)));
        }
        cells.push(("single", format!("D1/X-E/{}/no_selection", context.id)));
        for shape in cost_anchor_shapes() {
            cells.push(("pair", format!("D1/X-E/{}/{}", context.id, shape.name)));
        }
        if context.id == "C0" {
            for run_len in [8, 32, 128] {
                cells.push(("pair", format!("D1/X-G/C0/batch1024/f50_l{run_len}")));
            }
        }
    }
    for context_id in ["C0", "C4"] {
        for policy in ["Required", "Skip"] {
            for shape in [OracleShape::l_sweep(64), OracleShape::page_matched_bursty()] {
                cells.push((
                    "pair",
                    format!("D1/X-C/{context_id}/page-{policy}/{}", shape.name),
                ));
            }
        }
    }
    cells
}

fn list_d1_cells(role: Option<&str>, replay: bool) -> Result<(), String> {
    if let Some(role) = role
        && role != "training"
    {
        return Err(format!(
            "D1 discovery exposes only --role training, not {role:?}"
        ));
    }
    let cells = d1_cell_listing(replay);
    let pair_count = cells.iter().filter(|(kind, _)| *kind == "pair").count();
    let single_count = cells.len() - pair_count;
    let expected = if replay {
        (305, 285, 20)
    } else {
        (289, 269, 20)
    };
    if (cells.len(), pair_count, single_count) != expected {
        return Err(format!(
            "D1 listing drift: total={}, pairs={pair_count}, singles={single_count}",
            cells.len()
        ));
    }
    let cell_count = cells.len();
    for (kind, id) in cells {
        println!("training\t{kind}\t{id}");
    }
    eprintln!(
        "listed {} D1 training cells: {pair_count} forced pairs + {single_count} no-selection singles; {} RG units",
        cell_count,
        cell_count * ORACLE_ROW_GROUPS
    );
    Ok(())
}

fn tier_c_context_ids(smoke: bool) -> &'static [&'static str] {
    if smoke {
        TIER_C_SMOKE_CONTEXT_IDS
    } else {
        TIER_C_CONTEXT_IDS
    }
}

fn tier_c_pair_shapes(smoke: bool) -> Vec<(&'static str, OracleShape)> {
    let transition_runs: &[usize] = if smoke { &[32] } else { TIER_C_TRANSITION_RUNS };
    let gap_rows: &[usize] = if smoke { &[4_096] } else { TIER_C_GAP_ROWS };
    let selected_rows: &[usize] = if smoke {
        &[32_768]
    } else {
        TIER_C_SELECTED_ROWS
    };
    transition_runs
        .iter()
        .map(|value| ("TC-R", OracleShape::tier_c_transitions(*value)))
        .chain(
            gap_rows
                .iter()
                .map(|value| ("TC-W", OracleShape::tier_c_gap(*value))),
        )
        .chain(
            selected_rows
                .iter()
                .map(|value| ("TC-F", OracleShape::tier_c_selectivity(*value))),
        )
        .chain(std::iter::once(("TC-A", OracleShape::all_selected())))
        .collect()
}

fn tier_c_cell_id(prefix: &str, group: &str, context_id: &str, shape: &OracleShape) -> String {
    format!("{prefix}/{group}/{context_id}/{}", shape.name)
}

fn tier_c_cell_listing(smoke: bool) -> Vec<(&'static str, String, OracleShape)> {
    let prefix = if smoke { "TC/C0" } else { "TC/C1" };
    let mut cells = Vec::new();
    for context_id in tier_c_context_ids(smoke) {
        for (group, shape) in tier_c_pair_shapes(smoke) {
            cells.push((
                "pair",
                tier_c_cell_id(prefix, group, context_id, &shape),
                shape,
            ));
        }
        cells.push((
            "single",
            format!("{prefix}/TC-A/{context_id}/no_selection"),
            OracleShape::all_selected(),
        ));
    }
    cells
}

fn list_tier_c_cells(role: Option<&str>, smoke: bool) -> Result<(), String> {
    if let Some(role) = role
        && role != "training"
    {
        return Err(format!("Tier-C exposes only --role training, not {role:?}"));
    }
    for context_id in tier_c_context_ids(smoke) {
        context_by_id(context_id)?;
    }
    let cells = tier_c_cell_listing(smoke);
    let pair_count = cells.iter().filter(|(kind, _, _)| *kind == "pair").count();
    let single_count = cells.len() - pair_count;
    let expected = if smoke { (15, 12, 3) } else { (255, 240, 15) };
    if (cells.len(), pair_count, single_count) != expected {
        return Err(format!(
            "Tier-C listing drift: total={}, pairs={pair_count}, singles={single_count}",
            cells.len()
        ));
    }
    let cell_count = cells.len();
    for (kind, id, shape) in cells {
        let summary = shape.summary();
        let selected_span = summary.last_selected_row_exclusive - summary.first_selected_row;
        println!(
            "training\t{kind}\t{id}\t{}\tT={}\tW={}\tS={}",
            shape_invariant_sha256(&shape),
            summary.internal_transition_count(),
            selected_span - summary.selected_rows,
            summary.selected_rows,
        );
    }
    eprintln!(
        "listed {} Tier-C {} cells: {pair_count} forced pairs + {single_count} no-selection singles; {} RG units; factor-rank=4 scaled-condition=12.746109456977376",
        cell_count,
        if smoke { "C0 smoke" } else { "C1 training" },
        cell_count * ORACLE_ROW_GROUPS
    );
    Ok(())
}

fn run_tier_c(
    runtime: &Runtime,
    options: &Options,
    smoke: bool,
    rows: &mut Vec<CsvRow>,
) -> Result<(), String> {
    let prefix = if smoke { "TC/C0" } else { "TC/C1" };
    for context_id in tier_c_context_ids(smoke) {
        let context = context_by_id(context_id)?;
        let fixture = build_oracle_fixture(context, None)
            .map_err(|error| format!("cannot build Tier-C context {context_id}: {error}"))?;
        for (group, shape) in tier_c_pair_shapes(smoke) {
            run_named_pair_fixture_shape(
                runtime,
                options,
                &fixture,
                group,
                "training",
                tier_c_cell_id(prefix, group, context_id, &shape),
                "Skip",
                None,
                &shape,
                OracleSelectionSource::External,
                "selectors",
                rows,
            )?;
        }
        run_named_no_selection(
            runtime,
            options,
            &fixture,
            "TC-A",
            "training",
            format!("{prefix}/TC-A/{context_id}/no_selection"),
            "Skip",
            None,
            rows,
        )?;
    }
    validate_tier_c_rows(rows, smoke)
}

fn validate_tier_c_rows(rows: &[CsvRow], smoke: bool) -> Result<(), String> {
    if rows.iter().any(|row| {
        row.role != "training"
            || matches!(row.context.id, "H1" | "H2" | "H3" | "H4")
            || row.measurement.arm == OracleArm::Auto
    }) {
        return Err("Tier-C leaked a non-training context, role, or Auto arm".to_string());
    }
    let expected = tier_c_cell_listing(smoke);
    let expected_cells = expected
        .iter()
        .map(|(_, id, _)| id.clone())
        .collect::<BTreeSet<_>>();
    let actual_cells = rows
        .iter()
        .map(|row| row.cell_id.clone())
        .collect::<BTreeSet<_>>();
    if actual_cells != expected_cells {
        return Err("Tier-C executed cell IDs differ from the frozen listing".to_string());
    }
    let expected_digests = expected
        .iter()
        .map(|(_, id, shape)| (id.as_str(), shape_invariant_sha256(shape)))
        .collect::<BTreeMap<_, _>>();
    let mut arms_by_cell = BTreeMap::<&str, BTreeSet<&str>>::new();
    let mut arms_by_unit = BTreeMap::<(&str, usize), BTreeSet<&str>>::new();
    for row in rows {
        let expected_digest = expected_digests
            .get(row.cell_id.as_str())
            .ok_or_else(|| format!("unexpected Tier-C cell {}", row.cell_id))?;
        if &row.shape_invariant_sha256 != expected_digest {
            return Err(format!(
                "Tier-C shape invariant digest drift for {}",
                row.cell_id
            ));
        }
        arms_by_cell
            .entry(&row.cell_id)
            .or_default()
            .insert(row.measurement.arm.label());
        arms_by_unit
            .entry((&row.cell_id, row.row_group_index))
            .or_default()
            .insert(row.measurement.arm.label());
        match row.group {
            "TC-R" => {
                if row.shape.selected_rows != ROWS_PER_GROUP / 2
                    || row.shape.first_selected_row != 0
                    || row.shape.last_selected_row_exclusive != ROWS_PER_GROUP
                {
                    return Err(format!("invalid Tier-C R invariant for {}", row.cell_id));
                }
            }
            "TC-W" => {
                if row.shape.selected_rows != 2_048
                    || row.shape.first_selected_row != 0
                    || row.shape.internal_skip_run_count() != 1
                    || row.shape.internal_transition_count() != 2
                {
                    return Err(format!("invalid Tier-C W invariant for {}", row.cell_id));
                }
            }
            "TC-F" => {
                if row.shape.first_selected_row != 0
                    || row.shape.last_selected_row_exclusive != ROWS_PER_GROUP
                    || row.shape.selected_run_count != 64
                    || row.shape.internal_skip_run_count() != 63
                    || row.shape.internal_transition_count() != 126
                {
                    return Err(format!("invalid Tier-C F invariant for {}", row.cell_id));
                }
            }
            "TC-A" => {}
            other => return Err(format!("unexpected Tier-C group {other}")),
        }
    }
    let pair_count = arms_by_cell
        .values()
        .filter(|arms| arms.len() == 2 && arms.contains("mask") && arms.contains("selectors"))
        .count();
    let single_count = arms_by_cell
        .values()
        .filter(|arms| arms.len() == 1 && arms.contains("no_selection"))
        .count();
    let expected_counts = if smoke {
        (15, 12, 3, 108)
    } else {
        (255, 240, 15, 1_980)
    };
    if (arms_by_cell.len(), pair_count, single_count, rows.len()) != expected_counts {
        return Err(format!(
            "Tier-C output drift: cells={}, pairs={pair_count}, singles={single_count}, rows={}",
            arms_by_cell.len(),
            rows.len()
        ));
    }
    if arms_by_unit.len() != expected_counts.0 * ORACLE_ROW_GROUPS {
        return Err(format!(
            "Tier-C RG-unit drift: expected {}, got {}",
            expected_counts.0 * ORACLE_ROW_GROUPS,
            arms_by_unit.len()
        ));
    }
    let prefix = if smoke { "TC/C0" } else { "TC/C1" };
    for context_id in tier_c_context_ids(smoke) {
        for row_group_index in 0..ORACLE_ROW_GROUPS {
            let no_selection = rows
                .iter()
                .find(|row| {
                    row.cell_id == format!("{prefix}/TC-A/{context_id}/no_selection")
                        && row.row_group_index == row_group_index
                })
                .ok_or_else(|| format!("missing Tier-C no-selection {context_id}"))?;
            let all_true = rows
                .iter()
                .find(|row| {
                    row.cell_id == format!("{prefix}/TC-A/{context_id}/all_selected")
                        && row.row_group_index == row_group_index
                        && row.measurement.arm == OracleArm::Selectors
                })
                .ok_or_else(|| format!("missing Tier-C all-selected {context_id}"))?;
            if no_selection.measurement.content != all_true.measurement.content {
                return Err(format!(
                    "Tier-C no-selection/all-selected mismatch for {context_id}/rg{row_group_index}"
                ));
            }
        }
    }
    Ok(())
}

fn tier_d_context_ids(smoke: bool) -> &'static [&'static str] {
    if smoke {
        TIER_D_D0_CONTEXT_IDS
    } else {
        TIER_D_D1_CONTEXT_IDS
    }
}

fn tier_d_role(smoke: bool) -> &'static str {
    if smoke {
        "training-contract"
    } else {
        "training"
    }
}

fn tier_d_shapes_for_context(context_id: &str, smoke: bool) -> Vec<(&'static str, OracleShape)> {
    if smoke {
        return vec![
            (
                "TD-T",
                OracleShape::tier_d_transition_selectivity(512, 32_768),
            ),
            (
                "TD-F",
                OracleShape::tier_d_transition_selectivity(512, 8_192),
            ),
        ];
    }
    let mut shapes = TIER_D_TRANSITION_RUNS
        .iter()
        .map(|selected_runs| {
            (
                "TD-T",
                OracleShape::tier_d_transition_selectivity(*selected_runs, 32_768),
            )
        })
        .collect::<Vec<_>>();
    shapes.extend(
        TIER_D_SELECTED_ROWS
            .iter()
            .filter(|selected_rows| **selected_rows != 32_768)
            .map(|selected_rows| {
                (
                    "TD-F",
                    OracleShape::tier_d_transition_selectivity(512, *selected_rows),
                )
            }),
    );
    if TIER_D_INTERACTION_CONTEXT_IDS.contains(&context_id) {
        for selected_runs in [384, 640] {
            for selected_rows in [8_192, 57_344] {
                shapes.push((
                    "TD-X",
                    OracleShape::tier_d_transition_selectivity(selected_runs, selected_rows),
                ));
            }
        }
    }
    shapes
}

fn tier_d_cell_listing(smoke: bool) -> Vec<(&'static str, String, OracleShape)> {
    let prefix = if smoke { "TD/D0" } else { "TD/D1" };
    let mut cells = Vec::new();
    for context_id in tier_d_context_ids(smoke) {
        for (group, shape) in tier_d_shapes_for_context(context_id, smoke) {
            cells.push((
                group,
                format!("{prefix}/{group}/{context_id}/{}", shape.name),
                shape,
            ));
        }
    }
    cells
}

fn list_tier_d_cells(role: Option<&str>, smoke: bool) -> Result<(), String> {
    let expected_role = tier_d_role(smoke);
    if let Some(role) = role
        && role != expected_role
    {
        return Err(format!(
            "Tier-D {} exposes only --role {expected_role}, not {role:?}",
            if smoke { "D0" } else { "D1" }
        ));
    }
    for context_id in tier_d_context_ids(smoke) {
        context_by_id(context_id)?;
    }
    let cells = tier_d_cell_listing(smoke);
    let expected_count = if smoke { 8 } else { 134 };
    if cells.len() != expected_count {
        return Err(format!(
            "Tier-D listing drift: expected {expected_count}, got {}",
            cells.len()
        ));
    }
    let mut ids = BTreeSet::new();
    for (group, id, shape) in &cells {
        if !ids.insert(id) {
            return Err(format!("duplicate Tier-D cell {id}"));
        }
        let summary = shape.summary();
        println!(
            "{expected_role}\tpair\t{id}\t{group}\t{}\tT={}\tS={}\tf={:.9}",
            shape_invariant_sha256(shape),
            summary.internal_transition_count(),
            summary.selected_rows,
            summary.selected_fraction,
        );
    }
    eprintln!(
        "listed {} Tier-D {} forced-pair cells; {} RG units; {} CSV rows",
        cells.len(),
        if smoke { "D0 smoke" } else { "D1 boundary" },
        cells.len() * ORACLE_ROW_GROUPS,
        cells.len() * ORACLE_ROW_GROUPS * 2,
    );
    Ok(())
}

fn run_tier_d(
    runtime: &Runtime,
    options: &Options,
    smoke: bool,
    rows: &mut Vec<CsvRow>,
) -> Result<(), String> {
    let prefix = if smoke { "TD/D0" } else { "TD/D1" };
    let role = tier_d_role(smoke);
    for context_id in tier_d_context_ids(smoke) {
        let context = context_by_id(context_id)?;
        let fixture = build_oracle_fixture(context, None)
            .map_err(|error| format!("cannot build Tier-D context {context_id}: {error}"))?;
        for (group, shape) in tier_d_shapes_for_context(context_id, smoke) {
            run_named_pair_fixture_shape(
                runtime,
                options,
                &fixture,
                group,
                role,
                format!("{prefix}/{group}/{context_id}/{}", shape.name),
                "Skip",
                None,
                &shape,
                OracleSelectionSource::External,
                "selectors",
                rows,
            )?;
        }
    }
    validate_tier_d_rows(rows, smoke)
}

fn validate_tier_d_rows(rows: &[CsvRow], smoke: bool) -> Result<(), String> {
    let role = tier_d_role(smoke);
    if rows.iter().any(|row| {
        row.role != role
            || matches!(row.context.id, "H1" | "H2" | "H3" | "H4")
            || row.measurement.arm == OracleArm::Auto
    }) {
        return Err("Tier-D leaked a role, blind context, or Auto arm".to_string());
    }
    let expected = tier_d_cell_listing(smoke);
    let expected_cells = expected
        .iter()
        .map(|(_, id, _)| id.as_str())
        .collect::<BTreeSet<_>>();
    let actual_cells = rows
        .iter()
        .map(|row| row.cell_id.as_str())
        .collect::<BTreeSet<_>>();
    if actual_cells != expected_cells {
        return Err("Tier-D executed cell IDs differ from frozen listing".to_string());
    }
    let expected_shapes = expected
        .iter()
        .map(|(_, id, shape)| {
            (
                id.as_str(),
                (shape_invariant_sha256(shape), shape.summary()),
            )
        })
        .collect::<BTreeMap<_, _>>();
    let mut arms_by_cell = BTreeMap::<&str, BTreeSet<&str>>::new();
    let mut arms_by_unit = BTreeMap::<(&str, usize), BTreeSet<&str>>::new();
    for row in rows {
        let (digest, summary) = expected_shapes
            .get(row.cell_id.as_str())
            .ok_or_else(|| format!("unexpected Tier-D cell {}", row.cell_id))?;
        if row.shape_invariant_sha256.as_str() != digest.as_str()
            || row.shape.selected_rows != summary.selected_rows
            || row.shape.selected_run_count != summary.selected_run_count
            || row.shape.internal_transition_count() != summary.internal_transition_count()
        {
            return Err(format!("Tier-D shape contract drift for {}", row.cell_id));
        }
        if row.shape.first_selected_row != 0
            || row.shape.last_selected_row_exclusive != ROWS_PER_GROUP
            || row.shape.internal_skip_run_count() + 1 != row.shape.selected_run_count
            || row.metadata.base_mask_chunk_count == 0
            || row.metadata.base_mask_decoded_rows < row.shape.selected_rows
            || row.metadata.base_mask_decoded_rows > ROWS_PER_GROUP
            || row.source != OracleSelectionSource::External
            || row.backing != "selectors"
        {
            return Err(format!(
                "Tier-D execution invariant drift for {}",
                row.cell_id
            ));
        }
        arms_by_cell
            .entry(&row.cell_id)
            .or_default()
            .insert(row.measurement.arm.label());
        arms_by_unit
            .entry((&row.cell_id, row.row_group_index))
            .or_default()
            .insert(row.measurement.arm.label());
    }
    let expected_cells = if smoke { 8 } else { 134 };
    if arms_by_cell.len() != expected_cells
        || arms_by_unit.len() != expected_cells * ORACLE_ROW_GROUPS
        || rows.len() != expected_cells * ORACLE_ROW_GROUPS * 2
        || arms_by_cell
            .values()
            .any(|arms| arms != &BTreeSet::from(["mask", "selectors"]))
        || arms_by_unit
            .values()
            .any(|arms| arms != &BTreeSet::from(["mask", "selectors"]))
    {
        return Err("Tier-D cell/RG/arm/row count drift".to_string());
    }
    Ok(())
}

fn tier_d_d2_shapes() -> Vec<(&'static str, OracleShape)> {
    let mut shapes = TIER_D_D2_TRANSITION_RUNS
        .iter()
        .map(|selected_runs| {
            (
                "TD-T",
                OracleShape::tier_d_transition_selectivity(*selected_runs, 32_768),
            )
        })
        .collect::<Vec<_>>();
    shapes.extend(
        TIER_D_SELECTED_ROWS
            .iter()
            .filter(|selected_rows| **selected_rows != 32_768)
            .map(|selected_rows| {
                (
                    "TD-F",
                    OracleShape::tier_d_transition_selectivity(512, *selected_rows),
                )
            }),
    );
    for selected_runs in [384, 640] {
        for selected_rows in [8_192, 57_344] {
            shapes.push((
                "TD-X",
                OracleShape::tier_d_transition_selectivity(selected_runs, selected_rows),
            ));
        }
    }
    shapes
}

fn tier_d_d2_cell_listing() -> Vec<(&'static str, String, OracleShape)> {
    let mut cells = Vec::new();
    for context_id in TIER_D_D2_CONTEXT_IDS {
        for (group, shape) in tier_d_d2_shapes() {
            cells.push((
                group,
                format!("TD/D2/{group}/{context_id}/{}", shape.name),
                shape,
            ));
        }
    }
    cells
}

fn list_tier_d_d2_cells(role: Option<&str>) -> Result<(), String> {
    const ROLE: &str = "validation-blind";
    if let Some(role) = role
        && role != ROLE
    {
        return Err(format!(
            "Tier-D D2 exposes only --role {ROLE}, not {role:?}"
        ));
    }
    for context_id in TIER_D_D2_CONTEXT_IDS {
        context_by_id(context_id)?;
    }
    let cells = tier_d_d2_cell_listing();
    if cells.len() != 121 {
        return Err(format!(
            "Tier-D D2 listing drift: expected 121, got {}",
            cells.len()
        ));
    }
    let mut ids = BTreeSet::new();
    for (group, id, shape) in &cells {
        if !ids.insert(id) {
            return Err(format!("duplicate Tier-D D2 cell {id}"));
        }
        let summary = shape.summary();
        println!(
            "{ROLE}\tpair\t{id}\t{group}\t{}\tT={}\tS={}\tf={:.9}",
            shape_invariant_sha256(shape),
            summary.internal_transition_count(),
            summary.selected_rows,
            summary.selected_fraction,
        );
    }
    eprintln!(
        "listed 121 Tier-D D2 forced-pair cells; {} RG units; {} CSV rows",
        121 * ORACLE_ROW_GROUPS,
        121 * ORACLE_ROW_GROUPS * 2,
    );
    Ok(())
}

fn run_tier_d_d2(
    runtime: &Runtime,
    options: &Options,
    rows: &mut Vec<CsvRow>,
) -> Result<(), String> {
    for context_id in TIER_D_D2_CONTEXT_IDS {
        let context = context_by_id(context_id)?;
        let fixture = build_oracle_fixture(context, None)
            .map_err(|error| format!("cannot build Tier-D D2 context {context_id}: {error}"))?;
        for (group, shape) in tier_d_d2_shapes() {
            run_named_pair_fixture_shape(
                runtime,
                options,
                &fixture,
                group,
                "validation-blind",
                format!("TD/D2/{group}/{context_id}/{}", shape.name),
                "Skip",
                None,
                &shape,
                OracleSelectionSource::External,
                "selectors",
                rows,
            )?;
        }
    }
    validate_tier_d_d2_rows(rows)
}

fn validate_tier_d_d2_rows(rows: &[CsvRow]) -> Result<(), String> {
    if rows.iter().any(|row| {
        row.role != "validation-blind"
            || matches!(row.context.id, "H1" | "H2" | "H3" | "H4")
            || row.measurement.arm == OracleArm::Auto
    }) {
        return Err("Tier-D D2 leaked a role, H context, or Auto arm".to_string());
    }
    let expected = tier_d_d2_cell_listing();
    let expected_cells = expected
        .iter()
        .map(|(_, id, _)| id.as_str())
        .collect::<BTreeSet<_>>();
    let actual_cells = rows
        .iter()
        .map(|row| row.cell_id.as_str())
        .collect::<BTreeSet<_>>();
    if actual_cells != expected_cells {
        return Err("Tier-D D2 executed cell IDs differ from frozen listing".to_string());
    }
    let expected_shapes = expected
        .iter()
        .map(|(_, id, shape)| {
            (
                id.as_str(),
                (shape_invariant_sha256(shape), shape.summary()),
            )
        })
        .collect::<BTreeMap<_, _>>();
    let mut arms_by_cell = BTreeMap::<&str, BTreeSet<&str>>::new();
    let mut arms_by_unit = BTreeMap::<(&str, usize), BTreeSet<&str>>::new();
    let mut q_side_by_context_rg = BTreeMap::<(&str, usize), bool>::new();
    for row in rows {
        let (digest, summary) = expected_shapes
            .get(row.cell_id.as_str())
            .ok_or_else(|| format!("unexpected Tier-D D2 cell {}", row.cell_id))?;
        if row.shape_invariant_sha256.as_str() != digest.as_str()
            || row.shape.selected_rows != summary.selected_rows
            || row.shape.selected_run_count != summary.selected_run_count
            || row.shape.internal_transition_count() != summary.internal_transition_count()
        {
            return Err(format!(
                "Tier-D D2 shape contract drift for {}",
                row.cell_id
            ));
        }
        if row.shape.first_selected_row != 0
            || row.shape.last_selected_row_exclusive != ROWS_PER_GROUP
            || row.shape.internal_skip_run_count() + 1 != row.shape.selected_run_count
            || row.metadata.base_mask_chunk_count == 0
            || row.metadata.base_mask_decoded_rows < row.shape.selected_rows
            || row.metadata.base_mask_decoded_rows > ROWS_PER_GROUP
            || row.source != OracleSelectionSource::External
            || row.backing != "selectors"
        {
            return Err(format!(
                "Tier-D D2 execution invariant drift for {}",
                row.cell_id
            ));
        }
        let q_numerator = (row.metadata.projected_leaf_count as u64)
            .saturating_mul(row.metadata.compressed_bytes);
        let q_denominator =
            (ROWS_PER_GROUP as u64).saturating_mul(row.metadata.arrow_output_width_proxy);
        let q_at_or_above = q_numerator >= TIER_D_FROZEN_TAU_Q * q_denominator;
        let expected_q_at_or_above = matches!(row.context.id, "U2" | "U4" | "U6" | "U8" | "U10");
        if q_at_or_above != expected_q_at_or_above {
            return Err(format!(
                "Tier-D D2 Q exposure drift for {}/rg{}: expected_at_or_above={expected_q_at_or_above}",
                row.context.id, row.row_group_index
            ));
        }
        if let Some(previous) =
            q_side_by_context_rg.insert((row.context.id, row.row_group_index), q_at_or_above)
            && previous != q_at_or_above
        {
            return Err(format!(
                "Tier-D D2 Q side changed within {}/rg{}",
                row.context.id, row.row_group_index
            ));
        }
        arms_by_cell
            .entry(&row.cell_id)
            .or_default()
            .insert(row.measurement.arm.label());
        arms_by_unit
            .entry((&row.cell_id, row.row_group_index))
            .or_default()
            .insert(row.measurement.arm.label());
    }
    if arms_by_cell.len() != 121
        || arms_by_unit.len() != 121 * ORACLE_ROW_GROUPS
        || rows.len() != 121 * ORACLE_ROW_GROUPS * 2
        || q_side_by_context_rg.len() != TIER_D_D2_CONTEXT_IDS.len() * ORACLE_ROW_GROUPS
        || arms_by_cell
            .values()
            .any(|arms| arms != &BTreeSet::from(["mask", "selectors"]))
        || arms_by_unit
            .values()
            .any(|arms| arms != &BTreeSet::from(["mask", "selectors"]))
    {
        return Err("Tier-D D2 cell/RG/arm/Q-exposure/row count drift".to_string());
    }
    Ok(())
}

fn run_d1_training(
    runtime: &Runtime,
    options: &Options,
    replay: bool,
    rows: &mut Vec<CsvRow>,
) -> Result<(), String> {
    for context in training_contexts() {
        let fixture = build_oracle_fixture(context, None)
            .map_err(|error| format!("cannot build D1 context {}: {error}", context.id))?;
        for run_len in coarse_grid(context) {
            let shape = OracleShape::l_sweep(*run_len);
            run_named_pair_fixture_shape(
                runtime,
                options,
                &fixture,
                "X-A",
                "training",
                format!("D1/X-A/{}/{}", context.id, shape.name),
                "Skip",
                None,
                &shape,
                OracleSelectionSource::External,
                "selectors",
                rows,
            )?;
        }
        if replay {
            for run_len in adaptive_grid(context.id) {
                let shape = OracleShape::l_sweep(*run_len);
                run_named_pair_fixture_shape(
                    runtime,
                    options,
                    &fixture,
                    "X-A-adaptive",
                    "training",
                    format!("D1/X-A-adaptive/{}/{}", context.id, shape.name),
                    "Skip",
                    None,
                    &shape,
                    OracleSelectionSource::External,
                    "selectors",
                    rows,
                )?;
            }
        }
        for shape in boundary_shapes(context.id) {
            run_named_pair_fixture_shape(
                runtime,
                options,
                &fixture,
                "X-B",
                "training",
                format!("D1/X-B/{}/{}", context.id, shape.name),
                "Skip",
                None,
                &shape,
                OracleSelectionSource::External,
                "selectors",
                rows,
            )?;
        }

        run_named_no_selection(
            runtime,
            options,
            &fixture,
            "X-E",
            "training",
            format!("D1/X-E/{}/no_selection", context.id),
            "Skip",
            None,
            rows,
        )?;
        for shape in cost_anchor_shapes() {
            run_named_pair_fixture_shape(
                runtime,
                options,
                &fixture,
                "X-E",
                "training",
                format!("D1/X-E/{}/{}", context.id, shape.name),
                "Skip",
                None,
                &shape,
                OracleSelectionSource::External,
                "selectors",
                rows,
            )?;
        }

        if context.id == "C0" {
            let batch_fixture = fixture.with_batch_size(1_024);
            for run_len in [8, 32, 128] {
                let shape = OracleShape::l_sweep(run_len);
                run_named_pair_fixture_shape(
                    runtime,
                    options,
                    &batch_fixture,
                    "X-G",
                    "training",
                    format!("D1/X-G/C0/batch1024/{}", shape.name),
                    "Skip",
                    None,
                    &shape,
                    OracleSelectionSource::External,
                    "selectors",
                    rows,
                )?;
            }
        }
    }

    for context_id in ["C0", "C4"] {
        let page_context = context_by_id(context_id)?.with_page_index();
        let required = build_oracle_fixture(page_context, None)
            .map_err(|error| format!("cannot build D1 page context {context_id}: {error}"))?;
        let skip = required
            .with_page_index_policy(PageIndexPolicy::Skip)
            .map_err(|error| format!("cannot create D1 same-Bytes Skip view: {error}"))?;
        if required.bytes_sha256() != skip.bytes_sha256() {
            return Err(format!("D1 {context_id} page views have different bytes"));
        }
        for (fixture, policy) in [(&required, "Required"), (&skip, "Skip")] {
            for shape in [OracleShape::l_sweep(64), OracleShape::page_matched_bursty()] {
                run_named_pair_fixture_shape(
                    runtime,
                    options,
                    fixture,
                    "X-C",
                    "training",
                    format!("D1/X-C/{context_id}/page-{policy}/{}", shape.name),
                    policy,
                    Some(ORACLE_PAGE_ROWS),
                    &shape,
                    OracleSelectionSource::External,
                    "selectors",
                    rows,
                )?;
            }
        }
        validate_page_exposure_for(rows, "X-C", context_id)?;
    }
    validate_d1_rows(rows, replay)
}

fn validate_d1_rows(rows: &[CsvRow], replay: bool) -> Result<(), String> {
    if rows.iter().any(|row| {
        !row.role.eq("training")
            || matches!(row.context.id, "H1" | "H2" | "H3" | "H4")
            || row.measurement.arm == OracleArm::Auto
    }) {
        return Err("D1 matrix leaked a non-training context, role, or Auto arm".to_string());
    }
    let expected_cells = d1_cell_listing(replay)
        .into_iter()
        .map(|(_, id)| id)
        .collect::<BTreeSet<_>>();
    let actual_cells = rows
        .iter()
        .map(|row| row.cell_id.clone())
        .collect::<BTreeSet<_>>();
    if actual_cells != expected_cells {
        return Err("D1 executed cell IDs differ from the frozen listing".to_string());
    }
    let mut arms_by_cell = BTreeMap::<&str, BTreeSet<&str>>::new();
    let mut arms_by_unit = BTreeMap::<(&str, usize), BTreeSet<&str>>::new();
    for row in rows {
        arms_by_cell
            .entry(&row.cell_id)
            .or_default()
            .insert(row.measurement.arm.label());
        arms_by_unit
            .entry((&row.cell_id, row.row_group_index))
            .or_default()
            .insert(row.measurement.arm.label());
    }
    let pair_count = arms_by_cell
        .values()
        .filter(|arms| arms.len() == 2 && arms.contains("selectors") && arms.contains("mask"))
        .count();
    let single_count = arms_by_cell
        .values()
        .filter(|arms| arms.len() == 1 && arms.contains("no_selection"))
        .count();
    let expected_counts = if replay {
        (305, 285, 20, 2_360)
    } else {
        (289, 269, 20, 2_232)
    };
    if (arms_by_cell.len(), pair_count, single_count, rows.len()) != expected_counts {
        return Err(format!(
            "D1 output drift: cells={}, pairs={pair_count}, singles={single_count}, rows={}",
            arms_by_cell.len(),
            rows.len()
        ));
    }
    let expected_units = if replay { 1_220 } else { 1_156 };
    if arms_by_unit.len() != expected_units {
        return Err(format!(
            "D1 RG-unit drift: expected {expected_units}, got {}",
            arms_by_unit.len()
        ));
    }
    for context in training_contexts() {
        for row_group_index in 0..ORACLE_ROW_GROUPS {
            let no_selection = rows
                .iter()
                .find(|row| {
                    row.cell_id == format!("D1/X-E/{}/no_selection", context.id)
                        && row.row_group_index == row_group_index
                })
                .ok_or_else(|| format!("missing no-selection digest for {}", context.id))?;
            let all_true = rows
                .iter()
                .find(|row| {
                    row.cell_id == format!("D1/X-E/{}/all_selected", context.id)
                        && row.row_group_index == row_group_index
                        && row.measurement.arm == OracleArm::Selectors
                })
                .ok_or_else(|| format!("missing all-true digest for {}", context.id))?;
            if no_selection.measurement.content != all_true.measurement.content {
                return Err(format!(
                    "D1 no-selection/all-true content mismatch for {}/rg{}",
                    context.id, row_group_index
                ));
            }
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_fixture_shape(
    runtime: &Runtime,
    options: &Options,
    fixture: &OracleFixture,
    group: &'static str,
    role: &'static str,
    metadata_policy: &'static str,
    page_layout_rows: Option<usize>,
    shape: &OracleShape,
    rows: &mut Vec<CsvRow>,
) -> Result<(), String> {
    let context = fixture.context();
    let cell_id = if page_layout_rows.is_some() {
        format!("D0/{}/page-{metadata_policy}/{}", context.id, shape.name)
    } else {
        format!("D0/{}/{}", context.id, shape.name)
    };
    run_named_pair_fixture_shape(
        runtime,
        options,
        fixture,
        group,
        role,
        cell_id,
        metadata_policy,
        page_layout_rows,
        shape,
        OracleSelectionSource::External,
        "selectors",
        rows,
    )
}

#[allow(clippy::too_many_arguments)]
fn run_named_pair_fixture_shape(
    runtime: &Runtime,
    options: &Options,
    fixture: &OracleFixture,
    group: &'static str,
    role: &'static str,
    cell_id: String,
    metadata_policy: &'static str,
    page_layout_rows: Option<usize>,
    shape: &OracleShape,
    source: OracleSelectionSource,
    backing: &'static str,
    rows: &mut Vec<CsvRow>,
) -> Result<(), String> {
    let fixture_sha256 = fixture.bytes_sha256();
    let context = fixture.context();
    let shape_invariant_sha256 = shape_invariant_sha256(shape);
    eprintln!("measuring {cell_id}");
    for row_group_index in 0..ORACLE_ROW_GROUPS {
        let metadata = metadata_summary(fixture, row_group_index, shape)?;
        let pair = measure_pair(
            runtime,
            options.samples,
            fixture,
            row_group_index,
            shape,
            source,
            &cell_id,
        )?;
        for measurement in [pair.selectors, pair.mask] {
            rows.push(CsvRow {
                group,
                role,
                cell_id: cell_id.clone(),
                context,
                metadata_policy,
                page_layout_rows,
                fixture_sha256: fixture_sha256.clone(),
                row_group_index,
                shape_name: shape.name.clone(),
                shape_invariant_sha256: shape_invariant_sha256.clone(),
                shape: shape.summary(),
                source,
                backing,
                metadata: metadata.clone(),
                measurement,
            });
        }
    }
    Ok(())
}

fn measure_pair(
    runtime: &Runtime,
    samples: usize,
    fixture: &OracleFixture,
    row_group_index: usize,
    shape: &OracleShape,
    source: OracleSelectionSource,
    cell: &str,
) -> Result<PairMeasurement, String> {
    if source == OracleSelectionSource::None {
        return Err("forced pair cannot use no-selection source".to_string());
    }
    let selection =
        || (source == OracleSelectionSource::External).then(|| shape.selection_for_row_group());
    let selectors_check = runtime.block_on(run_oracle_row_group(
        fixture,
        row_group_index,
        selection(),
        source,
        OracleArm::Selectors,
        true,
    ));
    let mask_check = runtime.block_on(run_oracle_row_group(
        fixture,
        row_group_index,
        selection(),
        source,
        OracleArm::Mask,
        true,
    ));
    assert_equivalent(cell, row_group_index, shape, &selectors_check, &mask_check)?;

    for _ in 0..WARMUPS_PER_ARM {
        for arm in [OracleArm::Selectors, OracleArm::Mask] {
            let result = runtime.block_on(run_oracle_row_group(
                fixture,
                row_group_index,
                selection(),
                source,
                arm,
                false,
            ));
            if result.row_count != selectors_check.row_count {
                return Err(format!(
                    "{cell}/rg{row_group_index}/{arm:?} warmup row mismatch"
                ));
            }
        }
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
            let timestamp = unix_nanos();
            let started = Instant::now();
            let result = runtime.block_on(run_oracle_row_group(
                fixture,
                row_group_index,
                selection(),
                source,
                arm,
                false,
            ));
            let elapsed = started.elapsed().as_nanos() as u64;
            if result.row_count != selectors_check.row_count {
                return Err(format!(
                    "{cell}/rg{row_group_index}/{arm:?} timed row mismatch"
                ));
            }
            hint::black_box(result.row_count);
            target.push(elapsed);
            timestamps.push(timestamp);
        }
    }

    Ok(PairMeasurement {
        selectors: arm_measurement(
            OracleArm::Selectors,
            selector_samples,
            selector_timestamps,
            selectors_check,
        )?,
        mask: arm_measurement(OracleArm::Mask, mask_samples, mask_timestamps, mask_check)?,
    })
}

#[allow(clippy::too_many_arguments)]
fn run_named_no_selection(
    runtime: &Runtime,
    options: &Options,
    fixture: &OracleFixture,
    group: &'static str,
    role: &'static str,
    cell_id: String,
    metadata_policy: &'static str,
    page_layout_rows: Option<usize>,
    rows: &mut Vec<CsvRow>,
) -> Result<(), String> {
    let shape = OracleShape::all_selected();
    let shape_invariant_sha256 = shape_invariant_sha256(&shape);
    let fixture_sha256 = fixture.bytes_sha256();
    let context = fixture.context();
    eprintln!("measuring {cell_id}");
    for row_group_index in 0..ORACLE_ROW_GROUPS {
        let metadata = metadata_summary(fixture, row_group_index, &shape)?;
        let check = runtime.block_on(run_oracle_row_group(
            fixture,
            row_group_index,
            None,
            OracleSelectionSource::None,
            OracleArm::NoSelection,
            true,
        ));
        if check.row_count != ROWS_PER_GROUP {
            return Err(format!(
                "{cell_id}/rg{row_group_index}: no-selection returned {} rows",
                check.row_count
            ));
        }
        for _ in 0..WARMUPS_PER_ARM {
            let result = runtime.block_on(run_oracle_row_group(
                fixture,
                row_group_index,
                None,
                OracleSelectionSource::None,
                OracleArm::NoSelection,
                false,
            ));
            if result.row_count != ROWS_PER_GROUP {
                return Err(format!(
                    "{cell_id}/rg{row_group_index}: no-selection warmup row mismatch"
                ));
            }
        }
        let mut samples_ns = Vec::with_capacity(options.samples);
        let mut timestamps = Vec::with_capacity(options.samples);
        for _ in 0..options.samples {
            timestamps.push(unix_nanos());
            let started = Instant::now();
            let result = runtime.block_on(run_oracle_row_group(
                fixture,
                row_group_index,
                None,
                OracleSelectionSource::None,
                OracleArm::NoSelection,
                false,
            ));
            let elapsed = started.elapsed().as_nanos() as u64;
            if result.row_count != ROWS_PER_GROUP {
                return Err(format!(
                    "{cell_id}/rg{row_group_index}: no-selection timed row mismatch"
                ));
            }
            hint::black_box(result.row_count);
            samples_ns.push(elapsed);
        }
        let measurement = arm_measurement(OracleArm::NoSelection, samples_ns, timestamps, check)?;
        rows.push(CsvRow {
            group,
            role,
            cell_id: cell_id.clone(),
            context,
            metadata_policy,
            page_layout_rows,
            fixture_sha256: fixture_sha256.clone(),
            row_group_index,
            shape_name: "no_selection".to_string(),
            shape_invariant_sha256: shape_invariant_sha256.clone(),
            shape: shape.summary(),
            source: OracleSelectionSource::None,
            backing: "none",
            metadata,
            measurement,
        });
    }
    Ok(())
}

fn assert_equivalent(
    cell: &str,
    row_group_index: usize,
    shape: &OracleShape,
    selectors: &OracleRowGroupRunResult,
    mask: &OracleRowGroupRunResult,
) -> Result<(), String> {
    let expected = shape.summary().selected_rows;
    if selectors.row_count != expected || mask.row_count != expected {
        return Err(format!(
            "{cell}/rg{row_group_index}: expected {expected} rows, selectors={}, mask={}",
            selectors.row_count, mask.row_count
        ));
    }
    if selectors.content != mask.content {
        return Err(format!(
            "{cell}/rg{row_group_index}: projected leaf digests differ between arms"
        ));
    }
    Ok(())
}

fn arm_measurement(
    arm: OracleArm,
    samples_ns: Vec<u64>,
    sample_started_unix_ns: Vec<u64>,
    check: OracleRowGroupRunResult,
) -> Result<ArmMeasurement, String> {
    let median_ns = median(&samples_ns);
    let deviations = samples_ns
        .iter()
        .map(|sample| sample.abs_diff(median_ns))
        .collect::<Vec<_>>();
    Ok(ArmMeasurement {
        arm,
        samples_ns,
        sample_started_unix_ns,
        median_ns,
        mad_ns: median(&deviations),
        rows_out: check.row_count,
        content: check
            .content
            .ok_or_else(|| "attribution pass did not produce content digest".to_string())?,
        requested_ranges: check.requested_ranges,
        requested_bytes: check.requested_bytes,
    })
}

fn metadata_summary(
    fixture: &OracleFixture,
    row_group_index: usize,
    shape: &OracleShape,
) -> Result<MetadataSummary, String> {
    let context = fixture.context();
    let predicate_columns = usize::from(fixture.has_predicate_column());
    let row_group = fixture
        .metadata()
        .row_groups()
        .get(row_group_index)
        .ok_or_else(|| format!("missing row group {row_group_index}"))?;
    let projected_columns = predicate_columns..predicate_columns + context.payload_columns;
    let mut physical_types = BTreeMap::<String, usize>::new();
    let mut encoding_sets = Vec::new();
    let mut dictionary_leaf_count = 0usize;
    let mut compressed_bytes = 0u64;
    let mut uncompressed_bytes = 0u64;
    let mut per_leaf_compressed_bytes = Vec::new();
    let mut per_leaf_uncompressed_bytes = Vec::new();
    let mut per_leaf_num_values = Vec::new();
    let mut num_values_min = i64::MAX;
    let mut num_values_max = i64::MIN;
    for column_idx in projected_columns.clone() {
        let column = row_group.column(column_idx);
        *physical_types
            .entry(format!("{:?}", column.column_type()))
            .or_default() += 1;
        let mut encodings = column
            .encodings()
            .map(|encoding| format!("{encoding:?}"))
            .collect::<Vec<_>>();
        encodings.sort();
        encoding_sets.push(encodings.join("+"));
        dictionary_leaf_count += usize::from(column.dictionary_page_offset().is_some());
        compressed_bytes = compressed_bytes.saturating_add(column.compressed_size() as u64);
        uncompressed_bytes = uncompressed_bytes.saturating_add(column.uncompressed_size() as u64);
        per_leaf_compressed_bytes.push(column.compressed_size().to_string());
        per_leaf_uncompressed_bytes.push(column.uncompressed_size().to_string());
        per_leaf_num_values.push(column.num_values().to_string());
        num_values_min = num_values_min.min(column.num_values());
        num_values_max = num_values_max.max(column.num_values());
    }

    let page = page_summary(fixture, row_group_index, shape, projected_columns)?;
    let (base_mask_chunk_count, base_mask_decoded_rows) =
        simulate_mask_chunks(shape, context.batch_size, None)?;
    let (mask_chunk_count, mask_decoded_rows) =
        simulate_mask_chunks(shape, context.batch_size, page.loaded_ranges.as_deref())?;
    let base_run_count = effective_run_count(shape, None);
    let loaded_run_count = effective_run_count(shape, page.loaded_ranges.as_deref());
    let physical_type_histogram = physical_types
        .into_iter()
        .map(|(kind, count)| format!("{kind}:{count}"))
        .collect::<Vec<_>>()
        .join("|");
    Ok(MetadataSummary {
        projected_leaf_count: context.payload_columns,
        projection_signature: (0..context.payload_columns)
            .map(|column_idx| context.payload_at(column_idx).dtype())
            .collect::<Vec<_>>()
            .join("|"),
        physical_type_histogram,
        encoding_sets: encoding_sets.join("|"),
        dictionary_leaf_count,
        compressed_bytes,
        uncompressed_bytes,
        per_leaf_compressed_bytes: per_leaf_compressed_bytes.join("|"),
        per_leaf_uncompressed_bytes: per_leaf_uncompressed_bytes.join("|"),
        per_leaf_num_values: per_leaf_num_values.join("|"),
        compression_ratio: compressed_bytes as f64 / uncompressed_bytes.max(1) as f64,
        num_values_min,
        num_values_max,
        num_values_consistent: num_values_min == ROWS_PER_GROUP as i64
            && num_values_max == ROWS_PER_GROUP as i64,
        arrow_output_width_proxy: (0..context.payload_columns)
            .map(|column_idx| output_width_proxy(context.payload_at(column_idx)))
            .sum(),
        loaded_page_rows: page
            .loaded_ranges
            .as_ref()
            .map(|ranges| ranges.iter().map(|range| range.end - range.start).sum()),
        loaded_range_count: page.loaded_ranges.as_ref().map(Vec::len),
        per_leaf_page_rows: page.per_leaf_page_rows,
        per_leaf_page_first_rows: page.per_leaf_page_first_rows,
        per_leaf_predicted_fetched_bytes: page.per_leaf_predicted_fetched_bytes,
        predicted_fetched_bytes: page.predicted_fetched_bytes,
        base_mask_chunk_count,
        base_mask_decoded_rows,
        base_run_count,
        mask_chunk_count,
        mask_decoded_rows,
        loaded_run_count,
    })
}

struct PageSummary {
    loaded_ranges: Option<Vec<Range<usize>>>,
    per_leaf_page_rows: Option<String>,
    per_leaf_page_first_rows: Option<String>,
    per_leaf_predicted_fetched_bytes: Option<String>,
    predicted_fetched_bytes: Option<u64>,
}

fn page_summary(
    fixture: &OracleFixture,
    row_group_index: usize,
    shape: &OracleShape,
    projected_columns: Range<usize>,
) -> Result<PageSummary, String> {
    let Some(offset_index) = fixture.metadata().offset_index() else {
        return Ok(PageSummary {
            loaded_ranges: None,
            per_leaf_page_rows: None,
            per_leaf_page_first_rows: None,
            per_leaf_predicted_fetched_bytes: None,
            predicted_fetched_bytes: None,
        });
    };
    let row_group_indexes = offset_index
        .get(row_group_index)
        .ok_or_else(|| format!("missing offset-index row group {row_group_index}"))?;
    let row_group = &fixture.metadata().row_groups()[row_group_index];
    let selection = shape.selection_for_row_group();
    let mut per_leaf_ranges = Vec::new();
    let mut per_leaf_rows = Vec::new();
    let mut per_leaf_page_first_rows = Vec::new();
    let mut per_leaf_predicted_fetched_bytes = Vec::new();
    let mut predicted_fetched_bytes = 0u64;
    for column_idx in projected_columns {
        let locations = row_group_indexes
            .get(column_idx)
            .ok_or_else(|| format!("missing offset index for leaf {column_idx}"))?
            .page_locations();
        let selected_starts = selection
            .scan_ranges(locations)
            .into_iter()
            .map(|range| range.start)
            .collect::<BTreeSet<_>>();
        let mut row_ranges = Vec::new();
        let mut selected_page_bytes = 0u64;
        for (page_idx, page) in locations.iter().enumerate() {
            if !selected_starts.contains(&(page.offset as u64)) {
                continue;
            }
            let end = locations
                .get(page_idx + 1)
                .map(|next| next.first_row_index as usize)
                .unwrap_or(ROWS_PER_GROUP);
            row_ranges.push(page.first_row_index as usize..end);
            selected_page_bytes = selected_page_bytes
                .saturating_add(u64::try_from(page.compressed_page_size).unwrap());
        }
        let column = row_group.column(column_idx);
        if !row_ranges.is_empty()
            && let Some(dictionary_offset) = column.dictionary_page_offset()
            && let Some(first_page) = locations.first()
        {
            selected_page_bytes = selected_page_bytes.saturating_add(
                u64::try_from(first_page.offset.saturating_sub(dictionary_offset)).unwrap(),
            );
        }
        predicted_fetched_bytes = predicted_fetched_bytes.saturating_add(selected_page_bytes);
        per_leaf_predicted_fetched_bytes.push(selected_page_bytes.to_string());
        per_leaf_rows.push(
            row_ranges
                .iter()
                .map(|range| range.end - range.start)
                .sum::<usize>()
                .to_string(),
        );
        per_leaf_page_first_rows.push(
            row_ranges
                .iter()
                .map(|range| range.start.to_string())
                .collect::<Vec<_>>()
                .join("+"),
        );
        per_leaf_ranges.push(row_ranges);
    }
    let loaded_ranges = per_leaf_ranges
        .into_iter()
        .reduce(|left, right| intersect_ranges(&left, &right))
        .unwrap_or_default();
    Ok(PageSummary {
        loaded_ranges: Some(loaded_ranges),
        per_leaf_page_rows: Some(per_leaf_rows.join("|")),
        per_leaf_page_first_rows: Some(per_leaf_page_first_rows.join("|")),
        per_leaf_predicted_fetched_bytes: Some(per_leaf_predicted_fetched_bytes.join("|")),
        predicted_fetched_bytes: Some(predicted_fetched_bytes),
    })
}

fn intersect_ranges(left: &[Range<usize>], right: &[Range<usize>]) -> Vec<Range<usize>> {
    let mut intersections = Vec::new();
    let mut left_idx = 0usize;
    let mut right_idx = 0usize;
    while left_idx < left.len() && right_idx < right.len() {
        let start = left[left_idx].start.max(right[right_idx].start);
        let end = left[left_idx].end.min(right[right_idx].end);
        if start < end {
            intersections.push(start..end);
        }
        if left[left_idx].end <= right[right_idx].end {
            left_idx += 1;
        } else {
            right_idx += 1;
        }
    }
    intersections
}

fn simulate_mask_chunks(
    shape: &OracleShape,
    batch_size: usize,
    loaded_ranges: Option<&[Range<usize>]>,
) -> Result<(usize, usize), String> {
    let mask = shape.selected_mask();
    let end = mask
        .iter()
        .rposition(|selected| *selected)
        .map(|idx| idx + 1)
        .ok_or_else(|| "Tier-B shapes must select at least one row".to_string())?;
    let mut position = 0usize;
    let mut chunks = 0usize;
    let mut decoded_rows = 0usize;
    while position < end {
        while position < end && !mask[position] {
            position += 1;
        }
        if position == end {
            break;
        }
        let range_end = if let Some(ranges) = loaded_ranges {
            ranges
                .iter()
                .find(|range| range.contains(&position))
                .map(|range| range.end)
                .ok_or_else(|| format!("selected row {position} has no loaded page range"))?
        } else {
            end
        };
        let chunk_start = position;
        let mut selected_rows = 0usize;
        while position < end && position < range_end && selected_rows < batch_size {
            selected_rows += usize::from(mask[position]);
            position += 1;
        }
        chunks += 1;
        decoded_rows += position - chunk_start;
    }
    Ok((chunks, decoded_rows))
}

fn effective_run_count(shape: &OracleShape, loaded_ranges: Option<&[Range<usize>]>) -> usize {
    let mask = shape.selected_mask();
    let end = mask
        .iter()
        .rposition(|selected| *selected)
        .map(|idx| idx + 1)
        .expect("Tier-B shapes must select at least one row");
    let full_range = [0..end];
    let ranges = loaded_ranges.unwrap_or(&full_range);
    let mut runs = 0usize;
    for range in ranges {
        let start = range.start.min(end);
        let range_end = range.end.min(end);
        if start >= range_end {
            continue;
        }
        runs += 1;
        runs += mask[start..range_end]
            .windows(2)
            .filter(|pair| pair[0] != pair[1])
            .count();
    }
    runs
}

fn output_width_proxy(payload: OraclePayload) -> u64 {
    match payload {
        OraclePayload::Int32 => 4,
        OraclePayload::Int64 | OraclePayload::Float64 => 8,
        OraclePayload::Utf8View8 | OraclePayload::Utf8View32 | OraclePayload::Utf8View64 => 16,
        OraclePayload::Utf8Dictionary1k => 13,
    }
}

fn shape_invariant_sha256(shape: &OracleShape) -> String {
    let mut hasher = Sha256::new();
    hasher.update(shape.invariant_material().as_bytes());
    format!("sha256:{}", hex_digest(&hasher.finalize()))
}

fn experiment_contract(matrix: &str) -> Value {
    if is_tier_d_matrix(matrix) {
        return tier_d_experiment_contract(matrix);
    }
    if !is_tier_c_matrix(matrix) {
        return candidate_library();
    }
    let smoke = matrix == MATRIX_TIER_C_C0_SMOKE;
    let shapes = tier_c_pair_shapes(smoke)
        .into_iter()
        .map(|(group, shape)| {
            let summary = shape.summary();
            let selected_span = summary.last_selected_row_exclusive - summary.first_selected_row;
            json!({
                "group": group,
                "shape_name": shape.name,
                "shape_invariant_sha256": shape_invariant_sha256(&shape),
                "selected_span_rows": selected_span,
                "selected_rows": summary.selected_rows,
                "wasted_span_rows": selected_span - summary.selected_rows,
                "selected_run_count": summary.selected_run_count,
                "internal_skip_run_count": summary.internal_skip_run_count(),
                "internal_transition_count": summary.internal_transition_count()
            })
        })
        .collect::<Vec<_>>();
    json!({
        "schema_version": "arrow-row-selection-tier-c-estimand-contract-v1",
        "matrix": matrix,
        "status": "diagnostic-opened-contexts-only",
        "production_candidate": Value::Null,
        "blind_timing_opened": false,
        "blind_contexts_constructed": false,
        "context_ids": tier_c_context_ids(smoke),
        "shape_families": shapes,
        "diagnostic_surface": {
            "formula": "delta_ns = alpha0 + alphaT*T + alphaW*(M-S) + alphaS*S",
            "context_id_is_production_input": false,
            "predictors": ["1", "internal_transition_count", "selected_span_rows-selected_rows", "selected_rows"],
            "factor_matrix_rank": 4,
            "positive_median_scaled_condition_number": 12.746109456977376_f64
        },
        "freeze_policy": "no production formula or H1-H4 timing may be unlocked by this contract"
    })
}

fn tier_d_experiment_contract(matrix: &str) -> Value {
    if matrix == MATRIX_TIER_D_D2 {
        return tier_d_d2_experiment_contract();
    }
    let smoke = matrix == MATRIX_TIER_D_D0_SMOKE;
    let mut unique_shapes = BTreeMap::new();
    for (group, _, shape) in tier_d_cell_listing(smoke) {
        let summary = shape.summary();
        unique_shapes.entry(format!("{group}/{}", shape.name)).or_insert_with(|| {
            json!({
                "group": group,
                "shape_name": shape.name,
                "shape_invariant_sha256": shape_invariant_sha256(&shape),
                "selected_rows": summary.selected_rows,
                "selected_fraction": summary.selected_fraction,
                "selected_run_count": summary.selected_run_count,
                "internal_skip_run_count": summary.internal_skip_run_count(),
                "internal_transition_count": summary.internal_transition_count(),
                "selected_span_rows": summary.last_selected_row_exclusive - summary.first_selected_row,
            })
        });
    }
    json!({
        "schema_version": "arrow-row-selection-tier-d-guard-contract-v1",
        "matrix": matrix,
        "status": if smoke { "opened-context-contract-smoke" } else { "opened-context-boundary-discovery-replay" },
        "production_candidate": Value::Null,
        "blind_timing_opened": false,
        "blind_contexts_constructed": false,
        "context_ids": tier_d_context_ids(smoke),
        "shape_families": unique_shapes.into_values().collect::<Vec<_>>(),
        "guard_hypothesis": {
            "base": "M0",
            "override_direction": "selectors-to-mask-only",
            "selector_pressure_formula": "projected_leaf_count * (compressed_bytes / represented_rows) / arrow_output_width_proxy",
            "inputs": ["internal_transition_count", "selected_fraction", "projected_leaf_count", "compressed_bytes", "represented_rows", "arrow_output_width_proxy"],
            "context_id_is_production_input": false,
            "threshold_candidates": {
                "tau_t": [254, 510, 766, 1022, 1278, 1534, 2046],
                "tau_q": [8, 12, 16, 20, 24, 28, 30, 32, 34, 36, 40, 48, 64],
                "tau_f": [0.015625, 0.125, 0.5, 0.875, 0.984375]
            },
            "fit_objective": [
                "minimize decisive >5% new-harm count",
                "minimize context-by-shape-family equal-weight macro regret",
                "minimize absolute-time regret",
                "prefer larger conservative thresholds"
            ]
        },
        "freeze_policy": "D1 discovery may freeze only the registered three-threshold guard; replay cannot refit; H1-H4 remain closed"
    })
}

fn tier_d_d2_experiment_contract() -> Value {
    let shape_families = tier_d_d2_shapes()
        .into_iter()
        .map(|(group, shape)| {
            let summary = shape.summary();
            json!({
                "group": group,
                "shape_name": shape.name,
                "shape_invariant_sha256": shape_invariant_sha256(&shape),
                "selected_rows": summary.selected_rows,
                "selected_fraction": summary.selected_fraction,
                "selected_run_count": summary.selected_run_count,
                "internal_skip_run_count": summary.internal_skip_run_count(),
                "internal_transition_count": summary.internal_transition_count(),
                "selected_span_rows": summary.last_selected_row_exclusive - summary.first_selected_row,
            })
        })
        .collect::<Vec<_>>();
    json!({
        "schema_version": "arrow-row-selection-tier-d-d2-validation-contract-v1",
        "matrix": MATRIX_TIER_D_D2,
        "status": "unseen-metadata-interpolation-transfer-validation",
        "production_candidate": Value::Null,
        "blind_timing_opened": true,
        "blind_contexts_constructed": true,
        "h_contexts_opened": false,
        "role": "validation-blind",
        "context_ids": TIER_D_D2_CONTEXT_IDS,
        "shape_families": shape_families,
        "frozen_guard": {
            "base": "M0",
            "override_direction": "selectors-to-mask-only",
            "selector_pressure_formula": "projected_leaf_count * compressed_bytes / (represented_rows * arrow_output_width_proxy)",
            "context_id_is_production_input": false,
            "thresholds": {
                "tau_t": TIER_D_FROZEN_TAU_T,
                "tau_q": TIER_D_FROZEN_TAU_Q,
                "tau_f": {
                    "numerator": TIER_D_FROZEN_TAU_F_NUMERATOR,
                    "denominator": TIER_D_FROZEN_TAU_F_DENOMINATOR
                }
            },
            "frozen_guard_sha256": TIER_D_FROZEN_GUARD_SHA256,
            "decision_semantic_sha256": TIER_D_FROZEN_DECISION_SHA256,
            "d1_discovery_csv_sha256": TIER_D_D1_DISCOVERY_CSV_SHA256,
            "d1_replay_csv_sha256": TIER_D_D1_REPLAY_CSV_SHA256,
            "d1_validation_analysis_sha256": TIER_D_D1_VALIDATION_SHA256
        },
        "q_exposure": {
            "threshold": TIER_D_FROZEN_TAU_Q,
            "comparison": "exact integer cross multiplication per row group",
            "below_above_pairs": [
                {"family": "int32", "below": "U1", "above": "U2"},
                {"family": "utf8view8", "below": "U3", "above": "U4"},
                {"family": "utf8view32", "below": "U5", "above": "U6"},
                {"family": "utf8view64-none", "below": "U7", "above": "U8"},
                {"family": "utf8view64-zstd", "below": "U9", "above": "U10"}
            ],
            "dictionary_below_control": "U11",
            "post_hoc_context_replacement": false
        },
        "validation_gate": {
            "threshold_refit_allowed": false,
            "repeat_non_tie_winner_agreement_min": 0.98,
            "arm_ratio_drift_octaves_max_exclusive": 0.25,
            "decisive_new_harm_over_5pct_per_run_per_context": 0,
            "macro_regret_not_worse_than_m0": true,
            "pooled_regret_at_most_fraction_of_m0": 0.5,
            "max_oracle_relative_regret_exclusive": 0.15,
            "stable_non_tie_beneficial_switch_count_min": 8,
            "stable_beneficial_switch_decoder_family_count_min": 2
        },
        "freeze_policy": "D2 repeats consume this exact guard and context manifest; no threshold refit, post-hoc context replacement, or H1-H4 timing is allowed"
    })
}

fn candidate_library() -> Value {
    json!({
        "schema_version": CANDIDATE_SCHEMA_VERSION,
        "raw_symbols": {
            "N": "rows_per_group",
            "S": "selected_rows",
            "R0": "base_run_count after trailing-skip trim",
            "Rp": "loaded_run_count after loaded-page intersection and trailing-skip trim",
            "M0": "base_mask_decoded_rows from the side-effect-free MaskCursor simulator without page ranges",
            "Mp": "mask_decoded_rows from the same simulator with loaded page ranges",
            "P": "projected_leaf_count",
            "U": "aggregate projected-leaf uncompressed_bytes",
            "C": "aggregate projected-leaf compressed_bytes",
            "A": "aggregate Arrow output_width_proxy bytes per row",
            "B": "batch_size",
            "L": "loaded_page_rows from the all-projected-leaf range intersection",
            "Q": "predicted_fetched_bytes summed from each projected leaf selected-page set, including dictionary page bytes",
            "Z": "1 when compression is zstd, otherwise 0"
        },
        "decision": {
            "target": "delta_ns = time(mask) - time(selectors)",
            "choose_mask_when": "score < 0",
            "tie_break": "M0: choose mask iff N < 32 * run_count, otherwise selectors",
            "coefficient_constraints": "beta_j >= 0 and sum(beta_j) = 1",
            "coefficient_count": 4
        },
        "preprocessing": {
            "scale": "for each unsigned raw feature, divide by the median of its finite values strictly greater than zero on the candidate's D1 training RG units",
            "zero_median": "candidate invalid when a feature has no finite strictly-positive training value",
            "clamp_each_scaled_unsigned_feature": [0.0, 16.0],
            "missing_or_unsupported": "fallback M0"
        },
        "fit": {
            "solver": "exhaustive simplex grid",
            "grid_step": 0.01,
            "simplex_enumeration": "integer k0..k3 >= 0, sum(k)=100, beta=k/100",
            "tie_band_ns": "max(3 * sqrt((mad_selectors^2 + mad_mask^2)/2), 0.01 * mean(median_selectors, median_mask))",
            "primary_loss": "sum(max(0, chosen_median_ns - oracle_median_ns)) over non-tie training RG units",
            "secondary_loss": "maximum oracle-relative regret",
            "tertiary_tie_break": "lexicographically smallest beta vector",
            "cell_weight": "one per RG unit",
            "timing_tie_weight": 0,
            "regularization": "none"
        },
        "candidates": [
            {
                "id": "B1-selection-only",
                "training_domain": "all D1 external forced-pair RG units",
                "signed_features": [
                    {"sign": -1, "name": "run_density", "formula": "R0 / N"},
                    {"sign": 1, "name": "selected_fraction", "formula": "S / N"},
                    {"sign": 1, "name": "long_skip_share_4096", "formula": "long_skip_rows_4096 / max(skipped_rows, 1)"},
                    {"sign": 1, "name": "batch_scale", "formula": "log2(B) / 20"}
                ]
            },
            {
                "id": "B2-row-group-cost",
                "training_domain": "all D1 external forced-pair RG units; page fields are ignored",
                "signed_features": [
                    {"sign": 1, "name": "mask_wasted_uncompressed_bytes", "formula": "max(M0 - S, 0) * U / N"},
                    {"sign": 1, "name": "mask_filter_materialization_bytes_proxy", "formula": "M0 * ceil(P / 8) + S * A"},
                    {"sign": -1, "name": "run_count_x_leaf_decoder_work_proxy", "formula": "R0 * (U / N + A)"},
                    {"sign": 1, "name": "mask_wasted_decompression_bytes", "formula": "Z * max(M0 - S, 0) * C / N"}
                ]
            },
            {
                "id": "B3-page-aware-cost",
                "training_domain": "D1 X-C Required external forced-pair RG units with offset index coverage for every projected leaf",
                "signed_features": [
                    {"sign": 1, "name": "page_mask_wasted_uncompressed_bytes", "formula": "max(Mp - S, 0) * U / N"},
                    {"sign": 1, "name": "page_mask_filter_materialization_bytes_proxy", "formula": "Mp * ceil(P / 8) + S * A"},
                    {"sign": -1, "name": "loaded_run_count_x_leaf_decoder_work_proxy", "formula": "Rp * (U / N + A)"},
                    {"sign": 1, "name": "page_mask_predicted_decompression_bytes", "formula": "Z * max(Mp - S, 0) * Q / max(L, 1)"}
                ]
            }
        ],
        "applicability": {
            "allowed": ["flat non-null fixed-width", "Utf8View", "Utf8 dictionary fixture"],
            "fallback_m0": [
                "nested or nullable output",
                "num_values inconsistent with row-group rows",
                "unknown encoding stats",
                "cached predicate/output projection overlap",
                "unsupported codec or output layout",
                "B3 without offset indexes and predicted page bytes for every projected leaf"
            ]
        },
        "structural_expressibility_witness": {
            "signed_formula": "+b0*x_waste + b1*x_filter - b2*x_run_decoder + b3*x_decompress",
            "beta": [0.15, 0.15, 0.20, 0.50],
            "synthetic_scaled_rows": {
                "C3": [1.0, 1.0, 1.0, 0.0],
                "C4": [8.0, 8.0, 64.0, 0.0],
                "C13": [8.0, 8.0, 64.0, 80.0]
            },
            "required_signs": {"C3": "selectors", "C4": "mask", "C13": "selectors"},
            "purpose": "prove algebraic expressibility only; witness values are not timing labels or fitted parameters"
        }
    })
}

fn validate_structural_expressibility() -> Result<(), String> {
    let beta = [0.15, 0.15, 0.20, 0.50];
    let score = |features: [f64; 4]| {
        beta[0] * features[0] + beta[1] * features[1] - beta[2] * features[2]
            + beta[3] * features[3]
    };
    let c3 = score([1.0, 1.0, 1.0, 0.0]);
    let c4 = score([8.0, 8.0, 64.0, 0.0]);
    let c13 = score([8.0, 8.0, 64.0, 80.0]);
    if !(c3 > 0.0 && c4 < 0.0 && c13 > 0.0) {
        return Err("candidate library cannot express C3/C4/C13 directions".to_string());
    }
    let sum: f64 = beta.into_iter().sum();
    if (sum - 1.0).abs() > f64::EPSILON {
        return Err("structural witness coefficients are not normalized".to_string());
    }
    Ok(())
}

fn validate_page_exposure(rows: &[CsvRow]) -> Result<(), String> {
    validate_page_exposure_for(rows, "D0-PAGE", "C0")
}

fn validate_page_exposure_for(
    rows: &[CsvRow],
    group: &str,
    context_id: &str,
) -> Result<(), String> {
    for row_group_index in 0..ORACLE_ROW_GROUPS {
        let find = |policy: &str| {
            rows.iter().find(|row| {
                row.group == group
                    && row.context.id == context_id
                    && row.shape_name == "page_matched_bursty_f50_l64"
                    && row.metadata_policy == policy
                    && row.row_group_index == row_group_index
                    && row.measurement.arm == OracleArm::Mask
            })
        };
        let required = find("Required").ok_or_else(|| {
            format!("missing {group}/{context_id} Required exposure row for rg{row_group_index}")
        })?;
        let skip = find("Skip").ok_or_else(|| {
            format!("missing {group}/{context_id} Skip exposure row for rg{row_group_index}")
        })?;
        let decoded_separation =
            skip.metadata
                .mask_decoded_rows
                .saturating_sub(required.metadata.mask_decoded_rows) as f64
                / ROWS_PER_GROUP as f64;
        let byte_separation =
            skip.measurement
                .requested_bytes
                .saturating_sub(required.measurement.requested_bytes) as f64
                / skip.measurement.requested_bytes.max(1) as f64;
        if decoded_separation < 0.20 || byte_separation < 0.15 {
            return Err(format!(
                "invalid {group}/{context_id} page exposure rg{row_group_index}: decoded separation={decoded_separation:.3}, requested-byte separation={byte_separation:.3}"
            ));
        }
    }
    Ok(())
}

fn is_tier_c_matrix(matrix: &str) -> bool {
    matches!(matrix, MATRIX_TIER_C_C0_SMOKE | MATRIX_TIER_C_C1)
}

fn is_tier_d_matrix(matrix: &str) -> bool {
    matches!(
        matrix,
        MATRIX_TIER_D_D0_SMOKE | MATRIX_TIER_D_D1 | MATRIX_TIER_D_D2
    )
}

fn has_extended_csv(matrix: &str) -> bool {
    is_tier_c_matrix(matrix) || is_tier_d_matrix(matrix)
}

fn csv_columns(matrix: &str) -> Vec<&'static str> {
    let mut columns = CSV_COLUMNS.to_vec();
    if has_extended_csv(matrix) {
        let insertion = columns
            .iter()
            .position(|column| *column == "last_selected_row_exclusive")
            .expect("v2 columns contain selected span")
            + 1;
        columns.splice(insertion..insertion, TIER_C_EXTRA_COLUMNS.iter().copied());
    }
    columns
}

fn write_csv(options: &Options, rows: &[CsvRow]) -> Result<(), String> {
    let columns = csv_columns(&options.matrix);
    let file = File::create(&options.csv)
        .map_err(|error| format!("cannot create CSV {}: {error}", options.csv.display()))?;
    let mut writer = BufWriter::new(file);
    writeln!(writer, "{}", columns.join(","))
        .map_err(|error| format!("cannot write CSV header: {error}"))?;
    for row in rows {
        let mut fields = vec![
            if is_tier_d_matrix(&options.matrix) {
                CSV_SCHEMA_VERSION_TIER_D
            } else if is_tier_c_matrix(&options.matrix) {
                CSV_SCHEMA_VERSION_TIER_C
            } else {
                CSV_SCHEMA_VERSION
            }
            .to_string(),
            row.group.to_string(),
            row.role.to_string(),
            row.cell_id.clone(),
            row.context.id.to_string(),
            row.row_group_index.to_string(),
            row.context.payload.dtype().to_string(),
            row.metadata.projection_signature.clone(),
            row.context.output_layout().to_string(),
            row.context.payload_columns.to_string(),
            row.context.encoding().to_string(),
            row.context.compression.label().to_string(),
            row.metadata_policy.to_string(),
            optional_number(row.page_layout_rows),
            row.fixture_sha256.clone(),
            row.context.batch_size.to_string(),
            ROWS_PER_GROUP.to_string(),
            row.shape_name.clone(),
            ROWS_PER_GROUP.to_string(),
            row.shape.selected_rows.to_string(),
            row.shape.skipped_rows.to_string(),
            format!("{:.9}", row.shape.selected_fraction),
            format!("{:.9}", row.shape.avg_run_len),
            row.shape.run_count.to_string(),
            row.shape.selected_run_count.to_string(),
            row.shape.skipped_run_count.to_string(),
            row.shape.first_selected_row.to_string(),
            row.shape.last_selected_row_exclusive.to_string(),
            row.shape.max_skip_run.to_string(),
            row.shape.long_skip_rows_1024.to_string(),
            row.shape.long_skip_count_1024.to_string(),
            row.shape.long_skip_rows_4096.to_string(),
            row.shape.long_skip_count_4096.to_string(),
            format!("{:.9}", row.shape.long_skip_share_1024),
            format!("{:.9}", row.shape.long_skip_share_4096),
            row.source.label().to_string(),
            row.backing.to_string(),
            row.metadata.projected_leaf_count.to_string(),
            row.metadata.physical_type_histogram.clone(),
            row.metadata.encoding_sets.clone(),
            row.metadata.dictionary_leaf_count.to_string(),
            row.metadata.compressed_bytes.to_string(),
            row.metadata.uncompressed_bytes.to_string(),
            row.metadata.per_leaf_compressed_bytes.clone(),
            row.metadata.per_leaf_uncompressed_bytes.clone(),
            row.metadata.per_leaf_num_values.clone(),
            format!("{:.9}", row.metadata.compression_ratio),
            row.metadata.num_values_min.to_string(),
            row.metadata.num_values_max.to_string(),
            row.metadata.num_values_consistent.to_string(),
            row.metadata.arrow_output_width_proxy.to_string(),
            optional_number(row.metadata.loaded_page_rows),
            optional_number(row.metadata.loaded_range_count),
            row.metadata.per_leaf_page_rows.clone().unwrap_or_default(),
            row.metadata
                .per_leaf_page_first_rows
                .clone()
                .unwrap_or_default(),
            row.metadata
                .per_leaf_predicted_fetched_bytes
                .clone()
                .unwrap_or_default(),
            row.metadata
                .predicted_fetched_bytes
                .map(|value| value.to_string())
                .unwrap_or_default(),
            row.metadata.base_mask_chunk_count.to_string(),
            row.metadata.base_mask_decoded_rows.to_string(),
            row.metadata.base_run_count.to_string(),
            row.metadata.mask_chunk_count.to_string(),
            row.metadata.mask_decoded_rows.to_string(),
            row.metadata.loaded_run_count.to_string(),
            row.measurement.arm.label().to_string(),
            row.measurement.samples_ns.len().to_string(),
            join_numbers(&row.measurement.samples_ns),
            join_numbers(&row.measurement.sample_started_unix_ns),
            row.measurement.median_ns.to_string(),
            row.measurement.mad_ns.to_string(),
            row.measurement.rows_out.to_string(),
            row.measurement.content.schema_sha256.clone(),
            row.measurement.content.leaf_sha256.join("|"),
            row.measurement.requested_ranges.len().to_string(),
            row.measurement
                .requested_ranges
                .iter()
                .map(|range| format!("{}-{}", range.start, range.end))
                .collect::<Vec<_>>()
                .join("|"),
            row.measurement.requested_bytes.to_string(),
        ];
        if has_extended_csv(&options.matrix) {
            let insertion = CSV_COLUMNS
                .iter()
                .position(|column| *column == "last_selected_row_exclusive")
                .expect("v2 columns contain selected span")
                + 1;
            fields.splice(
                insertion..insertion,
                [
                    row.shape.leading_skip_present().to_string(),
                    row.shape.internal_skip_run_count().to_string(),
                    row.shape.internal_transition_count().to_string(),
                    row.shape_invariant_sha256.clone(),
                ],
            );
        }
        if fields.len() != columns.len() {
            return Err(format!(
                "CSV field drift: header={}, row={}",
                columns.len(),
                fields.len()
            ));
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
        .map_err(|error| format!("cannot write CSV row: {error}"))?;
    }
    writer
        .flush()
        .map_err(|error| format!("cannot flush CSV {}: {error}", options.csv.display()))
}

fn write_manifest(
    options: &Options,
    rows: &[CsvRow],
    started_unix_ns: u64,
    completed_unix_ns: u64,
    elapsed_ns: u64,
    candidate_library: &Value,
) -> Result<(), String> {
    let tier_c = is_tier_c_matrix(&options.matrix);
    let tier_d = is_tier_d_matrix(&options.matrix);
    let tier_d_d2 = options.matrix == MATRIX_TIER_D_D2;
    let columns = csv_columns(&options.matrix);
    let columns_sha256 = sha256_json(&json!(&columns));
    let contract_schema_version = candidate_library
        .get("schema_version")
        .and_then(Value::as_str)
        .unwrap_or(CANDIDATE_SCHEMA_VERSION);
    let cells = rows
        .iter()
        .map(|row| row.cell_id.as_str())
        .collect::<BTreeSet<_>>()
        .len();
    let fixture_digests = rows
        .iter()
        .map(|row| {
            (
                format!(
                    "{}/{}/page-{}/batch-{}",
                    row.context.id,
                    row.metadata_policy,
                    row.page_layout_rows
                        .map(|value| value.to_string())
                        .unwrap_or_else(|| "none".to_string()),
                    row.context.batch_size
                ),
                row.fixture_sha256.clone(),
            )
        })
        .collect::<BTreeMap<_, _>>();
    let mut arms_by_cell = BTreeMap::<&str, BTreeSet<&str>>::new();
    for row in rows {
        arms_by_cell
            .entry(&row.cell_id)
            .or_default()
            .insert(row.measurement.arm.label());
    }
    let logical_cells = arms_by_cell
        .iter()
        .map(|(cell_id, arms)| {
            let row = rows
                .iter()
                .find(|row| row.cell_id.as_str() == *cell_id)
                .expect("cell came from rows");
            json!({
                "id": cell_id,
                "group": row.group,
                "role": row.role,
                "context_id": row.context.id,
                "shape_name": row.shape_name,
                "shape_invariant_sha256": row.shape_invariant_sha256,
                "leading_skip_present": row.shape.leading_skip_present(),
                "internal_skip_run_count": row.shape.internal_skip_run_count(),
                "internal_transition_count": row.shape.internal_transition_count(),
                "selected_span_rows": row.shape.last_selected_row_exclusive - row.shape.first_selected_row,
                "selected_rows": row.shape.selected_rows,
                "selection_source": row.source.label(),
                "selection_backing": row.backing,
                "metadata_policy": row.metadata_policy,
                "page_layout_rows": row.page_layout_rows,
                "batch_size": row.context.batch_size,
                "arms": arms,
            })
        })
        .collect::<Vec<_>>();
    let logical_cell_manifest = Value::Array(logical_cells);
    let pair_count = arms_by_cell.values().filter(|arms| arms.len() == 2).count();
    let single_count = arms_by_cell.len() - pair_count;
    let classification = match options.matrix.as_str() {
        MATRIX_D0_SMOKE => "non-formal D0 contract smoke",
        MATRIX_D1_DISCOVERY => "non-formal D1 training-only discovery",
        MATRIX_D1_REPLAY => "non-formal D1 training-only adaptive-union replay",
        MATRIX_TIER_C_C0_SMOKE => "non-formal Tier-C C0 orthogonal-shape contract smoke",
        MATRIX_TIER_C_C1 => "non-formal Tier-C C1 opened-context cost-surface training",
        MATRIX_TIER_D_D0_SMOKE => "non-formal Tier-D D0 opened-context guard contract smoke",
        MATRIX_TIER_D_D1 => "non-formal Tier-D D1 opened-context boundary discovery/replay",
        MATRIX_TIER_D_D2 => "non-formal Tier-D D2 unseen-context interpolation transfer validation",
        _ => unreachable!(),
    };
    let adaptive_contract = if options.matrix == MATRIX_D1_REPLAY {
        json!({
            "schema_version": "arrow-row-selection-tier-b-adaptive-union-v1",
            "source_run_id": "run-20260811-rs8846-tierb-d1-discovery-r1-c1ddeeff",
            "source_csv_sha256": D1_DISCOVERY_CSV_SHA256,
            "semantic_sha256": D1_ADAPTIVE_UNION_SHA256,
            "cell_count": 16,
            "run_lengths": {
                "T1": adaptive_grid("T1"),
                "T2": adaptive_grid("T2"),
                "T3": adaptive_grid("T3"),
                "T4": adaptive_grid("T4"),
                "T5": adaptive_grid("T5"),
                "T6": adaptive_grid("T6"),
            }
        })
    } else {
        Value::Null
    };
    let page_control = if tier_c || tier_d {
        json!({
            "status": if tier_d { "not-applicable-to-tier-d-d0-d2" } else { "not-applicable-to-tier-c-c0-c1" },
            "same_bytes": false,
            "writer_page_rows": Value::Null,
            "views": []
        })
    } else {
        json!({
            "same_bytes": true,
            "writer_page_rows": ORACLE_PAGE_ROWS,
            "views": ["PageIndexPolicy::Required", "PageIndexPolicy::Skip"]
        })
    };
    let manifest = json!({
        "schema_version": if tier_d { MANIFEST_SCHEMA_VERSION_TIER_D } else if tier_c { MANIFEST_SCHEMA_VERSION_TIER_C } else { MANIFEST_SCHEMA_VERSION },
        "csv_schema_version": if tier_d { CSV_SCHEMA_VERSION_TIER_D } else if tier_c { CSV_SCHEMA_VERSION_TIER_C } else { CSV_SCHEMA_VERSION },
        "candidate_library_schema_version": contract_schema_version,
        "candidate_library_sha256": sha256_json(candidate_library),
        "csv_columns": &columns,
        "csv_columns_sha256": columns_sha256,
        "matrix": options.matrix,
        "benchmark": if tier_d { "arrow_reader_row_selection_oracle_tier_d" } else if tier_c { "arrow_reader_row_selection_oracle_tier_c" } else { "arrow_reader_row_selection_oracle_tier_b" },
        "git_sha": command_output("git", &["-C", env!("CARGO_MANIFEST_DIR"), "rev-parse", "HEAD"]),
        "rustc": command_output("rustc", &["-vV"]),
        "started_unix_ns": started_unix_ns,
        "completed_unix_ns": completed_unix_ns,
        "elapsed_ns": elapsed_ns,
        "samples_per_arm": options.samples,
        "warmups_per_arm": WARMUPS_PER_ARM,
        "logical_cell_count": cells,
        "forced_pair_cell_count": pair_count,
        "single_arm_cell_count": single_count,
        "row_group_unit_count": cells * ORACLE_ROW_GROUPS,
        "arm_row_count": rows.len(),
        "logical_cell_manifest_sha256": sha256_json(&logical_cell_manifest),
        "logical_cells": logical_cell_manifest,
        "fixture_sha256": fixture_digests,
        "measurement_unit": "(run, logical_cell, row_group_index)",
        "correctness": "arrow-projected-leaf-content-v1 schema plus every projected logical leaf SHA-256",
        "page_control": page_control,
        "timing_protocol": {
            "forced_arm_order": "selectors,mask,mask,selectors repeated",
            "statistic": "median",
            "dispersion": "median_absolute_deviation",
            "attribution_outside_timer": true
        },
        "blind_timing_opened": tier_d_d2,
        "blind_contexts_constructed": tier_d_d2,
        "h_contexts_opened": false,
        "adaptive_points_opened": if options.matrix == MATRIX_D1_REPLAY { 16 } else { 0 },
        "adaptive_contract": adaptive_contract,
        "tier_c_factor_contract": if tier_c {
            json!({
                "diagnostic_predictors": ["1", "internal_transition_count", "selected_span_rows-selected_rows", "selected_rows"],
                "rank": 4,
                "positive_median_scaled_condition_number": 12.746109456977376_f64,
                "production_candidate": Value::Null,
                "opened_contexts_only": true
            })
        } else {
            Value::Null
        },
        "tier_d_guard_contract": if tier_d {
            candidate_library
                .get(if tier_d_d2 { "frozen_guard" } else { "guard_hypothesis" })
                .cloned()
                .unwrap_or(Value::Null)
        } else {
            Value::Null
        },
        "classification": classification
    });
    write_json(&options.manifest, &manifest)
}

fn write_json(path: &Path, value: &Value) -> Result<(), String> {
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

fn context_by_id(id: &str) -> Result<OracleContext, String> {
    training_contexts()
        .into_iter()
        .chain(TIER_D_D2_CONTEXTS.iter().copied())
        .find(|context| context.id == id)
        .ok_or_else(|| format!("unknown context {id}"))
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

fn default_artifact_path(filename: &str) -> PathBuf {
    env::var_os("CARGO_TARGET_DIR")
        .map(PathBuf::from)
        .map(|target_dir| target_dir.join(filename))
        .unwrap_or_else(|| PathBuf::from(filename))
}

fn emit_artifact(kind: &str, path: &Path) -> Result<(), String> {
    let raw = fs::read_to_string(path)
        .map_err(|error| format!("cannot read {} for log embedding: {error}", path.display()))?;
    println!("DFEXP_SELECTION_ORACLE_V2_{kind}_BEGIN");
    print!("{raw}");
    if !raw.ends_with('\n') {
        println!();
    }
    println!("DFEXP_SELECTION_ORACLE_V2_{kind}_END");
    Ok(())
}

fn sha256_json(value: &Value) -> String {
    let bytes = serde_json::to_vec(value).unwrap();
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("sha256:{}", hex_digest(&hasher.finalize()))
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

fn optional_number(value: Option<usize>) -> String {
    value.map(|value| value.to_string()).unwrap_or_default()
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

fn hex_digest(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        output.push(HEX[(byte >> 4) as usize] as char);
        output.push(HEX[(byte & 0x0f) as usize] as char);
    }
    output
}
