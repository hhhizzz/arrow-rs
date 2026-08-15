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

//! PC-2 same-SHA benchmark harvesters.
//!
//! This module is compiled only with `test_common`: the legacy, forced-thin,
//! and R16 policies are diagnostic experiment controls, not product API.

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
use parquet::arrow::arrow_reader::metrics::{ArrowReaderMetrics, PerColumnDecisionMetrics};
use parquet::arrow::arrow_reader::{RowSelection, RowSelectionPolicy, RowSelector};
use parquet::arrow::{ParquetRecordBatchStreamBuilder, ProjectionMask};
use regex::Regex;
use serde_json::json;
use sha2::{Digest, Sha256};
use tokio::runtime::Runtime;

use super::fixture::{
    MR1_HOLDOUT_CONTEXTS, ORACLE_CONTEXTS, ORACLE_PAGE_ROWS, ORACLE_ROW_GROUPS, OracleContext,
    OracleFixture, OraclePayload, PC_MIXED_CONTEXTS, PC2_HOLDOUT_CONTEXTS, TT_CONTEXTS,
    build_oracle_fixture, build_oracle_fixture_with_dimensions,
};
use super::model::{BATCH_SIZE, ROWS_PER_GROUP};
use super::runner::{ProjectedContentDigest, ProjectedContentDigester};
use super::shapes::{OracleShape, OracleShapeSummary};

const CSV_SCHEMA_VERSION: &str = "arrow-row-selection-pc2-v1";
const MANIFEST_SCHEMA_VERSION: &str = "arrow-row-selection-pc2-manifest-v1";
const MR1_CSV_SCHEMA_VERSION: &str = "arrow-row-selection-mr1-v1";
const MR1_MANIFEST_SCHEMA_VERSION: &str = "arrow-row-selection-mr1-manifest-v1";
const DEFAULT_SAMPLES: usize = 12;
const WARMUPS_PER_ARM: usize = 2;
const PURE_DICTIONARY_THRESHOLD: usize = 4;
const PRODUCT_OTHER_THRESHOLD: usize = 32;
const LEGACY_OTHER_THRESHOLD: usize = 16;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Mode {
    Tax,
    Product,
    Smoke,
    IdentityA,
    Mr1Smoke,
    Mr1Protocol,
    Mr1Product,
}

impl Mode {
    const fn label(self) -> &'static str {
        match self {
            Self::Tax => "tax",
            Self::Product => "product",
            Self::Smoke => "smoke",
            Self::IdentityA => "identity-a",
            Self::Mr1Smoke => "mr1-smoke",
            Self::Mr1Protocol => "mr1-protocol",
            Self::Mr1Product => "mr1-product",
        }
    }

    const fn arms(self) -> &'static [Pc2Arm] {
        match self {
            Self::Tax => &[Pc2Arm::Auto32, Pc2Arm::PcBolton, Pc2Arm::Pc2Thin],
            Self::Product | Self::Smoke => &[Pc2Arm::Auto32, Pc2Arm::Pc2, Pc2Arm::Pc2R16],
            Self::IdentityA => &[Pc2Arm::Auto32, Pc2Arm::Pc2],
            Self::Mr1Smoke | Self::Mr1Protocol | Self::Mr1Product => &[],
        }
    }

    const fn timed(self) -> bool {
        matches!(
            self,
            Self::Tax | Self::Product | Self::IdentityA | Self::Mr1Protocol | Self::Mr1Product
        )
    }

    const fn is_mr1(self) -> bool {
        matches!(self, Self::Mr1Smoke | Self::Mr1Protocol | Self::Mr1Product)
    }
}

#[derive(Debug)]
struct Options {
    mode: Mode,
    list: bool,
    samples: usize,
    filter: Option<Regex>,
    filter_text: Option<String>,
    csv: PathBuf,
    manifest: PathBuf,
    emit_artifacts: bool,
    round: Option<u8>,
    prereg_sha256: Option<String>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Pc2Arm {
    Auto32,
    PcBolton,
    Pc2Thin,
    Pc2,
    Pc2R16,
    Mr1LegacySelectors,
    Mr1Selectors,
    Mr1BaseMask,
    Mr1Mask,
    Mr1Independent,
}

impl Pc2Arm {
    const fn label(self) -> &'static str {
        match self {
            Self::Auto32 => "auto32",
            Self::PcBolton => "pc-bolton",
            Self::Pc2Thin => "pc2-thin",
            Self::Pc2 => "pc2",
            Self::Pc2R16 => "pc2-r16",
            Self::Mr1LegacySelectors => "base-selectors",
            Self::Mr1Selectors => "mr1-selectors",
            Self::Mr1BaseMask => "base-mask",
            Self::Mr1Mask => "mr1-mask",
            Self::Mr1Independent => "base-independent-selectors",
        }
    }

    const fn policy(self) -> RowSelectionPolicy {
        match self {
            Self::Auto32 => RowSelectionPolicy::Auto { threshold: 32 },
            Self::PcBolton => RowSelectionPolicy::PerColumnLegacy,
            Self::Pc2Thin => RowSelectionPolicy::PerColumnForcedThin,
            Self::Pc2 => RowSelectionPolicy::PerColumn,
            Self::Pc2R16 => RowSelectionPolicy::PerColumnR16,
            Self::Mr1LegacySelectors | Self::Mr1Independent => RowSelectionPolicy::SelectorsLegacy,
            Self::Mr1Selectors => RowSelectionPolicy::Selectors,
            Self::Mr1BaseMask | Self::Mr1Mask => RowSelectionPolicy::Mask,
        }
    }
}

#[derive(Clone, Debug)]
struct ScanResult {
    row_count: usize,
    batch_count: usize,
    content: Option<ProjectedContentDigest>,
    decisions: PerColumnDecisionMetrics,
    read_selection_calls: usize,
    driver_batches: usize,
}

#[derive(Clone, Debug)]
struct ArmMeasurement {
    arm: Pc2Arm,
    samples_ns: Vec<u64>,
    sample_started_unix_ns: Vec<u64>,
    median_ns: u64,
    mad_ns: u64,
    rows_out: usize,
    batches_out: usize,
    content: ProjectedContentDigest,
    decisions: PerColumnDecisionMetrics,
    read_selection_calls: usize,
    driver_batches: usize,
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
        eprintln!("PC-2 row-selection evaluation failed: {error}");
        std::process::exit(2);
    }
}

fn try_main() -> Result<(), String> {
    let options = parse_options()?;
    if options.list {
        list_cells(&options);
        return Ok(());
    }

    let started_unix_ns = unix_nanos();
    // Smoke is intentionally timing-free: do not even sample Instant on that
    // code path. Identity-A times only the three public witness cells and
    // therefore never opens the H-PC2 holdout.
    let started = options.mode.timed().then(Instant::now);
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|error| format!("cannot build Tokio runtime: {error}"))?;
    let mut rows = Vec::new();
    let mut fixture_sha256 = BTreeMap::new();

    match options.mode {
        Mode::Tax => run_tax(&runtime, &options, &mut rows, &mut fixture_sha256)?,
        Mode::Product => run_product(&runtime, &options, &mut rows, &mut fixture_sha256)?,
        Mode::Smoke => run_product(&runtime, &options, &mut rows, &mut fixture_sha256)?,
        Mode::IdentityA => run_identity_a(&runtime, &options, &mut rows, &mut fixture_sha256)?,
        Mode::Mr1Smoke | Mode::Mr1Protocol => {
            run_mr1_protocol(&runtime, &options, &mut rows, &mut fixture_sha256)?
        }
        Mode::Mr1Product => run_mr1_product(&runtime, &options, &mut rows, &mut fixture_sha256)?,
    }
    if rows.is_empty() {
        return Err(format!(
            "cell filter selected no PC-2 {} benchmark cells",
            options.mode.label()
        ));
    }

    let completed_unix_ns = unix_nanos();
    write_csv(&options.csv, &rows, options.mode.is_mr1())?;
    write_manifest(
        &options,
        &rows,
        &fixture_sha256,
        started_unix_ns,
        completed_unix_ns,
        started
            .map(|started| started.elapsed().as_nanos() as u64)
            .unwrap_or_default(),
    )?;

    let cells = rows
        .iter()
        .map(|row| row.cell_id.as_str())
        .collect::<BTreeSet<_>>()
        .len();
    let output_prefix = if options.mode.is_mr1() {
        "DFEXP_MR1"
    } else {
        "DFEXP_PC2"
    };
    println!("{output_prefix}_CELLS={cells}");
    println!("{output_prefix}_ROWS={}", rows.len());
    if options.emit_artifacts {
        emit_artifact(output_prefix, "CSV", &options.csv)?;
        emit_artifact(output_prefix, "MANIFEST", &options.manifest)?;
    }
    Ok(())
}

fn run_tax(
    runtime: &Runtime,
    options: &Options,
    rows: &mut Vec<CsvRow>,
    fixture_sha256: &mut BTreeMap<String, String>,
) -> Result<(), String> {
    for (context_id, run_lengths) in [("C6", &[64][..]), ("C0", &[64, 4][..])] {
        let context = find_context(context_id)?;
        let selected = run_lengths
            .iter()
            .copied()
            .map(OracleShape::l_sweep)
            .filter(|shape| matches_filter(options, &format!("PC-2A/{context_id}/{}", shape.name)))
            .collect::<Vec<_>>();
        if selected.is_empty() {
            continue;
        }
        let fixture = build_fixture(context, fixture_sha256)?;
        for shape in selected {
            let cell = format!("PC-2A/{context_id}/{}", shape.name);
            eprintln!("measuring {cell}");
            rows.extend(measure_cell(
                runtime,
                &fixture,
                options.mode.arms(),
                &shape,
                &cell,
                "PC2-TAX",
                "homogeneous",
                Some(shape.selection()),
                None,
                options.samples,
                options.mode.timed(),
            )?);
        }
    }
    Ok(())
}

fn run_product(
    runtime: &Runtime,
    options: &Options,
    rows: &mut Vec<CsvRow>,
    fixture_sha256: &mut BTreeMap<String, String>,
) -> Result<(), String> {
    let holdout_requested = product_shapes(true)
        .into_iter()
        .any(|shape| matches_filter(options, &product_cell_id("H-PC2", &shape.name)));
    if options.mode == Mode::Product && holdout_requested {
        let valid_round = matches!(options.round, Some(1 | 2));
        let valid_prereg = options.prereg_sha256.as_deref().is_some_and(valid_sha256);
        if !valid_round || !valid_prereg {
            return Err(
                "H-PC2 timing is embargoed: product timing requires --round 1|2 and --prereg-sha256 <64 lowercase hex>"
                    .to_string(),
            );
        }
    }
    for (context, role) in product_contexts() {
        let shapes = product_shapes(context.id == "H-PC2");
        let selected_shapes = shapes
            .into_iter()
            .filter(|shape| matches_filter(options, &product_cell_id(context.id, &shape.name)))
            .collect::<Vec<_>>();
        let fast_requested = context.id == "PC-M2"
            && [
                "PC-2-FAST/PC-M2/no_selection_4rg",
                "PC-2-FAST/PC-M2/no_selection_1rg",
                "PC-2-FAST/PC-M2/explicit_all_selected_4rg",
                "PC-2-FAST/PC-M2/explicit_all_selected_1rg",
            ]
            .into_iter()
            .any(|cell| matches_filter(options, cell));
        if selected_shapes.is_empty() && !fast_requested {
            continue;
        }

        let fixture = build_fixture(context, fixture_sha256)?;
        for shape in selected_shapes {
            let cell = product_cell_id(context.id, &shape.name);
            eprintln!("measuring {cell}");
            let arms = product_arms(role);
            rows.extend(measure_cell(
                runtime,
                &fixture,
                arms,
                &shape,
                &cell,
                group_for_role(role),
                role,
                Some(shape.selection()),
                None,
                options.samples,
                options.mode.timed(),
            )?);
        }
        if fast_requested {
            rows.extend(measure_fast_controls(
                runtime,
                options,
                &fixture,
                role,
                options.samples,
            )?);
        }
    }
    Ok(())
}

fn run_identity_a(
    runtime: &Runtime,
    options: &Options,
    rows: &mut Vec<CsvRow>,
    fixture_sha256: &mut BTreeMap<String, String>,
) -> Result<(), String> {
    for (context_id, run_len) in [("C6", 64), ("C0", 64), ("C0", 4)] {
        let context = find_context(context_id)?;
        let shape = OracleShape::l_sweep(run_len);
        let cell = format!("PC-2A-ID/{context_id}/{}", shape.name);
        if !matches_filter(options, &cell) {
            continue;
        }
        let fixture = build_fixture(context, fixture_sha256)?;
        eprintln!("measuring product-path identity witness {cell}");
        let measured = measure_cell(
            runtime,
            &fixture,
            options.mode.arms(),
            &shape,
            &cell,
            "PC2-IDENTITY-A",
            "homogeneous",
            Some(shape.selection()),
            None,
            options.samples,
            options.mode.timed(),
        )?;
        let product = measured
            .iter()
            .find(|row| row.measurement.arm == Pc2Arm::Pc2)
            .ok_or_else(|| format!("{cell} has no pc2 identity arm"))?;
        let decisions = product.measurement.decisions;
        if decisions.fallback_auto != ORACLE_ROW_GROUPS
            || decisions.fallback_forced != 0
            || decisions.engaged != 0
            || decisions.loaded_row_ranges_fallback != 0
        {
            return Err(format!(
                "{cell} product identity decision drifted: expected {ORACLE_ROW_GROUPS} Auto32 fallbacks and no other decisions, got {decisions:?}"
            ));
        }
        rows.extend(measured);
    }
    Ok(())
}

fn run_mr1_protocol(
    runtime: &Runtime,
    options: &Options,
    rows: &mut Vec<CsvRow>,
    fixture_sha256: &mut BTreeMap<String, String>,
) -> Result<(), String> {
    const SELECTOR_ARMS: &[Pc2Arm] = &[Pc2Arm::Mr1LegacySelectors, Pc2Arm::Mr1Selectors];
    for (context_id, run_len) in [("C6", 64), ("C0", 64), ("C0", 4), ("C0", 2)] {
        let shape = OracleShape::l_sweep(run_len);
        let cell = format!("MR1-A/{context_id}/{}", shape.name);
        if !matches_filter(options, &cell) {
            continue;
        }
        let fixture = build_fixture(find_context(context_id)?, fixture_sha256)?;
        rows.extend(measure_cell(
            runtime,
            &fixture,
            SELECTOR_ARMS,
            &shape,
            &cell,
            "MR1-A-WITNESS",
            "homogeneous",
            Some(shape.selection()),
            None,
            options.samples,
            options.mode.timed(),
        )?);
    }

    let mask_shape = OracleShape::l_sweep(4);
    let mask_cell = "MR1-A-MASK/C0/f50_l4";
    if matches_filter(options, mask_cell) {
        let fixture = build_fixture(find_context("C0")?, fixture_sha256)?;
        rows.extend(measure_cell(
            runtime,
            &fixture,
            &[Pc2Arm::Mr1BaseMask, Pc2Arm::Mr1Mask],
            &mask_shape,
            mask_cell,
            "MR1-A-NOTOUCH",
            "homogeneous",
            Some(mask_shape.selection()),
            None,
            options.samples,
            options.mode.timed(),
        )?);
    }

    for run_len in [4, 64] {
        let shape = OracleShape::l_sweep(run_len);
        let cell = format!("MR1-A-INTERLEAVE/C0/{}", shape.name);
        if !matches_filter(options, &cell) {
            continue;
        }
        let fixture = build_fixture(find_context("C0")?, fixture_sha256)?;
        rows.extend(measure_cell(
            runtime,
            &fixture,
            &[
                Pc2Arm::Mr1LegacySelectors,
                Pc2Arm::Mr1Independent,
                Pc2Arm::Mr1Selectors,
            ],
            &shape,
            &cell,
            "MR1-A-INTERLEAVE",
            "homogeneous",
            Some(shape.selection()),
            None,
            options.samples,
            options.mode.timed(),
        )?);
    }

    for row_groups in [1usize, 16] {
        let shape = OracleShape::l_sweep(64);
        let cell = format!(
            "MR1-A-SCALE/C0/rg{row_groups}-rows{ROWS_PER_GROUP}-{}",
            shape.name
        );
        if !matches_filter(options, &cell) {
            continue;
        }
        let context = find_context("C0")?;
        let fixture =
            build_oracle_fixture_with_dimensions(context, None, row_groups, ROWS_PER_GROUP)
                .map_err(|error| format!("cannot build MR-1 scale fixture {cell}: {error}"))?;
        fixture_sha256.insert(cell.clone(), fixture.bytes_sha256());
        let groups = (0..row_groups).collect::<Vec<_>>();
        rows.extend(measure_cell(
            runtime,
            &fixture,
            SELECTOR_ARMS,
            &shape,
            &cell,
            "MR1-A-SCALE",
            "homogeneous",
            Some(shape.selection_for_row_groups(row_groups)),
            Some(&groups),
            options.samples,
            options.mode.timed(),
        )?);
    }

    if options.mode == Mode::Mr1Smoke {
        run_mr1_smoke_controls(runtime, options, rows, fixture_sha256)?;
    }
    Ok(())
}

fn run_mr1_smoke_controls(
    runtime: &Runtime,
    options: &Options,
    rows: &mut Vec<CsvRow>,
    fixture_sha256: &mut BTreeMap<String, String>,
) -> Result<(), String> {
    let context = find_context("C0")?.with_batch_size(BATCH_SIZE - 1);
    let fixture = build_fixture(context, fixture_sha256)?;
    fixture_sha256.insert("C0-smoke-batch8191".to_string(), fixture.bytes_sha256());
    for (name, shape, selection) in [
        ("all-cursor", OracleShape::all_selected(), None),
        (
            "short-last-and-cross-batch",
            OracleShape::l_sweep(64),
            Some(RowSelection::from(vec![
                RowSelector::skip(3),
                RowSelector::select(BATCH_SIZE * 2 + 7),
                RowSelector::skip(5),
            ])),
        ),
        (
            "leading-and-trailing",
            OracleShape::leading_only(),
            Some(OracleShape::leading_only().selection()),
        ),
    ] {
        let cell = format!("MR1-SMOKE/C0/{name}");
        rows.extend(measure_cell(
            runtime,
            &fixture,
            &[Pc2Arm::Mr1LegacySelectors, Pc2Arm::Mr1Selectors],
            &shape,
            &cell,
            "MR1-SMOKE",
            "edge",
            selection,
            None,
            options.samples,
            false,
        )?);
    }

    let empty_shape = OracleShape::l_sweep(4);
    let empty = RowSelection::from(Vec::<RowSelector>::new());
    rows.extend(measure_cell(
        runtime,
        &fixture,
        &[Pc2Arm::Mr1LegacySelectors, Pc2Arm::Mr1Selectors],
        &empty_shape,
        "MR1-SMOKE/C0/empty-selection",
        "MR1-SMOKE",
        "edge",
        Some(empty),
        None,
        options.samples,
        false,
    )?);

    let zero_normalized = RowSelection::from(vec![
        RowSelector::skip(0),
        RowSelector::skip(3),
        RowSelector::select(0),
        RowSelector::select(7),
        RowSelector::skip(0),
    ]);
    rows.extend(measure_cell(
        runtime,
        &fixture,
        &[Pc2Arm::Mr1LegacySelectors, Pc2Arm::Mr1Selectors],
        &empty_shape,
        "MR1-SMOKE/C0/zero-count-normalization",
        "MR1-SMOKE",
        "edge",
        Some(zero_normalized),
        None,
        options.samples,
        false,
    )?);

    let holdout = MR1_HOLDOUT_CONTEXTS[0];
    let holdout_fixture = build_fixture(holdout, fixture_sha256)?;
    for run_len in [4, 64] {
        let shape = OracleShape::l_sweep(run_len);
        let cell = format!("MR1-SMOKE/H-MR/{}", shape.name);
        rows.extend(measure_cell(
            runtime,
            &holdout_fixture,
            mr1_product_arms("holdout", &shape),
            &shape,
            &cell,
            "MR1-SMOKE-HOLDOUT",
            "holdout",
            Some(shape.selection()),
            None,
            options.samples,
            false,
        )?);
    }
    Ok(())
}

fn run_mr1_product(
    runtime: &Runtime,
    options: &Options,
    rows: &mut Vec<CsvRow>,
    fixture_sha256: &mut BTreeMap<String, String>,
) -> Result<(), String> {
    let holdout_requested = mr1_product_shapes(true)
        .into_iter()
        .any(|shape| matches_filter(options, &mr1_product_cell_id("H-MR", &shape.name)));
    if holdout_requested {
        let valid_round = matches!(options.round, Some(1 | 2));
        let valid_prereg = options.prereg_sha256.as_deref().is_some_and(valid_sha256);
        if !valid_round || !valid_prereg {
            return Err(
                "H-MR timing is embargoed: use --round 1|2 and --prereg-sha256 <64 lowercase hex>"
                    .to_string(),
            );
        }
    }

    for (context, role) in mr1_product_contexts() {
        let shapes = mr1_product_shapes(context.id == "H-MR");
        let selected = shapes
            .into_iter()
            .filter(|shape| matches_filter(options, &mr1_product_cell_id(context.id, &shape.name)))
            .collect::<Vec<_>>();
        let diagnostic_requested = context.id == "C0"
            && [8, 16, 24]
                .into_iter()
                .any(|run_len| matches_filter(options, &format!("MR1-B-DIAG/C0/f50_l{run_len}")));
        if selected.is_empty() && !diagnostic_requested {
            continue;
        }
        let fixture = build_fixture(context, fixture_sha256)?;
        for shape in selected {
            let cell = mr1_product_cell_id(context.id, &shape.name);
            rows.extend(measure_cell(
                runtime,
                &fixture,
                mr1_product_arms(role, &shape),
                &shape,
                &cell,
                mr1_group_for_role(role),
                role,
                Some(shape.selection()),
                None,
                options.samples,
                true,
            )?);
        }
        if diagnostic_requested {
            for run_len in [8, 16, 24] {
                let shape = OracleShape::l_sweep(run_len);
                let cell = format!("MR1-B-DIAG/C0/{}", shape.name);
                if !matches_filter(options, &cell) {
                    continue;
                }
                rows.extend(measure_cell(
                    runtime,
                    &fixture,
                    &[Pc2Arm::Mr1LegacySelectors, Pc2Arm::Mr1Selectors],
                    &shape,
                    &cell,
                    "MR1-B-DIAG",
                    "diagnostic",
                    Some(shape.selection()),
                    None,
                    options.samples,
                    true,
                )?);
            }
        }
    }
    Ok(())
}

fn mr1_product_contexts() -> Vec<(OracleContext, &'static str)> {
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
        .chain(
            PC_MIXED_CONTEXTS
                .iter()
                .copied()
                .map(|context| (context, "mixed")),
        )
        .chain(
            MR1_HOLDOUT_CONTEXTS
                .iter()
                .copied()
                .map(|context| (context, "holdout")),
        )
        .collect()
}

fn mr1_product_shapes(holdout: bool) -> Vec<OracleShape> {
    if holdout {
        return [4, 64].into_iter().map(OracleShape::l_sweep).collect();
    }
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

fn mr1_product_arms(role: &str, shape: &OracleShape) -> &'static [Pc2Arm] {
    if shape.summary().avg_run_len < PRODUCT_OTHER_THRESHOLD as f64 {
        if role == "mixed" {
            &[Pc2Arm::Mr1BaseMask, Pc2Arm::Mr1Mask, Pc2Arm::Pc2]
        } else {
            &[Pc2Arm::Mr1BaseMask, Pc2Arm::Mr1Mask]
        }
    } else if role == "mixed" {
        &[
            Pc2Arm::Mr1LegacySelectors,
            Pc2Arm::Mr1Selectors,
            Pc2Arm::Pc2,
        ]
    } else {
        &[Pc2Arm::Mr1LegacySelectors, Pc2Arm::Mr1Selectors]
    }
}

fn mr1_product_cell_id(context: &str, shape: &str) -> String {
    format!("MR1-B/{context}/{shape}")
}

fn mr1_group_for_role(role: &str) -> &'static str {
    match role {
        "mixed" => "MR1-B-MIX",
        "holdout" => "MR1-B-HOLDOUT",
        _ => "MR1-B-HOM",
    }
}

fn build_fixture(
    context: OracleContext,
    fixture_sha256: &mut BTreeMap<String, String>,
) -> Result<OracleFixture, String> {
    eprintln!("building PC-2 {} fixture", context.id);
    let fixture = build_oracle_fixture(context, None)
        .map_err(|error| format!("cannot build PC-2 {} fixture: {error}", context.id))?;
    fixture_sha256
        .entry(context.id.to_string())
        .or_insert_with(|| fixture.bytes_sha256());
    Ok(fixture)
}

fn find_context(id: &str) -> Result<OracleContext, String> {
    ORACLE_CONTEXTS
        .iter()
        .copied()
        .find(|context| context.id == id)
        .ok_or_else(|| format!("missing PC-2 context {id}"))
}

fn product_contexts() -> Vec<(OracleContext, &'static str)> {
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
        .chain(
            PC_MIXED_CONTEXTS
                .iter()
                .copied()
                .map(|context| (context, "mixed")),
        )
        .chain(
            PC2_HOLDOUT_CONTEXTS
                .iter()
                .copied()
                .map(|context| (context, "holdout")),
        )
        .collect()
}

fn product_shapes(holdout: bool) -> Vec<OracleShape> {
    if holdout {
        return [8, 16].into_iter().map(OracleShape::l_sweep).collect();
    }
    [4, 8, 16, 64, 256, 1_024]
        .into_iter()
        .map(OracleShape::l_sweep)
        .chain([
            OracleShape::selectivity(2, 64),
            OracleShape::selectivity(98, 64),
            OracleShape::pc_bursty03_l4(),
        ])
        .collect()
}

fn product_arms(role: &str) -> &'static [Pc2Arm] {
    if role == "mixed" {
        &[Pc2Arm::Auto32, Pc2Arm::Pc2, Pc2Arm::Pc2R16]
    } else {
        &[Pc2Arm::Auto32, Pc2Arm::Pc2]
    }
}

fn list_cells(options: &Options) {
    let mut count = 0usize;
    match options.mode {
        Mode::Tax => {
            for (context, run_len) in [("C6", 64), ("C0", 64), ("C0", 4)] {
                let cell = format!("PC-2A/{context}/f50_l{run_len}");
                if matches_filter(options, &cell) {
                    println!("{cell}");
                    count += 1;
                }
            }
        }
        Mode::Product | Mode::Smoke => {
            for (context, _) in product_contexts() {
                for shape in product_shapes(context.id == "H-PC2") {
                    let cell = product_cell_id(context.id, &shape.name);
                    if matches_filter(options, &cell) {
                        println!("{cell}");
                        count += 1;
                    }
                }
            }
            for cell in [
                "PC-2-FAST/PC-M2/no_selection_4rg",
                "PC-2-FAST/PC-M2/no_selection_1rg",
                "PC-2-FAST/PC-M2/explicit_all_selected_4rg",
                "PC-2-FAST/PC-M2/explicit_all_selected_1rg",
            ] {
                if matches_filter(options, cell) {
                    println!("{cell}");
                    count += 1;
                }
            }
        }
        Mode::IdentityA => {
            for (context, run_len) in [("C6", 64), ("C0", 64), ("C0", 4)] {
                let cell = format!("PC-2A-ID/{context}/f50_l{run_len}");
                if matches_filter(options, &cell) {
                    println!("{cell}");
                    count += 1;
                }
            }
        }
        Mode::Mr1Smoke | Mode::Mr1Protocol => {
            for (context, run_len) in [("C6", 64), ("C0", 64), ("C0", 4), ("C0", 2)] {
                let cell = format!("MR1-A/{context}/f50_l{run_len}");
                if matches_filter(options, &cell) {
                    println!("{cell}");
                    count += 1;
                }
            }
            for cell in [
                "MR1-A-MASK/C0/f50_l4",
                "MR1-A-INTERLEAVE/C0/f50_l4",
                "MR1-A-INTERLEAVE/C0/f50_l64",
                "MR1-A-SCALE/C0/rg1-rows65536-f50_l64",
                "MR1-A-SCALE/C0/rg16-rows65536-f50_l64",
            ] {
                if matches_filter(options, cell) {
                    println!("{cell}");
                    count += 1;
                }
            }
        }
        Mode::Mr1Product => {
            for (context, role) in mr1_product_contexts() {
                for shape in mr1_product_shapes(context.id == "H-MR") {
                    let cell = mr1_product_cell_id(context.id, &shape.name);
                    if matches_filter(options, &cell) {
                        println!("{cell}");
                        count += 1;
                    }
                }
                if role == "homogeneous" && context.id == "C0" {
                    for run_len in [8, 16, 24] {
                        let cell = format!("MR1-B-DIAG/C0/f50_l{run_len}");
                        if matches_filter(options, &cell) {
                            println!("{cell}");
                            count += 1;
                        }
                    }
                }
            }
        }
    }
    eprintln!("listed {count} PC-2 {} cells", options.mode.label());
}

#[allow(clippy::too_many_arguments)]
fn measure_cell(
    runtime: &Runtime,
    fixture: &OracleFixture,
    arms: &[Pc2Arm],
    shape: &OracleShape,
    cell: &str,
    group: &'static str,
    context_role: &'static str,
    selection: Option<RowSelection>,
    row_groups: Option<&[usize]>,
    samples: usize,
    timed: bool,
) -> Result<Vec<CsvRow>, String> {
    let expected_rows = selection.as_ref().map_or_else(
        || {
            row_groups.map_or_else(
                || shape.total_selected_rows(),
                |groups| groups.len() * shape.summary().selected_rows,
            )
        },
        RowSelection::row_count,
    );
    let checks = check_arms(
        runtime,
        fixture,
        selection.as_ref(),
        row_groups,
        arms,
        expected_rows,
        cell,
    )?;
    if !timed {
        let context = fixture.context();
        return Ok(arms
            .iter()
            .copied()
            .enumerate()
            .map(|(index, arm)| CsvRow {
                group,
                cell_id: cell.to_string(),
                context,
                context_role,
                shape_name: shape.name.clone(),
                nominal_skip: shape.nominal_skip,
                nominal_select: shape.nominal_select,
                summary: shape.summary(),
                selection_source: if selection.is_some() {
                    "external"
                } else {
                    "none"
                },
                row_groups: row_groups.map_or(ORACLE_ROW_GROUPS, <[usize]>::len),
                measurement: measurement(arm, Vec::new(), Vec::new(), &checks[index]),
            })
            .collect());
    }
    warm_arms(
        runtime,
        fixture,
        selection.as_ref(),
        row_groups,
        arms,
        expected_rows,
        cell,
    )?;

    let mut values = (0..arms.len())
        .map(|_| Vec::with_capacity(samples))
        .collect::<Vec<Vec<u64>>>();
    let mut timestamps = (0..arms.len())
        .map(|_| Vec::with_capacity(samples))
        .collect::<Vec<Vec<u64>>>();
    let order = (0..arms.len())
        .chain((0..arms.len()).rev())
        .collect::<Vec<_>>();
    while values.iter().any(|values| values.len() < samples) {
        for &index in &order {
            if values[index].len() == samples {
                continue;
            }
            let (elapsed, result, timestamp) = time_arm(
                runtime,
                fixture,
                selection.as_ref(),
                row_groups,
                arms[index],
            )?;
            if result.row_count != expected_rows {
                return Err(format!(
                    "{cell}/{} timed row mismatch: expected {expected_rows}, got {}",
                    arms[index].label(),
                    result.row_count
                ));
            }
            values[index].push(elapsed);
            timestamps[index].push(timestamp);
        }
    }

    let context = fixture.context();
    Ok(arms
        .iter()
        .copied()
        .enumerate()
        .map(|(index, arm)| CsvRow {
            group,
            cell_id: cell.to_string(),
            context,
            context_role,
            shape_name: shape.name.clone(),
            nominal_skip: shape.nominal_skip,
            nominal_select: shape.nominal_select,
            summary: shape.summary(),
            selection_source: if selection.is_some() {
                "external"
            } else {
                "none"
            },
            row_groups: row_groups.map_or(ORACLE_ROW_GROUPS, <[usize]>::len),
            measurement: measurement(
                arm,
                std::mem::take(&mut values[index]),
                std::mem::take(&mut timestamps[index]),
                &checks[index],
            ),
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
    for (cell, row_groups, explicit_selection) in [
        ("PC-2-FAST/PC-M2/no_selection_4rg", None, false),
        (
            "PC-2-FAST/PC-M2/no_selection_1rg",
            Some(vec![0usize]),
            false,
        ),
        ("PC-2-FAST/PC-M2/explicit_all_selected_4rg", None, true),
        (
            "PC-2-FAST/PC-M2/explicit_all_selected_1rg",
            Some(vec![0usize]),
            true,
        ),
    ] {
        if !matches_filter(options, cell) {
            continue;
        }
        let selected_row_groups = row_groups.as_ref().map_or(ORACLE_ROW_GROUPS, Vec::len);
        let selection = explicit_selection.then(|| {
            RowSelection::from(vec![RowSelector::select(
                selected_row_groups * ROWS_PER_GROUP,
            )])
        });
        rows.extend(measure_cell(
            runtime,
            fixture,
            &[Pc2Arm::Auto32, Pc2Arm::Pc2],
            &shape,
            cell,
            "PC2-FAST",
            context_role,
            selection,
            row_groups.as_deref(),
            samples,
            options.mode.timed(),
        )?);
    }
    Ok(rows)
}

fn check_arms(
    runtime: &Runtime,
    fixture: &OracleFixture,
    selection: Option<&RowSelection>,
    row_groups: Option<&[usize]>,
    arms: &[Pc2Arm],
    expected_rows: usize,
    cell: &str,
) -> Result<Vec<ScanResult>, String> {
    let mut checks: Vec<ScanResult> = Vec::with_capacity(arms.len());
    for &arm in arms {
        let result =
            runtime.block_on(run_scan(fixture, selection.cloned(), row_groups, arm, true))?;
        if result.row_count != expected_rows {
            return Err(format!(
                "{cell}/{} correctness row mismatch: expected {expected_rows}, got {}",
                arm.label(),
                result.row_count
            ));
        }
        let Some(content) = &result.content else {
            return Err(format!("{cell}/{} has no content digest", arm.label()));
        };
        if let Some(first) = checks.first()
            && (first.row_count != result.row_count || first.content.as_ref() != Some(content))
        {
            return Err(format!(
                "{cell}: {} differs from {} in row count or full projected content",
                arm.label(),
                arms[0].label()
            ));
        }
        match arm {
            Pc2Arm::Mr1Selectors if selection.is_some() => {
                if result.driver_batches != result.batch_count {
                    return Err(format!(
                        "{cell}/{} driver call contract failed: batches={}, driver_batches={}",
                        arm.label(),
                        result.batch_count,
                        result.driver_batches
                    ));
                }
                let expected_calls =
                    result.driver_batches * (fixture.context().payload_columns + 1);
                if result.read_selection_calls != expected_calls {
                    return Err(format!(
                        "{cell}/{} fanout contract failed: expected {expected_calls} read_selection calls, got {}",
                        arm.label(),
                        result.read_selection_calls
                    ));
                }
            }
            Pc2Arm::Mr1LegacySelectors
            | Pc2Arm::Mr1BaseMask
            | Pc2Arm::Mr1Mask
            | Pc2Arm::Mr1Independent => {
                if result.driver_batches != 0 || result.read_selection_calls != 0 {
                    return Err(format!(
                        "{cell}/{} no-touch call contract failed: driver_batches={}, read_selection_calls={}",
                        arm.label(),
                        result.driver_batches,
                        result.read_selection_calls
                    ));
                }
            }
            _ => {}
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
    arms: &[Pc2Arm],
    expected_rows: usize,
    cell: &str,
) -> Result<(), String> {
    for _ in 0..WARMUPS_PER_ARM {
        for &arm in arms {
            let result = runtime.block_on(run_scan(
                fixture,
                selection.cloned(),
                row_groups,
                arm,
                false,
            ))?;
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
    arm: Pc2Arm,
) -> Result<(u64, ScanResult, u64), String> {
    // Selection cloning and all metrics are deliberately outside the timed
    // scan. Decision counters come from the separate correctness pass.
    let selection = selection.cloned();
    let timestamp = unix_nanos();
    let started = Instant::now();
    let result = runtime.block_on(run_scan(fixture, selection, row_groups, arm, false))?;
    let elapsed = started.elapsed().as_nanos() as u64;
    hint::black_box(result.row_count);
    Ok((elapsed, result, timestamp))
}

async fn run_scan(
    fixture: &OracleFixture,
    selection: Option<RowSelection>,
    row_groups: Option<&[usize]>,
    arm: Pc2Arm,
    attribution: bool,
) -> Result<ScanResult, String> {
    let context = fixture.context();
    if arm == Pc2Arm::Mr1Independent {
        let mut row_count = None;
        let mut batch_count = None;
        for column_idx in 0..context.payload_columns {
            let result = run_scan_projection(
                fixture,
                selection.clone(),
                row_groups,
                Pc2Arm::Mr1LegacySelectors,
                false,
                column_idx..column_idx + 1,
            )
            .await?;
            if row_count.is_some_and(|expected| expected != result.row_count)
                || batch_count.is_some_and(|expected| expected != result.batch_count)
            {
                return Err(
                    "MR-1 independent projections lost lockstep row/batch counts".to_string(),
                );
            }
            row_count = Some(result.row_count);
            batch_count = Some(result.batch_count);
        }
        // The independent arm proves each projection reaches the same output
        // boundary. Its untimed correctness digest is taken from an additional
        // full legacy projection; only the eight independent scans are timed.
        let content = if attribution {
            run_scan_projection(
                fixture,
                selection,
                row_groups,
                Pc2Arm::Mr1LegacySelectors,
                true,
                0..context.payload_columns,
            )
            .await?
            .content
        } else {
            None
        };
        return Ok(ScanResult {
            row_count: row_count.unwrap_or_default(),
            batch_count: batch_count.unwrap_or_default(),
            content,
            decisions: PerColumnDecisionMetrics::default(),
            read_selection_calls: 0,
            driver_batches: 0,
        });
    }
    run_scan_projection(
        fixture,
        selection,
        row_groups,
        arm,
        attribution,
        0..context.payload_columns,
    )
    .await
}

async fn run_scan_projection(
    fixture: &OracleFixture,
    selection: Option<RowSelection>,
    row_groups: Option<&[usize]>,
    arm: Pc2Arm,
    attribution: bool,
    projection_roots: std::ops::Range<usize>,
) -> Result<ScanResult, String> {
    let projection = ProjectionMask::roots(fixture.schema_descr(), projection_roots);
    let metrics = if attribution {
        ArrowReaderMetrics::enabled()
    } else {
        ArrowReaderMetrics::disabled()
    };
    let mut builder = ParquetRecordBatchStreamBuilder::new(fixture.reader())
        .await
        .map_err(|error| format!("cannot build PC-2 stream metadata: {error}"))?
        .with_batch_size(context.batch_size)
        .with_projection(projection)
        .with_metrics(metrics.clone())
        .with_row_selection_policy(arm.policy());
    if let Some(row_groups) = row_groups {
        builder = builder.with_row_groups(row_groups.to_vec());
    }
    if let Some(selection) = selection {
        builder = builder.with_row_selection(selection);
    }
    let mut stream = builder
        .build()
        .map_err(|error| format!("cannot build PC-2 stream: {error}"))?;
    let mut row_count = 0usize;
    let mut batch_count = 0usize;
    let mut digester = attribution.then(ProjectedContentDigester::default);
    while let Some(batch) = stream.next().await {
        let batch = batch.map_err(|error| format!("PC-2 stream failed: {error}"))?;
        if let Some(digester) = &mut digester {
            digester.update(&batch);
        }
        row_count += batch.num_rows();
        batch_count += 1;
    }
    let snapshot = metrics.decomposition().unwrap_or_default();
    Ok(ScanResult {
        row_count,
        batch_count,
        content: digester.map(ProjectedContentDigester::finish),
        decisions: snapshot.per_column_decisions,
        read_selection_calls: snapshot.read_selection_calls,
        driver_batches: snapshot.driver_batches,
    })
}

fn measurement(
    arm: Pc2Arm,
    samples_ns: Vec<u64>,
    sample_started_unix_ns: Vec<u64>,
    check: &ScanResult,
) -> ArmMeasurement {
    let median_ns = (!samples_ns.is_empty())
        .then(|| median(&samples_ns))
        .unwrap_or_default();
    let deviations = samples_ns
        .iter()
        .map(|sample| sample.abs_diff(median_ns))
        .collect::<Vec<_>>();
    ArmMeasurement {
        arm,
        samples_ns,
        sample_started_unix_ns,
        median_ns,
        mad_ns: (!deviations.is_empty())
            .then(|| median(&deviations))
            .unwrap_or_default(),
        rows_out: check.row_count,
        batches_out: check.batch_count,
        content: check.content.clone().unwrap(),
        decisions: check.decisions,
        read_selection_calls: check.read_selection_calls,
        driver_batches: check.driver_batches,
    }
}

fn write_csv(path: &Path, rows: &[CsvRow], mr1: bool) -> Result<(), String> {
    let file = File::create(path)
        .map_err(|error| format!("cannot create PC-2 CSV {}: {error}", path.display()))?;
    let mut writer = BufWriter::new(file);
    let header = if mr1 {
        "schema_version,group,cell_id,context_id,context_role,output_layout,payload_spec,payload_columns,encoding,compression,page_index,batch_size,rows_per_group,row_groups,shape_name,skip_rows,select_rows,selected_fraction,avg_run_len,run_count,selection_source,arm,column_strategies,dict_threshold,plain_threshold,sample_count,samples_ns,sample_started_unix_ns,median_ns,mad_ns,rows_out,batches_out,schema_sha256,leaf_sha256,fallback_auto,fallback_forced,engaged,loaded_row_ranges_fallback,read_selection_calls,driver_batches"
    } else {
        "schema_version,group,cell_id,context_id,context_role,output_layout,payload_spec,payload_columns,encoding,compression,page_index,batch_size,rows_per_group,row_groups,shape_name,skip_rows,select_rows,selected_fraction,avg_run_len,run_count,selection_source,arm,column_strategies,dict_threshold,plain_threshold,sample_count,samples_ns,sample_started_unix_ns,median_ns,mad_ns,rows_out,schema_sha256,leaf_sha256,fallback_auto,fallback_forced,engaged,loaded_row_ranges_fallback"
    };
    writeln!(writer, "{header}")
        .map_err(|error| format!("cannot write PC-2 CSV header: {error}"))?;
    for row in rows {
        let measurement = &row.measurement;
        let decisions = measurement.decisions;
        let mut fields = vec![
            if mr1 {
                MR1_CSV_SCHEMA_VERSION.to_string()
            } else {
                CSV_SCHEMA_VERSION.to_string()
            },
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
                .map_or_else(String::new, |value| value.to_string()),
            row.nominal_select
                .map_or_else(String::new, |value| value.to_string()),
            format!("{:.9}", row.summary.selected_fraction),
            format!("{:.9}", row.summary.avg_run_len),
            row.summary.run_count.to_string(),
            row.selection_source.to_string(),
            measurement.arm.label().to_string(),
            column_strategies(row),
            dictionary_threshold(measurement.arm).to_string(),
            other_threshold(measurement.arm).to_string(),
            measurement.samples_ns.len().to_string(),
            join_numbers(&measurement.samples_ns),
            join_numbers(&measurement.sample_started_unix_ns),
            measurement.median_ns.to_string(),
            measurement.mad_ns.to_string(),
            measurement.rows_out.to_string(),
        ];
        if mr1 {
            fields.push(measurement.batches_out.to_string());
        }
        fields.extend([
            measurement.content.schema_sha256.clone(),
            measurement.content.leaf_sha256.join("|"),
            decisions.fallback_auto.to_string(),
            decisions.fallback_forced.to_string(),
            decisions.engaged.to_string(),
            decisions.loaded_row_ranges_fallback.to_string(),
        ]);
        if mr1 {
            fields.extend([
                measurement.read_selection_calls.to_string(),
                measurement.driver_batches.to_string(),
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
        .map_err(|error| format!("cannot write PC-2 CSV row: {error}"))?;
    }
    writer
        .flush()
        .map_err(|error| format!("cannot flush PC-2 CSV {}: {error}", path.display()))
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
        return "all-selected-fast-path".to_string();
    }
    let fixed_strategy = match row.measurement.arm {
        Pc2Arm::Mr1LegacySelectors | Pc2Arm::Mr1Selectors | Pc2Arm::Mr1Independent => {
            Some("selectors")
        }
        Pc2Arm::Mr1BaseMask | Pc2Arm::Mr1Mask => Some("mask"),
        _ => None,
    };
    (0..row.context.payload_columns)
        .map(|column_idx| {
            if let Some(strategy) = fixed_strategy {
                return format!("{column_idx}:{strategy}");
            }
            let threshold = match row.measurement.arm {
                Pc2Arm::Auto32 | Pc2Arm::Pc2Thin => PRODUCT_OTHER_THRESHOLD,
                Pc2Arm::PcBolton | Pc2Arm::Pc2R16 => {
                    if pure_dictionary(row.context.payload_at(column_idx)) {
                        PURE_DICTIONARY_THRESHOLD
                    } else {
                        LEGACY_OTHER_THRESHOLD
                    }
                }
                Pc2Arm::Pc2 => {
                    if pure_dictionary(row.context.payload_at(column_idx)) {
                        PURE_DICTIONARY_THRESHOLD
                    } else {
                        PRODUCT_OTHER_THRESHOLD
                    }
                }
                Pc2Arm::Mr1LegacySelectors
                | Pc2Arm::Mr1Selectors
                | Pc2Arm::Mr1BaseMask
                | Pc2Arm::Mr1Mask
                | Pc2Arm::Mr1Independent => unreachable!(),
            };
            let strategy = if row.summary.avg_run_len < threshold as f64 {
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

const fn other_threshold(arm: Pc2Arm) -> usize {
    match arm {
        Pc2Arm::PcBolton | Pc2Arm::Pc2R16 => LEGACY_OTHER_THRESHOLD,
        Pc2Arm::Auto32
        | Pc2Arm::Pc2Thin
        | Pc2Arm::Pc2
        | Pc2Arm::Mr1LegacySelectors
        | Pc2Arm::Mr1Selectors
        | Pc2Arm::Mr1BaseMask
        | Pc2Arm::Mr1Mask
        | Pc2Arm::Mr1Independent => PRODUCT_OTHER_THRESHOLD,
    }
}

const fn dictionary_threshold(arm: Pc2Arm) -> usize {
    match arm {
        Pc2Arm::Auto32 | Pc2Arm::Pc2Thin => PRODUCT_OTHER_THRESHOLD,
        Pc2Arm::PcBolton
        | Pc2Arm::Pc2
        | Pc2Arm::Pc2R16
        | Pc2Arm::Mr1LegacySelectors
        | Pc2Arm::Mr1Selectors
        | Pc2Arm::Mr1BaseMask
        | Pc2Arm::Mr1Mask
        | Pc2Arm::Mr1Independent => PURE_DICTIONARY_THRESHOLD,
    }
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
    let arm_labels = rows
        .iter()
        .map(|row| row.measurement.arm.label())
        .collect::<BTreeSet<_>>();
    let mr1 = options.mode.is_mr1();
    let manifest = json!({
        "schema_version": if mr1 {
            MR1_MANIFEST_SCHEMA_VERSION
        } else {
            MANIFEST_SCHEMA_VERSION
        },
        "benchmark": if mr1 {
            "arrow_reader_row_selection_mr1"
        } else {
            "arrow_reader_row_selection_pc2"
        },
        "mode": options.mode.label(),
        "csv_schema_version": if mr1 {
            MR1_CSV_SCHEMA_VERSION
        } else {
            CSV_SCHEMA_VERSION
        },
        "git_sha": git_sha,
        "git_status_porcelain": git_status,
        "same_sha_arms": arm_labels,
        "timed": options.mode.timed(),
        "round": options.round,
        "preregistration_sha256": options.prereg_sha256.as_deref(),
        "rustc": rustc,
        "cpu_model": cpu_model,
        "hostname": hostname,
        "os": os,
        "environment_fingerprint": format!("{:016x}", environment_hasher.finish()),
        "started_unix_ns": started_unix_ns,
        "completed_unix_ns": completed_unix_ns,
        "elapsed_ns": elapsed_ns,
        "samples_per_arm": if options.mode.timed() { options.samples } else { 0 },
        "warmups_per_arm": if options.mode.timed() { WARMUPS_PER_ARM } else { 0 },
        "filter": options.filter_text.as_deref(),
        "cell_count": cells.len(),
        "arm_row_count": rows.len(),
        "measured_context_ids": context_ids,
        "declared_matrix": {
            "tax_witnesses": ["PC-2A/C6/f50_l64", "PC-2A/C0/f50_l64", "PC-2A/C0/f50_l4"],
            "product_homogeneous_contexts": 44,
            "product_mixed_contexts": 6,
            "product_holdout": if mr1 {
                json!({"id": "H-MR", "shapes": ["f50_l4", "f50_l64"]})
            } else {
                json!({"id": "H-PC2", "shapes": ["f50_l8", "f50_l16"]})
            },
            "product_shapes_except_holdout": [
                "f50_l4", "f50_l8", "f50_l16", "f50_l64", "f50_l256",
                "f50_l1024", "f02_l64", "f98_l64", "bursty03_l4"
            ],
            "fast_controls": [
                "no_selection_4rg", "no_selection_1rg",
                "explicit_all_selected_4rg", "explicit_all_selected_1rg"
            ]
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
        "policy_contract": {
            "product": {
                "pure_dictionary_threshold": PURE_DICTIONARY_THRESHOLD,
                "other_column_threshold": PRODUCT_OTHER_THRESHOLD,
                "uniform_auto32": "fallback_auto",
                "uniform_non_auto32": "fallback_forced",
                "mixed_strategy": "engaged"
            },
            "pc_bolton": {"pure_dictionary_threshold": 4, "other_column_threshold": 16},
            "pc2_thin": "force every column to the Auto32 decision but retain the native coordinator",
            "pc2_r16": {"pure_dictionary_threshold": 4, "other_column_threshold": 16},
            "scope": "flat output projection; unsupported shapes and loaded row ranges fall back to Auto32"
        },
        "native_plan_contract": {
            "compile_frequency": "once per row group",
            "selector_form": "flat batch-sliced RowSelector instructions",
            "physical_span": "gap_skip, span_start, span_rows, selected per output batch",
            "mask_materialization": "once only when at least one projected column uses Mask",
            "forbidden_engaged_runtime": ["MaskRunIter", "next_chunk", "per-row mask.value"]
        },
        "shared_filter_contract": {
            "plan": "one optimized FilterPredicate per output batch",
            "reuse": "the same predicate is applied to every Mask column in that batch"
        },
        "decision_counter_contract": if mr1 {
            json!({
                "source": "separate untimed correctness scan with ArrowReaderMetrics enabled",
                "fields": [
                    "fallback_auto", "fallback_forced", "engaged", "loaded_row_ranges_fallback",
                    "read_selection_calls", "driver_batches"
                ],
                "mr1_selector_contract": "driver_batches == output batches; read_selection_calls == driver_batches * (1 + projected root children)",
                "mr1_mask_and_legacy_contract": "read_selection_calls == driver_batches == 0",
                "timed_scan_metrics": "disabled"
            })
        } else {
            json!({
                "source": "separate untimed correctness scan with ArrowReaderMetrics enabled",
                "fields": ["fallback_auto", "fallback_forced", "engaged", "loaded_row_ranges_fallback"],
                "timed_scan_metrics": "disabled"
            })
        },
        "timing_protocol": {
            "order": "all arms forward then reverse, repeated",
            "statistic": "median",
            "dispersion": "median_absolute_deviation",
            "clock": if options.mode.timed() { "std::time::Instant" } else { "disabled" },
            "sample_start_clock": if options.mode.timed() { "unix_epoch_nanoseconds" } else { "disabled" },
            "selection_clone": "outside timed region"
        },
        "correctness": {
            "hard_gate": "row count, schema digest, and every projected leaf digest equal across all same-SHA arms",
            "digest": "arrow-projected-leaf-content-v1"
        },
        "holdout_protocol": if mr1 {
            json!({
                "H-MR_timing_embargo": "mr1-product mode requires --round 1|2 plus the lowercase SHA-256 supplied by the retained dfexp request; mr1-smoke never calls Instant",
                "external_binding": "closure analysis must verify the retained request argument and this manifest value against the actual preregistration.json bytes"
            })
        } else {
            json!({
                "H-PC2_timing_embargo": "product mode requires --round 1|2 plus the lowercase SHA-256 supplied by the retained dfexp request; smoke never calls Instant and identity-a can address only C0/C6 witnesses",
                "external_binding": "closure analysis must verify the retained request argument and this manifest value against the actual preregistration.json bytes"
            })
        },
        "csv_sha256": sha256_path(&options.csv)?,
        "classification": "non-formal"
    });
    let file = File::create(&options.manifest).map_err(|error| {
        format!(
            "cannot create PC-2 manifest {}: {error}",
            options.manifest.display()
        )
    })?;
    serde_json::to_writer_pretty(file, &manifest)
        .map_err(|error| format!("cannot write PC-2 manifest: {error}"))?;
    let mut file = fs::OpenOptions::new()
        .append(true)
        .open(&options.manifest)
        .map_err(|error| format!("cannot reopen PC-2 manifest: {error}"))?;
    writeln!(file).map_err(|error| format!("cannot finish PC-2 manifest: {error}"))
}

fn parse_options() -> Result<Options, String> {
    let mut mode = None;
    let mut list = false;
    let mut samples = DEFAULT_SAMPLES;
    let mut filter_text = None;
    let mut csv = None;
    let mut manifest = None;
    let mut emit_artifacts = false;
    let mut round = None;
    let mut prereg_sha256 = None;
    let mut args = env::args().skip(1);
    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--selection-oracle" | "--bench" => {}
            "--pc2-tax" => set_mode(&mut mode, Mode::Tax)?,
            "--pc2-product" => set_mode(&mut mode, Mode::Product)?,
            "--pc2-smoke" => set_mode(&mut mode, Mode::Smoke)?,
            "--pc2-identity-a" => set_mode(&mut mode, Mode::IdentityA)?,
            "--mr1-smoke" => set_mode(&mut mode, Mode::Mr1Smoke)?,
            "--mr1-protocol" => set_mode(&mut mode, Mode::Mr1Protocol)?,
            "--mr1-product" => set_mode(&mut mode, Mode::Mr1Product)?,
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
            "--round" => {
                round = Some(
                    args.next()
                        .ok_or_else(|| "--round requires 1 or 2".to_string())?
                        .parse::<u8>()
                        .map_err(|_| "--round requires 1 or 2".to_string())?,
                );
            }
            "--prereg-sha256" => {
                prereg_sha256 = Some(
                    args.next()
                        .ok_or_else(|| "--prereg-sha256 requires a value".to_string())?,
                );
            }
            "--help" | "-h" => {
                println!(
                    "arrow_reader_row_selection_oracle (--pc2-tax | --pc2-product | --pc2-smoke | --pc2-identity-a | --mr1-smoke | --mr1-protocol | --mr1-product) \
                     [--list] [--filter REGEX] [--samples EVEN] \
                     [--round 1|2 --prereg-sha256 HEX] \
                     [--csv PATH] [--manifest PATH] [--emit-artifacts]"
                );
                std::process::exit(0);
            }
            _ => return Err(format!("unsupported PC-2 argument {argument:?}")),
        }
    }
    let mode = mode.ok_or_else(|| "one PC-2/MR-1 mode flag is required".to_string())?;
    if round.is_some_and(|round| !matches!(round, 1 | 2)) {
        return Err("--round requires 1 or 2".to_string());
    }
    if !(2..=100).contains(&samples) || !samples.is_multiple_of(2) {
        return Err("--samples must be an even integer in 2..=100".to_string());
    }
    let csv = csv.unwrap_or_else(|| default_artifact_path(&format!("pc2-{}.csv", mode.label())));
    let manifest = manifest
        .unwrap_or_else(|| default_artifact_path(&format!("pc2-{}-manifest.json", mode.label())));
    if csv == manifest {
        return Err("--csv and --manifest must name different paths".to_string());
    }
    let filter = filter_text
        .as_deref()
        .map(Regex::new)
        .transpose()
        .map_err(|error| format!("invalid --filter regex: {error}"))?;
    Ok(Options {
        mode,
        list,
        samples,
        filter,
        filter_text,
        csv,
        manifest,
        emit_artifacts,
        round,
        prereg_sha256,
    })
}

fn set_mode(mode: &mut Option<Mode>, candidate: Mode) -> Result<(), String> {
    if let Some(current) = mode
        && *current != candidate
    {
        return Err("PC-2 mode flags are mutually exclusive".to_string());
    }
    *mode = Some(candidate);
    Ok(())
}

fn product_cell_id(context: &str, shape: &str) -> String {
    format!("PC-2/{context}/{shape}")
}

fn group_for_role(role: &str) -> &'static str {
    match role {
        "mixed" => "PC2-MIX",
        "holdout" => "PC2-HOLDOUT",
        _ => "PC2-HOM",
    }
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || matches!(byte, b'a'..=b'f'))
}

fn matches_filter(options: &Options, cell: &str) -> bool {
    options
        .filter
        .as_ref()
        .is_none_or(|filter| filter.is_match(cell))
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

fn emit_artifact(prefix: &str, kind: &str, path: &Path) -> Result<(), String> {
    let raw = fs::read_to_string(path)
        .map_err(|error| format!("cannot read PC-2 artifact {}: {error}", path.display()))?;
    println!("{prefix}_{kind}_BEGIN");
    print!("{raw}");
    if !raw.ends_with('\n') {
        println!();
    }
    println!("{prefix}_{kind}_END");
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
        .unwrap_or_default()
        .as_nanos() as u64
}
