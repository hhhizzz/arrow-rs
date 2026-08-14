// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements. See the NOTICE file distributed
// with this work for additional information regarding copyright ownership.
// The ASF licenses this file to You under the Apache License, Version 2.0.

//! Frozen CAL-0 calibration harness for arrow-rs #8846.

use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use parquet::arrow::arrow_reader::metrics::ArrowReaderMetrics;
use serde_json::json;
use tokio::runtime::Runtime;

use super::fixture::{
    ORACLE_ROW_GROUPS, OracleCompression, OracleContext, OraclePayload, build_oracle_fixture,
};
use super::model::ROWS_PER_GROUP;
use super::runner::{
    OracleArm, OracleSelectionSource, run_oracle, run_oracle_row_group_with_metrics,
};
use super::shapes::OracleShape;

const SAMPLES: usize = 12;
const WARMUPS: usize = 2;

#[derive(Clone, Debug)]
struct CalContext {
    context: OracleContext,
    class: &'static str,
    role: &'static str,
}

pub(crate) fn main() {
    let output_dir = parse_output_dir();
    fs::create_dir_all(&output_dir).unwrap();
    let runtime = Runtime::new().unwrap();
    let contexts = contexts();
    assert_eq!(contexts.len(), 22);
    let mut csv = String::from(
        "schema_version,context_id,class,role,dtype,payload_columns,encoding,compression,shape_name,selected_rows,skipped_rows,run_count,arm,samples_ns,median_ns,mad_ns,rows_out,checksum,filter_record_batch_ns,selectors_to_mask_ns,skip_records_calls,skip_records_rows,read_records_calls,read_records_rows\n",
    );

    for (context_index, spec) in contexts.iter().enumerate() {
        eprintln!(
            "CAL0_CONTEXT {}/{} {}",
            context_index + 1,
            contexts.len(),
            spec.context.id
        );
        let fixture = build_oracle_fixture(spec.context, None).unwrap();
        for shape in shapes_for(spec.role) {
            let summary = shape.summary();
            let selection = shape.selection();
            let selector_check = runtime.block_on(run_oracle(
                &fixture,
                Some(selection.clone()),
                OracleSelectionSource::External,
                OracleArm::Selectors,
                true,
            ));
            let mask_check = runtime.block_on(run_oracle(
                &fixture,
                Some(selection.clone()),
                OracleSelectionSource::External,
                OracleArm::Mask,
                true,
            ));
            assert_eq!(selector_check, mask_check);

            for arm in [OracleArm::Selectors, OracleArm::Mask] {
                for _ in 0..WARMUPS {
                    runtime.block_on(run_oracle(
                        &fixture,
                        Some(selection.clone()),
                        OracleSelectionSource::External,
                        arm,
                        false,
                    ));
                }
            }
            let mut samples: BTreeMap<OracleArm, Vec<u64>> = [
                (OracleArm::Selectors, Vec::with_capacity(SAMPLES)),
                (OracleArm::Mask, Vec::with_capacity(SAMPLES)),
            ]
            .into_iter()
            .collect();
            for round in 0..SAMPLES {
                let order = if round % 2 == 0 {
                    [OracleArm::Selectors, OracleArm::Mask]
                } else {
                    [OracleArm::Mask, OracleArm::Selectors]
                };
                for arm in order {
                    let started = Instant::now();
                    let result = runtime.block_on(run_oracle(
                        &fixture,
                        Some(selection.clone()),
                        OracleSelectionSource::External,
                        arm,
                        false,
                    ));
                    assert_eq!(result.row_count, selector_check.row_count);
                    samples
                        .get_mut(&arm)
                        .unwrap()
                        .push(started.elapsed().as_nanos().min(u128::from(u64::MAX)) as u64);
                }
            }

            for arm in [OracleArm::Selectors, OracleArm::Mask] {
                let diagnostic = diagnostic(&runtime, &fixture, &shape, arm);
                assert_eq!(
                    diagnostic.rows_out,
                    if arm == OracleArm::Selectors {
                        selector_check.row_count
                    } else {
                        mask_check.row_count
                    }
                );
                let values = &samples[&arm];
                let med = median(values);
                let mad = median(&values.iter().map(|v| v.abs_diff(med)).collect::<Vec<_>>());
                csv.push_str(&format!(
                    "arrow-row-selection-cal0-v1,{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n",
                    spec.context.id,
                    spec.class,
                    spec.role,
                    spec.context.payload.dtype(),
                    spec.context.payload_columns,
                    spec.context.encoding(),
                    spec.context.compression.label(),
                    shape.name,
                    summary.selected_rows * ORACLE_ROW_GROUPS,
                    summary.skipped_rows * ORACLE_ROW_GROUPS,
                    summary.run_count * ORACLE_ROW_GROUPS,
                    arm.label(),
                    values.iter().map(u64::to_string).collect::<Vec<_>>().join(";"),
                    med,
                    mad,
                    diagnostic.rows_out,
                    selector_check.checksum,
                    diagnostic.filter_ns,
                    diagnostic.convert_ns,
                    diagnostic.skip_calls,
                    diagnostic.skip_rows,
                    diagnostic.read_calls,
                    diagnostic.read_rows,
                ));
            }
        }
    }

    let csv_path = output_dir.join("cal0.csv");
    fs::write(&csv_path, &csv).unwrap();
    let manifest = serde_json::to_string_pretty(&json!({
        "schema_version": "arrow-row-selection-cal0-manifest-v1",
        "matrix": "cal0-frozen-v1",
        "context_count": contexts.len(),
        "samples_per_arm": SAMPLES,
        "warmups_per_arm": WARMUPS,
        "rows_per_group": ROWS_PER_GROUP,
        "row_groups": ORACLE_ROW_GROUPS,
        "diagnostics": "one enabled pass per row group, summed",
        "evidence_class": "non-Formal",
        "csv": "cal0.csv",
    }))
    .unwrap()
        + "\n";
    fs::write(output_dir.join("cal0-manifest.json"), &manifest).unwrap();
    println!("CAL0_CSV_BEGIN\n{csv}CAL0_CSV_END");
    println!("CAL0_MANIFEST_BEGIN\n{manifest}CAL0_MANIFEST_END");
}

#[derive(Default)]
struct Diagnostic {
    rows_out: usize,
    filter_ns: u64,
    convert_ns: u64,
    skip_calls: usize,
    skip_rows: usize,
    read_calls: usize,
    read_rows: usize,
}

fn diagnostic(
    runtime: &Runtime,
    fixture: &super::fixture::OracleFixture,
    shape: &OracleShape,
    arm: OracleArm,
) -> Diagnostic {
    let mut out = Diagnostic::default();
    for row_group in 0..ORACLE_ROW_GROUPS {
        let metrics = ArrowReaderMetrics::enabled();
        let result = runtime.block_on(run_oracle_row_group_with_metrics(
            fixture,
            row_group,
            Some(shape.selection_for_row_group()),
            OracleSelectionSource::External,
            arm,
            false,
            metrics.clone(),
        ));
        let m = metrics.decomposition().unwrap();
        out.rows_out += result.row_count;
        out.filter_ns += m.filter_record_batch_ns;
        out.convert_ns += m.selectors_to_mask_ns;
        out.skip_calls += m.skip_records_calls;
        out.skip_rows += m.skip_records_rows;
        out.read_calls += m.read_records_calls;
        out.read_rows += m.read_records_rows;
    }
    out
}

fn context(
    id: &'static str,
    payload: OraclePayload,
    columns: usize,
    compression: OracleCompression,
    class: &'static str,
    role: &'static str,
) -> CalContext {
    CalContext {
        context: OracleContext {
            id,
            payload,
            payload_columns: columns,
            column_payloads: None,
            compression,
            page_index: false,
            batch_size: 8_192,
        },
        class,
        role,
    }
}

fn contexts() -> Vec<CalContext> {
    use OracleCompression::{Uncompressed as N, Zstd as Z};
    use OraclePayload::{Float64, Int32, Int64, Utf8Dictionary1k as Dict, Utf8View8, Utf8View64};
    vec![
        context("CAL-I32-N-K1", Int32, 1, N, "int32", "calibration"),
        context("CAL-I32-Z-K1", Int32, 1, Z, "int32", "calibration"),
        context("CAL-I64-N-K1", Int64, 1, N, "int64", "calibration"),
        context("CAL-F64-N-K1", Float64, 1, N, "float64", "calibration"),
        context("CAL-U8-N-K1", Utf8View8, 1, N, "utf8view-8b", "calibration"),
        context("CAL-U8-Z-K1", Utf8View8, 1, Z, "utf8view-8b", "calibration"),
        context(
            "CAL-U64-N-K1",
            Utf8View64,
            1,
            N,
            "utf8view-64b",
            "calibration",
        ),
        context(
            "CAL-U64-Z-K1",
            Utf8View64,
            1,
            Z,
            "utf8view-64b",
            "calibration",
        ),
        context("CAL-DICT-N-K1", Dict, 1, N, "dict-utf8-1k", "calibration"),
        context("CAL-DICT-Z-K1", Dict, 1, Z, "dict-utf8-1k", "calibration"),
        context("CAL-I32-N-K4", Int32, 4, N, "int32", "calibration"),
        context("CAL-I32-N-K8", Int32, 8, N, "int32", "calibration"),
        context("CAL-I32-N-K16", Int32, 16, N, "int32", "calibration"),
        context("CAL-I32-N-K32", Int32, 32, N, "int32", "calibration"),
        context("CAL-U8-N-K4", Utf8View8, 4, N, "utf8view-8b", "calibration"),
        context("CAL-U8-N-K8", Utf8View8, 8, N, "utf8view-8b", "calibration"),
        context(
            "CAL-U8-N-K16",
            Utf8View8,
            16,
            N,
            "utf8view-8b",
            "calibration",
        ),
        context(
            "CAL-U8-N-K32",
            Utf8View8,
            32,
            N,
            "utf8view-8b",
            "calibration",
        ),
        context(
            "CAL-U64-N-K8",
            Utf8View64,
            8,
            N,
            "utf8view-64b",
            "calibration",
        ),
        context(
            "CAL-U64-N-K32",
            Utf8View64,
            32,
            N,
            "utf8view-64b",
            "calibration",
        ),
        context("HOLD-I64-Z-K1", Int64, 1, Z, "int64", "holdout"),
        context("HOLD-F64-Z-K1", Float64, 1, Z, "float64", "holdout"),
    ]
}

fn shapes_for(role: &str) -> Vec<OracleShape> {
    if role == "holdout" {
        vec![
            OracleShape::l_sweep(1),
            OracleShape::l_sweep(8),
            OracleShape::l_sweep(64),
            OracleShape::l_sweep(512),
            OracleShape::selectivity(2, 64),
            OracleShape::selectivity(98, 64),
            OracleShape::bursty(30),
            OracleShape::all_selected(),
        ]
    } else {
        vec![
            OracleShape::all_selected(),
            OracleShape::leading_only(),
            OracleShape::l_sweep(8),
            OracleShape::l_sweep(64),
        ]
    }
}

fn median(values: &[u64]) -> u64 {
    let mut values = values.to_vec();
    values.sort_unstable();
    (values[values.len() / 2 - 1] / 2)
        + (values[values.len() / 2] / 2)
        + (values[values.len() / 2 - 1] % 2 + values[values.len() / 2] % 2) / 2
}

fn parse_output_dir() -> PathBuf {
    let mut args = env::args().skip(1);
    let mut output = None;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--selection-oracle" | "--cal" | "--bench" => {}
            "--output-dir" => {
                output = Some(PathBuf::from(args.next().expect("missing output dir")))
            }
            other => panic!("unexpected CAL argument: {other}"),
        }
    }
    output.unwrap_or_else(|| {
        env::var_os("CARGO_TARGET_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("target"))
            .join("cal0")
    })
}
