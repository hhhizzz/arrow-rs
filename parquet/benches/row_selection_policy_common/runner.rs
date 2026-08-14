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

use arrow::array::{
    Array, BinaryViewArray, DictionaryArray, Float64Array, Int32Array, Int64Array, RecordBatch,
    StringArray, StringViewArray,
};
use arrow::compute::kernels::cmp::eq;
use arrow::datatypes::{DataType, Int32Type};
use arrow_array::cast::AsArray;
use futures::StreamExt;
use parquet::arrow::arrow_reader::metrics::ArrowReaderMetrics;
use parquet::arrow::arrow_reader::{ArrowPredicateFn, RowFilter, RowSelection, RowSelectionPolicy};
use parquet::arrow::{ParquetRecordBatchStreamBuilder, ProjectionMask};
use sha2::{Digest, Sha256};

use super::fixture::{CaseFixture, OracleFixture, OraclePayload};
use super::model::{BATCH_SIZE, PAYLOAD_COLUMNS};

#[derive(Debug, Eq, PartialEq)]
pub(crate) struct RunResult {
    pub(crate) row_count: usize,
    pub(crate) payload0: Vec<i32>,
}

async fn run_with_consumer<F>(
    fixture: &CaseFixture,
    policy: RowSelectionPolicy,
    mut consume: F,
) -> usize
where
    F: FnMut(&RecordBatch),
{
    let predicate_projection = ProjectionMask::roots(fixture.schema_descr(), [0]);
    let output_projection = ProjectionMask::roots(fixture.schema_descr(), 1..=PAYLOAD_COLUMNS);
    let predicate = ArrowPredicateFn::new(predicate_projection, |batch: RecordBatch| {
        eq(batch.column(0), &Int32Array::new_scalar(1))
    });
    let row_filter = RowFilter::new(vec![Box::new(predicate)]);

    let mut stream = ParquetRecordBatchStreamBuilder::new(fixture.reader())
        .await
        .unwrap()
        .with_batch_size(BATCH_SIZE)
        .with_projection(output_projection)
        .with_row_filter(row_filter)
        .with_row_selection_policy(policy)
        .build()
        .unwrap();

    let mut rows = 0;
    while let Some(batch) = stream.next().await {
        let batch = batch.unwrap();
        rows += batch.num_rows();
        consume(&batch);
    }
    rows
}

pub(crate) async fn run(fixture: &CaseFixture, policy: RowSelectionPolicy) -> usize {
    run_with_consumer(fixture, policy, |_| {}).await
}

pub(crate) async fn run_auto(fixture: &CaseFixture) -> usize {
    run(fixture, RowSelectionPolicy::default()).await
}

pub(crate) async fn run_collect_payload0(
    fixture: &CaseFixture,
    policy: RowSelectionPolicy,
) -> RunResult {
    let mut payload0 = Vec::with_capacity(fixture.expected_rows);
    let row_count = run_with_consumer(fixture, policy, |batch| {
        let values = batch.column(0).as_primitive::<Int32Type>();
        payload0.extend(values.values().iter().copied());
    })
    .await;

    RunResult {
        row_count,
        payload0,
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(crate) enum OracleArm {
    Selectors,
    Mask,
    Auto,
    NoSelection,
}

impl OracleArm {
    pub(crate) const fn label(self) -> &'static str {
        match self {
            Self::Selectors => "selectors",
            Self::Mask => "mask",
            Self::Auto => "auto",
            Self::NoSelection => "no_selection",
        }
    }

    fn policy(self) -> Option<RowSelectionPolicy> {
        match self {
            Self::Selectors => Some(RowSelectionPolicy::Selectors),
            Self::Mask => Some(RowSelectionPolicy::Mask),
            Self::Auto => Some(RowSelectionPolicy::default()),
            Self::NoSelection => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum OracleSelectionSource {
    External,
    Predicate,
    None,
}

impl OracleSelectionSource {
    pub(crate) const fn label(self) -> &'static str {
        match self {
            Self::External => "external",
            Self::Predicate => "predicate",
            Self::None => "none",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct OracleRunResult {
    pub(crate) row_count: usize,
    pub(crate) checksum: u64,
}

pub(crate) async fn run_oracle(
    fixture: &OracleFixture,
    selection: Option<RowSelection>,
    source: OracleSelectionSource,
    arm: OracleArm,
    checksum: bool,
) -> OracleRunResult {
    assert_eq!(
        source == OracleSelectionSource::Predicate,
        fixture.has_predicate_column(),
        "predicate source and fixture shape must agree"
    );
    assert_eq!(
        source == OracleSelectionSource::External,
        selection.is_some(),
        "only the external source accepts a pre-built selection"
    );
    assert_eq!(
        source == OracleSelectionSource::None,
        arm == OracleArm::NoSelection,
        "the no-selection source has exactly one arm"
    );

    let context = fixture.context();
    let predicate_columns = usize::from(fixture.has_predicate_column());
    let output_projection = ProjectionMask::roots(
        fixture.schema_descr(),
        predicate_columns..predicate_columns + context.payload_columns,
    );
    let mut builder = ParquetRecordBatchStreamBuilder::new(fixture.reader())
        .await
        .unwrap()
        .with_batch_size(context.batch_size)
        .with_projection(output_projection);

    if let Some(policy) = arm.policy() {
        builder = builder.with_row_selection_policy(policy);
    }
    match source {
        OracleSelectionSource::External => {
            builder = builder.with_row_selection(selection.unwrap());
        }
        OracleSelectionSource::Predicate => {
            let predicate_projection = ProjectionMask::roots(fixture.schema_descr(), [0]);
            let predicate = ArrowPredicateFn::new(predicate_projection, |batch: RecordBatch| {
                eq(batch.column(0), &Int32Array::new_scalar(1))
            });
            builder = builder.with_row_filter(RowFilter::new(vec![Box::new(predicate)]));
        }
        OracleSelectionSource::None => {}
    }

    let mut stream = builder.build().unwrap();
    let mut row_count = 0usize;
    let mut payload_checksum = 0u64;
    while let Some(batch) = stream.next().await {
        let batch = batch.unwrap();
        if checksum {
            checksum_payload0(
                batch.column(0),
                context.payload,
                row_count,
                &mut payload_checksum,
            );
        }
        row_count += batch.num_rows();
    }
    OracleRunResult {
        row_count,
        checksum: payload_checksum,
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ProjectedContentDigest {
    pub(crate) schema_sha256: String,
    pub(crate) leaf_sha256: Vec<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct OracleRowGroupRunResult {
    pub(crate) row_count: usize,
    pub(crate) content: Option<ProjectedContentDigest>,
    pub(crate) requested_ranges: Vec<std::ops::Range<u64>>,
    pub(crate) requested_bytes: u64,
}

/// Execute exactly one row group. Attribution (logical content hashing and IO
/// range tracing) is used only for the untimed correctness pass so neither the
/// SHA-256 work nor the tracing mutex contaminates arm timings.
pub(crate) async fn run_oracle_row_group(
    fixture: &OracleFixture,
    row_group_index: usize,
    selection: Option<RowSelection>,
    source: OracleSelectionSource,
    arm: OracleArm,
    attribution: bool,
) -> OracleRowGroupRunResult {
    run_oracle_row_group_with_metrics(
        fixture,
        row_group_index,
        selection,
        source,
        arm,
        attribution,
        ArrowReaderMetrics::disabled(),
    )
    .await
}

pub(crate) async fn run_oracle_row_group_with_metrics(
    fixture: &OracleFixture,
    row_group_index: usize,
    selection: Option<RowSelection>,
    source: OracleSelectionSource,
    arm: OracleArm,
    attribution: bool,
    metrics: ArrowReaderMetrics,
) -> OracleRowGroupRunResult {
    assert!(row_group_index < fixture.metadata().num_row_groups());
    assert_eq!(
        source == OracleSelectionSource::Predicate,
        fixture.has_predicate_column(),
        "predicate source and fixture shape must agree"
    );
    assert_eq!(
        source == OracleSelectionSource::External,
        selection.is_some(),
        "only the external source accepts a pre-built selection"
    );
    assert_eq!(
        source == OracleSelectionSource::None,
        arm == OracleArm::NoSelection,
        "the no-selection source has exactly one arm"
    );

    let (reader, trace) = if attribution {
        let (reader, trace) = fixture.tracked_reader();
        (reader, Some(trace))
    } else {
        (fixture.reader(), None)
    };
    let context = fixture.context();
    let predicate_columns = usize::from(fixture.has_predicate_column());
    let output_projection = ProjectionMask::roots(
        fixture.schema_descr(),
        predicate_columns..predicate_columns + context.payload_columns,
    );
    let mut builder = ParquetRecordBatchStreamBuilder::new(reader)
        .await
        .unwrap()
        .with_row_groups(vec![row_group_index])
        .with_batch_size(context.batch_size)
        .with_projection(output_projection)
        .with_metrics(metrics);

    if let Some(policy) = arm.policy() {
        builder = builder.with_row_selection_policy(policy);
    }
    match source {
        OracleSelectionSource::External => {
            builder = builder.with_row_selection(selection.unwrap());
        }
        OracleSelectionSource::Predicate => {
            let predicate_projection = ProjectionMask::roots(fixture.schema_descr(), [0]);
            let predicate = ArrowPredicateFn::new(predicate_projection, |batch: RecordBatch| {
                eq(batch.column(0), &Int32Array::new_scalar(1))
            });
            builder = builder.with_row_filter(RowFilter::new(vec![Box::new(predicate)]));
        }
        OracleSelectionSource::None => {}
    }

    let mut stream = builder.build().unwrap();
    let mut row_count = 0usize;
    let mut digester = attribution.then(ProjectedContentDigester::default);
    while let Some(batch) = stream.next().await {
        let batch = batch.unwrap();
        if let Some(digester) = &mut digester {
            digester.update(&batch);
        }
        row_count += batch.num_rows();
    }

    let requested_ranges = trace.map(|trace| trace.ranges()).unwrap_or_default();
    let requested_bytes = requested_ranges
        .iter()
        .map(|range| range.end.saturating_sub(range.start))
        .sum();
    OracleRowGroupRunResult {
        row_count,
        content: digester.map(|digester| digester.finish()),
        requested_ranges,
        requested_bytes,
    }
}

#[derive(Default)]
struct ProjectedContentDigester {
    schema_sha256: Option<String>,
    leaf_hashers: Vec<Sha256>,
    rows: usize,
}

impl ProjectedContentDigester {
    fn update(&mut self, batch: &RecordBatch) {
        if self.schema_sha256.is_none() {
            let mut schema_hasher = Sha256::new();
            schema_hasher.update(b"arrow-projected-leaf-content-v1\0");
            for field in batch.schema().fields() {
                update_sized(&mut schema_hasher, field.name().as_bytes());
                update_sized(
                    &mut schema_hasher,
                    format!("{:?}", field.data_type()).as_bytes(),
                );
                schema_hasher.update([u8::from(field.is_nullable())]);
            }
            self.schema_sha256 = Some(hex_digest(&schema_hasher.finalize()));
            self.leaf_hashers = batch
                .schema()
                .fields()
                .iter()
                .enumerate()
                .map(|(column_idx, field)| {
                    let mut hasher = Sha256::new();
                    hasher.update(b"arrow-projected-leaf-content-v1\0");
                    hasher.update((column_idx as u64).to_le_bytes());
                    update_sized(&mut hasher, field.name().as_bytes());
                    update_sized(&mut hasher, format!("{:?}", field.data_type()).as_bytes());
                    hasher
                })
                .collect();
        }
        assert_eq!(self.leaf_hashers.len(), batch.num_columns());
        for (column, hasher) in batch.columns().iter().zip(&mut self.leaf_hashers) {
            update_logical_array(hasher, column.as_ref());
        }
        self.rows += batch.num_rows();
    }

    fn finish(mut self) -> ProjectedContentDigest {
        let leaf_sha256 = self
            .leaf_hashers
            .drain(..)
            .map(|mut hasher| {
                hasher.update((self.rows as u64).to_le_bytes());
                hex_digest(&hasher.finalize())
            })
            .collect();
        ProjectedContentDigest {
            schema_sha256: self.schema_sha256.unwrap_or_default(),
            leaf_sha256,
        }
    }
}

fn update_logical_array(hasher: &mut Sha256, array: &dyn Array) {
    for row_idx in 0..array.len() {
        if array.is_null(row_idx) {
            hasher.update([0]);
            continue;
        }
        hasher.update([1]);
        match array.data_type() {
            DataType::Int32 => hasher.update(
                array
                    .as_any()
                    .downcast_ref::<Int32Array>()
                    .unwrap()
                    .value(row_idx)
                    .to_le_bytes(),
            ),
            DataType::Int64 => hasher.update(
                array
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .unwrap()
                    .value(row_idx)
                    .to_le_bytes(),
            ),
            DataType::Float64 => hasher.update(
                array
                    .as_any()
                    .downcast_ref::<Float64Array>()
                    .unwrap()
                    .value(row_idx)
                    .to_bits()
                    .to_le_bytes(),
            ),
            DataType::Utf8View => update_sized(
                hasher,
                array
                    .as_any()
                    .downcast_ref::<StringViewArray>()
                    .unwrap()
                    .value(row_idx)
                    .as_bytes(),
            ),
            DataType::Utf8 => update_sized(
                hasher,
                array
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .unwrap()
                    .value(row_idx)
                    .as_bytes(),
            ),
            DataType::Dictionary(key, value)
                if key.as_ref() == &DataType::Int32 && value.as_ref() == &DataType::Utf8 =>
            {
                let dictionary = array
                    .as_any()
                    .downcast_ref::<DictionaryArray<Int32Type>>()
                    .unwrap();
                let values = dictionary
                    .values()
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .unwrap();
                update_sized(
                    hasher,
                    values
                        .value(dictionary.keys().value(row_idx) as usize)
                        .as_bytes(),
                );
            }
            data_type => panic!("unsupported oracle digest type {data_type:?}"),
        }
    }
}

fn update_sized(hasher: &mut Sha256, value: &[u8]) {
    hasher.update((value.len() as u64).to_le_bytes());
    hasher.update(value);
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

fn checksum_payload0(
    array: &dyn Array,
    payload: OraclePayload,
    output_offset: usize,
    checksum: &mut u64,
) {
    match payload {
        OraclePayload::Int32 => {
            let values = array.as_any().downcast_ref::<Int32Array>().unwrap();
            for (idx, value) in values.values().iter().enumerate() {
                xor_value(checksum, *value as u64, output_offset + idx);
            }
        }
        OraclePayload::Int64 => {
            let values = array.as_any().downcast_ref::<Int64Array>().unwrap();
            for (idx, value) in values.values().iter().enumerate() {
                xor_value(checksum, *value as u64, output_offset + idx);
            }
        }
        OraclePayload::Float64 => {
            let values = array.as_any().downcast_ref::<Float64Array>().unwrap();
            for (idx, value) in values.values().iter().enumerate() {
                xor_value(checksum, value.to_bits(), output_offset + idx);
            }
        }
        OraclePayload::Utf8View8
        | OraclePayload::Utf8View16
        | OraclePayload::Utf8View32
        | OraclePayload::Utf8View48
        | OraclePayload::Utf8View64 => {
            let values = array.as_any().downcast_ref::<StringViewArray>().unwrap();
            for idx in 0..values.len() {
                xor_value(
                    checksum,
                    hash_bytes(values.value(idx).as_bytes()),
                    output_offset + idx,
                );
            }
        }
        OraclePayload::BinaryView64 => {
            let values = array.as_any().downcast_ref::<BinaryViewArray>().unwrap();
            for idx in 0..values.len() {
                xor_value(checksum, hash_bytes(values.value(idx)), output_offset + idx);
            }
        }
        OraclePayload::Utf8Dictionary1k | OraclePayload::Utf8Dictionary { .. } => {
            checksum_dictionary_or_string(array, output_offset, checksum)
        }
    }
}

fn checksum_dictionary_or_string(array: &dyn Array, output_offset: usize, checksum: &mut u64) {
    if let Some(values) = array.as_any().downcast_ref::<StringArray>() {
        for idx in 0..values.len() {
            xor_value(
                checksum,
                hash_bytes(values.value(idx).as_bytes()),
                output_offset + idx,
            );
        }
        return;
    }
    if let Some(dictionary) = array.as_any().downcast_ref::<DictionaryArray<Int32Type>>() {
        let values = dictionary
            .values()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        for (idx, key) in dictionary.keys().values().iter().enumerate() {
            xor_value(
                checksum,
                hash_bytes(values.value(*key as usize).as_bytes()),
                output_offset + idx,
            );
        }
        return;
    }
    panic!("dictionary oracle payload decoded to unsupported type {array:?}");
}

fn xor_value(checksum: &mut u64, value: u64, output_row: usize) {
    *checksum ^= value
        .wrapping_add((output_row as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15))
        .rotate_left((output_row % 63) as u32);
}

fn hash_bytes(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}
