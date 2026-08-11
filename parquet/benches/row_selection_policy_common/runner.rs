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
    Array, DictionaryArray, Float64Array, Int32Array, Int64Array, RecordBatch, StringArray,
    StringViewArray,
};
use arrow::compute::kernels::cmp::eq;
use arrow::datatypes::Int32Type;
use arrow_array::cast::AsArray;
use futures::StreamExt;
use parquet::arrow::arrow_reader::{ArrowPredicateFn, RowFilter, RowSelection, RowSelectionPolicy};
use parquet::arrow::{ParquetRecordBatchStreamBuilder, ProjectionMask};

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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
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
        OraclePayload::Utf8View8 | OraclePayload::Utf8View64 => {
            let values = array.as_any().downcast_ref::<StringViewArray>().unwrap();
            for idx in 0..values.len() {
                xor_value(
                    checksum,
                    hash_bytes(values.value(idx).as_bytes()),
                    output_offset + idx,
                );
            }
        }
        OraclePayload::Utf8Dictionary1k => {
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
