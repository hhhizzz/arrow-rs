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

use std::ops::Range;
use std::sync::{Arc, Mutex};

use arrow::array::{
    ArrayRef, BinaryViewArray, Float64Array, Int32Array, Int64Array, RecordBatch, StringArray,
    StringViewArray,
};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use bytes::Bytes;
use futures::FutureExt;
use futures::future::BoxFuture;
use parquet::arrow::ArrowWriter;
use parquet::arrow::arrow_reader::ArrowReaderOptions;
use parquet::arrow::async_reader::AsyncFileReader;
use parquet::basic::{Compression, Encoding, ZstdLevel};
use parquet::errors::Result;
use parquet::file::metadata::{PageIndexPolicy, ParquetMetaData, ParquetMetaDataReader};
use parquet::file::properties::WriterProperties;
use parquet::schema::types::ColumnPath;
use sha2::{Digest, Sha256};

use super::model::{BATCH_SIZE, CaseSpec, PAYLOAD_COLUMNS, PAYLOAD_VALUE_MODULUS, ROWS_PER_GROUP};
use super::shapes::{expand_pattern, selected_rows};

pub(crate) const ORACLE_ROW_GROUPS: usize = 4;
pub(crate) const ORACLE_PAGE_ROWS: usize = 4_096;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum OraclePayload {
    Int32,
    Int64,
    Float64,
    Utf8View8,
    Utf8View16,
    Utf8View32,
    Utf8View48,
    Utf8View64,
    BinaryView64,
    Utf8Dictionary1k,
    Utf8Dictionary {
        cardinality: usize,
        value_width: usize,
        fallback_plain_percent: Option<usize>,
    },
}

impl OraclePayload {
    pub(crate) const fn dtype(self) -> &'static str {
        match self {
            Self::Int32 => "int32",
            Self::Int64 => "int64",
            Self::Float64 => "float64",
            Self::Utf8View8 => "utf8view-8b",
            Self::Utf8View16 => "utf8view-16b",
            Self::Utf8View32 => "utf8view-32b",
            Self::Utf8View48 => "utf8view-48b",
            Self::Utf8View64 => "utf8view-64b",
            Self::BinaryView64 => "binaryview-64b",
            Self::Utf8Dictionary1k | Self::Utf8Dictionary { .. } => "utf8",
        }
    }

    fn data_type(self) -> DataType {
        match self {
            Self::Int32 => DataType::Int32,
            Self::Int64 => DataType::Int64,
            Self::Float64 => DataType::Float64,
            Self::Utf8View8
            | Self::Utf8View16
            | Self::Utf8View32
            | Self::Utf8View48
            | Self::Utf8View64 => DataType::Utf8View,
            Self::BinaryView64 => DataType::BinaryView,
            Self::Utf8Dictionary1k | Self::Utf8Dictionary { .. } => DataType::Utf8,
        }
    }

    const fn dictionary_spec(self) -> Option<(usize, usize, Option<usize>)> {
        match self {
            Self::Utf8Dictionary {
                cardinality,
                value_width,
                fallback_plain_percent,
            } => Some((cardinality, value_width, fallback_plain_percent)),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum OracleCompression {
    Uncompressed,
    Snappy,
    Lz4,
    Zstd,
}

impl OracleCompression {
    pub(crate) const fn label(self) -> &'static str {
        match self {
            Self::Uncompressed => "none",
            Self::Snappy => "snappy",
            Self::Lz4 => "lz4",
            Self::Zstd => "zstd",
        }
    }

    fn parquet(self) -> Compression {
        match self {
            Self::Uncompressed => Compression::UNCOMPRESSED,
            Self::Snappy => Compression::SNAPPY,
            Self::Lz4 => Compression::LZ4_RAW,
            Self::Zstd => Compression::ZSTD(ZstdLevel::default()),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct OracleContext {
    pub(crate) id: &'static str,
    pub(crate) payload: OraclePayload,
    pub(crate) payload_columns: usize,
    /// Optional physical payload for every output column. `None` preserves the
    /// Phase-1 homogeneous projection represented by `payload`.
    pub(crate) column_payloads: Option<&'static [OraclePayload]>,
    pub(crate) compression: OracleCompression,
    pub(crate) page_index: bool,
    pub(crate) batch_size: usize,
}

impl OracleContext {
    pub(crate) fn encoding(self) -> String {
        match self.column_payloads {
            Some(payloads)
                if payloads
                    .iter()
                    .all(|payload| matches!(payload, OraclePayload::Utf8Dictionary1k)) =>
            {
                "dictionary-1k".to_string()
            }
            Some(payloads)
                if payloads.iter().any(|payload| {
                    matches!(
                        payload,
                        OraclePayload::Utf8Dictionary1k | OraclePayload::Utf8Dictionary { .. }
                    )
                }) =>
            {
                "mixed".to_string()
            }
            _ => match self.payload {
                OraclePayload::Utf8Dictionary1k => "dictionary-1k".to_string(),
                OraclePayload::Utf8Dictionary {
                    cardinality: _,
                    value_width,
                    fallback_plain_percent: Some(percent),
                } => format!("dictionary-fallback-p{percent}-w{value_width}"),
                OraclePayload::Utf8Dictionary {
                    cardinality,
                    value_width,
                    fallback_plain_percent: None,
                } => format!("dictionary-c{cardinality}-w{value_width}"),
                _ => "plain".to_string(),
            },
        }
    }

    pub(crate) fn payload_at(self, column_idx: usize) -> OraclePayload {
        assert!(column_idx < self.payload_columns);
        self.column_payloads
            .map(|payloads| {
                assert_eq!(payloads.len(), self.payload_columns);
                payloads[column_idx]
            })
            .unwrap_or(self.payload)
    }

    pub(crate) fn uses_dictionary(self) -> bool {
        (0..self.payload_columns).any(|column_idx| {
            matches!(
                self.payload_at(column_idx),
                OraclePayload::Utf8Dictionary1k | OraclePayload::Utf8Dictionary { .. }
            )
        })
    }

    fn dictionary_page_size_limit(self) -> Option<usize> {
        let specs = (0..self.payload_columns)
            .filter_map(|column_idx| self.payload_at(column_idx).dictionary_spec())
            .collect::<Vec<_>>();
        if specs.iter().any(|(_, _, fallback)| fallback.is_some()) {
            // The fallback fixtures repeat a 16-value dictionary up to the
            // requested transition, then introduce unique values. A 1 KiB
            // dictionary budget crosses within one 256-row writer batch.
            Some(1_024)
        } else if specs.is_empty() {
            None
        } else {
            // Keep the 64K x 64B calibration context dictionary-only.
            Some(16 * 1024 * 1024)
        }
    }

    fn has_dictionary_fallback(self) -> bool {
        (0..self.payload_columns).any(|column_idx| {
            self.payload_at(column_idx)
                .dictionary_spec()
                .is_some_and(|(_, _, fallback)| fallback.is_some())
        })
    }

    pub(crate) const fn output_layout(self) -> &'static str {
        match self.column_payloads {
            Some(_) => "mixed",
            None => match self.payload {
                OraclePayload::Int32 | OraclePayload::Int64 | OraclePayload::Float64 => "fixed",
                OraclePayload::Utf8View8
                | OraclePayload::Utf8View16
                | OraclePayload::Utf8View32
                | OraclePayload::Utf8View48
                | OraclePayload::Utf8View64 => "utf8view",
                OraclePayload::BinaryView64 => "binaryview",
                OraclePayload::Utf8Dictionary1k | OraclePayload::Utf8Dictionary { .. } => "utf8",
            },
        }
    }

    pub(crate) const fn with_page_index(mut self) -> Self {
        self.page_index = true;
        self
    }

    pub(crate) const fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }
}

pub(crate) const ORACLE_CONTEXTS: &[OracleContext] = &[
    OracleContext {
        id: "C0",
        payload: OraclePayload::Int32,
        payload_columns: 8,
        column_payloads: None,
        compression: OracleCompression::Uncompressed,
        page_index: false,
        batch_size: BATCH_SIZE,
    },
    OracleContext {
        id: "C1",
        payload: OraclePayload::Int64,
        payload_columns: 8,
        column_payloads: None,
        compression: OracleCompression::Uncompressed,
        page_index: false,
        batch_size: BATCH_SIZE,
    },
    OracleContext {
        id: "C2",
        payload: OraclePayload::Float64,
        payload_columns: 8,
        column_payloads: None,
        compression: OracleCompression::Uncompressed,
        page_index: false,
        batch_size: BATCH_SIZE,
    },
    OracleContext {
        id: "C3",
        payload: OraclePayload::Utf8View8,
        payload_columns: 8,
        column_payloads: None,
        compression: OracleCompression::Uncompressed,
        page_index: false,
        batch_size: BATCH_SIZE,
    },
    OracleContext {
        id: "C4",
        payload: OraclePayload::Utf8View64,
        payload_columns: 8,
        column_payloads: None,
        compression: OracleCompression::Uncompressed,
        page_index: false,
        batch_size: BATCH_SIZE,
    },
    OracleContext {
        id: "C5",
        payload: OraclePayload::Utf8Dictionary1k,
        payload_columns: 8,
        column_payloads: None,
        compression: OracleCompression::Uncompressed,
        page_index: false,
        batch_size: BATCH_SIZE,
    },
    OracleContext {
        id: "C6",
        payload: OraclePayload::Int32,
        payload_columns: 1,
        column_payloads: None,
        compression: OracleCompression::Uncompressed,
        page_index: false,
        batch_size: BATCH_SIZE,
    },
    OracleContext {
        id: "C7",
        payload: OraclePayload::Int32,
        payload_columns: 4,
        column_payloads: None,
        compression: OracleCompression::Uncompressed,
        page_index: false,
        batch_size: BATCH_SIZE,
    },
    OracleContext {
        id: "C8",
        payload: OraclePayload::Int32,
        payload_columns: 32,
        column_payloads: None,
        compression: OracleCompression::Uncompressed,
        page_index: false,
        batch_size: BATCH_SIZE,
    },
    OracleContext {
        id: "C9",
        payload: OraclePayload::Utf8View8,
        payload_columns: 1,
        column_payloads: None,
        compression: OracleCompression::Uncompressed,
        page_index: false,
        batch_size: BATCH_SIZE,
    },
    OracleContext {
        id: "C10",
        payload: OraclePayload::Utf8View8,
        payload_columns: 4,
        column_payloads: None,
        compression: OracleCompression::Uncompressed,
        page_index: false,
        batch_size: BATCH_SIZE,
    },
    OracleContext {
        id: "C11",
        payload: OraclePayload::Utf8View8,
        payload_columns: 32,
        column_payloads: None,
        compression: OracleCompression::Uncompressed,
        page_index: false,
        batch_size: BATCH_SIZE,
    },
    OracleContext {
        id: "C12",
        payload: OraclePayload::Int32,
        payload_columns: 8,
        column_payloads: None,
        compression: OracleCompression::Zstd,
        page_index: false,
        batch_size: BATCH_SIZE,
    },
    OracleContext {
        id: "C13",
        payload: OraclePayload::Utf8View64,
        payload_columns: 8,
        column_payloads: None,
        compression: OracleCompression::Zstd,
        page_index: false,
        batch_size: BATCH_SIZE,
    },
];

macro_rules! tt_context {
    ($id:literal, $payload:expr, $columns:expr, $compression:expr) => {
        OracleContext {
            id: $id,
            payload: $payload,
            payload_columns: $columns,
            column_payloads: None,
            compression: $compression,
            page_index: false,
            batch_size: BATCH_SIZE,
        }
    };
}

const fn tt_dictionary(
    cardinality: usize,
    value_width: usize,
    fallback_plain_percent: Option<usize>,
) -> OraclePayload {
    OraclePayload::Utf8Dictionary {
        cardinality,
        value_width,
        fallback_plain_percent,
    }
}

/// TT-1 follows the literal Cartesian product in the runbook. This is 34
/// contexts (17 variants x {1, 8} projected columns), despite the runbook's
/// approximate 22-context arithmetic note.
pub(crate) const TT_CONTEXTS: &[OracleContext] = &[
    tt_context!(
        "TT-D-C16-W8-P1",
        tt_dictionary(16, 8, None),
        1,
        OracleCompression::Uncompressed
    ),
    tt_context!(
        "TT-D-C16-W8-P8",
        tt_dictionary(16, 8, None),
        8,
        OracleCompression::Uncompressed
    ),
    tt_context!(
        "TT-D-C16-W64-P1",
        tt_dictionary(16, 64, None),
        1,
        OracleCompression::Uncompressed
    ),
    tt_context!(
        "TT-D-C16-W64-P8",
        tt_dictionary(16, 64, None),
        8,
        OracleCompression::Uncompressed
    ),
    tt_context!(
        "TT-D-C256-W8-P1",
        tt_dictionary(256, 8, None),
        1,
        OracleCompression::Uncompressed
    ),
    tt_context!(
        "TT-D-C256-W8-P8",
        tt_dictionary(256, 8, None),
        8,
        OracleCompression::Uncompressed
    ),
    tt_context!(
        "TT-D-C256-W64-P1",
        tt_dictionary(256, 64, None),
        1,
        OracleCompression::Uncompressed
    ),
    tt_context!(
        "TT-D-C256-W64-P8",
        tt_dictionary(256, 64, None),
        8,
        OracleCompression::Uncompressed
    ),
    tt_context!(
        "TT-D-C1024-W8-P1",
        tt_dictionary(1_024, 8, None),
        1,
        OracleCompression::Uncompressed
    ),
    tt_context!(
        "TT-D-C1024-W8-P8",
        tt_dictionary(1_024, 8, None),
        8,
        OracleCompression::Uncompressed
    ),
    tt_context!(
        "TT-D-C1024-W64-P1",
        tt_dictionary(1_024, 64, None),
        1,
        OracleCompression::Uncompressed
    ),
    tt_context!(
        "TT-D-C1024-W64-P8",
        tt_dictionary(1_024, 64, None),
        8,
        OracleCompression::Uncompressed
    ),
    tt_context!(
        "TT-D-C65536-W8-P1",
        tt_dictionary(65_536, 8, None),
        1,
        OracleCompression::Uncompressed
    ),
    tt_context!(
        "TT-D-C65536-W8-P8",
        tt_dictionary(65_536, 8, None),
        8,
        OracleCompression::Uncompressed
    ),
    tt_context!(
        "TT-D-C65536-W64-P1",
        tt_dictionary(65_536, 64, None),
        1,
        OracleCompression::Uncompressed
    ),
    tt_context!(
        "TT-D-C65536-W64-P8",
        tt_dictionary(65_536, 64, None),
        8,
        OracleCompression::Uncompressed
    ),
    tt_context!(
        "TT-F-P25-C1",
        tt_dictionary(65_536, 32, Some(25)),
        1,
        OracleCompression::Uncompressed
    ),
    tt_context!(
        "TT-F-P25-C8",
        tt_dictionary(65_536, 32, Some(25)),
        8,
        OracleCompression::Uncompressed
    ),
    tt_context!(
        "TT-F-P75-C1",
        tt_dictionary(65_536, 32, Some(75)),
        1,
        OracleCompression::Uncompressed
    ),
    tt_context!(
        "TT-F-P75-C8",
        tt_dictionary(65_536, 32, Some(75)),
        8,
        OracleCompression::Uncompressed
    ),
    tt_context!(
        "TT-V-U16-C1",
        OraclePayload::Utf8View16,
        1,
        OracleCompression::Uncompressed
    ),
    tt_context!(
        "TT-V-U16-C8",
        OraclePayload::Utf8View16,
        8,
        OracleCompression::Uncompressed
    ),
    tt_context!(
        "TT-V-U32-C1",
        OraclePayload::Utf8View32,
        1,
        OracleCompression::Uncompressed
    ),
    tt_context!(
        "TT-V-U32-C8",
        OraclePayload::Utf8View32,
        8,
        OracleCompression::Uncompressed
    ),
    tt_context!(
        "TT-V-U64-SNAPPY-C1",
        OraclePayload::Utf8View64,
        1,
        OracleCompression::Snappy
    ),
    tt_context!(
        "TT-V-U64-SNAPPY-C8",
        OraclePayload::Utf8View64,
        8,
        OracleCompression::Snappy
    ),
    tt_context!(
        "TT-V-U64-LZ4-C1",
        OraclePayload::Utf8View64,
        1,
        OracleCompression::Lz4
    ),
    tt_context!(
        "TT-V-U64-LZ4-C8",
        OraclePayload::Utf8View64,
        8,
        OracleCompression::Lz4
    ),
    tt_context!(
        "TT-V-B64-C1",
        OraclePayload::BinaryView64,
        1,
        OracleCompression::Uncompressed
    ),
    tt_context!(
        "TT-V-B64-C8",
        OraclePayload::BinaryView64,
        8,
        OracleCompression::Uncompressed
    ),
    tt_context!(
        "TT-H-D-C4096-W32-C1",
        tt_dictionary(4_096, 32, None),
        1,
        OracleCompression::Uncompressed
    ),
    tt_context!(
        "TT-H-D-C4096-W32-C8",
        tt_dictionary(4_096, 32, None),
        8,
        OracleCompression::Uncompressed
    ),
    tt_context!(
        "TT-H-U48-C1",
        OraclePayload::Utf8View48,
        1,
        OracleCompression::Uncompressed
    ),
    tt_context!(
        "TT-H-U48-C8",
        OraclePayload::Utf8View48,
        8,
        OracleCompression::Uncompressed
    ),
];

const PC_DICT_C1024_W8: OraclePayload = tt_dictionary(1_024, 8, None);
const PC_DICT_C4096_W64: OraclePayload = tt_dictionary(4_096, 64, None);

const PC_M1_PAYLOADS: &[OraclePayload] = &[
    PC_DICT_C1024_W8,
    PC_DICT_C1024_W8,
    PC_DICT_C1024_W8,
    PC_DICT_C1024_W8,
    OraclePayload::Utf8View8,
    OraclePayload::Utf8View8,
    OraclePayload::Utf8View8,
    OraclePayload::Utf8View8,
];
const PC_M2_PAYLOADS: &[OraclePayload] = &[
    PC_DICT_C1024_W8,
    PC_DICT_C1024_W8,
    PC_DICT_C1024_W8,
    PC_DICT_C1024_W8,
    OraclePayload::BinaryView64,
    OraclePayload::BinaryView64,
    OraclePayload::BinaryView64,
    OraclePayload::BinaryView64,
];
const PC_M3_PAYLOADS: &[OraclePayload] = &[
    OraclePayload::Int32,
    OraclePayload::Int32,
    OraclePayload::Float64,
    OraclePayload::Float64,
    PC_DICT_C1024_W8,
    PC_DICT_C1024_W8,
    OraclePayload::Utf8View32,
    OraclePayload::Utf8View32,
];
const PC_M4_PAYLOADS: &[OraclePayload] = &[
    PC_DICT_C1024_W8,
    PC_DICT_C1024_W8,
    PC_DICT_C1024_W8,
    PC_DICT_C1024_W8,
    PC_DICT_C1024_W8,
    PC_DICT_C1024_W8,
    PC_DICT_C1024_W8,
    PC_DICT_C1024_W8,
    OraclePayload::Utf8View8,
    OraclePayload::Utf8View8,
    OraclePayload::Utf8View8,
    OraclePayload::Utf8View8,
    OraclePayload::Utf8View8,
    OraclePayload::Utf8View8,
    OraclePayload::Utf8View8,
    OraclePayload::Utf8View8,
];
const PC_M5_PAYLOADS: &[OraclePayload] = &[
    OraclePayload::Int32,
    OraclePayload::Int32,
    OraclePayload::Int32,
    OraclePayload::Int32,
    OraclePayload::Utf8View32,
    OraclePayload::Utf8View32,
    OraclePayload::Utf8View32,
    OraclePayload::Utf8View32,
];
const PC_M6_PAYLOADS: &[OraclePayload] = &[
    PC_DICT_C4096_W64,
    PC_DICT_C4096_W64,
    PC_DICT_C4096_W64,
    OraclePayload::BinaryView64,
    OraclePayload::BinaryView64,
    OraclePayload::BinaryView64,
    OraclePayload::BinaryView64,
    OraclePayload::BinaryView64,
];

// PC-2's unopened mixed holdout deliberately combines three physical classes
// without reusing one of the PC-1 M-context layouts. The harness owns the
// timing embargo; exposing the fixture here also lets smoke runs validate its
// schema and decision counters without adding another fixture generator.
#[cfg(feature = "test_common")]
const PC2_H_PAYLOADS: &[OraclePayload] = &[
    PC_DICT_C1024_W8,
    PC_DICT_C1024_W8,
    PC_DICT_C1024_W8,
    OraclePayload::Utf8View32,
    OraclePayload::Utf8View32,
    OraclePayload::Utf8View32,
    OraclePayload::Int64,
    OraclePayload::Int64,
];

macro_rules! pc_mixed_context {
    ($id:literal, $payloads:expr) => {
        OracleContext {
            id: $id,
            payload: $payloads[0],
            payload_columns: $payloads.len(),
            column_payloads: Some($payloads),
            compression: OracleCompression::Uncompressed,
            page_index: false,
            batch_size: BATCH_SIZE,
        }
    };
}

/// PC-1 mixed projection environments. PC-M6 is the unopened holdout and
/// deliberately uses C4096/W64 rather than the C1024/W64 PC-0 proxy.
pub(crate) const PC_MIXED_CONTEXTS: &[OracleContext] = &[
    pc_mixed_context!("PC-M1", PC_M1_PAYLOADS),
    pc_mixed_context!("PC-M2", PC_M2_PAYLOADS),
    pc_mixed_context!("PC-M3", PC_M3_PAYLOADS),
    pc_mixed_context!("PC-M4", PC_M4_PAYLOADS),
    pc_mixed_context!("PC-M5", PC_M5_PAYLOADS),
    pc_mixed_context!("PC-M6", PC_M6_PAYLOADS),
];

/// PC-2 holdout. Formal timing is opened only by the experiment gate; the
/// fixture itself remains available for untimed correctness/counter smoke.
#[cfg(feature = "test_common")]
pub(crate) const PC2_HOLDOUT_CONTEXTS: &[OracleContext] =
    &[pc_mixed_context!("H-PC2", PC2_H_PAYLOADS)];

#[derive(Debug)]
pub(crate) struct CaseFixture {
    bytes: Bytes,
    metadata: Arc<ParquetMetaData>,
    pub(crate) expected_rows: usize,
}

impl CaseFixture {
    pub(crate) fn reader(&self) -> InMemoryAsyncReader {
        InMemoryAsyncReader {
            bytes: self.bytes.clone(),
            metadata: Arc::clone(&self.metadata),
            requested_ranges: None,
        }
    }

    pub(crate) fn schema_descr(&self) -> &parquet::schema::types::SchemaDescriptor {
        self.metadata.file_metadata().schema_descr()
    }
}

#[derive(Debug, Clone)]
pub(crate) struct InMemoryAsyncReader {
    bytes: Bytes,
    metadata: Arc<ParquetMetaData>,
    requested_ranges: Option<Arc<Mutex<Vec<Range<u64>>>>>,
}

impl AsyncFileReader for InMemoryAsyncReader {
    fn get_bytes(&mut self, range: Range<u64>) -> BoxFuture<'_, Result<Bytes>> {
        if let Some(requested_ranges) = &self.requested_ranges {
            requested_ranges.lock().unwrap().push(range.clone());
        }
        let bytes = self.bytes.slice(range.start as usize..range.end as usize);
        async move { Ok(bytes) }.boxed()
    }

    fn get_metadata<'a>(
        &'a mut self,
        _options: Option<&'a ArrowReaderOptions>,
    ) -> BoxFuture<'a, Result<Arc<ParquetMetaData>>> {
        let metadata = Arc::clone(&self.metadata);
        async move { Ok(metadata) }.boxed()
    }
}

#[derive(Clone, Debug)]
pub(crate) struct OracleReadTrace(Arc<Mutex<Vec<Range<u64>>>>);

impl OracleReadTrace {
    pub(crate) fn ranges(&self) -> Vec<Range<u64>> {
        self.0.lock().unwrap().clone()
    }
}

#[derive(Debug)]
pub(crate) struct OracleFixture {
    bytes: Bytes,
    metadata: Arc<ParquetMetaData>,
    context: OracleContext,
    predicate_column: bool,
}

impl OracleFixture {
    pub(crate) fn reader(&self) -> InMemoryAsyncReader {
        InMemoryAsyncReader {
            bytes: self.bytes.clone(),
            metadata: Arc::clone(&self.metadata),
            requested_ranges: None,
        }
    }

    pub(crate) fn tracked_reader(&self) -> (InMemoryAsyncReader, OracleReadTrace) {
        let requested_ranges = Arc::new(Mutex::new(Vec::new()));
        (
            InMemoryAsyncReader {
                bytes: self.bytes.clone(),
                metadata: Arc::clone(&self.metadata),
                requested_ranges: Some(Arc::clone(&requested_ranges)),
            },
            OracleReadTrace(requested_ranges),
        )
    }

    pub(crate) fn schema_descr(&self) -> &parquet::schema::types::SchemaDescriptor {
        self.metadata.file_metadata().schema_descr()
    }

    pub(crate) fn metadata(&self) -> &ParquetMetaData {
        &self.metadata
    }

    pub(crate) fn bytes_sha256(&self) -> String {
        let mut hasher = Sha256::new();
        hasher.update(&self.bytes);
        hex_digest(hasher.finalize().as_slice())
    }

    pub(crate) const fn context(&self) -> OracleContext {
        self.context
    }

    pub(crate) const fn has_predicate_column(&self) -> bool {
        self.predicate_column
    }

    pub(crate) fn with_batch_size(&self, batch_size: usize) -> Self {
        Self {
            bytes: self.bytes.clone(),
            metadata: Arc::clone(&self.metadata),
            context: self.context.with_batch_size(batch_size),
            predicate_column: self.predicate_column,
        }
    }

    /// Reparse the exact same Parquet bytes with a different page-index
    /// policy. This is the Tier-B same-Bytes control; it deliberately does not
    /// invoke the writer or change page layout.
    pub(crate) fn with_page_index_policy(&self, policy: PageIndexPolicy) -> Result<Self> {
        let mut metadata_reader = ParquetMetaDataReader::new().with_page_index_policy(policy);
        metadata_reader.try_parse(&self.bytes)?;
        let metadata = Arc::new(metadata_reader.finish()?);
        let mut context = self.context;
        context.page_index = !matches!(policy, PageIndexPolicy::Skip);
        Ok(Self {
            bytes: self.bytes.clone(),
            metadata,
            context,
            predicate_column: self.predicate_column,
        })
    }
}

pub(crate) fn build_oracle_fixture(
    context: OracleContext,
    predicate_values: Option<&[i32]>,
) -> Result<OracleFixture> {
    build_oracle_fixture_with_dimensions(
        context,
        predicate_values,
        ORACLE_ROW_GROUPS,
        ROWS_PER_GROUP,
    )
}

/// Builds the same deterministic oracle fixture at an explicit physical scale.
///
/// PC-1c varies row-group count independently from rows per row group to
/// distinguish per-scan, per-row-group, and per-physical-row overhead.  The
/// established oracle path remains the wrapper above so earlier experiments
/// retain their exact bytes and manifest contract.
pub(crate) fn build_oracle_fixture_with_dimensions(
    context: OracleContext,
    predicate_values: Option<&[i32]>,
    row_groups: usize,
    rows_per_group: usize,
) -> Result<OracleFixture> {
    assert!(row_groups > 0, "oracle fixture must contain a row group");
    assert!(rows_per_group > 0, "oracle row groups must contain rows");
    let total_rows = row_groups * rows_per_group;
    if let Some(values) = predicate_values {
        assert_eq!(
            values.len(),
            total_rows,
            "predicate values must cover every oracle row group"
        );
    }

    let predicate_column = predicate_values.is_some();
    let schema = build_oracle_schema(context, predicate_column);
    let mixed_payloads = context.column_payloads.is_some();
    let mut properties = WriterProperties::builder()
        .set_compression(context.compression.parquet())
        .set_dictionary_enabled(!mixed_payloads && context.uses_dictionary())
        .set_max_row_group_row_count(Some(rows_per_group));
    if mixed_payloads {
        // Mixed PC fixtures must not accidentally dictionary-encode their
        // view columns just because another column is the dictionary class.
        for column_idx in 0..context.payload_columns {
            let dictionary = matches!(
                context.payload_at(column_idx),
                OraclePayload::Utf8Dictionary1k | OraclePayload::Utf8Dictionary { .. }
            );
            properties = properties.set_column_dictionary_enabled(
                ColumnPath::from(format!("payload_{column_idx}")),
                dictionary,
            );
        }
    }
    if let Some(limit) = context.dictionary_page_size_limit() {
        properties = properties.set_dictionary_page_size_limit(limit);
    }
    if context.has_dictionary_fallback() {
        properties = properties.set_write_batch_size(256);
    }
    if context.page_index {
        properties = properties
            .set_data_page_row_count_limit(ORACLE_PAGE_ROWS)
            .set_write_batch_size(1_024);
    }

    let mut encoded = Vec::new();
    {
        let mut writer =
            ArrowWriter::try_new(&mut encoded, Arc::clone(&schema), Some(properties.build()))?;
        for row_group_idx in 0..row_groups {
            let start = row_group_idx * rows_per_group;
            let predicate = predicate_values.map(|values| &values[start..start + rows_per_group]);
            writer.write(&build_oracle_row_group_batch(
                Arc::clone(&schema),
                context,
                row_group_idx,
                rows_per_group,
                predicate,
            )?)?;
        }
        writer.close()?;
    }

    let bytes = Bytes::from(encoded);
    let policy = if context.page_index {
        PageIndexPolicy::Required
    } else {
        PageIndexPolicy::Skip
    };
    let mut metadata_reader = ParquetMetaDataReader::new().with_page_index_policy(policy);
    metadata_reader.try_parse(&bytes)?;
    let metadata = Arc::new(metadata_reader.finish()?);

    assert_eq!(metadata.num_row_groups(), row_groups);
    for row_group in metadata.row_groups() {
        assert_eq!(row_group.num_rows() as usize, rows_per_group);
    }
    assert_dictionary_encoding_contract(&metadata, context, predicate_column);
    if context.page_index {
        assert!(
            metadata.offset_index().is_some(),
            "page-index oracle fixture must preload its offset index"
        );
    } else {
        assert!(
            metadata.offset_index().is_none(),
            "non-page-index oracle fixture must not preload its offset index"
        );
    }

    Ok(OracleFixture {
        bytes,
        metadata,
        context,
        predicate_column,
    })
}

fn build_oracle_schema(context: OracleContext, predicate_column: bool) -> SchemaRef {
    let mut fields = Vec::with_capacity(context.payload_columns + usize::from(predicate_column));
    if predicate_column {
        fields.push(Field::new("predicate", DataType::Int32, false));
    }
    fields.extend((0..context.payload_columns).map(|column_idx| {
        Field::new(
            format!("payload_{column_idx}"),
            context.payload_at(column_idx).data_type(),
            false,
        )
    }));
    Arc::new(Schema::new(fields))
}

fn build_oracle_row_group_batch(
    schema: SchemaRef,
    context: OracleContext,
    row_group_idx: usize,
    rows_per_group: usize,
    predicate: Option<&[i32]>,
) -> Result<RecordBatch> {
    let mut columns = Vec::with_capacity(schema.fields().len());
    if let Some(values) = predicate {
        columns.push(Arc::new(Int32Array::from(values.to_vec())) as ArrayRef);
    }

    let mut cached = Vec::<(OraclePayload, ArrayRef)>::new();
    for column_idx in 0..context.payload_columns {
        let payload = context.payload_at(column_idx);
        let values = if let Some((_, values)) = cached.iter().find(|(kind, _)| *kind == payload) {
            Arc::clone(values)
        } else {
            let values = build_oracle_payload(payload, row_group_idx, rows_per_group);
            cached.push((payload, Arc::clone(&values)));
            values
        };
        columns.push(values);
    }
    Ok(RecordBatch::try_new(schema, columns)?)
}

fn build_oracle_payload(
    payload: OraclePayload,
    row_group_idx: usize,
    rows_per_group: usize,
) -> ArrayRef {
    let global_start = row_group_idx * rows_per_group;
    match payload {
        OraclePayload::Int32 => Arc::new(Int32Array::from_iter_values(
            (0..rows_per_group).map(|row_idx| mix64(global_start + row_idx) as i32),
        )),
        OraclePayload::Int64 => Arc::new(Int64Array::from_iter_values(
            (0..rows_per_group).map(|row_idx| mix64(global_start + row_idx) as i64),
        )),
        OraclePayload::Float64 => Arc::new(Float64Array::from_iter_values(
            (0..rows_per_group).map(|row_idx| {
                let bits = 0x3ff0_0000_0000_0000 | (mix64(global_start + row_idx) >> 12);
                f64::from_bits(bits) - 1.0
            }),
        )),
        OraclePayload::Utf8View8 => Arc::new(StringViewArray::from_iter_values(
            (0..rows_per_group).map(|row_idx| oracle_string(global_start + row_idx, 8)),
        )),
        OraclePayload::Utf8View16 => Arc::new(StringViewArray::from_iter_values(
            (0..rows_per_group).map(|row_idx| oracle_string(global_start + row_idx, 16)),
        )),
        OraclePayload::Utf8View32 => Arc::new(StringViewArray::from_iter_values(
            (0..rows_per_group).map(|row_idx| oracle_string(global_start + row_idx, 32)),
        )),
        OraclePayload::Utf8View48 => Arc::new(StringViewArray::from_iter_values(
            (0..rows_per_group).map(|row_idx| oracle_string(global_start + row_idx, 48)),
        )),
        OraclePayload::Utf8View64 => Arc::new(StringViewArray::from_iter_values(
            (0..rows_per_group).map(|row_idx| oracle_string(global_start + row_idx, 64)),
        )),
        OraclePayload::BinaryView64 => Arc::new(BinaryViewArray::from_iter_values(
            (0..rows_per_group)
                .map(|row_idx| oracle_string(global_start + row_idx, 64).into_bytes()),
        )),
        OraclePayload::Utf8Dictionary1k => Arc::new(StringArray::from_iter_values(
            (0..rows_per_group).map(|row_idx| format!("d{:04x}", (global_start + row_idx) % 1_024)),
        )),
        OraclePayload::Utf8Dictionary {
            cardinality,
            value_width,
            fallback_plain_percent,
        } => {
            assert!(cardinality > 0 && cardinality <= rows_per_group);
            assert!(value_width >= 8);
            let plain_start = fallback_plain_percent.map(|percent| {
                assert!(matches!(percent, 25 | 75));
                rows_per_group * (100 - percent) / 100
            });
            Arc::new(StringArray::from_iter_values((0..rows_per_group).map(
                |row_idx| {
                    let dictionary_key = match plain_start {
                        Some(start) if row_idx < start => row_idx % 16,
                        Some(_) => global_start + row_idx,
                        None => (global_start + row_idx) % cardinality,
                    };
                    oracle_dictionary_string(dictionary_key, value_width)
                },
            )))
        }
    }
}

fn assert_dictionary_encoding_contract(
    metadata: &ParquetMetaData,
    context: OracleContext,
    predicate_column: bool,
) {
    let first_payload = usize::from(predicate_column);
    for (row_group_idx, row_group) in metadata.row_groups().iter().enumerate() {
        for payload_idx in 0..context.payload_columns {
            let payload = context.payload_at(payload_idx);
            let fallback_plain_percent = match payload {
                OraclePayload::Utf8Dictionary1k => None,
                OraclePayload::Utf8Dictionary {
                    fallback_plain_percent,
                    ..
                } => fallback_plain_percent,
                _ => {
                    if context.column_payloads.is_some() {
                        let column_idx = first_payload + payload_idx;
                        let mask = row_group
                            .column(column_idx)
                            .page_encoding_stats_mask()
                            .unwrap_or_else(|| {
                                panic!(
                                    "mixed oracle fixture lacks data-page encoding stats: row_group={row_group_idx}, column={column_idx}"
                                )
                            });
                        assert!(
                            !mask.is_set(Encoding::RLE_DICTIONARY)
                                && !mask.is_set(Encoding::PLAIN_DICTIONARY),
                            "mixed oracle non-dictionary column was dictionary encoded: row_group={row_group_idx}, column={column_idx}, payload={payload:?}, mask={mask:?}"
                        );
                    }
                    continue;
                }
            };
            let column_idx = first_payload + payload_idx;
            let column = row_group.column(column_idx);
            let mask = column.page_encoding_stats_mask().unwrap_or_else(|| {
                panic!(
                    "oracle dictionary fixture lacks data-page encoding stats: row_group={row_group_idx}, column={column_idx}"
                )
            });
            let dictionary =
                mask.is_set(Encoding::RLE_DICTIONARY) || mask.is_set(Encoding::PLAIN_DICTIONARY);
            assert!(
                dictionary,
                "oracle dictionary fixture has no dictionary data page"
            );
            if fallback_plain_percent.is_some() {
                assert!(
                    mask.is_set(Encoding::PLAIN),
                    "oracle fallback fixture has no PLAIN data page"
                );
            } else {
                assert!(
                    mask.is_only(Encoding::RLE_DICTIONARY)
                        || mask.is_only(Encoding::PLAIN_DICTIONARY),
                    "oracle pure dictionary fixture contains a non-dictionary data page: {mask:?}"
                );
            }
        }
    }
}

fn mix64(value: usize) -> u64 {
    let mut value = (value as u64).wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn oracle_string(row: usize, len: usize) -> String {
    let mut value = String::with_capacity(len);
    let mut lane = 0usize;
    while value.len() < len {
        value.push_str(&format!("{:016x}", mix64(row.wrapping_add(lane))));
        lane = lane.wrapping_add(0x1_0001);
    }
    value.truncate(len);
    value
}

fn oracle_dictionary_string(value: usize, len: usize) -> String {
    let mut output = format!("{value:08x}");
    let mut lane = 0usize;
    while output.len() < len {
        output.push_str(&format!("{:016x}", mix64(value.wrapping_add(lane))));
        lane = lane.wrapping_add(0x1_0001);
    }
    output.truncate(len);
    output
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

pub(crate) fn build_fixture(case: &CaseSpec) -> Result<CaseFixture> {
    assert!(
        !case.row_groups.is_empty(),
        "benchmark case must contain at least one row group"
    );

    let schema = build_schema();
    let properties = WriterProperties::builder()
        .set_compression(Compression::UNCOMPRESSED)
        .set_dictionary_enabled(false)
        .set_max_row_group_row_count(Some(ROWS_PER_GROUP))
        .build();

    let mut encoded = Vec::new();
    {
        let mut writer = ArrowWriter::try_new(&mut encoded, Arc::clone(&schema), Some(properties))?;
        for (row_group_idx, pattern) in case.row_groups.iter().copied().enumerate() {
            writer.write(&build_row_group_batch(
                Arc::clone(&schema),
                pattern,
                row_group_idx,
            )?)?;
        }
        writer.close()?;
    }

    let bytes = Bytes::from(encoded);
    let mut metadata_reader =
        ParquetMetaDataReader::new().with_page_index_policy(PageIndexPolicy::Skip);
    metadata_reader.try_parse(&bytes)?;
    let metadata = Arc::new(metadata_reader.finish()?);

    assert_eq!(
        metadata.num_row_groups(),
        case.row_groups.len(),
        "writer did not preserve the requested row-group layout"
    );
    for row_group in metadata.row_groups() {
        assert_eq!(row_group.num_rows() as usize, ROWS_PER_GROUP);
    }

    let expected_rows = case
        .row_groups
        .iter()
        .copied()
        .map(|pattern| selected_rows(pattern, ROWS_PER_GROUP))
        .sum();

    Ok(CaseFixture {
        bytes,
        metadata,
        expected_rows,
    })
}

fn build_schema() -> SchemaRef {
    let mut fields = Vec::with_capacity(PAYLOAD_COLUMNS + 1);
    fields.push(Field::new("predicate", DataType::Int32, false));
    fields.extend(
        (0..PAYLOAD_COLUMNS)
            .map(|column_idx| Field::new(format!("payload_{column_idx}"), DataType::Int32, false)),
    );
    Arc::new(Schema::new(fields))
}

fn build_row_group_batch(
    schema: SchemaRef,
    pattern: super::model::RowGroupPattern,
    row_group_idx: usize,
) -> Result<RecordBatch> {
    let predicate = expand_pattern(pattern, ROWS_PER_GROUP);
    let mut columns = Vec::with_capacity(PAYLOAD_COLUMNS + 1);
    columns.push(Arc::new(Int32Array::from(predicate)) as ArrayRef);

    for column_idx in 0..PAYLOAD_COLUMNS {
        let values = Int32Array::from_iter_values((0..ROWS_PER_GROUP).map(|row_idx| {
            let global_row = row_group_idx * ROWS_PER_GROUP + row_idx;
            global_row
                .wrapping_add(column_idx * 17)
                .wrapping_rem(PAYLOAD_VALUE_MODULUS) as i32
        }));
        columns.push(Arc::new(values) as ArrayRef);
    }

    Ok(RecordBatch::try_new(schema, columns)?)
}
