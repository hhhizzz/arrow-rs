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

//! End-to-end tests for reading `Utf8`/`Binary` columns through 64-bit offset
//! staging. A small byte capacity models the `i32::MAX` limit of a 32-bit
//! offset array, not a reader memory budget.

use crate::arrow::arrow_reader::{
    ParquetRecordBatchReaderBuilder, RowSelection, RowSelectionPolicy, RowSelector,
};
use crate::arrow::schema::ArrowSchemaConverter;
use crate::arrow::{ArrowWriter, ProjectionMask};
use crate::basic::{Encoding, Type as PhysicalType};
use crate::file::properties::WriterProperties;
use crate::file::reader::{FileReader, SerializedFileReader};
use arrow_array::builder::{LargeListBuilder, ListBuilder, MapBuilder, StringBuilder};
use arrow_array::types::Int32Type;
use arrow_array::{
    Array, ArrayRef, BinaryArray, BinaryViewArray, Decimal128Array, DictionaryArray, Int64Array,
    LargeBinaryArray, LargeListViewArray, LargeStringArray, ListArray, ListViewArray, MapArray,
    RecordBatch, StringArray, StringViewArray, StructArray,
};
use arrow_schema::{DataType, Field, FieldRef, Schema};
use arrow_select::concat::concat_batches;
use bytes::Bytes;
use std::collections::HashMap;
use std::sync::Arc;

const CODECS: [Encoding; 4] = [
    Encoding::PLAIN,
    Encoding::RLE_DICTIONARY,
    Encoding::DELTA_LENGTH_BYTE_ARRAY,
    Encoding::DELTA_BYTE_ARRAY,
];

fn field(name: &str, data_type: DataType, nullable: bool) -> FieldRef {
    Arc::new(
        Field::new(name, data_type, nullable)
            .with_metadata(HashMap::from([("test-owner".into(), name.into())])),
    )
}

fn write(batch: &RecordBatch, encoding: Encoding) -> Bytes {
    let descriptor = ArrowSchemaConverter::new()
        .convert(batch.schema().as_ref())
        .unwrap();
    let mut properties = WriterProperties::builder()
        .set_dictionary_enabled(false)
        .set_max_row_group_row_count(Some(16))
        .set_write_batch_size(3)
        .set_data_page_row_count_limit(3);
    for column in descriptor.columns() {
        if column.physical_type() == PhysicalType::BYTE_ARRAY {
            properties = if encoding == Encoding::RLE_DICTIONARY {
                properties.set_column_dictionary_enabled(column.path().clone(), true)
            } else {
                properties.set_column_encoding(column.path().clone(), encoding)
            };
        }
    }
    let mut data = vec![];
    let mut writer =
        ArrowWriter::try_new(&mut data, batch.schema(), Some(properties.build())).unwrap();
    writer.write(batch).unwrap();
    writer.close().unwrap();
    let bytes = Bytes::from(data);
    let file = SerializedFileReader::new(bytes.clone()).unwrap();
    let first_byte_array = file
        .metadata()
        .file_metadata()
        .schema_descr()
        .columns()
        .iter()
        .position(|column| column.physical_type() == PhysicalType::BYTE_ARRAY)
        .unwrap();
    assert!(
        file.metadata()
            .row_group(0)
            .column(first_byte_array)
            .encodings()
            .any(|e| e == encoding),
        "fixture must contain the requested {encoding:?} value encoding"
    );
    bytes
}

fn mixed_batch() -> RecordBatch {
    let rows = 12;
    let texts: Vec<_> = (0..rows)
        .map(|i| format!("row-{i:02}-{}", "界".repeat(5 + i % 3)))
        .collect();
    let strings = StringArray::from_iter(
        texts
            .iter()
            .enumerate()
            .map(|(i, value)| (i % 5 != 0).then_some(value.as_str())),
    );
    let binary_values: Vec<_> = (0..rows)
        .map(|i| vec![0xff, i as u8, 0, 0, 7, 8, 9, 10, 11, 12, 13, 14, 15])
        .collect();
    let binary = BinaryArray::from_iter(binary_values.iter().map(|value| Some(value.as_slice())));
    let mut lists =
        ListBuilder::new(StringBuilder::new()).with_field(field("element", DataType::Utf8, true));
    let mut maps = MapBuilder::new(None, StringBuilder::new(), StringBuilder::new())
        .with_keys_field(field("keys", DataType::Utf8, false))
        .with_values_field(field("values", DataType::Utf8, true));
    for (i, text) in texts.iter().enumerate() {
        match i % 4 {
            0 => lists.append(false),
            1 => lists.append(true),
            2 => {
                lists.values().append_value(text);
                lists.values().append_null();
                lists.append(true);
            }
            _ => {
                lists.values().append_value(text);
                lists.values().append_value("second-long-element");
                lists.append(true);
            }
        }
        if i % 5 != 0 {
            maps.keys().append_value(format!("k{i:02}"));
            maps.values().append_value(text);
            maps.append(true).unwrap();
        } else {
            maps.append(false).unwrap();
        }
    }
    let nested = StructArray::new(
        vec![
            field("nested_text", DataType::Utf8, true),
            field("id", DataType::Int64, false),
        ]
        .into(),
        vec![
            Arc::new(strings.clone()),
            Arc::new(Int64Array::from_iter_values(0..rows as i64)),
        ],
        None,
    );
    let arrays: Vec<ArrayRef> = vec![
        Arc::new(strings),
        Arc::new(binary),
        Arc::new(lists.finish()),
        Arc::new(nested),
        Arc::new(maps.finish()),
    ];
    let names = ["text", "binary", "list", "nested", "map"];
    let fields: Vec<_> = names
        .into_iter()
        .zip(&arrays)
        .map(|(name, array)| field(name, array.data_type().clone(), true))
        .collect();
    RecordBatch::try_new(Arc::new(Schema::new(fields)), arrays).unwrap()
}

/// Read with the given byte capacity; `None` reads through the unstaged
/// readers, which is the reference behavior.
fn read_sync(
    data: &Bytes,
    limit: Option<usize>,
    projection: Option<&[usize]>,
    selection: Option<RowSelection>,
    policy: RowSelectionPolicy,
    batch_size: usize,
) -> Vec<RecordBatch> {
    let mut builder = ParquetRecordBatchReaderBuilder::try_new(data.clone())
        .unwrap()
        .with_batch_size(batch_size)
        .with_row_selection_policy(policy);
    if let Some(leaves) = projection {
        let mask = ProjectionMask::leaves(builder.parquet_schema(), leaves.iter().copied());
        builder = builder.with_projection(mask);
    }
    if let Some(selection) = selection {
        builder = builder.with_row_selection(selection);
    }
    builder = builder.with_byte_array_capacity(limit);
    builder.build().unwrap().map(Result::unwrap).collect()
}

fn flatten(batches: &[RecordBatch]) -> RecordBatch {
    concat_batches(&batches[0].schema(), batches.iter()).unwrap()
}

// Visible byte-array payload per leaf in this top-level row interval. Lists and
// maps retain whole records; offsets may reference a slice of a shared child.
fn leaf_payloads(array: &dyn Array, start: usize, end: usize) -> Vec<usize> {
    match array.data_type() {
        DataType::Utf8 => {
            let offsets = array
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap()
                .value_offsets();
            vec![(offsets[end] - offsets[start]) as usize]
        }
        DataType::Binary => {
            let offsets = array
                .as_any()
                .downcast_ref::<BinaryArray>()
                .unwrap()
                .value_offsets();
            vec![(offsets[end] - offsets[start]) as usize]
        }
        DataType::List(_) => {
            let list = array.as_any().downcast_ref::<ListArray>().unwrap();
            leaf_payloads(
                list.values().as_ref(),
                list.value_offsets()[start] as usize,
                list.value_offsets()[end] as usize,
            )
        }
        DataType::Map(_, _) => {
            let map = array.as_any().downcast_ref::<MapArray>().unwrap();
            leaf_payloads(
                map.entries(),
                map.value_offsets()[start] as usize,
                map.value_offsets()[end] as usize,
            )
        }
        DataType::Struct(_) => array
            .as_any()
            .downcast_ref::<StructArray>()
            .unwrap()
            .columns()
            .iter()
            .flat_map(|child| leaf_payloads(child.as_ref(), start, end))
            .collect(),
        _ => vec![],
    }
}

fn assert_legal_payloads(batches: &[RecordBatch], limit: usize) {
    for batch in batches {
        assert!(batch.num_rows() > 0);
        for array in batch.columns() {
            assert!(
                leaf_payloads(array.as_ref(), 0, batch.num_rows())
                    .into_iter()
                    .all(|bytes| bytes <= limit)
            );
        }
    }
}

#[test]
fn test_capacity_staging_matches_native_all_codecs_and_deep_projection() {
    let batch = mixed_batch();
    for encoding in CODECS {
        let data = write(&batch, encoding);
        for projection in [None, Some([0, 2, 3].as_slice())] {
            let native = read_sync(&data, None, projection, None, RowSelectionPolicy::Mask, 4);
            let staged = read_sync(
                &data,
                Some(i32::MAX as usize),
                projection,
                None,
                RowSelectionPolicy::Mask,
                4,
            );
            assert_eq!(
                staged, native,
                "{encoding:?}: includes batch shape and nested Field metadata"
            );
        }
    }
}

#[test]
fn test_capacity_staging_small_limit_keeps_complete_nested_records() {
    let batch = mixed_batch();
    for encoding in CODECS {
        let data = write(&batch, encoding);
        let native = read_sync(&data, None, None, None, RowSelectionPolicy::Mask, 8);
        let staged = read_sync(&data, Some(64), None, None, RowSelectionPolicy::Mask, 8);
        assert!(
            staged.len() > native.len(),
            "{encoding:?}: capacity must split actual API output"
        );
        assert_legal_payloads(&staged, 64);
        assert_eq!(flatten(&staged), flatten(&native), "{encoding:?}");
    }
}

#[test]
fn test_capacity_staging_mask_discards_heavy_unselected_rows_before_cast() {
    let strings = StringArray::from_iter_values((0..20).map(|i| {
        if i % 2 == 0 {
            "h".repeat(64)
        } else {
            "x".into()
        }
    }));
    let batch = RecordBatch::try_from_iter(vec![
        ("text", Arc::new(strings) as ArrayRef),
        (
            "id",
            Arc::new(Int64Array::from_iter_values(0..20)) as ArrayRef,
        ),
    ])
    .unwrap();
    let selection = RowSelection::from(
        (0..20)
            .map(|i| {
                if i % 2 == 0 {
                    RowSelector::skip(1)
                } else {
                    RowSelector::select(1)
                }
            })
            .collect::<Vec<_>>(),
    );
    for encoding in CODECS {
        let data = write(&batch, encoding);
        let mask = read_sync(
            &data,
            Some(4),
            None,
            Some(selection.clone()),
            RowSelectionPolicy::Mask,
            8,
        );
        let selectors = read_sync(
            &data,
            Some(4),
            None,
            Some(selection.clone()),
            RowSelectionPolicy::Selectors,
            8,
        );
        assert_eq!(flatten(&mask), flatten(&selectors));
        assert_legal_payloads(&mask, 4);
        let merged = flatten(&mask);
        let ids = merged
            .column(1)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        assert_eq!(ids.values().as_ref(), &[1, 3, 5, 7, 9, 11, 13, 15, 17, 19]);
    }
}

#[test]
fn test_capacity_staging_single_selected_record_error() {
    let batch = RecordBatch::try_from_iter(vec![(
        "text",
        Arc::new(StringArray::from(vec!["a-single-too-large-value"])) as ArrayRef,
    )])
    .unwrap();
    let data = write(&batch, Encoding::DELTA_BYTE_ARRAY);
    let mut reader = ParquetRecordBatchReaderBuilder::try_new(data)
        .unwrap()
        .with_byte_array_capacity(Some(5))
        .build()
        .unwrap();
    let error = reader.next().unwrap().unwrap_err().to_string();
    assert!(
        error.contains("a single row needs 24 bytes of Utf8 data for column text"),
        "{error}"
    );
    assert!(
        reader.next().is_none(),
        "terminal capacity error must release pending output"
    );
}

#[test]
fn test_capacity_staging_preserves_original_view_large_dictionary_decimal_types() {
    let wide = [
        "original-long-string-a",
        "original-long-string-b",
        "original-long-string-c",
    ];
    let byte_slices: Vec<_> = wide.iter().map(|s| s.as_bytes()).collect();
    let arrays: Vec<ArrayRef> = vec![
        Arc::new(StringArray::from(vec!["a", "b", "c"])),
        Arc::new(StringViewArray::from(wide.to_vec())),
        Arc::new(BinaryViewArray::from(byte_slices.clone())),
        Arc::new(LargeStringArray::from(wide.to_vec())),
        Arc::new(LargeBinaryArray::from(byte_slices)),
        Arc::new(wide.iter().copied().collect::<DictionaryArray<Int32Type>>()),
        Arc::new(
            Decimal128Array::from(vec![123, 456, 789])
                .with_precision_and_scale(10, 2)
                .unwrap(),
        ),
    ];
    let fields: Vec<_> = arrays
        .iter()
        .enumerate()
        .map(|(i, a)| field(&format!("c{i}"), a.data_type().clone(), true))
        .collect();
    let batch = RecordBatch::try_new(Arc::new(Schema::new(fields)), arrays).unwrap();
    let data = write(&batch, Encoding::RLE_DICTIONARY);
    let native = read_sync(&data, None, None, None, RowSelectionPolicy::Selectors, 3);
    let staged = read_sync(&data, Some(2), None, None, RowSelectionPolicy::Selectors, 3);
    assert_eq!(staged[0].schema(), native[0].schema());
    assert_eq!(flatten(&staged), flatten(&native));
    assert_eq!(
        staged.iter().map(RecordBatch::num_rows).collect::<Vec<_>>(),
        [2, 1]
    );
}

#[test]
fn test_capacity_staging_zero_projection_preserves_row_counts_and_shape() {
    let batch = mixed_batch();
    let data = write(&batch, Encoding::PLAIN);
    let native = read_sync(&data, None, Some(&[]), None, RowSelectionPolicy::Mask, 5);
    let staged = read_sync(&data, Some(0), Some(&[]), None, RowSelectionPolicy::Mask, 5);
    assert_eq!(staged, native);
    assert!(staged.iter().all(|batch| batch.num_columns() == 0));
    assert_eq!(
        staged.iter().map(RecordBatch::num_rows).sum::<usize>(),
        batch.num_rows()
    );
}

#[test]
fn test_capacity_staging_list_view_subtrees_keep_native_behavior() {
    let mut lists = ListBuilder::new(StringBuilder::new());
    let mut large_lists = LargeListBuilder::new(StringBuilder::new());
    for row in 0..4 {
        match row {
            0 | 3 => {
                lists
                    .values()
                    .append_value("native-list-view-value-exceeds-the-testing-limit");
                large_lists
                    .values()
                    .append_value("native-large-list-view-value-exceeds-the-testing-limit");
                lists.append(true);
                large_lists.append(true);
            }
            1 => {
                lists.append(true);
                large_lists.append(true);
            }
            _ => {
                lists.append(false);
                large_lists.append(false);
            }
        }
    }
    let arrays: Vec<ArrayRef> = vec![
        Arc::new(ListViewArray::from(lists.finish())),
        Arc::new(LargeListViewArray::from(large_lists.finish())),
    ];
    let fields: Vec<_> = arrays
        .iter()
        .enumerate()
        .map(|(i, array)| field(&format!("list_view_{i}"), array.data_type().clone(), true))
        .collect();
    let batch = RecordBatch::try_new(Arc::new(Schema::new(fields)), arrays).unwrap();
    let data = write(&batch, Encoding::PLAIN);
    let native = read_sync(&data, None, None, None, RowSelectionPolicy::Mask, 3);
    // These unsupported container subtrees deliberately retain native decoding:
    // enabling staging must neither reject them nor promise this byte limit.
    let staged = read_sync(&data, Some(1), None, None, RowSelectionPolicy::Mask, 3);
    assert_eq!(staged, native);
    assert_eq!(staged[0].schema(), batch.schema());
}

/// A real >i32::MAX decoded logical window with configurable short output
/// batches. This intentionally does not request a single >2 GiB Utf8 output
/// batch and does not run the native decoder's large-allocation failure path.
/// The dictionary keeps the encoded file near 3 MiB; streamed output is dropped
/// after each <=32 MiB batch instead of collecting or concatenating 2 GiB.
#[test]
#[ignore = "streams over 2 GiB of logical string output; run explicitly"]
fn test_capacity_staging_real_i32_logical_window_dictionary_streamed() {
    use crate::column::page::Page;
    use crate::schema::types::ColumnPath;

    const VALUE_BYTES: usize = 3 * 1024 * 1024;
    const ROWS_PER_WRITE: usize = 10;
    const ROWS: usize = 700;
    const OUTPUT_LIMIT: usize = 32 * 1024 * 1024;
    assert!(ROWS * VALUE_BYTES > i32::MAX as usize);

    let schema = Arc::new(Schema::new(vec![
        Field::new("text", DataType::Utf8, false),
        Field::new("id", DataType::Int64, false),
    ]));
    let value = "p".repeat(VALUE_BYTES);
    // This 30 MiB array is reused for all 70 writes. No builder ever contains
    // the full 700-row expanded payload.
    let strings: ArrayRef = Arc::new(StringArray::from_iter_values(std::iter::repeat_n(
        value.as_str(),
        ROWS_PER_WRITE,
    )));
    let properties = WriterProperties::builder()
        .set_dictionary_enabled(false)
        .set_column_dictionary_enabled(ColumnPath::from("text"), true)
        .set_dictionary_page_size_limit(8 * 1024 * 1024)
        .set_max_row_group_row_count(Some(ROWS))
        .set_write_batch_size(ROWS_PER_WRITE)
        .set_data_page_row_count_limit(ROWS)
        .build();
    let mut data = vec![];
    let mut writer = ArrowWriter::try_new(&mut data, schema.clone(), Some(properties)).unwrap();
    for start in (0..ROWS).step_by(ROWS_PER_WRITE) {
        let ids: ArrayRef = Arc::new(Int64Array::from_iter_values(
            start as i64..(start + ROWS_PER_WRITE) as i64,
        ));
        writer
            .write(&RecordBatch::try_new(schema.clone(), vec![strings.clone(), ids]).unwrap())
            .unwrap();
    }
    writer.close().unwrap();
    drop(strings);
    drop(value);
    let data = Bytes::from(data);
    assert!(
        data.len() < 4 * 1024 * 1024,
        "dictionary fixture unexpectedly expanded: {} bytes",
        data.len()
    );

    let file = SerializedFileReader::new(data.clone()).unwrap();
    assert_eq!(file.metadata().num_row_groups(), 1);
    assert_eq!(file.metadata().row_group(0).num_rows(), ROWS as i64);
    assert!(
        file.metadata()
            .row_group(0)
            .column(0)
            .encodings()
            .any(|encoding| encoding == Encoding::RLE_DICTIONARY)
    );
    let row_group = file.get_row_group(0).unwrap();
    let mut pages = row_group.get_column_page_reader(0).unwrap();
    let mut data_pages = 0;
    while let Some(page) = pages.get_next_page().unwrap() {
        match page {
            Page::DataPage { encoding, .. } | Page::DataPageV2 { encoding, .. } => {
                assert_eq!(
                    encoding,
                    Encoding::RLE_DICTIONARY,
                    "dictionary must not fall back to expanded PLAIN pages"
                );
                data_pages += 1;
            }
            Page::DictionaryPage { num_values, .. } => assert_eq!(num_values, 1),
        }
    }
    assert!(data_pages > 0);
    drop(pages);
    drop(row_group);
    drop(file);

    let reader = ParquetRecordBatchReaderBuilder::try_new(data)
        .unwrap()
        .with_batch_size(ROWS)
        .with_byte_array_capacity(Some(OUTPUT_LIMIT))
        .build()
        .unwrap();
    let mut rows = 0;
    let mut logical_bytes = 0usize;
    let mut batches = 0;
    for batch in reader {
        let batch = batch.unwrap();
        let strings = batch
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let ids = batch
            .column(1)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        assert_eq!(batch.num_rows(), ROWS_PER_WRITE);
        assert!(strings.value_data().len() <= OUTPUT_LIMIT);
        for i in 0..batch.num_rows() {
            assert_eq!(ids.value(i), (rows + i) as i64);
            let bytes = strings.value(i).as_bytes();
            assert_eq!(bytes.len(), VALUE_BYTES);
            assert!(bytes[..32].iter().all(|&byte| byte == b'p'));
            assert!(bytes[VALUE_BYTES - 32..].iter().all(|&byte| byte == b'p'));
            assert_eq!(bytes[VALUE_BYTES / 2], b'p');
        }
        rows += batch.num_rows();
        logical_bytes += batch.num_rows() * VALUE_BYTES;
        batches += 1;
        // `batch` and its copied output payload are released on every iteration.
    }
    assert_eq!(rows, ROWS);
    assert!(logical_bytes > i32::MAX as usize);
    assert_eq!(batches, 70);
}

#[cfg(feature = "async")]
#[tokio::test]
async fn test_capacity_staging_async_predicate_cache_and_policies() {
    use crate::arrow::ParquetRecordBatchStreamBuilder;
    use crate::arrow::arrow_reader::{ArrowPredicateFn, RowFilter, metrics::ArrowReaderMetrics};
    use arrow_array::BooleanArray;
    use futures::TryStreamExt;
    use std::io::Cursor;
    use std::sync::Mutex;

    let batch = RecordBatch::try_from_iter(vec![
        (
            "text",
            Arc::new(StringArray::from_iter_values(
                (0..24).map(|i| format!("value-{i:03}")),
            )) as ArrayRef,
        ),
        (
            "id",
            Arc::new(Int64Array::from_iter_values(0..24)) as ArrayRef,
        ),
    ])
    .unwrap();
    let data = write(&batch, Encoding::DELTA_BYTE_ARRAY);
    let mut baseline = None;
    for policy in [RowSelectionPolicy::Selectors, RowSelectionPolicy::Mask] {
        for cache_size in [0, usize::MAX] {
            let builder = ParquetRecordBatchStreamBuilder::new(Cursor::new(data.clone()))
                .await
                .unwrap();
            let projection = ProjectionMask::leaves(builder.parquet_schema(), [0]);
            let counts = Arc::new(Mutex::new(vec![0usize; 24]));
            let batch_sizes = Arc::new(Mutex::new(vec![]));
            let observed = counts.clone();
            let observed_sizes = batch_sizes.clone();
            let predicate = ArrowPredicateFn::new(projection, move |batch: RecordBatch| {
                // The callback must see the requested type, not the staged one.
                let strings = batch
                    .column(0)
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .expect("predicate requires ordinary Utf8");
                observed_sizes.lock().unwrap().push(batch.num_rows());
                let mut counts = observed.lock().unwrap();
                Ok(BooleanArray::from_iter(strings.iter().map(|value| {
                    let id: usize = value.unwrap()[6..].parse().unwrap();
                    counts[id] += 1;
                    !id.is_multiple_of(3)
                })))
            });
            let metrics = ArrowReaderMetrics::enabled();
            let batches: Vec<RecordBatch> = builder
                .with_batch_size(8)
                .with_row_selection_policy(policy)
                .with_max_predicate_cache_size(cache_size)
                .with_metrics(metrics.clone())
                .with_row_filter(RowFilter::new(vec![Box::new(predicate)]))
                .with_byte_array_capacity(Some(20))
                .build()
                .unwrap()
                .try_collect()
                .await
                .unwrap();
            assert_eq!(
                *counts.lock().unwrap(),
                vec![1; 24],
                "{policy:?}, cache={cache_size}: each input row evaluated once"
            );
            assert!(batch_sizes.lock().unwrap().iter().all(|&rows| rows <= 2));
            assert!(
                batch_sizes.lock().unwrap().len() > 3,
                "capacity must also split predicate input"
            );
            assert_legal_payloads(&batches, 20);
            let merged = flatten(&batches);
            let ids = merged
                .column(1)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap();
            assert_eq!(
                ids.values().as_ref(),
                &(0..24).filter(|i| i % 3 != 0).collect::<Vec<_>>()
            );
            if let Some(expected) = baseline.as_ref() {
                assert_eq!(&merged, expected);
            } else {
                baseline = Some(merged);
            }
            let cache_reads = metrics.records_read_from_cache().unwrap();
            if cache_size == 0 {
                assert_eq!(cache_reads, 0);
            } else {
                assert!(
                    cache_reads > 0,
                    "{policy:?}: enabled cache must actually be exercised"
                );
            }
        }
    }
}
