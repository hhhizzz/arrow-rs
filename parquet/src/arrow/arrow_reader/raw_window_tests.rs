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

//! Real Parquet regressions for Mask windows made of whole batch-size grids.
//! These tests never concatenate output: Dictionary<View> and ListView output
//! must remain legal independently, without a test-side coalescer hiding errors.
//!
//! The reader merges window results with fewer than `batch_size / 2` rows,
//! unless the projection contains a dictionary or a list view.

use super::{
    FIRST_MASK_WINDOW_GRIDS, MASK_WINDOW_BYTES, ParquetRecordBatchReaderBuilder, RowSelection,
    RowSelectionPolicy,
};
use crate::arrow::ArrowWriter;
use crate::arrow::schema::ArrowSchemaConverter;
use crate::basic::{Encoding, Type as PhysicalType};
use crate::file::properties::WriterProperties;
use arrow_array::builder::{LargeListBuilder, ListBuilder, MapBuilder, StringBuilder};
use arrow_array::cast::AsArray;
use arrow_array::types::Int8Type;
use arrow_array::{
    Array, ArrayRef, BooleanArray, DictionaryArray, Int8Array, Int64Array, LargeListViewArray,
    ListArray, ListViewArray, RecordBatch, StringArray, StringViewArray, StructArray,
};
use arrow_buffer::{NullBuffer, OffsetBuffer};
use arrow_schema::{DataType, Field, Schema};
use arrow_select::filter::filter_record_batch;
use bytes::Bytes;
use std::sync::Arc;

const GRID: usize = 8;
const CODECS: [Encoding; 4] = [
    Encoding::PLAIN,
    Encoding::RLE_DICTIONARY,
    Encoding::DELTA_LENGTH_BYTE_ARRAY,
    Encoding::DELTA_BYTE_ARRAY,
];

fn encode(source: &RecordBatch, encoding: Encoding, page_rows: usize, group_rows: usize) -> Bytes {
    let descriptor = ArrowSchemaConverter::new()
        .convert(source.schema().as_ref())
        .unwrap();
    let mut props = WriterProperties::builder()
        .set_dictionary_enabled(false)
        .set_max_row_group_row_count(Some(group_rows))
        .set_write_batch_size(page_rows)
        .set_data_page_row_count_limit(page_rows);
    for column in descriptor.columns() {
        if column.physical_type() == PhysicalType::BYTE_ARRAY {
            props = if encoding == Encoding::RLE_DICTIONARY {
                props.set_column_dictionary_enabled(column.path().clone(), true)
            } else {
                props.set_column_encoding(column.path().clone(), encoding)
            };
        }
    }
    let mut bytes = vec![];
    let mut writer =
        ArrowWriter::try_new(&mut bytes, source.schema(), Some(props.build())).unwrap();
    writer.write(source).unwrap();
    writer.close().unwrap();
    Bytes::from(bytes)
}

fn flat_source(rows: usize) -> RecordBatch {
    RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("text", DataType::Utf8, false),
        ])),
        vec![
            Arc::new(Int64Array::from_iter_values(0..rows as i64)),
            Arc::new(StringArray::from_iter_values(
                (0..rows).map(|id| format!("row-{id:04}")),
            )),
        ],
    )
    .unwrap()
}

fn selection(mask: &[bool]) -> RowSelection {
    RowSelection::from_filters(&[BooleanArray::from(mask.to_vec())])
}

fn assert_rows(batches: &[RecordBatch], expected: &RecordBatch) {
    let mut position = 0;
    for batch in batches {
        assert!(
            batch.num_rows() > 0,
            "an empty raw window must not be emitted as EOF/data"
        );
        assert_eq!(batch.schema(), expected.schema());
        let expected = expected.slice(position, batch.num_rows());
        for (actual, expected) in batch.columns().iter().zip(expected.columns()) {
            if let DataType::Dictionary(_, values) = actual.data_type() {
                // Different legal dictionary domains may encode equal values.
                let actual = arrow::compute::cast(actual, values).unwrap();
                let expected = arrow::compute::cast(expected, values).unwrap();
                assert_eq!(actual.as_ref(), expected.as_ref());
            } else {
                assert_eq!(actual.as_ref(), expected.as_ref());
            }
        }
        position += batch.num_rows();
    }
    assert_eq!(
        position,
        expected.num_rows(),
        "no lost or duplicated selected rows"
    );
}

fn row_counts(batches: &[RecordBatch]) -> Vec<usize> {
    batches.iter().map(RecordBatch::num_rows).collect()
}

fn selected_ids(mask: &[bool]) -> impl Iterator<Item = usize> + '_ {
    mask.iter()
        .enumerate()
        .filter_map(|(id, &keep)| keep.then_some(id))
}

/// One Mask window of one reader, as modeled by [`model_windows`].
struct ModelWindow {
    /// The first selected raw row
    first: usize,
    /// The end of the last grid the window may span
    end: usize,
    /// Selected rows
    rows: usize,
}

/// Mask windows of one reader for selected raw rows `ids`, when no page gap
/// splits a window: a window starts at the next selected row and ends after
/// `grid` selected rows or at the end of the last grid it may span.
///
/// The first window may span [`FIRST_MASK_WINDOW_GRIDS`] grids and each later
/// window twice as many as the one before: these tests decode a few hundred
/// bytes per grid, far below [`MASK_WINDOW_BYTES`], so the size estimate never
/// limits the growth.
fn model_windows(ids: impl IntoIterator<Item = usize>, grid: usize) -> Vec<ModelWindow> {
    let mut windows: Vec<ModelWindow> = vec![];
    let mut grids = FIRST_MASK_WINDOW_GRIDS as usize;
    for id in ids {
        match windows.last_mut() {
            Some(window) if id < window.end && window.rows < grid => window.rows += 1,
            last => {
                if last.is_some() {
                    grids *= 2;
                }
                windows.push(ModelWindow {
                    first: id,
                    end: (id / grid + grids) * grid,
                    rows: 1,
                });
            }
        }
    }
    // Even at 1 KiB per raw row, the last window's grids fit the budget.
    assert!(grid * grids * 1024 < MASK_WINDOW_BYTES);
    windows
}

/// Selected rows of each Mask window; see [`model_windows`].
fn expected_windows(ids: impl IntoIterator<Item = usize>, grid: usize) -> Vec<usize> {
    model_windows(ids, grid)
        .into_iter()
        .map(|window| window.rows)
        .collect()
}

/// Batch sizes for `ids` (see [`expected_windows`]): window results with at
/// least `grid / 2` rows are returned unchanged, smaller ones are merged.
fn expected_merged_rows(ids: impl IntoIterator<Item = usize>, grid: usize) -> Vec<usize> {
    let mut batches = vec![];
    let mut buffered = 0;
    for rows in expected_windows(ids, grid) {
        if rows >= grid / 2 {
            if buffered > 0 {
                batches.push(buffered);
                buffered = 0;
            }
            batches.push(rows);
            continue;
        }
        buffered += rows;
        if buffered >= grid {
            batches.push(grid);
            buffered -= grid;
        }
    }
    if buffered > 0 {
        batches.push(buffered);
    }
    batches
}

/// Holds whatever the window boundaries are: no batch exceeds `grid` rows,
/// and a batch of fewer than `grid / 2` rows is followed only by a large
/// window result, so consecutive small results were merged.
fn assert_merged_batches(batches: &[RecordBatch], grid: usize) {
    for (batch, next) in batches.iter().zip(batches.iter().skip(1)) {
        if batch.num_rows() < grid / 2 {
            assert!(
                next.num_rows() >= grid / 2,
                "small results were not merged: {:?}",
                row_counts(batches)
            );
        }
    }
    assert!(batches.iter().all(|batch| batch.num_rows() <= grid));
}

/// The modeled window that each batch of unmerged Mask output lies within; a
/// batch has at most `grid` rows.
fn batch_windows<'a>(
    batches: &[RecordBatch],
    windows: &'a [ModelWindow],
    grid: usize,
) -> Vec<&'a ModelWindow> {
    batches
        .iter()
        .map(|batch| {
            let ids = batch
                .column(batch.schema().index_of("id").unwrap())
                .as_primitive::<arrow_array::types::Int64Type>();
            let first = ids.value(0) as usize;
            let last = ids.value(ids.len() - 1) as usize;
            let window = windows
                .iter()
                .find(|window| (window.first..window.end).contains(&first))
                .unwrap_or_else(|| panic!("batch {first}..={last} starts outside every window"));
            assert!(
                last < window.end,
                "batch {first}..={last} crosses the window ending at {}",
                window.end
            );
            assert!(batch.num_rows() <= grid);
            window
        })
        .collect()
}

#[test]
fn test_raw_window_mask_real_codecs_bounded_grid_and_false_gaps() {
    let source = flat_source(48);
    let mask = (0..48)
        .map(|id| (id % 3 == 0 || id == 7 || id == 8) && !(16..32).contains(&id))
        .collect::<Vec<_>>();
    let expected = filter_record_batch(&source, &BooleanArray::from(mask.clone())).unwrap();
    for codec in CODECS {
        let batches = ParquetRecordBatchReaderBuilder::try_new(encode(&source, codec, 8, 48))
            .unwrap()
            .with_batch_size(GRID)
            .with_row_selection_policy(RowSelectionPolicy::Mask)
            .with_row_selection(selection(&mask))
            .build()
            .unwrap()
            .map(Result::unwrap)
            .collect::<Vec<_>>();
        assert_rows(&batches, &expected);
        // The first window spans grid 0 and the second grids 1 and 2, with
        // 4 rows each; the third may span four grids and covers 33..48 after
        // the unselected gap.
        assert_eq!(row_counts(&batches), [4, 4, 5]);
        assert_eq!(
            row_counts(&batches),
            expected_merged_rows(selected_ids(&mask), GRID)
        );
    }
}

#[test]
fn test_mask_merge_small_windows_around_large_windows() {
    let source = flat_source(712);
    // Windows of 1, 2, 4, ... grids starting at rows 0, 8, 24, 56, 120, 248
    // and 504 select 3, 2, 6, 2, 3, 3 and 3 rows.
    let selected = [
        0, 2, 4, 8, 12, 24, 28, 32, 36, 40, 44, 56, 88, 120, 160, 200, 248, 300, 400, 504, 600, 700,
    ];
    let mask = (0..712)
        .map(|id| selected.contains(&id))
        .collect::<Vec<_>>();
    let expected = filter_record_batch(&source, &BooleanArray::from(mask.clone())).unwrap();
    assert_eq!(expected_windows(selected, GRID), [3, 2, 6, 2, 3, 3, 3]);
    for codec in [Encoding::PLAIN, Encoding::DELTA_BYTE_ARRAY] {
        let batches = ParquetRecordBatchReaderBuilder::try_new(encode(&source, codec, 8, 712))
            .unwrap()
            .with_batch_size(GRID)
            .with_row_selection_policy(RowSelectionPolicy::Mask)
            .with_row_selection(selection(&mask))
            .build()
            .unwrap()
            .map(Result::unwrap)
            .collect::<Vec<_>>();
        assert_rows(&batches, &expected);
        // The first two windows merge, the 6-row window is returned on its
        // own after them, the next three fill one batch, and the last one
        // ends the stream.
        assert_eq!(row_counts(&batches), [5, 6, 8, 3]);
        assert_eq!(row_counts(&batches), expected_merged_rows(selected, GRID));
    }
}

#[test]
fn test_mask_merge_returns_windows_of_exactly_half_a_batch_unchanged() {
    let source = flat_source(112);
    // Windows of 1, 2, 4 and 8 grids starting at rows 0, 8, 24 and 56.
    let selected = [0, 2, 4, 6, 8, 12, 16, 24, 32, 40, 56, 72, 88, 104];
    let mask = (0..112)
        .map(|id| selected.contains(&id))
        .collect::<Vec<_>>();
    assert_eq!(expected_windows(selected, GRID), [4, 3, 3, 4]);
    let batches =
        ParquetRecordBatchReaderBuilder::try_new(encode(&source, Encoding::PLAIN, 8, 112))
            .unwrap()
            .with_batch_size(GRID)
            .with_row_selection_policy(RowSelectionPolicy::Mask)
            .with_row_selection(selection(&mask))
            .build()
            .unwrap()
            .map(Result::unwrap)
            .collect::<Vec<_>>();
    let expected = filter_record_batch(&source, &BooleanArray::from(mask)).unwrap();
    assert_rows(&batches, &expected);
    // The two 3-row windows are merged, and the windows of exactly half a
    // batch are returned unchanged.
    assert_eq!(row_counts(&batches), [4, 6, 4]);
    assert_eq!(row_counts(&batches), expected_merged_rows(selected, GRID));
}

#[test]
fn test_raw_window_all_selected_keeps_full_batch_shape() {
    let source = flat_source(35);
    let bytes = encode(&source, Encoding::PLAIN, GRID, 35);
    let selected = ParquetRecordBatchReaderBuilder::try_new(bytes.clone())
        .unwrap()
        .with_batch_size(GRID)
        .with_row_selection_policy(RowSelectionPolicy::Mask)
        .with_row_selection(selection(&[true; 35]))
        .build()
        .unwrap()
        .map(Result::unwrap)
        .collect::<Vec<_>>();
    let ordinary = ParquetRecordBatchReaderBuilder::try_new(bytes)
        .unwrap()
        .with_batch_size(GRID)
        .build()
        .unwrap()
        .map(Result::unwrap)
        .collect::<Vec<_>>();
    assert_eq!(selected, ordinary);
    assert_eq!(
        selected
            .iter()
            .map(RecordBatch::num_rows)
            .collect::<Vec<_>>(),
        [8, 8, 8, 8, 3]
    );
}

fn nested_source(rows: usize) -> RecordBatch {
    let mut inner = ListBuilder::new(StringBuilder::new());
    let mut names = StringBuilder::new();
    let mut struct_valid = vec![];
    let mut outer_offsets = vec![0i32];
    let mut outer_valid = vec![];
    let mut maps = MapBuilder::new(None, StringBuilder::new(), StringBuilder::new());
    for row in 0..rows {
        let count = if row % 7 < 2 { 0 } else { 1 + row % 3 };
        outer_valid.push(row % 7 != 0);
        for item in 0..count {
            let valid = (row + item) % 5 != 0;
            struct_valid.push(valid);
            if !valid {
                inner.append(false);
                names.append_null();
                continue;
            }
            match (row + item) % 4 {
                0 => inner.append(false),
                1 => inner.append(true),
                _ => {
                    inner.values().append_value(format!("a-{row:02}-{item}"));
                    inner.values().append_null();
                    inner.append(true);
                }
            }
            names.append_value(format!("b-{row:02}-{item}"));
        }
        outer_offsets.push(*outer_offsets.last().unwrap() + count as i32);
        if row % 6 == 0 {
            maps.append(false).unwrap();
        } else {
            for entry in 0..row % 3 {
                maps.keys().append_value(format!("k{entry}"));
                if (row + entry) % 4 == 0 {
                    maps.values().append_null();
                } else {
                    maps.values().append_value(format!("v{row:02}"));
                }
            }
            maps.append(true).unwrap();
        }
    }
    let inner: ArrayRef = Arc::new(inner.finish());
    let names: ArrayRef = Arc::new(names.finish());
    let children = StructArray::new(
        vec![
            Arc::new(Field::new("a", inner.data_type().clone(), true)),
            Arc::new(Field::new("b", DataType::Utf8, true)),
        ]
        .into(),
        vec![inner, names],
        Some(NullBuffer::from(struct_valid)),
    );
    let nested: ArrayRef = Arc::new(ListArray::new(
        Arc::new(Field::new("item", children.data_type().clone(), true)),
        OffsetBuffer::new(outer_offsets.into()),
        Arc::new(children),
        Some(NullBuffer::from(outer_valid)),
    ));
    let map: ArrayRef = Arc::new(maps.finish());
    let fields = vec![
        Field::new("id", DataType::Int64, false),
        Field::new("text", DataType::Utf8, false),
        Field::new("nested", nested.data_type().clone(), true).with_metadata(
            std::collections::HashMap::from([("owner".to_owned(), "raw-window".to_owned())]),
        ),
        Field::new("map", map.data_type().clone(), true),
    ];
    RecordBatch::try_new(
        Arc::new(Schema::new(fields)),
        vec![
            Arc::new(Int64Array::from_iter_values(0..rows as i64)),
            Arc::new(StringArray::from_iter_values(
                (0..rows).map(|row| format!("text-r{row:02}")),
            )),
            nested,
            map,
        ],
    )
    .unwrap()
}

#[test]
fn test_raw_window_deep_list_struct_map_nulls_and_byte_split() {
    let source = nested_source(64);
    // Dense grids are returned unchanged; the sparse mask merges windows
    // across the row group boundary at row 32.
    let dense = (0..64)
        .map(|row| row % 3 != 1 && !(16..24).contains(&row))
        .collect::<Vec<_>>();
    let sparse = (0..64)
        .map(|row| (row % 4 == 1 || row == 30) && !(16..24).contains(&row))
        .collect::<Vec<_>>();
    for mask in [dense, sparse] {
        let expected = filter_record_batch(&source, &BooleanArray::from(mask.clone())).unwrap();
        for codec in CODECS {
            let data = encode(&source, codec, 2, 32);
            for cap in [i32::MAX as usize, 20] {
                for policy in [RowSelectionPolicy::Mask, RowSelectionPolicy::Selectors] {
                    let batches = ParquetRecordBatchReaderBuilder::try_new(data.clone())
                        .unwrap()
                        .with_batch_size(GRID)
                        .with_byte_array_capacity(Some(cap))
                        .with_row_selection_policy(policy)
                        .with_row_selection(selection(&mask))
                        .build()
                        .unwrap()
                        .map(Result::unwrap)
                        .collect::<Vec<_>>();
                    assert_rows(&batches, &expected);
                    if policy == RowSelectionPolicy::Mask && cap == i32::MAX as usize {
                        assert_eq!(
                            row_counts(&batches),
                            expected_merged_rows(selected_ids(&mask), GRID)
                        );
                    }
                    for batch in &batches {
                        assert!(batch.num_rows() <= GRID);
                        // Each physical batch must independently validate all nested offsets,
                        // nulls and dictionary keys, rather than only compare visible IDs.
                        for array in batch.columns() {
                            array.to_data().validate_full().unwrap();
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn test_raw_window_dictionary_view_narrow_keys_never_coalesced() {
    let rows = 128;
    let values: ArrayRef = Arc::new(StringViewArray::from_iter_values(
        (0..100).map(|value| format!("dictionary-value-{value:03}")),
    ));
    let dictionary: ArrayRef = Arc::new(DictionaryArray::<Int8Type>::new(
        Int8Array::from_iter_values((0..rows).map(|row| (row % 100) as i8)),
        values,
    ));
    let source = RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("text", DataType::Utf8, false),
            Field::new("dict", dictionary.data_type().clone(), false),
        ])),
        vec![
            Arc::new(Int64Array::from_iter_values(0..rows as i64)),
            Arc::new(StringArray::from_iter_values(
                (0..rows).map(|row| format!("text-{row:03}")),
            )),
            dictionary,
        ],
    )
    .unwrap();
    let data = encode(&source, Encoding::RLE_DICTIONARY, 32, rows);
    // Every grid of the sparse mask is a small window, which would be merged
    // without the dictionary.
    for stride in [2, 4] {
        let mask = (0..rows).map(|row| row % stride == 0).collect::<Vec<_>>();
        let expected = filter_record_batch(&source, &BooleanArray::from(mask.clone())).unwrap();
        for policy in [RowSelectionPolicy::Mask, RowSelectionPolicy::Selectors] {
            let batches = ParquetRecordBatchReaderBuilder::try_new(data.clone())
                .unwrap()
                .with_batch_size(GRID)
                .with_row_selection_policy(policy)
                .with_row_selection(selection(&mask))
                .build()
                .unwrap()
                .map(Result::unwrap)
                .collect::<Vec<_>>();
            assert_rows(&batches, &expected);
            if policy == RowSelectionPolicy::Mask {
                batch_windows(&batches, &model_windows(selected_ids(&mask), GRID), GRID);
                assert_eq!(
                    row_counts(&batches),
                    expected_windows(selected_ids(&mask), GRID)
                );
            }
            for batch in &batches {
                let dictionary = batch.column(2).as_dictionary::<Int8Type>();
                assert_eq!(dictionary.values().data_type(), &DataType::Utf8View);
                assert!(dictionary.values().len() <= 128);
                dictionary.to_data().validate_full().unwrap();
            }
        }
    }
}

#[test]
fn test_raw_window_list_view_subtrees_stay_independently_representable() {
    let rows = 64;
    let mut lists = ListBuilder::new(StringBuilder::new());
    let mut large = LargeListBuilder::new(StringBuilder::new());
    for row in 0..rows {
        if row % 5 == 0 {
            lists.append(false);
            large.append(false);
        } else {
            let count = if row % 5 == 1 { 0 } else { 1 + row % 3 };
            for item in 0..count {
                if item == 1 && row % 2 == 0 {
                    lists.values().append_null();
                    large.values().append_null();
                } else {
                    let value = format!("unstaged-list-view-child-{row:02}-{item}");
                    lists.values().append_value(&value);
                    large.values().append_value(value);
                }
            }
            lists.append(true);
            large.append(true);
        }
    }
    let lists: ArrayRef = Arc::new(ListViewArray::from(lists.finish()));
    let large: ArrayRef = Arc::new(LargeListViewArray::from(large.finish()));
    let source = RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("text", DataType::Utf8, false),
            Field::new("view", lists.data_type().clone(), true),
            Field::new("large_view", large.data_type().clone(), true),
        ])),
        vec![
            Arc::new(Int64Array::from_iter_values(0..rows as i64)),
            Arc::new(StringArray::from_iter_values(
                (0..rows).map(|row| format!("row-{row:04}")),
            )),
            lists,
            large,
        ],
    )
    .unwrap();
    let dense = (0..rows)
        .map(|row| (row % 2 == 0 || row == 1 || row == 41) && !(16..40).contains(&row))
        .collect::<Vec<_>>();
    // Small windows, which would be merged without the list views.
    let sparse = (0..rows).map(|row| row % 4 == 1).collect::<Vec<_>>();
    let data = encode(&source, Encoding::PLAIN, 4, 32);
    for mask in [dense, sparse] {
        let expected = filter_record_batch(&source, &BooleanArray::from(mask.clone())).unwrap();
        // A small capacity also splits batches on the `text` column.
        for cap in [i32::MAX as usize, 20] {
            for policy in [RowSelectionPolicy::Mask, RowSelectionPolicy::Selectors] {
                let batches = ParquetRecordBatchReaderBuilder::try_new(data.clone())
                    .unwrap()
                    .with_batch_size(GRID)
                    .with_byte_array_capacity(Some(cap))
                    .with_row_selection_policy(policy)
                    .with_row_selection(selection(&mask))
                    .build()
                    .unwrap()
                    .map(Result::unwrap)
                    .collect::<Vec<_>>();
                assert_rows(&batches, &expected);
                if policy == RowSelectionPolicy::Mask {
                    let windows = model_windows(selected_ids(&mask), GRID);
                    let containing = batch_windows(&batches, &windows, GRID);
                    if cap == i32::MAX as usize {
                        assert_eq!(
                            row_counts(&batches),
                            expected_windows(selected_ids(&mask), GRID)
                        );
                    }
                    // Filtering a list view keeps the child values of every raw
                    // row its window decoded, at most 3 per row.
                    for (batch, window) in batches.iter().zip(containing) {
                        let bound = 3 * (window.end - window.first);
                        assert!(batch.column(2).as_list_view::<i32>().values().len() <= bound);
                        assert!(batch.column(3).as_list_view::<i64>().values().len() <= bound);
                    }
                }
            }
        }
    }
}

#[test]
fn test_raw_window_offset_limit_crosses_false_grids_and_row_groups() {
    let source = flat_source(64);
    let mask = (0..64)
        .map(|row| row % 5 == 0 && !(17..33).contains(&row))
        .collect::<Vec<_>>();
    let selected = filter_record_batch(&source, &BooleanArray::from(mask.clone())).unwrap();
    let expected = selected.slice(3, 5);
    let data = encode(&source, Encoding::DELTA_LENGTH_BYTE_ARRAY, 4, 16);
    for policy in [RowSelectionPolicy::Mask, RowSelectionPolicy::Selectors] {
        let batches = ParquetRecordBatchReaderBuilder::try_new(data.clone())
            .unwrap()
            .with_batch_size(GRID)
            .with_row_selection_policy(policy)
            .with_row_selection(selection(&mask))
            .with_offset(3)
            .with_limit(5)
            .build()
            .unwrap()
            .map(Result::unwrap)
            .collect::<Vec<_>>();
        assert_rows(&batches, &expected);
        if policy == RowSelectionPolicy::Mask {
            // Rows 15, 35, 40, 45 and 50 come from four small windows.
            assert_eq!(row_counts(&batches), [5]);
        }
    }
}

#[cfg(feature = "async")]
#[tokio::test]
async fn test_raw_window_async_cache_and_stateful_predicates_visit_each_row_once() {
    use super::{ArrowPredicateFn, RowFilter, metrics::ArrowReaderMetrics};
    use crate::arrow::{ParquetRecordBatchStreamBuilder, ProjectionMask};
    use futures::TryStreamExt;
    use std::io::Cursor;
    use std::sync::Mutex;

    let source = flat_source(64);
    let data = encode(&source, Encoding::DELTA_BYTE_ARRAY, 4, 32);
    let initial = (0..64)
        .map(|row| row % 2 == 0 || row % 7 == 0)
        .collect::<Vec<_>>();
    let candidates = (0..64).filter(|&row| initial[row]).collect::<Vec<_>>();
    let after_first = candidates
        .iter()
        .copied()
        .enumerate()
        .filter_map(|(index, row)| {
            (row % 3 != 0 && (index + 1) % 5 != 0 && row % 11 != 0).then_some(row)
        })
        .collect::<Vec<_>>();
    let accepted = after_first
        .iter()
        .copied()
        .enumerate()
        .filter_map(|(index, row)| (row % 4 != 0 && (index + 1) % 3 != 0).then_some(row))
        .collect::<Vec<_>>();
    let expected_mask = (0..64)
        .map(|row| accepted.contains(&row))
        .collect::<Vec<_>>();
    let expected = filter_record_batch(&source, &BooleanArray::from(expected_mask)).unwrap();

    for policy in [RowSelectionPolicy::Mask, RowSelectionPolicy::Selectors] {
        for cache_size in [0, usize::MAX] {
            for capacity in [i32::MAX as usize, 20] {
                let builder = ParquetRecordBatchStreamBuilder::new(Cursor::new(data.clone()))
                    .await
                    .unwrap();
                let projection = ProjectionMask::leaves(builder.parquet_schema(), [0, 1]);
                let first_seen = Arc::new(Mutex::new(vec![]));
                let first_trace = first_seen.clone();
                let mut first_position = 0usize;
                let first = ArrowPredicateFn::new(projection.clone(), move |batch: RecordBatch| {
                    let ids = batch
                        .column(0)
                        .as_primitive::<arrow_array::types::Int64Type>();
                    let texts = batch.column(1).as_string::<i32>();
                    let mut trace = first_trace.lock().unwrap();
                    Ok(BooleanArray::from_iter((0..ids.len()).map(|index| {
                        let id = ids.value(index) as usize;
                        assert_eq!(texts.value(index), format!("row-{id:04}"));
                        trace.push(id);
                        first_position += 1;
                        (!id.is_multiple_of(11))
                            .then_some(!id.is_multiple_of(3) && !first_position.is_multiple_of(5))
                    })))
                });
                let second_seen = Arc::new(Mutex::new(vec![]));
                let second_trace = second_seen.clone();
                let mut second_position = 0usize;
                let second = ArrowPredicateFn::new(projection, move |batch: RecordBatch| {
                    let ids = batch
                        .column(0)
                        .as_primitive::<arrow_array::types::Int64Type>();
                    batch.column(1).as_string::<i32>(); // requested Utf8 at both predicate seams
                    let mut trace = second_trace.lock().unwrap();
                    Ok(BooleanArray::from_iter((0..ids.len()).map(|index| {
                        let id = ids.value(index) as usize;
                        trace.push(id);
                        second_position += 1;
                        !id.is_multiple_of(4) && !second_position.is_multiple_of(3)
                    })))
                });
                let metrics = ArrowReaderMetrics::enabled();
                let batches = builder
                    .with_batch_size(GRID)
                    .with_row_selection_policy(policy)
                    .with_row_selection(selection(&initial))
                    .with_row_filter(RowFilter::new(vec![Box::new(first), Box::new(second)]))
                    .with_max_predicate_cache_size(cache_size)
                    .with_metrics(metrics.clone())
                    .with_byte_array_capacity(Some(capacity))
                    .build()
                    .unwrap()
                    .try_collect::<Vec<RecordBatch>>()
                    .await
                    .unwrap();
                assert_eq!(*first_seen.lock().unwrap(), candidates);
                assert_eq!(*second_seen.lock().unwrap(), after_first);
                assert_rows(&batches, &expected);
                if policy == RowSelectionPolicy::Mask && capacity == i32::MAX as usize {
                    assert_merged_batches(&batches, GRID);
                }
                let hits = metrics.records_read_from_cache().unwrap();
                if cache_size == 0 {
                    assert_eq!(hits, 0);
                } else {
                    assert!(hits > 0, "cache-on must execute actual cache reads");
                }
            }
        }
    }
}

#[cfg(all(feature = "async", feature = "snap"))]
#[tokio::test]
async fn test_raw_window_async_merged_nested_views_release_rejected_page_values() {
    use super::{ArrowReaderOptions, RowSelector};
    use crate::arrow::ParquetRecordBatchStreamBuilder;
    use crate::basic::Compression;
    use crate::file::metadata::PageIndexPolicy;
    use futures::TryStreamExt;
    use std::io::Cursor;

    const BATCH: usize = 16;
    const PAGE: usize = 16;
    const ROWS: usize = BATCH * PAGE * 2;
    const VALUE_BYTES: usize = 128;
    let values: ArrayRef = Arc::new(StringArray::from_iter_values(
        (0..ROWS).map(|row| format!("{row:0128}")),
    ));
    let nested = StructArray::from(vec![(
        Arc::new(Field::new("value", DataType::Utf8, false)),
        values,
    )]);
    let source = RecordBatch::try_from_iter([("nested", Arc::new(nested) as ArrayRef)]).unwrap();
    let props = WriterProperties::builder()
        .set_dictionary_enabled(false)
        .set_compression(Compression::SNAPPY)
        .set_write_batch_size(PAGE)
        .set_data_page_row_count_limit(PAGE)
        .set_max_row_group_row_count(Some(ROWS))
        .build();
    let mut bytes = Vec::new();
    let mut writer = ArrowWriter::try_new(&mut bytes, source.schema(), Some(props)).unwrap();
    writer.write(&source).unwrap();
    writer.close().unwrap();

    let schema = Arc::new(Schema::new(vec![Field::new(
        "nested",
        DataType::Struct(vec![Field::new("value", DataType::Utf8View, false)].into()),
        false,
    )]));
    let builder = ParquetRecordBatchStreamBuilder::new_with_options(
        Cursor::new(Bytes::from(bytes)),
        ArrowReaderOptions::new()
            .with_schema(schema.clone())
            .with_page_index_policy(PageIndexPolicy::Required),
    )
    .await
    .unwrap();
    assert_eq!(
        builder
            .metadata()
            .page_index()
            .unwrap()
            .page_locations(0, 0)
            .unwrap()
            .len(),
        ROWS / PAGE
    );
    // Loading alternating pages creates separate one-row Mask windows even
    // after their grid count grows. Snappy gives each page its own backing
    // allocation, and only one value from each page should remain buffered.
    let selection = RowSelection::from(
        (0..BATCH)
            .flat_map(|_| [RowSelector::select(1), RowSelector::skip(2 * PAGE - 1)])
            .collect::<Vec<_>>(),
    );
    let batches: Vec<RecordBatch> = builder
        .with_batch_size(BATCH)
        .with_row_selection_policy(RowSelectionPolicy::Mask)
        .with_row_selection(selection)
        .build()
        .unwrap()
        .try_collect()
        .await
        .unwrap();
    assert_eq!(row_counts(&batches), [BATCH]);
    assert_eq!(batches[0].schema(), schema);
    let views = batches[0].column(0).as_struct().column(0).as_string_view();
    for row in 0..BATCH {
        assert_eq!(views.value(row), format!("{:0128}", row * 2 * PAGE));
    }
    let retained_bytes: usize = views.data_buffers().iter().map(|b| b.capacity()).sum();
    assert!(
        retained_bytes <= 4 * BATCH * VALUE_BYTES,
        "merged nested views retained {retained_bytes} bytes for {} selected bytes",
        BATCH * VALUE_BYTES
    );
}

#[test]
fn test_mask_window_grids_grow_twofold_and_shrink_to_the_budget() {
    use super::next_mask_window_grids;
    const BATCH: usize = 8192;
    // At 10 bytes per row the budget allows hundreds of grids: grow twofold.
    assert_eq!(next_mask_window_grids(1, 10 * BATCH, BATCH, BATCH), 2);
    assert_eq!(next_mask_window_grids(64, 10 * BATCH, BATCH, BATCH), 128);
    // At 1 KiB per row a grid is 8 MiB, so the budget allows four.
    assert_eq!(next_mask_window_grids(64, 1024 * BATCH, BATCH, BATCH), 4);
    // Rows wider than the budget per grid shrink to one grid at once.
    assert_eq!(next_mask_window_grids(128, 6000 * BATCH, BATCH, BATCH), 1);
    // A window shorter than its grids is scaled by its raw rows.
    assert_eq!(next_mask_window_grids(2, 1024 * 100, 100, BATCH), 4);
    // An empty estimate does not divide by zero.
    assert_eq!(next_mask_window_grids(3, 0, 0, BATCH), 6);
}

#[test]
fn test_window_bytes_counts_view_bytes_referenced_not_buffer_capacity() {
    use super::window_bytes;
    let views = StringViewArray::from_iter_values((0..10_000).map(|row| format!("{row:0100}")));
    let views = views.slice(5000, 2);
    // The slice shares every data buffer of the 10,000 values.
    assert!(views.get_array_memory_size() > 1_000_000);
    assert_eq!(window_bytes(&views), 2 * 16 + 2 * 100);
    let nested = StructArray::from(vec![(
        Arc::new(Field::new("views", DataType::Utf8View, false)),
        Arc::new(views) as ArrayRef,
    )]);
    assert_eq!(window_bytes(&nested), 2 * 16 + 2 * 100);
    // Other layouts count the slice of their buffers.
    let strings = StringArray::from_iter_values((0..10_000).map(|row| format!("{row:0100}")));
    assert_eq!(window_bytes(&strings.slice(5000, 2)), 3 * 4 + 2 * 100);
    let ints = Int64Array::from_iter_values(0..10_000);
    assert_eq!(window_bytes(&ints.slice(5000, 2)), 2 * 8);
}
