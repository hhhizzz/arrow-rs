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

use super::*;
use arrow_array::types::Int32Type;
use arrow_array::{
    BinaryArray, DictionaryArray, Int32Array, LargeBinaryArray, LargeListArray, LargeStringArray,
    ListArray, StringArray, StringViewArray,
};
use arrow_buffer::NullBuffer;
use arrow_schema::{Field, Fields, Schema};
use rand::RngExt;
use std::collections::HashMap;

fn field(name: &str, data_type: DataType) -> FieldRef {
    Arc::new(Field::new(name, data_type, true))
}

fn batch(fields: Vec<FieldRef>, columns: Vec<ArrayRef>) -> RecordBatch {
    RecordBatch::try_new(Arc::new(Schema::new(fields)), columns).unwrap()
}

fn drain(splitter: &mut CapacitySplitter) -> Vec<RecordBatch> {
    let mut out = vec![];
    while let Some(batch) = splitter.next_batch().unwrap() {
        out.push(batch);
    }
    out
}

fn large_strings(values: Vec<Option<&str>>) -> ArrayRef {
    Arc::new(LargeStringArray::from(values))
}

#[test]
fn test_capacity_splitter_flat_split_and_fast_path() {
    let strings = large_strings(vec![Some("aa"), None, Some("bbb"), Some("c")]);
    let binary: ArrayRef = Arc::new(LargeBinaryArray::from_iter_values([
        b"111".as_slice(),
        b"22",
        b"3",
        b"4",
    ]));
    let source = batch(
        vec![
            field("b", DataType::LargeBinary),
            field("s", DataType::LargeUtf8),
        ],
        vec![binary, strings],
    );
    let target = Arc::new(Schema::new(vec![
        field("b", DataType::Binary),
        field("s", DataType::Utf8),
    ]));
    let mut splitter = CapacitySplitter::new(target.clone(), 5);
    splitter.push(source.clone()).unwrap();
    let output = drain(&mut splitter);
    assert_eq!(
        output.iter().map(RecordBatch::num_rows).collect::<Vec<_>>(),
        vec![2, 2]
    );
    assert_eq!(
        output[0].column(0).as_binary::<i32>(),
        &BinaryArray::from_vec(vec![b"111", b"22"])
    );
    assert_eq!(
        output[0].column(1).as_string::<i32>(),
        &StringArray::from(vec![Some("aa"), None])
    );
    assert_eq!(
        output[1].column(1).as_string::<i32>(),
        &StringArray::from(vec![Some("bbb"), Some("c")])
    );
    assert_eq!(output[0].schema(), target);

    let mut fits = CapacitySplitter::new(target, 100);
    fits.push(source.clone()).unwrap();
    assert_eq!(drain(&mut fits)[0].num_rows(), 4);

    let wrong_projection = Arc::new(Schema::new(vec![field("s", DataType::Utf8)]));
    assert!(
        CapacitySplitter::new(wrong_projection, 5)
            .push(source)
            .is_err()
    );
}

#[test]
fn test_capacity_splitter_shares_fitting_window_and_compacts_split() {
    let source = large_strings(vec![Some("aa"), None, Some("bbb"), Some("cccc")]);
    let source_values = source.as_string::<i64>().values().as_ptr() as usize;
    let staged = batch(vec![field("s", DataType::LargeUtf8)], vec![source]);
    let target = Arc::new(Schema::new(vec![field("s", DataType::Utf8)]));

    // A window that fits is restored as one batch sharing the payload.
    let mut splitter = CapacitySplitter::new(target.clone(), 1024);
    splitter.push(staged.clone()).unwrap();
    let whole = drain(&mut splitter);
    assert_eq!(whole.len(), 1);
    let whole = whole[0].column(0).as_string::<i32>();
    assert_eq!(whole.value_offsets(), &[0, 2, 2, 5, 9]);
    assert_eq!(whole.values().as_ptr() as usize, source_values);

    // A split window is compacted: offsets rebased and payload copied.
    let mut splitter = CapacitySplitter::new(target, 5);
    splitter.push(staged).unwrap();
    let parts = drain(&mut splitter);
    assert_eq!(
        parts[0].column(0).as_string::<i32>(),
        &StringArray::from(vec![Some("aa"), None, Some("bbb")])
    );
    let second = parts[1].column(0).as_string::<i32>();
    assert_eq!(second.value_offsets(), &[0, 4]);
    assert_eq!(second.values().len(), 4);
    assert_ne!(second.values().as_ptr() as usize, source_values + 5);
}

#[test]
fn test_capacity_splitter_restores_sliced_input() {
    let source: ArrayRef = Arc::new(LargeBinaryArray::from_iter_values([
        b"0123".as_slice(),
        b"45",
        b"6789",
    ]));
    let source = batch(
        vec![field("b", DataType::LargeBinary)],
        vec![source.slice(1, 2)],
    );
    let target = Arc::new(Schema::new(vec![field("b", DataType::Binary)]));
    let mut splitter = CapacitySplitter::new(target, 6);
    splitter.push(source).unwrap();
    let output = drain(&mut splitter);
    assert_eq!(output.len(), 1);
    let restored = output[0].column(0).as_binary::<i32>();
    assert_eq!(restored.value_offsets(), &[0, 2, 6]);
    assert_eq!(restored.value(0), b"45");
    assert_eq!(restored.value(1), b"6789");
}

#[test]
fn test_capacity_splitter_list_trims_sibling_child() {
    let values = large_strings(vec![
        Some("a"),
        Some("abcdefghijklmnopqrstuvwxyz"),
        Some("bb"),
    ]);
    let source = ListArray::new(
        field("item", DataType::LargeUtf8),
        OffsetBuffer::new(vec![0i32, 1, 2, 3].into()),
        values,
        None,
    );
    // `slice` keeps the whole child; only the first row's child is needed.
    let tiny = source.slice(0, 1);
    let source = batch(
        vec![field("l", tiny.data_type().clone())],
        vec![Arc::new(tiny)],
    );
    let target = Arc::new(Schema::new(vec![field(
        "l",
        DataType::List(field("item", DataType::Utf8)),
    )]));
    let mut splitter = CapacitySplitter::new(target, 2);
    splitter.push(source).unwrap();
    let output = drain(&mut splitter);
    let list = output[0].column(0).as_list::<i32>();
    assert_eq!(list.value_offsets(), &[0, 1]);
    assert_eq!(list.values().len(), 1);
    assert_eq!(list.values().as_string::<i32>().value_data(), b"a");
}

#[test]
fn test_capacity_splitter_nested_list_records_nulls_and_metadata() {
    let metadata = HashMap::from([("nested".to_owned(), "preserved".to_owned())]);
    let source_item =
        Arc::new(Field::new("item", DataType::LargeUtf8, true).with_metadata(metadata.clone()));
    let target_item =
        Arc::new(Field::new("item", DataType::Utf8, true).with_metadata(metadata.clone()));
    let values = large_strings(vec![
        Some("aa"),
        None,
        Some("bbb"),
        Some("cccc"),
        Some("dd"),
    ]);
    let list = ListArray::new(
        source_item,
        OffsetBuffer::new(vec![0i32, 3, 3, 3, 5].into()),
        values,
        Some(NullBuffer::from(vec![true, true, false, true])),
    );
    let source_fields = Fields::from(vec![
        field("l", list.data_type().clone()),
        field("id", DataType::Int32),
    ]);
    let structure = StructArray::new(
        source_fields,
        vec![Arc::new(list), Arc::new(Int32Array::from(vec![1, 2, 3, 4]))],
        Some(NullBuffer::from(vec![true, true, false, true])),
    );
    let source = batch(
        vec![field("s", structure.data_type().clone())],
        vec![Arc::new(structure)],
    );
    let target_struct = DataType::Struct(Fields::from(vec![
        field("l", DataType::List(target_item)),
        field("id", DataType::Int32),
    ]));
    let target_field =
        Arc::new(Field::new("s", target_struct, true).with_metadata(metadata.clone()));
    let target = Arc::new(Schema::new_with_metadata(vec![target_field], metadata));
    let mut splitter = CapacitySplitter::new(target.clone(), 6);
    splitter.push(source).unwrap();
    let output = drain(&mut splitter);
    assert_eq!(
        output.iter().map(RecordBatch::num_rows).collect::<Vec<_>>(),
        vec![3, 1]
    );
    assert_eq!(output[0].schema(), target);
    let first = output[0].column(0).as_struct();
    assert!(first.is_null(2));
    let list = first.column(0).as_list::<i32>();
    assert_eq!(list.value_offsets(), &[0, 3, 3, 3]);
    assert!(list.is_null(2));
    assert!(list.values().is_null(1));
    assert_eq!(
        list.value(0).as_string::<i32>().iter().collect::<Vec<_>>(),
        vec![Some("aa"), None, Some("bbb")]
    );
    assert_eq!(
        output[1]
            .column(0)
            .as_struct()
            .column(0)
            .as_list::<i32>()
            .value_offsets(),
        &[0, 2]
    );
}

#[test]
fn test_capacity_splitter_oversized_row_errors_and_releases_pending() {
    let list = ListArray::new(
        field("item", DataType::LargeUtf8),
        OffsetBuffer::new(vec![0i32, 2].into()),
        large_strings(vec![Some("aaaa"), Some("bbbb")]),
        None,
    );
    let source = batch(
        vec![field("l", list.data_type().clone())],
        vec![Arc::new(list)],
    );
    let target = Arc::new(Schema::new(vec![field(
        "l",
        DataType::List(field("item", DataType::Utf8)),
    )]));
    let mut splitter = CapacitySplitter::new(target, 6);
    splitter.push(source).unwrap();
    let error = splitter.next_batch().unwrap_err().to_string();
    assert!(
        error.contains("a single row needs 8 bytes of Utf8 data for column l"),
        "{error}"
    );
    assert!(splitter.next_batch().unwrap().is_none());
}

#[test]
fn test_capacity_splitter_large_list_fixed_size_list_and_map() {
    let strings = large_strings(vec![Some("aaa"), Some("b"), Some("cc"), Some("dd")]);
    let large: ArrayRef = Arc::new(LargeListArray::new(
        field("item", DataType::LargeUtf8),
        OffsetBuffer::new(vec![0i64, 2, 4].into()),
        strings.clone(),
        None,
    ));
    let fixed: ArrayRef = Arc::new(FixedSizeListArray::new(
        field("item", DataType::LargeUtf8),
        2,
        strings,
        None,
    ));
    let entries_fields = Fields::from(vec![
        Arc::new(Field::new("key", DataType::LargeUtf8, false)),
        field("value", DataType::LargeUtf8),
    ]);
    let entries = StructArray::new(
        entries_fields,
        vec![
            large_strings(vec![Some("k"), Some("j"), Some("m")]),
            large_strings(vec![Some("aa"), Some("bbb"), Some("c")]),
        ],
        None,
    );
    let map: ArrayRef = Arc::new(MapArray::new(
        Arc::new(Field::new("entries", entries.data_type().clone(), false)),
        OffsetBuffer::new(vec![0i32, 2, 3].into()),
        entries,
        None,
        false,
    ));
    let source = batch(
        vec![
            field("large", large.data_type().clone()),
            field("fixed", fixed.data_type().clone()),
            field("map", map.data_type().clone()),
        ],
        vec![large, fixed, map],
    );
    let target_entries = Arc::new(Field::new(
        "entries",
        DataType::Struct(Fields::from(vec![
            Arc::new(Field::new("key", DataType::Utf8, false)),
            field("value", DataType::Utf8),
        ])),
        false,
    ));
    let target = Arc::new(Schema::new(vec![
        field("large", DataType::LargeList(field("item", DataType::Utf8))),
        field(
            "fixed",
            DataType::FixedSizeList(field("item", DataType::Utf8), 2),
        ),
        field("map", DataType::Map(target_entries, false)),
    ]));
    let mut splitter = CapacitySplitter::new(target, 5);
    splitter.push(source).unwrap();
    let output = drain(&mut splitter);
    assert_eq!(output.len(), 2);
    assert_eq!(
        output[0].column(0).as_list::<i64>().value_offsets(),
        &[0, 2]
    );
    assert_eq!(
        output[1]
            .column(0)
            .as_list::<i64>()
            .values()
            .as_string::<i32>()
            .value_data(),
        b"ccdd"
    );
    assert_eq!(
        output[1]
            .column(1)
            .as_fixed_size_list()
            .values()
            .as_string::<i32>()
            .value_data(),
        b"ccdd"
    );
    assert_eq!(output[1].column(2).as_map().value_offsets(), &[0, 1]);
    assert_eq!(
        output[1]
            .column(2)
            .as_map()
            .values()
            .as_string::<i32>()
            .value_data(),
        b"c"
    );
}

#[test]
fn test_capacity_splitter_leaves_unstaged_types_and_empty_projection_unchanged() {
    let dict: ArrayRef = Arc::new(DictionaryArray::<Int32Type>::new(
        Int32Array::from(vec![0, 0]),
        Arc::new(StringArray::from(vec!["long dictionary value"])),
    ));
    let columns: Vec<ArrayRef> = vec![
        Arc::new(StringViewArray::from(vec!["long", "values"])),
        // A requested LargeUtf8 column keeps its type and has no limit.
        large_strings(vec![Some("more long"), Some("values")]),
        dict,
    ];
    let fields = columns
        .iter()
        .enumerate()
        .map(|(i, a)| field(&format!("c{i}"), a.data_type().clone()))
        .collect::<Vec<_>>();
    let source = batch(fields, columns);
    let mut splitter = CapacitySplitter::new(source.schema(), 1);
    splitter.push(source.clone()).unwrap();
    assert_eq!(drain(&mut splitter), vec![source]);

    let schema = Arc::new(Schema::empty());
    let zero_columns = RecordBatch::try_new_with_options(
        schema.clone(),
        vec![],
        &RecordBatchOptions::new().with_row_count(Some(3)),
    )
    .unwrap();
    let mut splitter = CapacitySplitter::new(schema, 1);
    splitter.push(zero_columns).unwrap();
    assert_eq!(drain(&mut splitter)[0].num_rows(), 3);
}

#[test]
fn test_capacity_splitter_sliced_nonzero_payload_offsets() {
    let source: ArrayRef = Arc::new(LargeBinaryArray::from_iter_values([
        vec![0xff; 1024],
        b"ab".to_vec(),
        b"c".to_vec(),
        b"de".to_vec(),
    ]));
    let source = source.slice(1, 3);
    assert!(source.as_binary::<i64>().value_offsets()[0] > 3);
    let staged = batch(vec![field("b", DataType::LargeBinary)], vec![source]);
    let target = Arc::new(Schema::new(vec![field("b", DataType::Binary)]));
    let mut splitter = CapacitySplitter::new(target, 3);
    splitter.push(staged).unwrap();
    let output = drain(&mut splitter);
    assert_eq!(
        output.iter().map(RecordBatch::num_rows).collect::<Vec<_>>(),
        [2, 1]
    );
    assert_eq!(
        output[0].column(0).as_binary::<i32>().value_offsets(),
        &[0, 2, 3]
    );
    assert_eq!(
        output[1].column(0).as_binary::<i32>().value_offsets(),
        &[0, 2]
    );
    assert_eq!(output[0].column(0).as_binary::<i32>().value_data(), b"abc");
    assert_eq!(output[1].column(0).as_binary::<i32>().value_data(), b"de");
}

#[test]
fn test_capacity_splitter_zero_width_fixed_list_survives_companion_split() {
    let fixed: ArrayRef = Arc::new(
        FixedSizeListArray::try_new_with_length(
            field("item", DataType::LargeUtf8),
            0,
            large_strings(vec![]),
            Some(NullBuffer::from(vec![true, false, true, true, false])),
            5,
        )
        .unwrap(),
    );
    let source = batch(
        vec![
            field("zero", fixed.data_type().clone()),
            field("s", DataType::LargeUtf8),
        ],
        vec![
            fixed,
            large_strings(vec![
                Some("a"),
                Some("bb"),
                Some("ccc"),
                Some(""),
                Some("d"),
            ]),
        ],
    );
    let target = Arc::new(Schema::new(vec![
        field(
            "zero",
            DataType::FixedSizeList(field("item", DataType::Utf8), 0),
        ),
        field("s", DataType::Utf8),
    ]));
    let mut splitter = CapacitySplitter::new(target, 3);
    splitter.push(source.slice(1, 4)).unwrap();
    let output = drain(&mut splitter);
    assert_eq!(
        output.iter().map(RecordBatch::num_rows).collect::<Vec<_>>(),
        [1, 2, 1]
    );
    for output in &output {
        let fixed = output.column(0).as_fixed_size_list();
        assert_eq!(fixed.len(), output.num_rows());
        assert_eq!(fixed.value_length(), 0);
        assert!(fixed.values().is_empty());
        assert_eq!(fixed.value_type(), DataType::Utf8);
        fixed.to_data().validate_full().unwrap();
    }
    assert!(output[0].column(0).is_null(0));
    assert!(output[2].column(0).is_null(0));
}

#[test]
fn test_capacity_splitter_null_list_retained_child_span_matches_restore() {
    // A valid null List can retain nonempty child storage. The capacity
    // describes the arrays actually restored, including retained storage;
    // null parents must not make the planner underestimate that storage.
    let list = ListArray::new(
        field("item", DataType::LargeUtf8),
        OffsetBuffer::new(vec![0i32, 1, 2, 3].into()),
        large_strings(vec![
            Some("abcdefghijkl"),
            Some("ok"),
            Some("unused-sibling"),
        ]),
        Some(NullBuffer::from(vec![false, true, false])),
    );
    let staged = batch(
        vec![field("l", list.data_type().clone())],
        vec![Arc::new(list)],
    );
    let target = Arc::new(Schema::new(vec![field(
        "l",
        DataType::List(field("item", DataType::Utf8)),
    )]));
    let mut splitter = CapacitySplitter::new(target, 14);
    splitter.push(staged.slice(0, 2)).unwrap();
    let output = drain(&mut splitter);
    assert_eq!(output.len(), 1);
    let list = output[0].column(0).as_list::<i32>();
    assert!(list.is_null(0));
    assert_eq!(list.value_offsets(), &[0, 1, 2]);
    assert_eq!(
        list.values().as_string::<i32>().value_data(),
        b"abcdefghijklok"
    );
    assert_eq!(list.values().len(), 2, "unused sibling must be trimmed");
    list.to_data().validate_full().unwrap();
}

#[test]
fn test_capacity_splitter_later_error_releases_pending_without_repeating_prefix() {
    let source = Arc::new(LargeStringArray::from(vec!["a", "too-large", "b"]));
    let weak = Arc::downgrade(&source);
    let staged = batch(vec![field("s", DataType::LargeUtf8)], vec![source]);
    let target = Arc::new(Schema::new(vec![field("s", DataType::Utf8)]));
    let mut splitter = CapacitySplitter::new(target, 3);
    splitter.push(staged).unwrap();
    let first = splitter.next_batch().unwrap().unwrap();
    assert_eq!(
        first.column(0).as_string::<i32>(),
        &StringArray::from(vec!["a"])
    );
    assert!(
        weak.upgrade().is_some(),
        "unreturned suffix remains pending"
    );
    let error = splitter.next_batch().unwrap_err().to_string();
    assert!(error.contains("single row needs 9 bytes"), "{error}");
    assert!(
        weak.upgrade().is_none(),
        "error releases the original source array"
    );
    assert!(splitter.next_batch().unwrap().is_none());
}

fn generated_pair(
    rng: &mut impl rand::Rng,
    depth: usize,
    rows: usize,
    forced: Option<usize>,
) -> (ArrayRef, ArrayRef) {
    let kind = forced.unwrap_or_else(|| {
        if depth == 0 {
            rng.random_range(0..3)
        } else {
            rng.random_range(0..9)
        }
    });
    match kind {
        0 => {
            let values: Vec<_> = (0..rows)
                .map(|_| {
                    if rng.random_bool(0.2) {
                        None
                    } else {
                        Some("é".repeat(rng.random_range(0..5)))
                    }
                })
                .collect();
            (
                Arc::new(LargeStringArray::from_iter(
                    values.iter().map(|s| s.as_deref()),
                )),
                Arc::new(StringArray::from_iter(values.iter().map(|s| s.as_deref()))),
            )
        }
        1 => {
            let values: Vec<_> = (0..rows)
                .map(|_| {
                    if rng.random_bool(0.2) {
                        None
                    } else {
                        Some(vec![0xff; rng.random_range(0..7)])
                    }
                })
                .collect();
            (
                Arc::new(LargeBinaryArray::from_iter(
                    values.iter().map(|s| s.as_deref()),
                )),
                Arc::new(BinaryArray::from_iter(values.iter().map(|s| s.as_deref()))),
            )
        }
        2 => {
            let array: ArrayRef = Arc::new(Int32Array::from_iter_values(0..rows as i32));
            (array.clone(), array)
        }
        3 => {
            let (a, b) = generated_pair(rng, depth - 1, rows, None);
            let (c, d) = generated_pair(rng, depth - 1, rows, None);
            let nulls = Some(NullBuffer::from(
                (0..rows).map(|_| rng.random_bool(0.8)).collect::<Vec<_>>(),
            ));
            (
                Arc::new(StructArray::new(
                    vec![
                        field("a", a.data_type().clone()),
                        field("b", c.data_type().clone()),
                    ]
                    .into(),
                    vec![a, c],
                    nulls.clone(),
                )),
                Arc::new(StructArray::new(
                    vec![
                        field("a", b.data_type().clone()),
                        field("b", d.data_type().clone()),
                    ]
                    .into(),
                    vec![b, d],
                    nulls,
                )),
            )
        }
        4 | 5 | 7 => {
            let mut offsets = vec![0i32];
            for _ in 0..rows {
                offsets.push(offsets.last().unwrap() + rng.random_range(0..4));
            }
            let children = *offsets.last().unwrap() as usize;
            let (a, b) = generated_pair(rng, depth - 1, children, None);
            let nulls = Some(NullBuffer::from(
                (0..rows).map(|_| rng.random_bool(0.8)).collect::<Vec<_>>(),
            ));
            if kind == 4 {
                (
                    Arc::new(ListArray::new(
                        field("item", a.data_type().clone()),
                        OffsetBuffer::new(offsets.clone().into()),
                        a,
                        nulls.clone(),
                    )),
                    Arc::new(ListArray::new(
                        field("item", b.data_type().clone()),
                        OffsetBuffer::new(offsets.into()),
                        b,
                        nulls,
                    )),
                )
            } else if kind == 5 {
                let offsets: Vec<i64> = offsets.into_iter().map(i64::from).collect();
                (
                    Arc::new(LargeListArray::new(
                        field("item", a.data_type().clone()),
                        OffsetBuffer::new(offsets.clone().into()),
                        a,
                        nulls.clone(),
                    )),
                    Arc::new(LargeListArray::new(
                        field("item", b.data_type().clone()),
                        OffsetBuffer::new(offsets.into()),
                        b,
                        nulls,
                    )),
                )
            } else {
                let keys: Vec<_> = (0..children).map(|i| format!("k{i}")).collect();
                let make_entries = |keys: ArrayRef, values: ArrayRef| {
                    StructArray::new(
                        vec![
                            Arc::new(Field::new("key", keys.data_type().clone(), false)),
                            field("value", values.data_type().clone()),
                        ]
                        .into(),
                        vec![keys, values],
                        None,
                    )
                };
                let source = make_entries(
                    Arc::new(LargeStringArray::from_iter_values(
                        keys.iter().map(String::as_str),
                    )),
                    a,
                );
                let target = make_entries(
                    Arc::new(StringArray::from_iter_values(
                        keys.iter().map(String::as_str),
                    )),
                    b,
                );
                (
                    Arc::new(MapArray::new(
                        Arc::new(Field::new("entries", source.data_type().clone(), false)),
                        OffsetBuffer::new(offsets.clone().into()),
                        source,
                        nulls.clone(),
                        false,
                    )),
                    Arc::new(MapArray::new(
                        Arc::new(Field::new("entries", target.data_type().clone(), false)),
                        OffsetBuffer::new(offsets.into()),
                        target,
                        nulls,
                        false,
                    )),
                )
            }
        }
        6 => {
            let size = rng.random_range(0..4);
            let (a, b) = generated_pair(rng, depth - 1, rows * size, None);
            let nulls = Some(NullBuffer::from(
                (0..rows).map(|_| rng.random_bool(0.8)).collect::<Vec<_>>(),
            ));
            (
                Arc::new(
                    FixedSizeListArray::try_new_with_length(
                        field("item", a.data_type().clone()),
                        size as i32,
                        a,
                        nulls.clone(),
                        rows,
                    )
                    .unwrap(),
                ),
                Arc::new(
                    FixedSizeListArray::try_new_with_length(
                        field("item", b.data_type().clone()),
                        size as i32,
                        b,
                        nulls,
                        rows,
                    )
                    .unwrap(),
                ),
            )
        }
        8 => {
            // A logical Large target is deliberately unstaged and unlimited.
            let strings: Vec<_> = (0..rows)
                .map(|_| "L".repeat(rng.random_range(0..40)))
                .collect();
            let array: ArrayRef = Arc::new(LargeStringArray::from_iter_values(strings));
            (array.clone(), array)
        }
        _ => unreachable!(),
    }
}

// Independent oracle: inspect each actual target value, instead of using
// the staging planner's offset differences or prefix-search implementation.
fn target_leaf_bytes(array: &ArrayRef) -> Vec<usize> {
    use DataType::*;
    match array.data_type() {
        Utf8 => vec![
            (0..array.len())
                .map(|i| array.as_string::<i32>().value(i).len())
                .sum(),
        ],
        Binary => vec![
            (0..array.len())
                .map(|i| array.as_binary::<i32>().value(i).len())
                .sum(),
        ],
        Struct(_) => array
            .as_struct()
            .columns()
            .iter()
            .flat_map(target_leaf_bytes)
            .collect(),
        List(_) | LargeList(_) | FixedSizeList(_, _) | Map(_, _) => {
            let child = |row| match array.data_type() {
                List(_) => array.as_list::<i32>().value(row),
                LargeList(_) => array.as_list::<i64>().value(row),
                FixedSizeList(_, _) => array.as_fixed_size_list().value(row),
                Map(_, _) => Arc::new(array.as_map().value(row)) as ArrayRef,
                _ => unreachable!(),
            };
            let empty_child = match array.data_type() {
                List(_) => array.as_list::<i32>().values().slice(0, 0),
                LargeList(_) => array.as_list::<i64>().values().slice(0, 0),
                FixedSizeList(_, _) => array.as_fixed_size_list().values().slice(0, 0),
                Map(_, _) => Arc::new(array.as_map().entries().slice(0, 0)) as ArrayRef,
                _ => unreachable!(),
            };
            let mut bytes = target_leaf_bytes(&empty_child);
            for row in 0..array.len() {
                for (sum, next) in bytes.iter_mut().zip(target_leaf_bytes(&child(row))) {
                    *sum += next;
                }
            }
            bytes
        }
        _ => vec![],
    }
}

#[test]
fn test_capacity_splitter_seeded_nested_differential_slices() {
    use rand::{SeedableRng, rngs::StdRng};
    let mut rng = StdRng::seed_from_u64(0x7973_11478);
    for case in 0..126 {
        let (source, target) = generated_pair(&mut rng, 3, 8, Some(case % 9));
        let (extra_source, extra_target) = generated_pair(&mut rng, 0, 8, Some(0));
        let source = batch(
            vec![
                field("root", source.data_type().clone()),
                field("extra", extra_source.data_type().clone()),
            ],
            vec![source, extra_source],
        )
        .slice(1, 6);
        let target = batch(
            vec![
                field("root", target.data_type().clone()),
                field("extra", extra_target.data_type().clone()),
            ],
            vec![target, extra_target],
        )
        .slice(1, 6);
        let cap = [0, 1, 3, 8, 17, 1000][(case / 9) % 6];
        let mut splitter = CapacitySplitter::new(target.schema(), cap);
        splitter.push(source).unwrap();
        let costs: Vec<Vec<usize>> = (0..target.num_rows())
            .map(|row| {
                target
                    .columns()
                    .iter()
                    .flat_map(|a| target_leaf_bytes(&a.slice(row, 1)))
                    .collect()
            })
            .collect();
        let mut position = 0;
        while position < target.num_rows() {
            let mut accumulated = vec![0; costs[position].len()];
            let mut end = position;
            while end < target.num_rows()
                && accumulated
                    .iter()
                    .zip(&costs[end])
                    .all(|(sum, next)| sum + next <= cap)
            {
                for (sum, next) in accumulated.iter_mut().zip(&costs[end]) {
                    *sum += next;
                }
                end += 1;
            }
            if end == position {
                assert!(
                    splitter.next_batch().is_err(),
                    "case {case}, cap {cap}, row {position}"
                );
                break;
            }
            let actual = splitter.next_batch().unwrap().unwrap();
            assert_eq!(
                actual,
                target.slice(position, end - position),
                "case {case}, cap {cap}, row {position}"
            );
            for array in actual.columns() {
                array.to_data().validate_full().unwrap();
                assert!(target_leaf_bytes(array).iter().all(|&bytes| bytes <= cap));
            }
            position = end;
        }
        assert!(splitter.next_batch().unwrap().is_none());
    }
}
