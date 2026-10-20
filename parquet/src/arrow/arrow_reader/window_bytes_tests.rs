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

use super::{next_mask_window_grids, window_bytes};
use arrow_array::{
    Array, ArrayRef, FixedSizeListArray, Int32Array, LargeListArray, ListArray, MapArray,
    StringViewArray, StructArray,
};
use arrow_buffer::OffsetBuffer;
use arrow_schema::{DataType, Field};
use std::sync::Arc;

fn views() -> ArrayRef {
    Arc::new(StringViewArray::from_iter_values(
        (0..10_000).map(|row| format!("{row:0100}")),
    ))
}

#[test]
fn test_window_bytes_nested_views_trim_child_spans() {
    let views = views();
    let item = Arc::new(Field::new_list_field(DataType::Utf8View, false));
    let list = ListArray::new(
        item.clone(),
        OffsetBuffer::new(vec![100, 101, 103, 106].into()),
        views.clone(),
        None,
    )
    .slice(1, 1);
    let large_list = LargeListArray::new(
        item.clone(),
        OffsetBuffer::new(vec![100i64, 101, 103, 106].into()),
        views.clone(),
        None,
    )
    .slice(1, 1);
    let fixed_list = FixedSizeListArray::new(item, 2, views.slice(100, 6), None).slice(1, 1);
    let entries = StructArray::from(vec![
        (
            Arc::new(Field::new("key", DataType::Int32, false)),
            Arc::new(Int32Array::from_iter_values(0..10_000)) as ArrayRef,
        ),
        (
            Arc::new(Field::new("value", DataType::Utf8View, false)),
            views,
        ),
    ]);
    let map = MapArray::new(
        Arc::new(Field::new("entries", entries.data_type().clone(), false)),
        OffsetBuffer::new(vec![100, 101, 103, 106].into()),
        entries,
        None,
        false,
    )
    .slice(1, 1);

    let child_bytes = 2 * (16 + 100);
    assert_eq!(window_bytes(&list), 2 * 4 + child_bytes);
    assert_eq!(window_bytes(&large_list), 2 * 8 + child_bytes);
    assert_eq!(window_bytes(&fixed_list), child_bytes);
    assert_eq!(window_bytes(&map), 2 * 4 + 2 * 4 + child_bytes);
}

#[test]
fn test_window_bytes_sparse_nested_views_allow_grid_growth() {
    let list = ListArray::new(
        Arc::new(Field::new_list_field(DataType::Utf8View, false)),
        OffsetBuffer::new(vec![5000, 5001].into()),
        views(),
        None,
    );
    // One selected row at the start of each grid leaves a one-row window.
    // Its page buffer is shared, so it must not count as one row's payload.
    let bytes = window_bytes(&list);
    assert_eq!(bytes, 2 * 4 + 16 + 100);
    assert_eq!(next_mask_window_grids(1, bytes, 1, 8192), 2);
}
