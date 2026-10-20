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

//! Merges the filtered results of small Mask windows.
//!
//! Bounding the decoded size of each Mask window makes sparse selections of
//! wide columns, and the first windows of a reader, produce small batches.
//! Window results with fewer than
//! `batch_size / 2` selected rows are merged into batches of `batch_size`
//! rows. Larger results are returned unchanged, after the rows buffered before
//! them, so they are never copied again: joining two halves would copy a full
//! batch of rows to save one batch. Each window is filtered before the next
//! one is decoded, so only selected rows are buffered.
//!
//! A merged batch stays within the data byte limit of each byte array and the
//! offset range of each list or map: the buffered rows are returned first when
//! the next result would exceed one. A result that exceeds a limit on its own
//! is returned unchanged, for the capacity splitter.

use super::{MaskCursor, read_mask_window};
use crate::arrow::array_reader::ArrayReader;
use crate::errors::Result;
use arrow_array::cast::AsArray;
use arrow_array::{Array, OffsetSizeTrait, RecordBatch};
use arrow_schema::{DataType, SchemaRef};
use arrow_select::coalesce::BatchCoalescer;

/// Whether the results of small Mask windows can be merged for `data_type`.
///
/// Dictionaries and list views keep the whole window's values after
/// filtering, so concatenating filtered windows would copy all of them,
/// rejected rows included. Concatenated dictionaries can also overflow a
/// narrow key type. Layouts not listed in [`collect_limits`] are not merged.
pub(super) fn can_merge(data_type: &DataType) -> bool {
    collect_limits(data_type, 0, &mut vec![])
}

/// Buffers the results of small Mask windows; see [`read_merged_mask_batch`].
#[derive(Debug)]
pub(super) struct MaskWindowMerger {
    coalescer: BatchCoalescer,
    batch_size: usize,
    /// Data bytes of each byte array and child count of each list or map, in
    /// schema preorder.
    limits: Vec<u64>,
    /// Usage of the rows buffered in `coalescer`, in the order of `limits`.
    buffered: Vec<u64>,
    /// A result returned unchanged, after the buffered rows that precede it.
    pending: Option<RecordBatch>,
}

impl MaskWindowMerger {
    fn new(schema: SchemaRef, batch_size: usize, byte_capacity: usize) -> Self {
        let byte_capacity = byte_capacity.min(i32::MAX as usize) as u64;
        let mut limits = vec![];
        for field in schema.fields() {
            let mergeable = collect_limits(field.data_type(), byte_capacity, &mut limits);
            debug_assert!(mergeable, "Mask windows of {field} cannot be merged");
        }
        Self {
            coalescer: BatchCoalescer::new(schema, batch_size),
            batch_size,
            buffered: vec![0; limits.len()],
            limits,
            pending: None,
        }
    }

    fn next_ready(&mut self) -> Option<RecordBatch> {
        self.coalescer
            .next_completed_batch()
            .or_else(|| self.pending.take())
    }

    /// Complete a batch from the buffered rows, if any.
    fn flush(&mut self) -> Result<()> {
        self.coalescer.finish_buffered_batch()?;
        self.buffered.fill(0);
        Ok(())
    }

    /// Return `batch` unchanged, after the buffered rows.
    fn pass_through(&mut self, batch: RecordBatch) -> Result<()> {
        self.flush()?;
        self.pending = Some(batch);
        Ok(())
    }

    /// Buffer a filtered result of fewer than `batch_size / 2` rows.
    fn merge(&mut self, batch: RecordBatch) -> Result<()> {
        let rows = batch.num_rows();
        let usage = batch_usage(&batch, 0, rows, self.limits.len());
        if !within(&usage, &self.limits) {
            return self.pass_through(batch);
        }
        let joined = self
            .buffered
            .iter()
            .zip(&usage)
            .map(|(buffered, incoming)| buffered.saturating_add(*incoming))
            .collect::<Vec<_>>();
        let joined = if within(&joined, &self.limits) {
            joined
        } else {
            self.flush()?;
            usage
        };
        let total = self.coalescer.get_buffered_rows() + rows;
        self.coalescer.push_batch(batch.clone())?;
        // Each result has fewer than `batch_size / 2` rows, so a push completes
        // at most one batch, and the rows still buffered are the end of `batch`.
        self.buffered = match total.checked_sub(self.batch_size) {
            Some(tail) => batch_usage(&batch, rows - tail, tail, self.limits.len()),
            None => joined,
        };
        debug_assert_eq!(self.coalescer.get_buffered_rows(), total % self.batch_size);
        Ok(())
    }
}

/// Read Mask windows until a batch is ready, merging small window results.
///
/// `merger` is created, with the decoded schema, for the first small result.
/// `byte_capacity` limits the data bytes of each merged byte array.
/// `window_grids` sizes each window; see [`read_mask_window`].
pub(super) fn read_merged_mask_batch(
    merger: &mut Option<Box<MaskWindowMerger>>,
    array_reader: &mut dyn ArrayReader,
    mask_cursor: &mut MaskCursor,
    batch_size: usize,
    byte_capacity: usize,
    window_grids: &mut u32,
) -> Result<Option<RecordBatch>> {
    loop {
        if let Some(batch) = merger.as_mut().and_then(|merger| merger.next_ready()) {
            return Ok(Some(batch));
        }
        if mask_cursor.is_empty() {
            let Some(merger) = merger.as_mut() else {
                return Ok(None);
            };
            merger.flush()?;
            return Ok(merger.next_ready());
        }

        let window = read_mask_window(array_reader, mask_cursor, batch_size, window_grids)?;
        let small = window.selected_rows < batch_size / 2;
        // Filter first, so the decoded window is released before any copy.
        let batch = window.into_filtered()?;
        match merger {
            Some(merger) if small => merger.merge(batch)?,
            Some(merger) => merger.pass_through(batch)?,
            None if small => merger
                .insert(Box::new(MaskWindowMerger::new(
                    batch.schema(),
                    batch_size,
                    byte_capacity,
                )))
                .merge(batch)?,
            None => return Ok(Some(batch)),
        }
    }
}

fn within(usage: &[u64], limits: &[u64]) -> bool {
    usage.iter().zip(limits).all(|(used, limit)| used <= limit)
}

/// Push the limits of `data_type` in preorder. Returns false for layouts
/// whose filtered windows are not merged.
fn collect_limits(data_type: &DataType, byte_capacity: u64, limits: &mut Vec<u64>) -> bool {
    match data_type {
        DataType::Utf8 | DataType::Binary | DataType::LargeUtf8 | DataType::LargeBinary => {
            limits.push(byte_capacity)
        }
        DataType::List(field) | DataType::Map(field, _) => {
            limits.push(i32::MAX as u64);
            return collect_limits(field.data_type(), byte_capacity, limits);
        }
        DataType::LargeList(field) => {
            limits.push(i64::MAX as u64);
            return collect_limits(field.data_type(), byte_capacity, limits);
        }
        DataType::FixedSizeList(field, size) if *size >= 0 => {
            return collect_limits(field.data_type(), byte_capacity, limits);
        }
        DataType::Struct(fields) => {
            return fields
                .iter()
                .all(|field| collect_limits(field.data_type(), byte_capacity, limits));
        }
        DataType::Null | DataType::Boolean | DataType::Utf8View | DataType::BinaryView => {}
        DataType::FixedSizeBinary(size) if *size >= 0 => {}
        other if other.is_primitive() => {}
        _ => return false,
    }
    true
}

/// The usage of rows `start..start + len` of `batch`, in the order of the
/// limits of its schema.
fn batch_usage(batch: &RecordBatch, start: usize, len: usize, count: usize) -> Vec<u64> {
    let mut usage = Vec::with_capacity(count);
    for column in batch.columns() {
        collect_usage(column.as_ref(), start, start + len, &mut usage);
    }
    debug_assert_eq!(usage.len(), count);
    usage
}

fn offset_range<O: OffsetSizeTrait>(offsets: &[O], start: usize, end: usize) -> (usize, usize) {
    (offsets[start].as_usize(), offsets[end].as_usize())
}

fn collect_usage(array: &dyn Array, start: usize, end: usize, usage: &mut Vec<u64>) {
    let bytes = |(first, last): (usize, usize)| (last - first) as u64;
    match array.data_type() {
        DataType::Utf8 => usage.push(bytes(offset_range(
            array.as_string::<i32>().value_offsets(),
            start,
            end,
        ))),
        DataType::Binary => usage.push(bytes(offset_range(
            array.as_binary::<i32>().value_offsets(),
            start,
            end,
        ))),
        DataType::LargeUtf8 => usage.push(bytes(offset_range(
            array.as_string::<i64>().value_offsets(),
            start,
            end,
        ))),
        DataType::LargeBinary => usage.push(bytes(offset_range(
            array.as_binary::<i64>().value_offsets(),
            start,
            end,
        ))),
        DataType::List(_) => {
            let list = array.as_list::<i32>();
            let (first, last) = offset_range(list.value_offsets(), start, end);
            usage.push((last - first) as u64);
            collect_usage(list.values().as_ref(), first, last, usage);
        }
        DataType::LargeList(_) => {
            let list = array.as_list::<i64>();
            let (first, last) = offset_range(list.value_offsets(), start, end);
            usage.push((last - first) as u64);
            collect_usage(list.values().as_ref(), first, last, usage);
        }
        DataType::Map(_, _) => {
            let map = array.as_map();
            let (first, last) = offset_range(map.value_offsets(), start, end);
            usage.push((last - first) as u64);
            collect_usage(map.entries(), first, last, usage);
        }
        DataType::FixedSizeList(_, size) => {
            let size = *size as usize;
            collect_usage(
                array.as_fixed_size_list().values().as_ref(),
                start * size,
                end * size,
                usage,
            )
        }
        DataType::Struct(_) => {
            for child in array.as_struct().columns() {
                collect_usage(child.as_ref(), start, end, usage);
            }
        }
        // Fixed width and view arrays have no limit; see `collect_limits`.
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::builder::{FixedSizeListBuilder, ListBuilder, MapBuilder, StringBuilder};
    use arrow_array::{ArrayRef, Int32Array, LargeStringArray, ListArray, NullArray, StructArray};
    use arrow_buffer::OffsetBuffer;
    use arrow_schema::Field;
    use std::sync::Arc;

    fn strings(values: &[&str]) -> RecordBatch {
        let array: ArrayRef = Arc::new(LargeStringArray::from(values.to_vec()));
        RecordBatch::try_from_iter([("s", array)]).unwrap()
    }

    fn drain(merger: &mut MaskWindowMerger) -> Vec<Vec<String>> {
        merger.flush().unwrap();
        std::iter::from_fn(|| merger.next_ready())
            .map(|batch| {
                let values = batch.column(0).as_string::<i64>();
                values.iter().map(|v| v.unwrap().to_string()).collect()
            })
            .collect()
    }

    #[test]
    fn test_mask_merge_limits_and_usage_follow_schema_preorder() {
        let mut list = ListBuilder::new(StringBuilder::new());
        list.values().append_value("ab");
        list.append(true);
        list.values().append_value("cde");
        list.values().append_value("f");
        list.append(true);
        let mut map = MapBuilder::new(None, StringBuilder::new(), StringBuilder::new());
        map.keys().append_value("k");
        map.values().append_value("vv");
        map.append(true).unwrap();
        map.append(true).unwrap();
        let mut fixed = FixedSizeListBuilder::new(StringBuilder::new(), 2);
        for row in ["gh", "i"] {
            fixed.values().append_value(row);
            fixed.values().append_value("j");
            fixed.append(true);
        }
        let ids: ArrayRef = Arc::new(Int32Array::from(vec![1, 2]));
        let names: ArrayRef = Arc::new(LargeStringArray::from(vec!["xy", "zzz"]));
        let row: ArrayRef = Arc::new(StructArray::from(vec![(
            Arc::new(Field::new("name", DataType::LargeUtf8, false)),
            names,
        )]));
        let batch = RecordBatch::try_from_iter([
            ("id", ids),
            ("list", Arc::new(list.finish()) as ArrayRef),
            ("map", Arc::new(map.finish()) as ArrayRef),
            ("fixed", Arc::new(fixed.finish()) as ArrayRef),
            ("row", row),
        ])
        .unwrap();

        let mut limits = vec![];
        assert!(collect_limits(
            &DataType::Struct(batch.schema().fields().clone()),
            100,
            &mut limits
        ));
        let child = i32::MAX as u64;
        // list, list item, map, map key, map value, fixed item, struct name
        assert_eq!(limits, [child, 100, child, 100, 100, 100, 100]);
        assert_eq!(batch_usage(&batch, 0, 2, 7), [3, 6, 1, 1, 2, 5, 5]);
        assert_eq!(batch_usage(&batch, 1, 1, 7), [2, 4, 0, 0, 0, 2, 3]);
    }

    #[test]
    fn test_mask_merge_declines_layouts_that_keep_window_values() {
        let item = Arc::new(Field::new("item", DataType::Utf8, true));
        for data_type in [
            DataType::Dictionary(Box::new(DataType::Int8), Box::new(DataType::Utf8)),
            DataType::ListView(item.clone()),
            DataType::LargeListView(item.clone()),
            DataType::List(Arc::new(Field::new(
                "item",
                DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8)),
                true,
            ))),
        ] {
            assert!(!can_merge(&data_type), "{data_type}");
        }
        for data_type in [
            DataType::Decimal128(38, 2),
            DataType::Utf8View,
            DataType::FixedSizeBinary(4),
            DataType::Map(
                Arc::new(Field::new(
                    "entries",
                    DataType::Struct(
                        vec![
                            Field::new("key", DataType::Utf8, false),
                            Field::new("value", DataType::Int64, true),
                        ]
                        .into(),
                    ),
                    false,
                )),
                false,
            ),
        ] {
            assert!(can_merge(&data_type), "{data_type}");
        }
    }

    #[test]
    fn test_mask_merge_returns_buffered_rows_before_byte_capacity() {
        let schema = strings(&[]).schema();
        let mut merger = MaskWindowMerger::new(schema, 8, 10);
        merger.merge(strings(&["aaaa", "bbbb"])).unwrap();
        // 8 + 4 bytes would exceed the 10 byte capacity.
        merger.merge(strings(&["cccc"])).unwrap();
        merger.merge(strings(&["dd"])).unwrap();
        assert_eq!(
            drain(&mut merger),
            [vec!["aaaa", "bbbb"], vec!["cccc", "dd"]]
        );
    }

    #[test]
    fn test_mask_merge_passes_oversized_result_through_unchanged() {
        let schema = strings(&[]).schema();
        let mut merger = MaskWindowMerger::new(schema, 8, 10);
        merger.merge(strings(&["a"])).unwrap();
        let oversized = strings(&["bbbbbb", "cccccc"]);
        merger.merge(oversized.clone()).unwrap();
        assert_eq!(merger.next_ready().unwrap().num_rows(), 1);
        let passed = merger.next_ready().unwrap();
        assert_eq!(passed, oversized);
        assert_eq!(
            passed.column(0).to_data().buffers()[1].as_ptr(),
            oversized.column(0).to_data().buffers()[1].as_ptr(),
            "an oversized result is returned without a copy"
        );
        assert!(merger.next_ready().is_none());
    }

    #[test]
    fn test_mask_merge_recounts_the_tail_after_a_full_batch() {
        let schema = strings(&[]).schema();
        let mut merger = MaskWindowMerger::new(schema, 4, 100);
        merger.merge(strings(&["a", "b"])).unwrap();
        merger.merge(strings(&["c"])).unwrap();
        let (forty, thirty) = ("e".repeat(40), "f".repeat(30));
        // The first row completes a batch of 4; only the 40 byte row stays.
        merger.merge(strings(&["d", &forty])).unwrap();
        // 40 + 60 bytes fit when the tail is recounted.
        merger.merge(strings(&[&thirty, &thirty])).unwrap();
        let batches = drain(&mut merger);
        assert_eq!(
            batches.iter().map(Vec::len).collect::<Vec<_>>(),
            [4, 3],
            "{batches:?}"
        );
    }

    #[test]
    fn test_mask_merge_list_child_count_without_large_allocation() {
        // Two legal single-row lists whose children together exceed i32 offsets.
        let len = i32::MAX as usize / 2 + 1;
        let list = |len: usize| -> RecordBatch {
            let field = Arc::new(Field::new("item", DataType::Null, true));
            let array = ListArray::new(
                field,
                OffsetBuffer::from_lengths([len]),
                Arc::new(NullArray::new(len)),
                None,
            );
            RecordBatch::try_from_iter([("list", Arc::new(array) as ArrayRef)]).unwrap()
        };
        let schema = list(0).schema();
        let mut merger = MaskWindowMerger::new(schema, 8, i32::MAX as usize);
        merger.merge(list(len)).unwrap();
        merger.merge(list(len)).unwrap();
        merger.flush().unwrap();
        let batches = std::iter::from_fn(|| merger.next_ready()).collect::<Vec<_>>();
        assert_eq!(batches.len(), 2);
        for batch in batches {
            assert_eq!(batch.column(0).as_list::<i32>().values().len(), len);
        }
    }
}
