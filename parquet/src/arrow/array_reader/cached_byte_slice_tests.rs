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

//! Byte-slice retention tests using the production PLAIN byte-array reader.

use super::{BatchID, CacheRole, CachedArrayReader};
use crate::arrow::array_reader::test_util::utf8_column;
use crate::arrow::array_reader::{ArrayReader, RowGroupCache, make_byte_array_reader};
use crate::arrow::arrow_reader::metrics::ArrowReaderMetrics;
use crate::basic::Encoding;
use crate::data_type::{ByteArray, ByteArrayType as ParquetByteArrayType};
use crate::util::InMemoryPageIterator;
use crate::util::test_common::page_util::{DataPageBuilder, DataPageBuilderImpl};
use arrow_array::{Array, ArrayRef, GenericByteArray, cast::AsArray, types::ByteArrayType};
use arrow_buffer::Buffer;
use arrow_schema::DataType;
use std::sync::{Arc, RwLock};

const GRID: usize = 8;

fn plain_reader(
    data_type: DataType,
    rows: &[Option<Vec<u8>>],
    grid: usize,
) -> Box<dyn ArrayReader> {
    let descriptor = utf8_column();
    let levels: Vec<i16> = rows.iter().map(|row| i16::from(row.is_some())).collect();
    let values: Vec<ByteArray> = rows
        .iter()
        .flatten()
        .cloned()
        .map(ByteArray::from)
        .collect();
    let mut page = DataPageBuilderImpl::new(descriptor.clone(), rows.len() as u32, true);
    page.add_def_levels(1, &levels);
    page.add_values::<ParquetByteArrayType>(Encoding::PLAIN, &values);
    let pages = InMemoryPageIterator::new(vec![vec![page.consume()]]);
    make_byte_array_reader(Box::new(pages), descriptor, Some(data_type), grid, None).unwrap()
}

fn cached_reader(data_type: DataType, rows: &[Option<Vec<u8>>], grid: usize) -> CachedArrayReader {
    CachedArrayReader::new(
        plain_reader(data_type, rows, grid),
        Arc::new(RwLock::new(RowGroupCache::new(grid, 0))),
        0,
        CacheRole::Producer,
        ArrowReaderMetrics::disabled(),
    )
}

fn byte_buffers(array: &dyn Array) -> (&Buffer, &Buffer) {
    fn buffers<T: ByteArrayType>(array: &GenericByteArray<T>) -> (&Buffer, &Buffer) {
        (array.values(), array.offsets().inner().inner())
    }
    match array.data_type() {
        DataType::Utf8 => buffers(array.as_string::<i32>()),
        DataType::LargeUtf8 => buffers(array.as_string::<i64>()),
        DataType::Binary => buffers(array.as_binary::<i32>()),
        DataType::LargeBinary => buffers(array.as_binary::<i64>()),
        other => panic!("unexpected byte-array type: {other}"),
    }
}

fn byte_values(array: &dyn Array) -> Vec<Option<Vec<u8>>> {
    match array.data_type() {
        DataType::Utf8 => array
            .as_string::<i32>()
            .iter()
            .map(|value| value.map(|value| value.as_bytes().to_vec()))
            .collect(),
        DataType::LargeUtf8 => array
            .as_string::<i64>()
            .iter()
            .map(|value| value.map(|value| value.as_bytes().to_vec()))
            .collect(),
        DataType::Binary => array
            .as_binary::<i32>()
            .iter()
            .map(|v| v.map(<[u8]>::to_vec))
            .collect(),
        DataType::LargeBinary => array
            .as_binary::<i64>()
            .iter()
            .map(|v| v.map(<[u8]>::to_vec))
            .collect(),
        other => panic!("unexpected byte-array type: {other}"),
    }
}

fn completed_piece(reader: &CachedArrayReader) -> &ArrayRef {
    assert_eq!(reader.completed_selected.len(), 1);
    &reader.completed_selected[0]
}

/// The only completed piece, whose compaction is deferred: until
/// consume_batch it still shares the payload of `raw`, its source grid.
fn deferred_piece<'a>(reader: &'a CachedArrayReader, raw: &ArrayRef) -> &'a ArrayRef {
    let piece = completed_piece(reader);
    assert_eq!(reader.deferred.as_ref().map(|d| d.piece), Some(0));
    assert_eq!(
        byte_buffers(piece.as_ref()).0.as_ptr(),
        byte_buffers(raw.as_ref()).0.as_ptr(),
        "compaction waits until consume_batch knows whether concat copies the piece"
    );
    piece
}

#[test]
fn test_cached_byte_slice_near_full_suffix_shares_payload() {
    for data_type in [
        DataType::Utf8,
        DataType::Binary,
        DataType::LargeUtf8,
        DataType::LargeBinary,
    ] {
        let rows: Vec<_> = (0..2 * GRID)
            .map(|row| Some(format!("{row:04}{}", "x".repeat(124)).into_bytes()))
            .collect();
        let mut reader = cached_reader(data_type.clone(), &rows, GRID);
        assert_eq!(reader.fetch_batch(BatchID { val: 0 }).unwrap(), GRID);
        // This extra owner is used only for pointer identity, never as a live-memory measurement.
        let raw = reader.local_cache[&BatchID { val: 0 }].clone();
        assert_eq!(reader.skip_records(1).unwrap(), 1);
        assert_eq!(reader.read_records(GRID - 1).unwrap(), GRID - 1);
        let piece = completed_piece(&reader);
        assert_eq!(byte_values(piece.as_ref()), rows[1..GRID]);
        assert_eq!(
            byte_buffers(piece.as_ref()).0.as_ptr(),
            byte_buffers(raw.as_ref()).0.as_ptr(),
            "a near-full {data_type} suffix must not copy its payload before concat"
        );
        assert!(
            reader.deferred.is_none(),
            "a near-full suffix never needs compaction"
        );
        assert!(reader.local_cache.is_empty());
        assert_eq!(reader.read_records(1).unwrap(), 1);
        let output = reader.consume_batch().unwrap();
        assert_eq!(byte_values(output.as_ref()), rows[1..GRID + 1]);
    }
}

#[test]
fn test_cached_byte_slice_wide_excluded_prefix_still_copies_dense_suffix() {
    for data_type in [
        DataType::Utf8,
        DataType::Binary,
        DataType::LargeUtf8,
        DataType::LargeBinary,
    ] {
        let mut rows = vec![Some(vec![b'x'; 600 * 1024])];
        rows.extend((1..GRID).map(|row| Some(vec![b'0' + row as u8])));
        let mut reader = cached_reader(data_type.clone(), &rows, GRID);
        assert_eq!(reader.fetch_batch(BatchID { val: 0 }).unwrap(), GRID);
        let raw = reader.local_cache[&BatchID { val: 0 }].clone();
        assert_eq!(reader.skip_records(1).unwrap(), 1);
        assert_eq!(reader.read_records(GRID - 1).unwrap(), GRID - 1);
        let piece = deferred_piece(&reader, &raw);
        assert_eq!(byte_values(piece.as_ref()), rows[1..]);
        assert!(reader.local_cache.is_empty());
        // Returned alone, the piece is compacted rather than concatenated.
        let output = reader.consume_batch().unwrap();
        assert_eq!(byte_values(output.as_ref()), rows[1..]);
        let (selected_values, _) = byte_buffers(output.as_ref());
        assert_ne!(
            selected_values.as_ptr(),
            byte_buffers(raw.as_ref()).0.as_ptr(),
            "keeping seven of eight {data_type} rows must not retain the excluded wide value"
        );
        assert_eq!(selected_values.len(), GRID - 1);
        assert!(selected_values.capacity() <= 2 * (GRID - 1));
        assert!(reader.deferred.is_none());
        assert_eq!(reader.read_records(1).unwrap(), 0);
    }
}

#[test]
fn test_cached_byte_slice_tiny_completed_tail_does_not_pin_payload() {
    for data_type in [
        DataType::Utf8,
        DataType::Binary,
        DataType::LargeUtf8,
        DataType::LargeBinary,
    ] {
        let rows = vec![Some(vec![b'x'; 128]); GRID];
        let mut reader = cached_reader(data_type, &rows, GRID);
        assert_eq!(reader.fetch_batch(BatchID { val: 0 }).unwrap(), GRID);
        let raw = reader.local_cache[&BatchID { val: 0 }].clone();
        assert_eq!(reader.skip_records(GRID - 1).unwrap(), GRID - 1);
        assert_eq!(reader.read_records(1).unwrap(), 1);
        let piece = deferred_piece(&reader, &raw);
        assert_eq!(byte_values(piece.as_ref()), rows[GRID - 1..]);
        assert!(reader.local_cache.is_empty());
        let output = reader.consume_batch().unwrap();
        assert_eq!(byte_values(output.as_ref()), rows[GRID - 1..]);
        let (selected_values, _) = byte_buffers(output.as_ref());
        assert_ne!(
            selected_values.as_ptr(),
            byte_buffers(raw.as_ref()).0.as_ptr()
        );
        assert_eq!(selected_values.len(), 128);
        assert!(selected_values.capacity() <= 256);
    }
}

#[test]
fn test_cached_byte_slice_empty_values_compact_oversized_offset_owner() {
    // With only eight empty rows, the array header alone can keep the retained
    // footprint below twice the selected footprint. This larger grid isolates
    // offsets retention without introducing any payload allocation.
    const EMPTY_GRID: usize = 256;
    for (data_type, offset_width) in [
        (DataType::Utf8, 4),
        (DataType::Binary, 4),
        (DataType::LargeUtf8, 8),
        (DataType::LargeBinary, 8),
    ] {
        let rows = vec![Some(Vec::new()); EMPTY_GRID];
        let mut reader = cached_reader(data_type, &rows, EMPTY_GRID);
        assert_eq!(reader.fetch_batch(BatchID { val: 0 }).unwrap(), EMPTY_GRID);
        let raw = reader.local_cache[&BatchID { val: 0 }].clone();
        assert_eq!(reader.skip_records(EMPTY_GRID - 1).unwrap(), EMPTY_GRID - 1);
        assert_eq!(reader.read_records(1).unwrap(), 1);
        let piece = completed_piece(&reader);
        assert_eq!(byte_values(piece.as_ref()), vec![Some(Vec::new())]);
        let (_, raw_offsets) = byte_buffers(raw.as_ref());
        assert!(reader.deferred.is_some());
        assert!(
            byte_buffers(piece.as_ref()).1.capacity() >= (EMPTY_GRID + 1) * offset_width,
            "compaction waits until consume_batch knows whether concat copies the piece"
        );
        assert!(reader.local_cache.is_empty());
        let output = reader.consume_batch().unwrap();
        assert_eq!(byte_values(output.as_ref()), vec![Some(Vec::new())]);
        let (selected_values, selected_offsets) = byte_buffers(output.as_ref());
        assert!(selected_values.is_empty());
        assert_ne!(selected_offsets.as_ptr(), raw_offsets.as_ptr());
        assert!(selected_offsets.capacity() <= 2 * offset_width);
        assert!(raw_offsets.capacity() >= (EMPTY_GRID + 1) * offset_width);
        assert!(reader.local_cache.is_empty());
    }
}

#[test]
fn test_cached_byte_slice_nullable_nonzero_source_offsets_preserve_values() {
    for data_type in [
        DataType::Utf8,
        DataType::Binary,
        DataType::LargeUtf8,
        DataType::LargeBinary,
    ] {
        let rows = vec![
            Some(vec![b'x'; 600 * 1024]),
            Some(b"excluded".to_vec()),
            None,
            Some(b"three".to_vec()),
            Some(Vec::new()),
            None,
            Some(b"six".to_vec()),
            Some(b"seven".to_vec()),
            Some(b"eight".to_vec()),
        ];
        let mut source_reader = plain_reader(data_type.clone(), &rows, GRID);
        assert_eq!(source_reader.read_records(GRID + 1).unwrap(), GRID + 1);
        let owner = source_reader.consume_batch().unwrap();
        let source = owner.slice(1, GRID);
        // Seed one valid cached grid with a slice of production-decoded data.
        // This isolates already-sliced source buffers and a non-byte-aligned
        // validity bitmap; no mock ArrayReader supplies the decoded values.
        let mut reader = cached_reader(data_type.clone(), &rows[1..], GRID);
        reader
            .local_cache
            .insert(BatchID { val: 0 }, source.clone());
        assert_eq!(reader.skip_records(1).unwrap(), 1);
        assert_eq!(reader.read_records(GRID - 1).unwrap(), GRID - 1);
        let piece = deferred_piece(&reader, &source);
        assert_eq!(piece.data_type(), &data_type);
        assert_eq!(piece.null_count(), 2);
        assert_eq!(byte_values(piece.as_ref()), rows[2..]);
        assert!(reader.local_cache.is_empty());
        let output = reader.consume_batch().unwrap();
        assert_eq!(output.data_type(), &data_type);
        assert_eq!(byte_values(output.as_ref()), rows[2..]);
        assert_eq!(output.null_count(), 2);
        let (selected_values, _) = byte_buffers(output.as_ref());
        assert_ne!(
            selected_values.as_ptr(),
            byte_buffers(source.as_ref()).0.as_ptr(),
            "a sliced source must not hide the retained wide prefix"
        );
        let expected_payload: usize = rows[2..].iter().flatten().map(Vec::len).sum();
        assert_eq!(selected_values.len(), expected_payload);
        assert!(selected_values.capacity() <= 2 * expected_payload);
        assert_eq!(reader.read_records(1).unwrap(), 0);
    }
}

#[test]
fn test_cached_byte_slice_partial_grid_output_does_not_pin_payload() {
    for data_type in [
        DataType::Utf8,
        DataType::Binary,
        DataType::LargeUtf8,
        DataType::LargeBinary,
    ] {
        for initial_skip in [0, 2] {
            let rows: Vec<_> = (0..2 * GRID)
                .map(|row| Some(format!("{row:04}{}", "x".repeat(60)).into_bytes()))
                .collect();
            let metrics = ArrowReaderMetrics::enabled();
            let shared_cache = Arc::new(RwLock::new(RowGroupCache::new(GRID, 0)));
            let mut reader = CachedArrayReader::new(
                plain_reader(data_type.clone(), &rows, GRID),
                shared_cache.clone(),
                0,
                CacheRole::Producer,
                metrics.clone(),
            );
            assert_eq!(reader.skip_records(initial_skip).unwrap(), initial_skip);
            // A one-row read fetches the complete first grid even when the
            // shared cache rejects it. Keep an owner only to compare pointers.
            assert_eq!(reader.read_records(1).unwrap(), 1);
            assert_eq!(metrics.records_read_from_inner(), Some(GRID));
            let raw = reader.local_cache[&BatchID { val: 0 }].clone();
            assert_eq!(byte_buffers(raw.as_ref()).0.len(), GRID * 64);
            assert!(
                shared_cache
                    .read()
                    .unwrap()
                    .get(0, BatchID { val: 0 })
                    .is_none()
            );

            let output = reader.consume_batch().unwrap();
            assert_eq!(
                byte_values(output.as_ref()),
                rows[initial_skip..initial_skip + 1]
            );
            let (selected_values, _) = byte_buffers(output.as_ref());
            assert_ne!(
                selected_values.as_ptr(),
                byte_buffers(raw.as_ref()).0.as_ptr(),
                "an early one-row {data_type} output must not pin the complete current grid"
            );
            assert_eq!(selected_values.len(), 64);
            assert!(selected_values.capacity() <= 128);
            assert_eq!(reader.local_cache.len(), 1);
            assert!(reader.completed_selected.is_empty());

            // The reader still owns the current grid for its unread tail. It
            // must reuse that grid rather than decode it again after consume.
            let remaining = GRID - initial_skip - 1;
            assert_eq!(reader.read_records(remaining).unwrap(), remaining);
            let tail = reader.consume_batch().unwrap();
            assert_eq!(byte_values(tail.as_ref()), rows[initial_skip + 1..GRID]);
            assert_eq!(metrics.records_read_from_inner(), Some(GRID));
            assert_eq!(metrics.records_read_from_cache(), Some(remaining));
            assert!(reader.local_cache.is_empty());

            assert_eq!(reader.read_records(1).unwrap(), 1);
            let next = reader.consume_batch().unwrap();
            assert_eq!(byte_values(next.as_ref()), rows[GRID..GRID + 1]);
            assert_eq!(metrics.records_read_from_inner(), Some(2 * GRID));
            assert!(reader.local_cache.len() <= 1);
        }
    }
}

#[test]
fn test_cached_byte_slice_multi_grid_read_defers_compaction_to_concat() {
    for data_type in [
        DataType::Utf8,
        DataType::Binary,
        DataType::LargeUtf8,
        DataType::LargeBinary,
    ] {
        let rows: Vec<_> = (0..2 * GRID)
            .map(|row| Some(format!("{row:04}{}", "x".repeat(60)).into_bytes()))
            .collect();
        let mut reader = cached_reader(data_type.clone(), &rows, GRID);
        assert_eq!(reader.fetch_batch(BatchID { val: 0 }).unwrap(), GRID);
        let raw = reader.local_cache[&BatchID { val: 0 }].clone();
        // Like a Mask window spanning two grids: the one-row tail of grid 0
        // would be compacted if returned alone, but the read continues.
        assert_eq!(reader.skip_records(GRID - 1).unwrap(), GRID - 1);
        assert_eq!(reader.read_records(3).unwrap(), 3);
        let piece = deferred_piece(&reader, &raw);
        assert_eq!(byte_values(piece.as_ref()), rows[GRID - 1..GRID]);
        let output = reader.consume_batch().unwrap();
        assert_eq!(byte_values(output.as_ref()), rows[GRID - 1..GRID + 2]);
        let (values, _) = byte_buffers(output.as_ref());
        assert_ne!(
            values.as_ptr(),
            byte_buffers(raw.as_ref()).0.as_ptr(),
            "concat copies the deferred {data_type} piece"
        );
        assert_eq!(values.len(), 3 * 64);
        assert!(values.capacity() <= 2 * 3 * 64);
        assert!(reader.deferred.is_none());
    }
}

#[test]
fn test_cached_byte_slice_second_deferred_piece_compacts_the_first() {
    for data_type in [
        DataType::Utf8,
        DataType::Binary,
        DataType::LargeUtf8,
        DataType::LargeBinary,
    ] {
        let rows: Vec<_> = (0..2 * GRID)
            .map(|row| Some(format!("{row:04}{}", "x".repeat(60)).into_bytes()))
            .collect();
        let mut reader = cached_reader(data_type.clone(), &rows, GRID);
        assert_eq!(reader.fetch_batch(BatchID { val: 0 }).unwrap(), GRID);
        let raw0 = reader.local_cache[&BatchID { val: 0 }].clone();
        // Sparse reads of the last row of each grid, as in a Selectors read
        // accumulating one output batch across many grids.
        assert_eq!(reader.skip_records(GRID - 1).unwrap(), GRID - 1);
        assert_eq!(reader.read_records(1).unwrap(), 1);
        deferred_piece(&reader, &raw0);
        assert_eq!(reader.skip_records(GRID - 1).unwrap(), GRID - 1);
        assert_eq!(reader.read_records(1).unwrap(), 1);
        assert_eq!(reader.completed_selected.len(), 2);
        let deferred = reader.deferred.as_ref().unwrap();
        assert_eq!(
            deferred.piece, 1,
            "only the latest {data_type} piece may share its raw grid"
        );
        let raw1 = Arc::clone(&deferred.grid);
        let (first, _) = byte_buffers(reader.completed_selected[0].as_ref());
        assert_ne!(
            first.as_ptr(),
            byte_buffers(raw0.as_ref()).0.as_ptr(),
            "deferring a second {data_type} piece must release the first piece's raw grid"
        );
        assert_eq!(first.len(), 64);
        assert!(first.capacity() <= 128);
        assert_eq!(
            byte_buffers(reader.completed_selected[1].as_ref())
                .0
                .as_ptr(),
            byte_buffers(raw1.as_ref()).0.as_ptr()
        );
        let output = reader.consume_batch().unwrap();
        assert_eq!(
            byte_values(output.as_ref()),
            vec![rows[GRID - 1].clone(), rows[2 * GRID - 1].clone()]
        );
        let (values, _) = byte_buffers(output.as_ref());
        assert_eq!(values.len(), 2 * 64);
        assert!(values.capacity() <= 2 * 2 * 64);
        assert!(reader.deferred.is_none());
    }
}
