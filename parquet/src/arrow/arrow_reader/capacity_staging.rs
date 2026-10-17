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

//! Restore 32-bit offset byte arrays after decoding with 64-bit offsets.
//!
//! When the projected schema contains `Utf8` or `Binary` leaves, the array
//! readers decode them as `LargeUtf8`/`LargeBinary` (see
//! [`ArrayReaderBuilder::with_byte_array_staging`]), so decoding a window can
//! never overflow 32-bit offsets, however many rows or bytes the window spans.
//! [`CapacitySplitter`] turns each decoded and filtered window back into the
//! requested schema:
//!
//! * If every `Utf8`/`Binary` leaf of the window fits within the limit
//!   (`i32::MAX` bytes outside of tests), the window is emitted as a single
//!   batch. Offsets are narrowed to 32 bits and the values buffers are shared,
//!   so no string data is copied and the batch shape is unchanged.
//! * Otherwise the window is split at complete top-level rows into batches
//!   whose leaves each fit. Each piece's data is copied, so the outputs do not
//!   keep the whole window's buffers alive.
//!
//! A single top-level row whose data cannot fit in one array is an error.
//! Lists and maps keep their 32-bit child-count limit, and `ListView` /
//! `LargeListView` subtrees are not staged (see the builder).
//!
//! [`ArrayReaderBuilder::with_byte_array_staging`]: crate::arrow::array_reader::ArrayReaderBuilder

use crate::errors::{ParquetError, Result};
use arrow_array::cast::AsArray;
use arrow_array::types::{ByteArrayType, GenericBinaryType, GenericStringType};
use arrow_array::{
    Array, ArrayRef, FixedSizeListArray, GenericByteArray, GenericListArray, MapArray,
    OffsetSizeTrait, RecordBatch, RecordBatchOptions, StructArray,
};
use arrow_buffer::{ArrowNativeType, Buffer, OffsetBuffer, ScalarBuffer};
use arrow_schema::{DataType, FieldRef, SchemaRef};
use std::sync::Arc;

/// Splits decoded windows so that every `Utf8`/`Binary` leaf of each output
/// batch fits in a 32-bit offset array, then restores the requested schema.
pub(crate) struct CapacitySplitter {
    target_schema: SchemaRef,
    max_bytes: u64,
    pending: Option<PendingBatch>,
}

/// The remainder of one decoded window that has not been emitted yet.
struct PendingBatch {
    batch: RecordBatch,
    /// First row of `batch` that has not been emitted.
    offset: usize,
    /// Per-row byte prefixes of the leaves that do not fit. `None` when the
    /// whole window fits, which is the common case and needs no per-row work.
    prefixes: Option<Vec<LeafPrefix>>,
}

/// Cumulative data bytes of one oversized leaf at each top-level row boundary.
struct LeafPrefix {
    path: String,
    kind: &'static str,
    bytes: Vec<u64>,
}

impl CapacitySplitter {
    pub(crate) fn new(target_schema: SchemaRef, max_bytes: usize) -> Self {
        Self {
            target_schema,
            max_bytes: max_bytes.min(i32::MAX as usize) as u64,
            pending: None,
        }
    }

    pub(crate) fn has_pending(&self) -> bool {
        self.pending.is_some()
    }

    /// Accept one decoded window. [`Self::next_batch`] must drain it before
    /// the next call.
    pub(crate) fn push(&mut self, batch: RecordBatch) -> Result<()> {
        if self.pending.is_some() {
            return Err(general_err!(
                "Internal Error: capacity splitter must be drained before push"
            ));
        }
        validate_batch(&batch, &self.target_schema)?;
        let rows = batch.num_rows();
        let mut fits = true;
        for (array, field) in batch.columns().iter().zip(self.target_schema.fields()) {
            fits &= span_fits(array, field.data_type(), 0, rows, self.max_bytes)?;
        }
        let prefixes = if fits {
            None
        } else {
            let boundaries = (0..=rows).collect::<Vec<_>>();
            let mut prefixes = vec![];
            for (array, field) in batch.columns().iter().zip(self.target_schema.fields()) {
                collect_prefixes(
                    array,
                    field.data_type(),
                    &boundaries,
                    self.max_bytes,
                    field.name(),
                    &mut prefixes,
                )?;
            }
            Some(prefixes)
        };
        self.pending = Some(PendingBatch {
            batch,
            offset: 0,
            prefixes,
        });
        Ok(())
    }

    /// Return the next batch in the requested schema. On error the pending
    /// window is released rather than retrying the same unrepresentable row.
    pub(crate) fn next_batch(&mut self) -> Result<Option<RecordBatch>> {
        let Some(mut pending) = self.pending.take() else {
            return Ok(None);
        };
        let rows = pending.batch.num_rows();
        let mut end = rows;
        if let Some(prefixes) = &pending.prefixes {
            for leaf in prefixes {
                let start = leaf.bytes[pending.offset];
                let limit = start.saturating_add(self.max_bytes);
                let count = leaf.bytes[pending.offset + 1..].partition_point(|&b| b <= limit);
                if count == 0 {
                    return Err(general_err!(
                        "a single row needs {} bytes of {} data for column {}, which exceeds \
                         the {} byte limit of one {} array; read this column as a Large or \
                         View type instead",
                        leaf.bytes[pending.offset + 1] - start,
                        leaf.kind,
                        leaf.path,
                        self.max_bytes,
                        leaf.kind
                    ));
                }
                end = end.min(pending.offset + count);
            }
        }
        // Copy the data of each piece only when the window is split, so that
        // the outputs neither pin the window's buffers nor each report the
        // whole window through `get_array_memory_size`.
        let compact = pending.prefixes.is_some();
        let slice = pending.batch.slice(pending.offset, end - pending.offset);
        let output = restore_batch(&slice, self.target_schema.clone(), compact)?;
        pending.offset = end;
        if end < rows {
            self.pending = Some(pending);
        }
        Ok(Some(output))
    }
}

/// The type the array readers produce for `logical` when `Utf8`/`Binary`
/// leaves are staged. `ListView`/`LargeListView` subtrees are not staged.
pub(crate) fn staged_type(logical: &DataType) -> DataType {
    use DataType::*;
    let staged_field = |field: &FieldRef| {
        Arc::new(
            field
                .as_ref()
                .clone()
                .with_data_type(staged_type(field.data_type())),
        )
    };
    match logical {
        Utf8 => LargeUtf8,
        Binary => LargeBinary,
        Struct(fields) => Struct(fields.iter().map(staged_field).collect()),
        List(field) => List(staged_field(field)),
        LargeList(field) => LargeList(staged_field(field)),
        FixedSizeList(field, size) => FixedSizeList(staged_field(field), *size),
        Map(field, sorted) => Map(staged_field(field), *sorted),
        other => other.clone(),
    }
}

fn validate_batch(batch: &RecordBatch, target: &SchemaRef) -> Result<()> {
    if batch.num_columns() != target.fields().len() {
        return Err(general_err!(
            "Internal Error: staged batch has {} columns, expected {}",
            batch.num_columns(),
            target.fields().len()
        ));
    }
    for (from, to) in batch.schema().fields().iter().zip(target.fields()) {
        validate_field(from, to)?;
    }
    Ok(())
}

fn validate_field(from: &FieldRef, to: &FieldRef) -> Result<()> {
    if from.name() != to.name() {
        return Err(general_err!(
            "Internal Error: staged field {} does not match {}",
            from.name(),
            to.name()
        ));
    }
    validate_type(from.data_type(), to.data_type())
}

fn validate_type(from: &DataType, to: &DataType) -> Result<()> {
    use DataType::*;
    if from == to {
        return Ok(());
    }
    match (from, to) {
        (LargeUtf8, Utf8) | (LargeBinary, Binary) => Ok(()),
        (Struct(from), Struct(to)) if from.len() == to.len() => {
            for (from, to) in from.iter().zip(to) {
                validate_field(from, to)?;
            }
            Ok(())
        }
        (List(from), List(to)) | (LargeList(from), LargeList(to)) => validate_field(from, to),
        (FixedSizeList(from, n), FixedSizeList(to, m)) if n == m => validate_field(from, to),
        (Map(from, a), Map(to, b)) if a == b => validate_field(from, to),
        _ => Err(general_err!(
            "Internal Error: cannot restore staged {from} as {to}"
        )),
    }
}

/// Data bytes of rows `start..end` of a byte array, from its offsets.
fn span_bytes(array: &ArrayRef, start: usize, end: usize) -> Result<u64> {
    fn span<O: OffsetSizeTrait>(offsets: &[O], start: usize, end: usize) -> u64 {
        (offsets[end].as_usize() - offsets[start].as_usize()) as u64
    }
    Ok(match array.data_type() {
        DataType::Utf8 => span(array.as_string::<i32>().value_offsets(), start, end),
        DataType::Binary => span(array.as_binary::<i32>().value_offsets(), start, end),
        DataType::LargeUtf8 => span(array.as_string::<i64>().value_offsets(), start, end),
        DataType::LargeBinary => span(array.as_binary::<i64>().value_offsets(), start, end),
        other => {
            return Err(general_err!(
                "Internal Error: expected a staged byte array, got {other}"
            ));
        }
    })
}

/// Whether every `Utf8`/`Binary` leaf of rows `start..end` fits in `cap`.
fn span_fits(
    array: &ArrayRef,
    target: &DataType,
    start: usize,
    end: usize,
    cap: u64,
) -> Result<bool> {
    use DataType::*;
    match target {
        Utf8 | Binary => Ok(span_bytes(array, start, end)? <= cap),
        Struct(fields) => {
            let mut fits = true;
            for (child, field) in array.as_struct().columns().iter().zip(fields) {
                fits &= span_fits(child, field.data_type(), start, end, cap)?;
            }
            Ok(fits)
        }
        List(field) => {
            let list = array.as_list::<i32>();
            let offsets = list.value_offsets();
            span_fits(
                list.values(),
                field.data_type(),
                offsets[start].as_usize(),
                offsets[end].as_usize(),
                cap,
            )
        }
        LargeList(field) => {
            let list = array.as_list::<i64>();
            let offsets = list.value_offsets();
            span_fits(
                list.values(),
                field.data_type(),
                offsets[start].as_usize(),
                offsets[end].as_usize(),
                cap,
            )
        }
        FixedSizeList(field, size) => span_fits(
            array.as_fixed_size_list().values(),
            field.data_type(),
            start * *size as usize,
            end * *size as usize,
            cap,
        ),
        Map(field, _) => {
            let map = array.as_map();
            let offsets = map.value_offsets();
            let entries: ArrayRef = Arc::new(map.entries().clone());
            span_fits(
                &entries,
                field.data_type(),
                offsets[start].as_usize(),
                offsets[end].as_usize(),
                cap,
            )
        }
        // Everything else, including requested Large and View types,
        // dictionaries and unstaged ListView subtrees, has no 32-bit byte limit.
        _ => Ok(true),
    }
}

/// Cumulative data bytes of a byte array at each of `boundaries`, relative to
/// the first boundary.
fn byte_prefix(array: &ArrayRef, boundaries: &[usize]) -> Result<Vec<u64>> {
    fn prefix<O: OffsetSizeTrait>(offsets: &[O], boundaries: &[usize]) -> Vec<u64> {
        let base = offsets[boundaries[0]].as_usize();
        boundaries
            .iter()
            .map(|&row| (offsets[row].as_usize() - base) as u64)
            .collect()
    }
    Ok(match array.data_type() {
        DataType::Utf8 => prefix(array.as_string::<i32>().value_offsets(), boundaries),
        DataType::Binary => prefix(array.as_binary::<i32>().value_offsets(), boundaries),
        DataType::LargeUtf8 => prefix(array.as_string::<i64>().value_offsets(), boundaries),
        DataType::LargeBinary => prefix(array.as_binary::<i64>().value_offsets(), boundaries),
        other => {
            return Err(general_err!(
                "Internal Error: expected a staged byte array, got {other}"
            ));
        }
    })
}

/// Collect a per-row byte prefix for every `Utf8`/`Binary` leaf that does not
/// fit, mapping top-level row boundaries through list and map offsets.
fn collect_prefixes(
    array: &ArrayRef,
    target: &DataType,
    boundaries: &[usize],
    cap: u64,
    path: &str,
    prefixes: &mut Vec<LeafPrefix>,
) -> Result<()> {
    use DataType::*;
    match target {
        Utf8 | Binary => {
            let (start, end) = (boundaries[0], *boundaries.last().unwrap());
            if span_bytes(array, start, end)? <= cap {
                return Ok(());
            }
            prefixes.push(LeafPrefix {
                path: path.to_owned(),
                kind: if matches!(target, Utf8) {
                    "Utf8"
                } else {
                    "Binary"
                },
                bytes: byte_prefix(array, boundaries)?,
            });
        }
        Struct(fields) => {
            for (child, field) in array.as_struct().columns().iter().zip(fields) {
                collect_prefixes(
                    child,
                    field.data_type(),
                    boundaries,
                    cap,
                    &format!("{path}.{}", field.name()),
                    prefixes,
                )?;
            }
        }
        List(field) => {
            let list = array.as_list::<i32>();
            let mapped = map_boundaries(list.value_offsets(), boundaries);
            collect_prefixes(
                list.values(),
                field.data_type(),
                &mapped,
                cap,
                path,
                prefixes,
            )?;
        }
        LargeList(field) => {
            let list = array.as_list::<i64>();
            let mapped = map_boundaries(list.value_offsets(), boundaries);
            collect_prefixes(
                list.values(),
                field.data_type(),
                &mapped,
                cap,
                path,
                prefixes,
            )?;
        }
        FixedSizeList(field, size) => {
            let size = *size as usize;
            let mapped = boundaries.iter().map(|&row| row * size).collect::<Vec<_>>();
            collect_prefixes(
                array.as_fixed_size_list().values(),
                field.data_type(),
                &mapped,
                cap,
                path,
                prefixes,
            )?;
        }
        Map(field, _) => {
            let map = array.as_map();
            let mapped = map_boundaries(map.value_offsets(), boundaries);
            let entries: ArrayRef = Arc::new(map.entries().clone());
            collect_prefixes(&entries, field.data_type(), &mapped, cap, path, prefixes)?;
        }
        _ => {}
    }
    Ok(())
}

/// Map row boundaries of a list or map to boundaries of its child.
fn map_boundaries<O: OffsetSizeTrait>(offsets: &[O], boundaries: &[usize]) -> Vec<usize> {
    boundaries
        .iter()
        .map(|&row| offsets[row].as_usize())
        .collect()
}

fn restore_batch(batch: &RecordBatch, schema: SchemaRef, compact: bool) -> Result<RecordBatch> {
    let arrays = batch
        .columns()
        .iter()
        .zip(schema.fields())
        .map(|(array, field)| restore_array(array, field.data_type(), compact))
        .collect::<Result<Vec<_>>>()?;
    Ok(RecordBatch::try_new_with_options(
        schema,
        arrays,
        &RecordBatchOptions::new().with_row_count(Some(batch.num_rows())),
    )?)
}

/// Restore `array` to `target`. Lists and maps only slice their offsets, so
/// their child span is trimmed and offsets rebased before restoring the
/// children; otherwise rows outside this slice would also be converted.
fn restore_array(array: &ArrayRef, target: &DataType, compact: bool) -> Result<ArrayRef> {
    use DataType::*;
    if array.data_type() == target {
        return Ok(array.clone());
    }
    match target {
        Utf8 if array.data_type() == &LargeUtf8 => {
            narrow_large::<GenericStringType<i64>, GenericStringType<i32>>(array, compact)
        }
        Binary if array.data_type() == &LargeBinary => {
            narrow_large::<GenericBinaryType<i64>, GenericBinaryType<i32>>(array, compact)
        }
        Struct(fields) => {
            let source = array.as_struct();
            let children = source
                .columns()
                .iter()
                .zip(fields)
                .map(|(a, f)| restore_array(a, f.data_type(), compact))
                .collect::<Result<Vec<_>>>()?;
            Ok(Arc::new(StructArray::try_new_with_length(
                fields.clone(),
                children,
                source.nulls().cloned(),
                source.len(),
            )?))
        }
        List(field) => restore_list(array.as_list::<i32>(), field, compact),
        LargeList(field) => restore_list(array.as_list::<i64>(), field, compact),
        FixedSizeList(field, size) => {
            let source = array.as_fixed_size_list();
            let values = restore_array(source.values(), field.data_type(), compact)?;
            Ok(Arc::new(FixedSizeListArray::try_new_with_length(
                field.clone(),
                *size,
                values,
                source.nulls().cloned(),
                source.len(),
            )?))
        }
        Map(field, sorted) => {
            let source = array.as_map();
            let offsets = source.value_offsets();
            let start = offsets[0] as usize;
            let end = *offsets.last().unwrap() as usize;
            let entries: ArrayRef = Arc::new(source.entries().slice(start, end - start));
            let entries = restore_array(&entries, field.data_type(), compact)?;
            let offsets = OffsetBuffer::new(ScalarBuffer::from(
                offsets
                    .iter()
                    .map(|offset| *offset - start as i32)
                    .collect::<Vec<_>>(),
            ));
            Ok(Arc::new(MapArray::try_new(
                field.clone(),
                offsets,
                entries.as_struct().clone(),
                source.nulls().cloned(),
                *sorted,
            )?))
        }
        _ => Err(general_err!(
            "Internal Error: cannot restore staged {} as {target}",
            array.data_type()
        )),
    }
}

/// Convert a (possibly sliced) 64-bit offset byte array into its 32-bit form
/// by rebasing this slice's offsets. The values buffer is shared, or copied
/// when `compact` is set. Unlike arrow-cast's container cast this only touches
/// the slice's own offsets and does not require the whole backing buffer to
/// fit in 32 bits.
fn narrow_large<F, T>(array: &ArrayRef, compact: bool) -> Result<ArrayRef>
where
    F: ByteArrayType<Offset = i64>,
    T: ByteArrayType<Offset = i32, Native = F::Native>,
{
    let source = array.as_bytes::<F>();
    let offsets = source.value_offsets();
    let first = offsets[0];
    let span = offsets[offsets.len() - 1] - first;
    if span > i32::MAX as i64 {
        return Err(general_err!(
            "Internal Error: staged slice of {span} bytes exceeds 32-bit offsets"
        ));
    }
    let narrowed: Vec<i32> = offsets.iter().map(|o| (*o - first) as i32).collect();
    let (first, span) = (first as usize, span as usize);
    let values = if compact {
        Buffer::from(&source.values()[first..first + span])
    } else {
        source.values().slice_with_length(first, span)
    };
    // SAFETY: the offsets come from a valid array and are only rebased, so
    // they stay monotonic and within the re-sliced values. Each value's bytes,
    // and therefore UTF-8 validity for strings, are unchanged.
    let narrowed = unsafe {
        GenericByteArray::<T>::new_unchecked(
            OffsetBuffer::new_unchecked(ScalarBuffer::from(narrowed)),
            values,
            source.nulls().cloned(),
        )
    };
    Ok(Arc::new(narrowed))
}

fn restore_list<O: OffsetSizeTrait>(
    source: &GenericListArray<O>,
    field: &FieldRef,
    compact: bool,
) -> Result<ArrayRef> {
    let offsets = source.value_offsets();
    let start = offsets[0].as_usize();
    let end = offsets.last().unwrap().as_usize();
    let values = source.values().slice(start, end - start);
    let values = restore_array(&values, field.data_type(), compact)?;
    let offsets = if start == 0 {
        source.offsets().clone()
    } else {
        OffsetBuffer::new(ScalarBuffer::from(
            offsets
                .iter()
                .map(|offset| O::from_usize(offset.as_usize() - start).unwrap())
                .collect::<Vec<_>>(),
        ))
    };
    Ok(Arc::new(GenericListArray::<O>::try_new(
        field.clone(),
        offsets,
        values,
        source.nulls().cloned(),
    )?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::types::Int32Type;
    use arrow_array::{
        BinaryArray, DictionaryArray, Int32Array, LargeBinaryArray, LargeListArray,
        LargeStringArray, ListArray, StringArray, StringViewArray,
    };
    use arrow_buffer::NullBuffer;
    use arrow_schema::{Field, Fields, Schema};
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
        assert!(fits.pending.as_ref().unwrap().prefixes.is_none());
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
        let restored = restore_array(&source.slice(1, 2), &DataType::Binary, false).unwrap();
        let restored = restored.as_binary::<i32>();
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
        assert!(!splitter.has_pending());
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
}
