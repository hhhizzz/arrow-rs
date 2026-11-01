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

//! Restore top-level 32-bit byte arrays after decoding with 64-bit offsets.
//!
//! A fitting window shares its payload buffers. Oversized windows are split
//! at complete rows, copying staged payloads so they do not retain the whole
//! decoded payload. Other columns and null buffers may still share storage.
//! Nested columns retain their requested representation.

use crate::errors::{ParquetError, Result};
use arrow_array::cast::AsArray;
use arrow_array::types::{ByteArrayType, GenericBinaryType, GenericStringType};
use arrow_array::{Array, ArrayRef, GenericByteArray, RecordBatch, RecordBatchOptions};
use arrow_buffer::{Buffer, OffsetBuffer, ScalarBuffer};
use arrow_schema::{DataType, SchemaRef};
use std::sync::Arc;

/// Restores staged top-level columns and splits their payloads at `max_bytes`.
pub(crate) struct CapacitySplitter {
    schema: SchemaRef,
    max_bytes: i64,
    pending: Option<PendingBatch>,
}

struct PendingBatch {
    batch: RecordBatch,
    offset: usize,
    split: bool,
}

impl CapacitySplitter {
    pub(crate) fn new(schema: SchemaRef, max_bytes: usize) -> Self {
        Self {
            schema,
            max_bytes: max_bytes.min(i32::MAX as usize) as i64,
            pending: None,
        }
    }

    /// Accept a decoded window after the previous window has been drained.
    pub(crate) fn push(&mut self, batch: RecordBatch) -> Result<()> {
        if self.pending.is_some() {
            return Err(general_err!(
                "Internal Error: capacity splitter must be drained before push"
            ));
        }
        if batch.num_columns() != self.schema.fields().len() {
            return Err(general_err!(
                "Internal Error: staged batch has {} columns, expected {}",
                batch.num_columns(),
                self.schema.fields().len()
            ));
        }
        let mut split = false;
        for ((array, from), to) in batch
            .columns()
            .iter()
            .zip(batch.schema().fields())
            .zip(self.schema.fields())
        {
            let offsets = staged_offsets(array.as_ref(), to.data_type());
            if from.name() != to.name()
                || (array.data_type() != to.data_type() && offsets.is_none())
            {
                return Err(general_err!(
                    "Internal Error: cannot restore staged field {from} as {to}"
                ));
            }
            if let Some(offsets) = offsets {
                split |= offsets[offsets.len() - 1] - offsets[0] > self.max_bytes;
            }
        }
        self.pending = Some(PendingBatch {
            batch,
            offset: 0,
            split,
        });
        Ok(())
    }

    /// Emit the next batch. An error releases the pending window.
    pub(crate) fn next_batch(&mut self) -> Result<Option<RecordBatch>> {
        let Some(mut pending) = self.pending.take() else {
            return Ok(None);
        };
        let mut end = pending.batch.num_rows();
        if pending.split {
            for (array, field) in pending.batch.columns().iter().zip(self.schema.fields()) {
                let Some(offsets) = staged_offsets(array.as_ref(), field.data_type()) else {
                    continue;
                };
                let start = offsets[pending.offset];
                let limit = start.saturating_add(self.max_bytes);
                let count = offsets[pending.offset + 1..].partition_point(|&o| o <= limit);
                if count == 0 {
                    return Err(general_err!(
                        "a single row needs {} bytes of {} data for column {}, which exceeds \
                         the {} byte limit of one {} array; read this column as a Large or \
                         View type instead",
                        offsets[pending.offset + 1] - start,
                        field.data_type(),
                        field.name(),
                        self.max_bytes,
                        field.data_type()
                    ));
                }
                end = end.min(pending.offset + count);
            }
        }
        let slice = pending.batch.slice(pending.offset, end - pending.offset);
        let columns = slice
            .columns()
            .iter()
            .zip(self.schema.fields())
            .map(
                |(array, field)| match (array.data_type(), field.data_type()) {
                    (DataType::LargeUtf8, DataType::Utf8) => {
                        narrow::<GenericStringType<i64>, GenericStringType<i32>>(
                            array,
                            pending.split,
                        )
                    }
                    (DataType::LargeBinary, DataType::Binary) => {
                        narrow::<GenericBinaryType<i64>, GenericBinaryType<i32>>(
                            array,
                            pending.split,
                        )
                    }
                    _ => Ok(array.clone()),
                },
            )
            .collect::<Result<Vec<_>>>()?;
        let output = RecordBatch::try_new_with_options(
            self.schema.clone(),
            columns,
            &RecordBatchOptions::new().with_row_count(Some(slice.num_rows())),
        )?;
        pending.offset = end;
        if end < pending.batch.num_rows() {
            self.pending = Some(pending);
        }
        Ok(Some(output))
    }
}

/// The reader root's type after staging only its direct byte-array fields.
pub(crate) fn staged_type(logical: &DataType) -> DataType {
    let DataType::Struct(fields) = logical else {
        return logical.clone();
    };
    DataType::Struct(
        fields
            .iter()
            .map(|field| {
                let data_type = match field.data_type() {
                    DataType::Utf8 => DataType::LargeUtf8,
                    DataType::Binary => DataType::LargeBinary,
                    _ => return field.clone(),
                };
                Arc::new(field.as_ref().clone().with_data_type(data_type))
            })
            .collect(),
    )
}

fn staged_offsets<'a>(array: &'a dyn Array, target: &DataType) -> Option<&'a [i64]> {
    match (array.data_type(), target) {
        (DataType::LargeUtf8, DataType::Utf8) => Some(array.as_string::<i64>().value_offsets()),
        (DataType::LargeBinary, DataType::Binary) => Some(array.as_binary::<i64>().value_offsets()),
        _ => None,
    }
}

/// Rebase and narrow only the visible slice's offsets, preserving its validity.
fn narrow<F, T>(array: &ArrayRef, copy_values: bool) -> Result<ArrayRef>
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
    let narrowed = offsets
        .iter()
        .map(|o| (*o - first) as i32)
        .collect::<Vec<_>>();
    let (first, span) = (first as usize, span as usize);
    let values = if copy_values {
        Buffer::from(&source.values()[first..first + span])
    } else {
        source.values().slice_with_length(first, span)
    };
    // SAFETY: valid source offsets are rebased and fit in i32. The sliced
    // payload and unchanged validity preserve each value, including UTF-8.
    Ok(Arc::new(unsafe {
        GenericByteArray::<T>::new_unchecked(
            OffsetBuffer::new_unchecked(ScalarBuffer::from(narrowed)),
            values,
            source.nulls().cloned(),
        )
    }))
}

#[cfg(test)]
#[path = "capacity_staging_unit_tests.rs"]
mod tests;
