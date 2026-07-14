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

use crate::arrow::arrow_reader::RowSelection;
use crate::arrow::push_decoder::{RowGroupReaderResult, reader_builder::RowGroupReaderBuilder};
use crate::errors::ParquetError;
use crate::file::metadata::ParquetMetaData;
use bytes::Bytes;
use std::collections::VecDeque;
use std::ops::Range;
use std::sync::Arc;

/// State machine that tracks the remaining high level chunks (row groups) of
/// Parquet data are left to read.
///
/// This is currently a row group, but the author aspires to extend the pattern
/// to data boundaries other than RowGroups in the future.
#[derive(Debug)]
pub(crate) struct RemainingRowGroups {
    /// The underlying Parquet metadata
    parquet_metadata: Arc<ParquetMetaData>,

    /// The row groups that have not yet been read
    row_groups: VecDeque<usize>,

    /// The row group currently being built. It stays set across all
    /// `NeedsData` phases and is cleared only when that row group completes.
    active_row_group: Option<usize>,

    /// Remaining selection to apply to the next row groups
    selection: Option<RowSelection>,

    /// State for building the reader for the current row group
    row_group_reader_builder: RowGroupReaderBuilder,
}

impl RemainingRowGroups {
    pub fn new(
        parquet_metadata: Arc<ParquetMetaData>,
        row_groups: Vec<usize>,
        selection: Option<RowSelection>,
        row_group_reader_builder: RowGroupReaderBuilder,
    ) -> Self {
        Self {
            parquet_metadata,
            row_groups: VecDeque::from(row_groups),
            active_row_group: None,
            selection,
            row_group_reader_builder,
        }
    }

    /// Push new data buffers that can be used to satisfy pending requests
    pub fn push_data(&mut self, ranges: Vec<Range<u64>>, buffers: Vec<Bytes>) {
        self.row_group_reader_builder.push_data(ranges, buffers);
    }

    /// Return the total number of bytes buffered so far
    pub fn buffered_bytes(&self) -> u64 {
        self.row_group_reader_builder.buffered_bytes()
    }

    /// Clear any staged ranges currently buffered for future decode work
    pub fn clear_all_ranges(&mut self) {
        self.row_group_reader_builder.clear_all_ranges();
    }

    /// Returns the next queued row group with selected rows without advancing state.
    pub fn peek_next_row_group(&self) -> Result<Option<usize>, ParquetError> {
        let mut selection = self.selection.clone();

        for &row_group_idx in &self.row_groups {
            let row_group = self
                .parquet_metadata
                .row_groups()
                .get(row_group_idx)
                .ok_or_else(|| {
                    ParquetError::General(format!(
                        "Invalid row group index {row_group_idx}; file contains {} row groups",
                        self.parquet_metadata.num_row_groups()
                    ))
                })?;
            let row_count: usize = row_group
                .num_rows()
                .try_into()
                .map_err(|e| ParquetError::General(format!("Row count overflow: {e}")))?;

            let selected_rows = selection
                .as_mut()
                .map(|selection| selection.split_off(row_count).row_count())
                .unwrap_or(row_count);
            if selected_rows != 0 {
                return Ok(Some(row_group_idx));
            }
        }

        Ok(None)
    }

    /// returns [`ParquetRecordBatchReader`] suitable for reading the next
    /// group of rows from the Parquet data, or the list of data ranges still
    /// needed to proceed
    pub fn try_next_reader_with_row_group(&mut self) -> Result<RowGroupReaderResult, ParquetError> {
        loop {
            // Are we ready yet to start reading?
            let result = self.row_group_reader_builder.try_build()?;
            match result {
                crate::DecodeResult::Finished => {
                    // reader is done, proceed to the next row group
                    // fall through to the next row group
                    // This happens if the row group was completely filtered out
                    self.active_row_group = None;
                }
                crate::DecodeResult::NeedsData(ranges) => {
                    // need more data to proceed
                    let row_group_index = self.active_row_group.ok_or_else(|| {
                        ParquetError::General(
                            "Internal Error: missing active row group for data request".to_string(),
                        )
                    })?;
                    return Ok(RowGroupReaderResult::NeedsData {
                        row_group_index,
                        ranges,
                    });
                }
                crate::DecodeResult::Data(batch_reader) => {
                    // ready to read the row group
                    let row_group_index = self.active_row_group.take().ok_or_else(|| {
                        ParquetError::General(
                            "Internal Error: missing active row group for reader".to_string(),
                        )
                    })?;
                    return Ok(RowGroupReaderResult::Data {
                        row_group_index,
                        data: batch_reader,
                    });
                }
            }

            // No current reader, proceed to the next row group if any
            let row_group_idx = match self.row_groups.pop_front() {
                None => return Ok(RowGroupReaderResult::Finished),
                Some(idx) => idx,
            };

            let row_count: usize = self
                .parquet_metadata
                .row_group(row_group_idx)
                .num_rows()
                .try_into()
                .map_err(|e| ParquetError::General(format!("Row count overflow: {e}")))?;

            let selection = self.selection.as_mut().map(|s| s.split_off(row_count));
            self.row_group_reader_builder
                .next_row_group(row_group_idx, row_count, selection)?;
            self.active_row_group = Some(row_group_idx);
            // the next iteration will try to build the reader for the new row group
        }
    }
}
