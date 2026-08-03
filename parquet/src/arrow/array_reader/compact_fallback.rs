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

use std::any::Any;

use arrow_array::{Array, ArrayRef, BooleanArray, new_empty_array};
use arrow_buffer::BooleanBuffer;
use arrow_schema::DataType as ArrowType;

use super::{ArrayReader, EncodedSelectionSupport};
use crate::errors::{ParquetError, Result};

/// Adapts an unsupported flat leaf to the compact-output contract by fully
/// decoding its logical rows and filtering only that leaf.
///
/// This is intentionally a fallback child, not a root admission strategy. It
/// lets a native sibling keep the root Struct in the selected lane without
/// claiming that an all-fallback projection should filter each leaf.
pub(crate) struct CompactFallbackArrayReader {
    inner: Box<dyn ArrayReader>,
    selected_chunks: Vec<ArrayRef>,
    pending_def_levels: Option<Vec<i16>>,
    def_levels_buffer: Option<Vec<i16>>,
    pending_mode: PendingMode,
    consumed_selected: bool,
}

#[derive(Debug, Clone, Copy, Default, Eq, PartialEq)]
enum PendingMode {
    #[default]
    Empty,
    Full,
    Selected,
}

impl CompactFallbackArrayReader {
    pub(crate) fn new(inner: Box<dyn ArrayReader>) -> Self {
        Self {
            inner,
            selected_chunks: Vec::new(),
            pending_def_levels: None,
            def_levels_buffer: None,
            pending_mode: PendingMode::Empty,
            consumed_selected: false,
        }
    }
}

impl ArrayReader for CompactFallbackArrayReader {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn get_data_type(&self) -> &ArrowType {
        self.inner.get_data_type()
    }

    fn read_records(&mut self, batch_size: usize) -> Result<usize> {
        if self.pending_mode == PendingMode::Selected {
            return Err(general_err!(
                "cannot mix full and compact selected reads before consume_batch"
            ));
        }
        let read = self.inner.read_records(batch_size)?;
        if read != 0 {
            self.pending_mode = PendingMode::Full;
        }
        Ok(read)
    }

    fn encoded_selection_support(&self) -> EncodedSelectionSupport {
        EncodedSelectionSupport::Fallback
    }

    fn read_records_selected(&mut self, selection: &BooleanBuffer) -> Result<usize> {
        if self.pending_mode == PendingMode::Full {
            return Err(general_err!(
                "cannot mix full and compact selected reads before consume_batch"
            ));
        }
        if self.inner.max_def_level() > 1 {
            return Err(general_err!(
                "compact fallback only supports flat readers with max definition level <= 1"
            ));
        }

        self.pending_mode = PendingMode::Selected;
        let read = self.inner.read_records(selection.len())?;
        if read == 0 {
            return Ok(0);
        }

        let decoded = self.inner.consume_batch()?;
        if decoded.len() != read {
            return Err(general_err!(
                "compact fallback decoded {read} logical rows into {} array rows",
                decoded.len()
            ));
        }
        if self.inner.get_rep_levels().is_some() {
            return Err(general_err!(
                "compact fallback does not support repeated readers"
            ));
        }

        let consumed = selection.slice(0, read);
        let selected = consumed.count_set_bits();
        if let Some(levels) = self.inner.get_def_levels() {
            if levels.len() != read {
                return Err(general_err!(
                    "compact fallback decoded {read} logical rows with {} definition levels",
                    levels.len()
                ));
            }
            let compact_levels = self.pending_def_levels.get_or_insert_default();
            compact_levels.reserve(selected);
            compact_levels.extend(
                levels
                    .iter()
                    .enumerate()
                    .filter_map(|(idx, level)| consumed.value(idx).then_some(*level)),
            );
        }

        let filtered = arrow_select::filter::filter(
            decoded.as_ref(),
            &BooleanArray::from(consumed),
        )?;
        if filtered.len() != selected {
            return Err(general_err!(
                "compact fallback emitted {} rows, expected {selected}",
                filtered.len()
            ));
        }
        self.selected_chunks.push(filtered);
        Ok(read)
    }

    fn consume_batch(&mut self) -> Result<ArrayRef> {
        match std::mem::take(&mut self.pending_mode) {
            PendingMode::Empty | PendingMode::Full => {
                self.consumed_selected = false;
                self.def_levels_buffer = None;
                self.inner.consume_batch()
            }
            PendingMode::Selected => {
                self.consumed_selected = true;
                self.def_levels_buffer = self.pending_def_levels.take();
                let chunks = std::mem::take(&mut self.selected_chunks);
                match chunks.len() {
                    0 => Ok(new_empty_array(self.inner.get_data_type())),
                    1 => Ok(chunks.into_iter().next().unwrap()),
                    _ => Ok(arrow_select::concat::concat(
                        &chunks
                            .iter()
                            .map(|chunk| chunk.as_ref())
                            .collect::<Vec<_>>(),
                    )?),
                }
            }
        }
    }

    fn skip_records(&mut self, num_records: usize) -> Result<usize> {
        self.inner.skip_records(num_records)
    }

    fn get_def_levels(&self) -> Option<&[i16]> {
        if self.consumed_selected {
            self.def_levels_buffer.as_deref()
        } else {
            self.inner.get_def_levels()
        }
    }

    fn get_rep_levels(&self) -> Option<&[i16]> {
        if self.consumed_selected {
            None
        } else {
            self.inner.get_rep_levels()
        }
    }

    fn max_def_level(&self) -> i16 {
        self.inner.max_def_level()
    }
}
