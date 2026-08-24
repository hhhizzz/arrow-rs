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

//! One-shot [`ArrayReader`] over already materialized arrays.

use crate::arrow::array_reader::ArrayReader;
use crate::errors::{ParquetError, Result};
use arrow_array::{ArrayRef, new_empty_array};
use arrow_schema::DataType;
use std::any::Any;
use std::collections::VecDeque;

/// An [`ArrayReader`] that returns already materialized arrays without
/// re-batching them.
///
/// This is deliberately small and currently used only by the experimental
/// fused predicate/output path. Each call to [`Self::read_records`] stages at
/// most one source array, even when the caller requested more rows. Returning
/// fewer than the requested maximum is permitted by [`ArrayReader`] and lets
/// the filtered predicate batch flow directly to the caller without another
/// concat pass.
pub(crate) struct BufferedArrayReader {
    data_type: DataType,
    batches: VecDeque<ArrayRef>,
    front_offset: usize,
    pending: Option<ArrayRef>,
}

impl BufferedArrayReader {
    pub(crate) fn new(data_type: DataType, batches: Vec<ArrayRef>) -> Self {
        Self {
            data_type,
            batches: batches.into(),
            front_offset: 0,
            pending: None,
        }
    }

    fn discard_empty_fronts(&mut self) {
        while self
            .batches
            .front()
            .is_some_and(|array| self.front_offset == array.len())
        {
            self.batches.pop_front();
            self.front_offset = 0;
        }
    }
}

impl ArrayReader for BufferedArrayReader {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn get_data_type(&self) -> &DataType {
        &self.data_type
    }

    fn read_records(&mut self, num_records: usize) -> Result<usize> {
        if self.pending.is_some() {
            return Err(ParquetError::General(
                "BufferedArrayReader read_records called before consume_batch".to_string(),
            ));
        }
        if num_records == 0 {
            return Ok(0);
        }

        self.discard_empty_fronts();
        let Some(front) = self.batches.front() else {
            return Ok(0);
        };

        let read = num_records.min(front.len() - self.front_offset);
        self.pending = Some(front.slice(self.front_offset, read));
        self.front_offset += read;
        Ok(read)
    }

    fn consume_batch(&mut self) -> Result<ArrayRef> {
        Ok(self
            .pending
            .take()
            .unwrap_or_else(|| new_empty_array(&self.data_type)))
    }

    fn skip_records(&mut self, num_records: usize) -> Result<usize> {
        let mut skipped = 0;
        while skipped < num_records {
            self.discard_empty_fronts();
            let Some(front) = self.batches.front() else {
                break;
            };

            let available = front.len() - self.front_offset;
            let to_skip = (num_records - skipped).min(available);
            self.front_offset += to_skip;
            skipped += to_skip;
        }
        Ok(skipped)
    }

    fn get_def_levels(&self) -> Option<&[i16]> {
        None
    }

    fn get_rep_levels(&self) -> Option<&[i16]> {
        None
    }

    fn max_def_level(&self) -> i16 {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{Array, Int32Array};
    use std::sync::Arc;

    #[test]
    fn preserves_source_batch_boundaries_and_order() {
        let batches: Vec<ArrayRef> = vec![
            Arc::new(Int32Array::from(vec![1, 3])),
            Arc::new(Int32Array::from(vec![5, 7, 9])),
        ];
        let mut reader = BufferedArrayReader::new(DataType::Int32, batches);

        assert_eq!(reader.read_records(8).unwrap(), 2);
        assert_eq!(reader.consume_batch().unwrap().len(), 2);
        assert_eq!(reader.read_records(8).unwrap(), 3);
        let batch = reader.consume_batch().unwrap();
        assert_eq!(batch.as_any().downcast_ref::<Int32Array>().unwrap().values(), &[5, 7, 9]);
        assert_eq!(reader.read_records(8).unwrap(), 0);
    }
}
