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

//! [`CachedArrayReader`] wrapper around [`ArrayReader`]

use crate::arrow::array_reader::row_group_cache::BatchID;
use crate::arrow::array_reader::{ArrayReader, row_group_cache::RowGroupCache};
use crate::arrow::arrow_reader::metrics::ArrowReaderMetrics;
use crate::errors::Result;
use arrow_array::cast::AsArray;
use arrow_array::types::ByteArrayType;
use arrow_array::{Array, ArrayRef, BooleanArray, GenericByteArray, new_empty_array};
use arrow_buffer::{ArrowNativeType, BooleanBufferBuilder};
use arrow_schema::DataType as ArrowType;
use std::any::Any;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// Limit for sharing a partial byte-array slice instead of compacting it. This
/// bounds that slice's Arrow footprint relative to its compact representation,
/// not the reader's total memory or RSS. Nearly full slices avoid an extra copy.
const MAX_BYTE_SLICE_AMPLIFICATION: usize = 2;

/// Whether an all-selected partial grid should be filtered against the whole
/// grid to copy its selected values. None leaves other types' policy unchanged.
fn byte_slice_copy_policy(array: &dyn Array, offset: usize, rows: usize) -> Option<bool> {
    match array.data_type() {
        ArrowType::Utf8 => Some(byte_slice_needs_copy(
            array.as_string::<i32>(),
            offset,
            rows,
        )),
        ArrowType::Binary => Some(byte_slice_needs_copy(
            array.as_binary::<i32>(),
            offset,
            rows,
        )),
        ArrowType::LargeUtf8 => Some(byte_slice_needs_copy(
            array.as_string::<i64>(),
            offset,
            rows,
        )),
        ArrowType::LargeBinary => Some(byte_slice_needs_copy(
            array.as_binary::<i64>(),
            offset,
            rows,
        )),
        _ => None,
    }
}

fn byte_slice_needs_copy<T: ByteArrayType>(
    array: &GenericByteArray<T>,
    offset: usize,
    rows: usize,
) -> bool {
    let offsets_capacity = array.offsets().inner().inner().capacity();
    let values_capacity = array.values().capacity();
    let nulls_capacity = array.nulls().map(|nulls| nulls.buffer().capacity());
    // Externally owned buffers can report zero capacity. Conservatively copy
    // when any backing allocation's size is unknown, including empty buffers.
    if offsets_capacity == 0 || values_capacity == 0 || nulls_capacity == Some(0) {
        return true;
    }

    let compact_size = (|| {
        let end = offset.checked_add(rows)?;
        let offsets = array.value_offsets();
        let start_bytes = offsets.get(offset)?.to_usize()?;
        let end_bytes = offsets.get(end)?.to_usize()?;
        let values_size = end_bytes.checked_sub(start_bytes)?;
        let offsets_size = rows
            .checked_add(1)?
            .checked_mul(std::mem::size_of::<T::Offset>())?;
        let nulls_size = if nulls_capacity.is_some() {
            rows.div_ceil(8)
        } else {
            0
        };
        std::mem::size_of::<GenericByteArray<T>>()
            .checked_add(values_size)?
            .checked_add(offsets_size)?
            .checked_add(nulls_size)
    })();
    let Some(compact_size) = compact_size else {
        return true;
    };

    array.get_array_memory_size() > compact_size.saturating_mul(MAX_BYTE_SLICE_AMPLIFICATION)
}

/// An all-selected partial grid kept as a zero-copy slice although it would
/// retain too much of its raw grid if returned alone. `consume_batch`'s concat
/// already copies every piece of a multi-piece output, so compacting such a
/// piece when its grid completes would copy it twice.
struct DeferredCompaction {
    /// Index of the piece in `completed_selected`
    piece: usize,
    /// The raw grid the piece is sliced from
    grid: ArrayRef,
    offset: usize,
    rows: usize,
}

/// Copies `rows` rows of `grid` starting at `offset` into compact storage.
fn compact_grid_slice(grid: &ArrayRef, offset: usize, rows: usize) -> Result<ArrayRef> {
    // Filtering an all-selected slice takes Arrow's zero-copy All path, so
    // mark the excluded grid prefix/suffix as false to copy only these rows.
    let mut mask = BooleanBufferBuilder::new(grid.len());
    mask.append_n(offset, false);
    mask.append_n(rows, true);
    mask.append_n(grid.len() - offset - rows, false);
    Ok(arrow_select::filter::filter(
        grid,
        &BooleanArray::from(mask.finish()),
    )?)
}

/// Role of the cached array reader
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheRole {
    /// Producer role: inserts data into the cache during filter phase
    Producer,
    /// Consumer role: removes consumed data from the cache during output building phase
    Consumer,
}

/// A cached wrapper around an ArrayReader that avoids duplicate decoding
/// when the same column appears in both filter predicates and output projection.
///
/// This reader acts as a transparent layer over the inner reader, using a cache
/// to avoid redundant work when the same data is needed multiple times.
///
/// The reader can operate in two roles:
/// - Producer: During filter phase, inserts decoded data into the cache
/// - Consumer: During output building, consumes and removes data from the cache
///
/// This means the memory consumption of the cache has two stages:
/// 1. During the filter phase, the memory increases as the cache is populated
/// 2. It peaks when filters are built.
/// 3. It decreases as the cached data is consumed.
///
/// ```text
///    ▲
///    │     ╭─╮
///    │    ╱   ╲
///    │   ╱     ╲
///    │  ╱       ╲
///    │ ╱         ╲
///    │╱           ╲
///    └─────────────╲──────► Time
///    │      │      │
///    Filter  Peak  Consume
///    Phase (Built) (Decrease)
/// ```
pub struct CachedArrayReader {
    /// The underlying array reader
    inner: Box<dyn ArrayReader>,
    /// Shared cache for this row group
    shared_cache: Arc<RwLock<RowGroupCache>>,
    /// Column index for cache key generation
    column_idx: usize,
    /// Current logical position in the data stream for this reader (for cache key generation)
    outer_position: usize,
    /// Current position in `inner`
    inner_position: usize,
    /// Batch size for the cache
    batch_size: usize,
    /// Reads and intervening skips within the current grid. Once that grid is
    /// crossed, its selected values move to completed_selected and these bits
    /// are reset. Skips with no current-grid reads need no bits.
    selections: BooleanBufferBuilder,
    /// Filtered pieces from completed grids, in logical row order. Keeping
    /// selected values rather than raw grids bounds sparse Selectors reads.
    /// Shared dictionary/View backing storage retains its existing semantics.
    completed_selected: Vec<ArrayRef>,
    /// The one piece in `completed_selected` that may still share its raw
    /// grid. A later piece needing compaction compacts it first, so a read
    /// crossing many grids retains at most one extra raw grid.
    deferred: Option<DeferredCompaction>,
    /// Role of this reader (Producer or Consumer)
    role: CacheRole,
    /// The raw current grid, including any unread tail. Completed grids are
    /// filtered before releasing these references, independently of shared-cache
    /// rejection or eviction, except for the `deferred` piece's grid. Skips
    /// release grids wholly behind their position.
    local_cache: HashMap<BatchID, ArrayRef>,
    /// Statistics to report on the Cache behavior
    metrics: ArrowReaderMetrics,
    /// Exclusive upper bound of the batch ids already removed from the shared
    /// cache by [`Self::cleanup_consumed_batches`].
    cleaned_up_to: usize,
}

impl CachedArrayReader {
    /// Creates a new cached array reader with the specified role
    pub fn new(
        inner: Box<dyn ArrayReader>,
        cache: Arc<RwLock<RowGroupCache>>,
        column_idx: usize,
        role: CacheRole,
        metrics: ArrowReaderMetrics,
    ) -> Self {
        let batch_size = cache.read().unwrap().batch_size();

        Self {
            inner,
            shared_cache: cache,
            column_idx,
            outer_position: 0,
            inner_position: 0,
            batch_size,
            selections: BooleanBufferBuilder::new(0),
            completed_selected: Vec::new(),
            deferred: None,
            role,
            local_cache: HashMap::new(),
            metrics,
            cleaned_up_to: 0,
        }
    }

    fn get_batch_id_from_position(&self, row_id: usize) -> BatchID {
        BatchID {
            val: row_id / self.batch_size,
        }
    }

    /// Loads the batch with the given ID (first row offset) from the inner
    /// reader
    ///
    /// After this call the required batch will be available in
    /// `self.local_cache` and may also be stored in `self.shared_cache`.
    ///
    fn fetch_batch(&mut self, batch_id: BatchID) -> Result<usize> {
        let first_row_offset = batch_id.val * self.batch_size;
        if self.inner_position < first_row_offset {
            let to_skip = first_row_offset - self.inner_position;
            let skipped = self.inner.skip_records(to_skip)?;
            assert_eq!(skipped, to_skip);
            self.inner_position += skipped;
        }

        // For sparse mask reads, this full-batch fallback relies on `MaskCursor`
        // ending every chunk at a selected row. Predicate fetch expands cached
        // columns to batch boundaries, so the batch containing that row is loaded.
        let read = self.inner.read_records(self.batch_size)?;

        // If there are no remaining records (EOF), return immediately without
        // attempting to cache an empty batch. This prevents inserting zero-length
        // arrays into the cache which can later cause panics when slicing.
        if read == 0 {
            return Ok(0);
        }

        let array = self.inner.consume_batch()?;

        // Store in both shared cache and local cache
        // The shared cache is used to reuse results between readers
        // The local cache ensures data is available for our consume_batch call
        let _cached =
            self.shared_cache
                .write()
                .unwrap()
                .insert(self.column_idx, batch_id, array.clone());
        // Note: if the shared cache is full (_cached == false), we continue without caching
        // The local cache will still store the data for this reader's use

        self.local_cache.insert(batch_id, array);

        self.inner_position += read;
        Ok(read)
    }

    /// Remove batches from cache that have been completely consumed
    /// This is only called for Consumer role readers
    fn cleanup_consumed_batches(&mut self) {
        let current_batch_id = self.get_batch_id_from_position(self.outer_position);
        // `outer_position` only moves forward, so the watermark can never run
        // ahead of the batch the reader is currently on.
        debug_assert!(current_batch_id.val >= self.cleaned_up_to);

        // Remove batches that are at least one batch behind the current position
        // This ensures we don't remove batches that might still be needed for the current batch
        // We can safely remove batch_id if current_batch_id > batch_id + 1
        if current_batch_id.val <= 1 {
            return;
        }
        let end = current_batch_id.val - 1;
        // Everything below `cleaned_up_to` was removed by an earlier call.
        if end <= self.cleaned_up_to {
            return;
        }
        let start = self.cleaned_up_to;
        self.cleaned_up_to = end;
        let mut cache = self.shared_cache.write().unwrap();
        for val in start..end {
            cache.remove(self.column_idx, BatchID { val });
        }
    }

    /// Commit the selected part of the current grid before advancing to another
    /// one. This is the same per-grid slice/filter previously done by
    /// consume_batch, but sparse callers no longer retain every raw grid until
    /// they have accumulated a full output batch; at most the deferred piece's
    /// grid stays alive.
    fn finish_current_grid(&mut self) -> Result<()> {
        let rows = self.selections.len();
        if rows == 0 {
            return Ok(());
        }
        let start = self.outer_position - rows;
        let batch_id = self.get_batch_id_from_position(start);
        let offset = start % self.batch_size;
        debug_assert!(rows <= self.batch_size - offset);
        let mask = self.selections.finish();
        let selected_rows = mask.count_set_bits();
        if selected_rows != 0 {
            let cached = Arc::clone(
                self.local_cache
                    .get(&batch_id)
                    .expect("selected current grid must already be locally cached"),
            );
            let completed_grid = self.outer_position.is_multiple_of(self.batch_size);
            if selected_rows == rows
                && (offset != 0 || rows != cached.len())
                && (completed_grid || self.completed_selected.is_empty())
                && byte_slice_copy_policy(cached.as_ref(), offset, rows).unwrap_or(completed_grid)
            {
                // An all-selected slice would retain too much raw backing
                // storage if returned alone. Defer its compaction: when more
                // pieces follow, consume_batch's concat copies it anyway.
                // Nearly full byte-array slices and whole-grid reads skip this.
                self.compact_deferred()?;
                self.deferred = Some(DeferredCompaction {
                    piece: self.completed_selected.len(),
                    grid: Arc::clone(&cached),
                    offset,
                    rows,
                });
                self.completed_selected.push(cached.slice(offset, rows));
            } else {
                // Nearly full byte-array slices can share their backing array.
                // An unfinished tail after earlier pieces will be compacted by
                // consume_batch's concat, so copying it here would do so twice.
                let slice = cached.slice(offset, rows);
                self.completed_selected.push(arrow_select::filter::filter(
                    &slice,
                    &BooleanArray::from(mask),
                )?);
            }
        }
        let current_batch_id = self.get_batch_id_from_position(self.outer_position);
        self.local_cache
            .retain(|id, _| id.val >= current_batch_id.val);
        Ok(())
    }

    /// Replace the deferred piece, if any, with a compact copy, releasing its
    /// raw grid.
    fn compact_deferred(&mut self) -> Result<()> {
        if let Some(deferred) = self.deferred.take() {
            self.completed_selected[deferred.piece] =
                compact_grid_slice(&deferred.grid, deferred.offset, deferred.rows)?;
        }
        Ok(())
    }
}

impl ArrayReader for CachedArrayReader {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn get_data_type(&self) -> &ArrowType {
        self.inner.get_data_type()
    }

    fn read_records(&mut self, num_records: usize) -> Result<usize> {
        let mut read = 0;
        while read < num_records {
            let batch_id = self.get_batch_id_from_position(self.outer_position);

            // Check local cache first
            let cached = if let Some(array) = self.local_cache.get(&batch_id) {
                Some(Arc::clone(array))
            } else {
                // If not in local cache, i.e., we are consumer, check shared cache
                let cache_content = self
                    .shared_cache
                    .read()
                    .unwrap()
                    .get(self.column_idx, batch_id);
                if let Some(array) = cache_content.as_ref() {
                    // Store in local cache for later use in consume_batch
                    self.local_cache.insert(batch_id, Arc::clone(array));
                }
                cache_content
            };

            match cached {
                Some(array) => {
                    let array_len = array.len();
                    if array_len + batch_id.val * self.batch_size > self.outer_position {
                        // the cache batch has some records that we can select
                        let v = array_len + batch_id.val * self.batch_size - self.outer_position;
                        let select_cnt = std::cmp::min(num_records - read, v);
                        read += select_cnt;
                        self.metrics.increment_cache_reads(select_cnt);
                        self.outer_position += select_cnt;
                        self.selections.append_n(select_cnt, true);
                    } else {
                        // this is last batch and we have used all records from it
                        break;
                    }
                }
                None => {
                    let read_from_inner = self.fetch_batch(batch_id)?;
                    // Reached end-of-file, no more records to read
                    if read_from_inner == 0 {
                        break;
                    }
                    self.metrics.increment_inner_reads(read_from_inner);
                    let select_from_this_batch = std::cmp::min(
                        num_records - read,
                        self.inner_position - self.outer_position,
                    );
                    read += select_from_this_batch;
                    self.outer_position += select_from_this_batch;
                    self.selections.append_n(select_from_this_batch, true);
                    if read_from_inner < self.batch_size {
                        // this is last batch from inner reader
                        break;
                    }
                }
            }
            if self.outer_position.is_multiple_of(self.batch_size) {
                self.finish_current_grid()?;
            }
        }
        Ok(read)
    }

    fn skip_records(&mut self, num_records: usize) -> Result<usize> {
        let mut skipped = 0;
        while skipped < num_records {
            if self.selections.is_empty() {
                // Earlier grids already have selected pieces; at most the
                // deferred one still shares its raw grid. Whole skipped grids
                // need no mask, fetch, or local reference.
                self.outer_position += num_records - skipped;
                let current_batch_id = self.get_batch_id_from_position(self.outer_position);
                self.local_cache
                    .retain(|id, _| id.val >= current_batch_id.val);
                // Keep inner_position: a full-grid fetch can already be ahead
                // of outer_position; fetch_batch advances it only when needed.
                return Ok(num_records);
            }
            let to_boundary = self.batch_size - self.outer_position % self.batch_size;
            let size = std::cmp::min(num_records - skipped, to_boundary);
            skipped += size;
            self.selections.append_n(size, false);
            self.outer_position += size;
            if self.outer_position.is_multiple_of(self.batch_size) {
                self.finish_current_grid()?;
            }
        }
        Ok(num_records)
    }

    fn consume_batch(&mut self) -> Result<ArrayRef> {
        if self.selections.is_empty() && self.completed_selected.is_empty() {
            return Ok(new_empty_array(self.inner.get_data_type()));
        }
        self.finish_current_grid()?;
        // concat copies every piece of a multi-piece output, so a deferred
        // piece needs its own copy only when it is returned alone.
        if self.completed_selected.len() == 1 {
            self.compact_deferred()?;
        }
        self.deferred = None;
        let selected_arrays = std::mem::take(&mut self.completed_selected);

        // For consumers, cleanup batches that have been completely consumed
        // This reduces the memory usage of the shared cache
        if self.role == CacheRole::Consumer {
            self.cleanup_consumed_batches();
        }

        match selected_arrays.len() {
            0 => Ok(new_empty_array(self.inner.get_data_type())),
            1 => Ok(selected_arrays.into_iter().next().unwrap()),
            _ => Ok(arrow_select::concat::concat(
                &selected_arrays
                    .iter()
                    .map(|a| a.as_ref())
                    .collect::<Vec<_>>(),
            )?),
        }
    }

    fn get_def_levels(&self) -> Option<&[i16]> {
        None // we don't allow nullable parent for now.
    }

    fn get_rep_levels(&self) -> Option<&[i16]> {
        None
    }

    fn max_def_level(&self) -> i16 {
        self.inner.max_def_level()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arrow::array_reader::ArrayReader;
    use crate::arrow::array_reader::row_group_cache::RowGroupCache;
    use arrow_array::{ArrayRef, Int32Array};
    use std::sync::{Arc, RwLock};

    // Mock ArrayReader for testing
    struct MockArrayReader {
        data: Vec<i32>,
        position: usize,
        records_to_consume: usize,
        data_type: ArrowType,
    }

    impl MockArrayReader {
        fn new(data: Vec<i32>) -> Self {
            Self {
                data,
                position: 0,
                records_to_consume: 0,
                data_type: ArrowType::Int32,
            }
        }
    }

    impl ArrayReader for MockArrayReader {
        fn as_any(&self) -> &dyn Any {
            self
        }

        fn get_data_type(&self) -> &ArrowType {
            &self.data_type
        }

        fn read_records(&mut self, batch_size: usize) -> Result<usize> {
            let remaining = self.data.len() - self.position;
            let to_read = std::cmp::min(batch_size, remaining);
            self.records_to_consume += to_read;
            Ok(to_read)
        }

        fn consume_batch(&mut self) -> Result<ArrayRef> {
            let start = self.position;
            let end = start + self.records_to_consume;
            let slice = &self.data[start..end];
            self.position = end;
            self.records_to_consume = 0;
            Ok(Arc::new(Int32Array::from(slice.to_vec())))
        }

        fn skip_records(&mut self, num_records: usize) -> Result<usize> {
            let remaining = self.data.len() - self.position;
            let to_skip = std::cmp::min(num_records, remaining);
            self.position += to_skip;
            Ok(to_skip)
        }

        fn get_def_levels(&self) -> Option<&[i16]> {
            None
        }

        fn get_rep_levels(&self) -> Option<&[i16]> {
            None
        }
    }

    #[test]
    fn test_cached_reader_basic() {
        let metrics = ArrowReaderMetrics::disabled();
        let mock_reader = MockArrayReader::new(vec![1, 2, 3, 4, 5]);
        let cache = Arc::new(RwLock::new(RowGroupCache::new(3, usize::MAX))); // Batch size 3
        let mut cached_reader = CachedArrayReader::new(
            Box::new(mock_reader),
            cache,
            0,
            CacheRole::Producer,
            metrics,
        );

        // Read 3 records
        let records_read = cached_reader.read_records(3).unwrap();
        assert_eq!(records_read, 3);

        let array = cached_reader.consume_batch().unwrap();
        assert_eq!(array.len(), 3);

        let int32_array = array.as_any().downcast_ref::<Int32Array>().unwrap();
        assert_eq!(int32_array.values(), &[1, 2, 3]);

        // Read 3 more records
        let records_read = cached_reader.read_records(3).unwrap();
        assert_eq!(records_read, 2);
    }

    #[test]
    fn test_read_skip_pattern() {
        let metrics = ArrowReaderMetrics::disabled();
        let mock_reader = MockArrayReader::new(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
        let cache = Arc::new(RwLock::new(RowGroupCache::new(5, usize::MAX))); // Batch size 5
        let mut cached_reader = CachedArrayReader::new(
            Box::new(mock_reader),
            cache,
            0,
            CacheRole::Consumer,
            metrics,
        );

        let read1 = cached_reader.read_records(2).unwrap();
        assert_eq!(read1, 2);

        let array1 = cached_reader.consume_batch().unwrap();
        assert_eq!(array1.len(), 2);
        let int32_array = array1.as_any().downcast_ref::<Int32Array>().unwrap();
        assert_eq!(int32_array.values(), &[1, 2]);

        let skipped = cached_reader.skip_records(2).unwrap();
        assert_eq!(skipped, 2);

        let read2 = cached_reader.read_records(1).unwrap();
        assert_eq!(read2, 1);

        // Consume it (should be the 5th element after skipping 3,4)
        let array2 = cached_reader.consume_batch().unwrap();
        assert_eq!(array2.len(), 1);
        let int32_array = array2.as_any().downcast_ref::<Int32Array>().unwrap();
        assert_eq!(int32_array.values(), &[5]);
    }

    #[test]
    fn test_multiple_reads_before_consume() {
        let metrics = ArrowReaderMetrics::disabled();
        let mock_reader = MockArrayReader::new(vec![1, 2, 3, 4, 5, 6]);
        let cache = Arc::new(RwLock::new(RowGroupCache::new(3, usize::MAX))); // Batch size 3
        let mut cached_reader = CachedArrayReader::new(
            Box::new(mock_reader),
            cache,
            0,
            CacheRole::Consumer,
            metrics,
        );

        // Multiple reads should accumulate
        let read1 = cached_reader.read_records(2).unwrap();
        assert_eq!(read1, 2);

        let read2 = cached_reader.read_records(1).unwrap();
        assert_eq!(read2, 1);

        // Consume should return all accumulated records
        let array = cached_reader.consume_batch().unwrap();
        assert_eq!(array.len(), 3);
        let int32_array = array.as_any().downcast_ref::<Int32Array>().unwrap();
        assert_eq!(int32_array.values(), &[1, 2, 3]);
    }

    #[test]
    fn test_eof_behavior() {
        let metrics = ArrowReaderMetrics::disabled();
        let mock_reader = MockArrayReader::new(vec![1, 2, 3]);
        let cache = Arc::new(RwLock::new(RowGroupCache::new(5, usize::MAX))); // Batch size 5
        let mut cached_reader = CachedArrayReader::new(
            Box::new(mock_reader),
            cache,
            0,
            CacheRole::Consumer,
            metrics,
        );

        // Try to read more than available
        let read1 = cached_reader.read_records(5).unwrap();
        assert_eq!(read1, 3); // Should only get 3 records (all available)

        let array1 = cached_reader.consume_batch().unwrap();
        assert_eq!(array1.len(), 3);

        // Further reads should return 0
        let read2 = cached_reader.read_records(1).unwrap();
        assert_eq!(read2, 0);

        let array2 = cached_reader.consume_batch().unwrap();
        assert_eq!(array2.len(), 0);
    }

    #[test]
    fn test_cache_sharing() {
        let metrics = ArrowReaderMetrics::disabled();
        let cache = Arc::new(RwLock::new(RowGroupCache::new(5, usize::MAX))); // Batch size 5

        // First reader - populate cache
        let mock_reader1 = MockArrayReader::new(vec![1, 2, 3, 4, 5]);
        let mut cached_reader1 = CachedArrayReader::new(
            Box::new(mock_reader1),
            cache.clone(),
            0,
            CacheRole::Producer,
            metrics.clone(),
        );

        cached_reader1.read_records(3).unwrap();
        let array1 = cached_reader1.consume_batch().unwrap();
        assert_eq!(array1.len(), 3);

        // Second reader with different column index should not interfere
        let mock_reader2 = MockArrayReader::new(vec![10, 20, 30, 40, 50]);
        let mut cached_reader2 = CachedArrayReader::new(
            Box::new(mock_reader2),
            cache.clone(),
            1,
            CacheRole::Consumer,
            metrics.clone(),
        );

        cached_reader2.read_records(2).unwrap();
        let array2 = cached_reader2.consume_batch().unwrap();
        assert_eq!(array2.len(), 2);

        // Verify the second reader got its own data, not from cache
        let int32_array = array2.as_any().downcast_ref::<Int32Array>().unwrap();
        assert_eq!(int32_array.values(), &[10, 20]);
    }

    #[test]
    fn test_consumer_removes_batches() {
        let metrics = ArrowReaderMetrics::disabled();
        let mock_reader = MockArrayReader::new(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
        let cache = Arc::new(RwLock::new(RowGroupCache::new(3, usize::MAX))); // Batch size 3
        let mut consumer_reader = CachedArrayReader::new(
            Box::new(mock_reader),
            cache.clone(),
            0,
            CacheRole::Consumer,
            metrics,
        );

        // Read first batch (positions 0-2, batch 0)
        let read1 = consumer_reader.read_records(3).unwrap();
        assert_eq!(read1, 3);
        assert_eq!(consumer_reader.outer_position, 3);
        // Check that batch 0 is in cache after read_records
        assert!(cache.read().unwrap().get(0, BatchID { val: 0 }).is_some());

        let array1 = consumer_reader.consume_batch().unwrap();
        assert_eq!(array1.len(), 3);

        // After first consume_batch, batch 0 should still be in cache
        // (current_batch_id = 3/3 = 1, cleanup only happens if current_batch_id > 1)
        assert!(cache.read().unwrap().get(0, BatchID { val: 0 }).is_some());
        // Nothing was removed, so the watermark has not moved
        assert_eq!(consumer_reader.cleaned_up_to, 0);

        // Read second batch (positions 3-5, batch 1)
        let read2 = consumer_reader.read_records(3).unwrap();
        assert_eq!(read2, 3);
        assert_eq!(consumer_reader.outer_position, 6);
        let array2 = consumer_reader.consume_batch().unwrap();
        assert_eq!(array2.len(), 3);

        // After second consume_batch, batch 0 should be removed
        // (current_batch_id = 6/3 = 2, cleanup removes batches 0..(2-1) = 0..1, so removes batch 0)
        assert!(cache.read().unwrap().get(0, BatchID { val: 0 }).is_none());
        assert!(cache.read().unwrap().get(0, BatchID { val: 1 }).is_some());
        // Watermark records that batch 0 is gone
        assert_eq!(consumer_reader.cleaned_up_to, 1);

        // Read third batch (positions 6-8, batch 2)
        let read3 = consumer_reader.read_records(3).unwrap();
        assert_eq!(read3, 3);
        assert_eq!(consumer_reader.outer_position, 9);
        let array3 = consumer_reader.consume_batch().unwrap();
        assert_eq!(array3.len(), 3);

        // After third consume_batch, batches 0 and 1 should be removed
        // (current_batch_id = 9/3 = 3, cleanup removes batches 0..(3-1) = 0..2, so removes batches 0 and 1)
        assert!(cache.read().unwrap().get(0, BatchID { val: 0 }).is_none());
        assert!(cache.read().unwrap().get(0, BatchID { val: 1 }).is_none());
        assert!(cache.read().unwrap().get(0, BatchID { val: 2 }).is_some());
        // Watermark advanced by exactly one batch: only batch 1 was newly removed
        assert_eq!(consumer_reader.cleaned_up_to, 2);

        // Read one more record (position 9, still batch 3) and consume again.
        // current_batch_id is unchanged, so cleanup has nothing new to remove:
        // the watermark must stay put and batch 2 must remain cached.
        let read4 = consumer_reader.read_records(1).unwrap();
        assert_eq!(read4, 1);
        assert_eq!(consumer_reader.outer_position, 10);
        let array4 = consumer_reader.consume_batch().unwrap();
        assert_eq!(array4.len(), 1);
        assert_eq!(consumer_reader.cleaned_up_to, 2);
        assert!(cache.read().unwrap().get(0, BatchID { val: 2 }).is_some());
    }

    #[test]
    fn test_consumer_cleanup_after_skip() {
        // A consumer that skips several batches (e.g. rows filtered out by a
        // predicate) must still remove the skipped batches from the shared
        // cache, even though it never fetched them itself: one cleanup call
        // covers the whole skipped range.
        let metrics = ArrowReaderMetrics::disabled();
        let cache = Arc::new(RwLock::new(RowGroupCache::new(3, usize::MAX))); // Batch size 3

        // Producer populates batches 0..=3 (12 values).
        let producer_values: Vec<i32> = (1..=12).collect();
        let mut producer = CachedArrayReader::new(
            Box::new(MockArrayReader::new(producer_values.clone())),
            cache.clone(),
            0,
            CacheRole::Producer,
            metrics.clone(),
        );
        for _ in 0..4 {
            producer.read_records(3).unwrap();
            producer.consume_batch().unwrap();
        }
        for batch in 0..4 {
            assert!(
                cache
                    .read()
                    .unwrap()
                    .get(0, BatchID { val: batch })
                    .is_some()
            );
        }

        // Consumer skips batches 0..=2 outright and reads batch 3.
        let mut consumer = CachedArrayReader::new(
            Box::new(MockArrayReader::new(producer_values)),
            cache.clone(),
            0,
            CacheRole::Consumer,
            metrics,
        );
        assert_eq!(consumer.skip_records(9).unwrap(), 9);
        assert_eq!(consumer.read_records(3).unwrap(), 3);
        // Skipping and reading do not touch the shared cache; only consume does
        assert_eq!(consumer.cleaned_up_to, 0);
        let array = consumer.consume_batch().unwrap();
        assert_eq!(array.len(), 3);

        // current_batch_id = 12 / 3 = 4, so cleanup covers batches 0..3 in a
        // single call, including the ones the consumer never fetched.
        for batch in 0..3 {
            assert!(
                cache
                    .read()
                    .unwrap()
                    .get(0, BatchID { val: batch })
                    .is_none()
            );
        }
        assert!(cache.read().unwrap().get(0, BatchID { val: 3 }).is_some());
        // Watermark jumped over the skipped batches in one step
        assert_eq!(consumer.cleaned_up_to, 3);
    }

    #[test]
    fn test_producer_keeps_batches() {
        let metrics = ArrowReaderMetrics::disabled();
        let mock_reader = MockArrayReader::new(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
        let cache = Arc::new(RwLock::new(RowGroupCache::new(3, usize::MAX))); // Batch size 3
        let mut producer_reader = CachedArrayReader::new(
            Box::new(mock_reader),
            cache.clone(),
            0,
            CacheRole::Producer,
            metrics,
        );

        // Read first batch (positions 0-2)
        let read1 = producer_reader.read_records(3).unwrap();
        assert_eq!(read1, 3);
        let array1 = producer_reader.consume_batch().unwrap();
        assert_eq!(array1.len(), 3);

        // Verify batch 0 is in cache
        assert!(cache.read().unwrap().get(0, BatchID { val: 0 }).is_some());

        // Read second batch (positions 3-5) - producer should NOT remove batch 0
        let read2 = producer_reader.read_records(3).unwrap();
        assert_eq!(read2, 3);
        let array2 = producer_reader.consume_batch().unwrap();
        assert_eq!(array2.len(), 3);

        // Verify both batch 0 and batch 1 are still present (no removal for producer)
        assert!(cache.read().unwrap().get(0, BatchID { val: 0 }).is_some());
        assert!(cache.read().unwrap().get(0, BatchID { val: 1 }).is_some());
    }

    #[test]
    fn test_local_cache_protects_against_eviction() {
        let metrics = ArrowReaderMetrics::disabled();
        let mock_reader = MockArrayReader::new(vec![1, 2, 3, 4, 5, 6]);
        let cache = Arc::new(RwLock::new(RowGroupCache::new(3, usize::MAX))); // Batch size 3
        let mut cached_reader = CachedArrayReader::new(
            Box::new(mock_reader),
            cache.clone(),
            0,
            CacheRole::Consumer,
            metrics,
        );

        // Read records which should populate both shared and local cache
        let records_read = cached_reader.read_records(3).unwrap();
        assert_eq!(records_read, 3);

        // The complete grid is already filtered into a pending selected piece.
        // Raw local storage is no longer needed before consume_batch.
        assert!(cache.read().unwrap().get(0, BatchID { val: 0 }).is_some());
        assert!(cached_reader.local_cache.is_empty());

        // Simulate cache eviction by manually removing from shared cache
        cache.write().unwrap().remove(0, BatchID { val: 0 });
        assert!(cache.read().unwrap().get(0, BatchID { val: 0 }).is_none());

        // Shared eviction cannot invalidate the pending selected piece.
        let array = cached_reader.consume_batch().unwrap();
        assert_eq!(array.len(), 3);

        let int32_array = array.as_any().downcast_ref::<Int32Array>().unwrap();
        assert_eq!(int32_array.values(), &[1, 2, 3]);

        // Local cache should be cleared after consume_batch
        assert!(cached_reader.local_cache.is_empty());
    }

    #[test]
    fn test_local_cache_is_cleared_properly() {
        let metrics = ArrowReaderMetrics::disabled();
        let mock_reader = MockArrayReader::new(vec![1, 2, 3, 4]);
        let cache = Arc::new(RwLock::new(RowGroupCache::new(3, 0))); // Batch size 3, cache 0
        let mut cached_reader = CachedArrayReader::new(
            Box::new(mock_reader),
            cache.clone(),
            0,
            CacheRole::Consumer,
            metrics,
        );

        // Read records which should populate both shared and local cache
        let records_read = cached_reader.read_records(1).unwrap();
        assert_eq!(records_read, 1);
        let array = cached_reader.consume_batch().unwrap();
        assert_eq!(array.len(), 1);

        let records_read = cached_reader.read_records(3).unwrap();
        assert_eq!(records_read, 3);
        let array = cached_reader.consume_batch().unwrap();
        assert_eq!(array.len(), 3);
    }

    #[test]
    fn test_cached_reader_idle_skip_does_not_buffer_gap_or_old_local_grid() {
        for role in [CacheRole::Producer, CacheRole::Consumer] {
            for shared_limit in [0, usize::MAX] {
                let cache = Arc::new(RwLock::new(RowGroupCache::new(4, shared_limit)));
                let mut reader = CachedArrayReader::new(
                    Box::new(MockArrayReader::new((0..128).collect())),
                    cache,
                    0,
                    role,
                    ArrowReaderMetrics::disabled(),
                );
                assert_eq!(reader.read_records(1).unwrap(), 1);
                let output = reader.consume_batch().unwrap();
                assert_eq!(
                    output
                        .as_any()
                        .downcast_ref::<Int32Array>()
                        .unwrap()
                        .values(),
                    &[0]
                );
                assert_eq!(
                    reader.local_cache.len(),
                    1,
                    "the current grid still has unread rows"
                );

                // There is no pending output after consume. A long initial gap
                // between windows needs neither a false mask nor the old grid.
                assert_eq!(reader.skip_records(59).unwrap(), 59);
                assert_eq!(reader.outer_position, 60);
                assert_eq!(
                    reader.selections.len(),
                    0,
                    "an idle skip must not allocate a gap-sized mask"
                );
                assert!(
                    reader.local_cache.is_empty(),
                    "skipped grids must release their local references before the next read"
                );

                assert_eq!(reader.read_records(4).unwrap(), 4);
                assert!(reader.local_cache.len() <= 1);
                let output = reader.consume_batch().unwrap();
                assert_eq!(
                    output
                        .as_any()
                        .downcast_ref::<Int32Array>()
                        .unwrap()
                        .values(),
                    &[60, 61, 62, 63]
                );
                assert!(reader.local_cache.is_empty());
            }
        }
    }

    #[test]
    fn test_cached_reader_fixed_grid_windows_reuse_local_when_shared_rejects() {
        for shared_limit in [0, usize::MAX] {
            let cache = Arc::new(RwLock::new(RowGroupCache::new(4, shared_limit)));
            let metrics = ArrowReaderMetrics::enabled();
            let mut reader = CachedArrayReader::new(
                Box::new(MockArrayReader::new((0..16).collect())),
                cache.clone(),
                0,
                CacheRole::Producer,
                metrics.clone(),
            );
            let mut actual = vec![];
            // Every read/consume is an independent raw window within one grid.
            // Some windows continue the same grid; others skip to a later one.
            for (skip, count) in [(3, 1), (1, 2), (0, 1), (3, 1), (0, 4)] {
                assert_eq!(reader.skip_records(skip).unwrap(), skip);
                assert_eq!(reader.selections.len(), 0);
                assert_eq!(
                    reader.read_records(count).unwrap(),
                    count,
                    "a window boundary is not EOF"
                );
                assert!(reader.local_cache.len() <= 1);
                let output = reader.consume_batch().unwrap();
                actual.extend_from_slice(
                    output
                        .as_any()
                        .downcast_ref::<Int32Array>()
                        .unwrap()
                        .values(),
                );
                assert!(reader.local_cache.len() <= 1);
            }
            assert_eq!(actual, vec![3, 5, 6, 7, 11, 12, 13, 14, 15]);
            assert_eq!(metrics.records_read_from_inner(), Some(16));
            assert_eq!(
                metrics.records_read_from_cache(),
                Some(1),
                "row 7 reuses the retained local grid"
            );
            assert!(reader.local_cache.is_empty());
            if shared_limit == 0 {
                for id in 0..4 {
                    assert!(cache.read().unwrap().get(0, BatchID { val: id }).is_none());
                }
            }
            assert_eq!(
                reader.read_records(1).unwrap(),
                0,
                "only actual input exhaustion is EOF"
            );
            assert_eq!(reader.consume_batch().unwrap().len(), 0);
        }
    }

    #[test]
    fn test_cached_reader_pending_skip_preserves_prior_rows_across_grids() {
        let cache = Arc::new(RwLock::new(RowGroupCache::new(4, 0)));
        let mut reader = CachedArrayReader::new(
            Box::new(MockArrayReader::new((0..16).collect())),
            cache,
            0,
            CacheRole::Consumer,
            ArrowReaderMetrics::disabled(),
        );
        assert_eq!(reader.read_records(2).unwrap(), 2);
        assert_eq!(reader.skip_records(5).unwrap(), 5);
        assert!(reader.selections.len() <= 4);
        assert!(
            !reader.local_cache.contains_key(&BatchID { val: 0 }),
            "the selected prefix survives without pinning the completed raw grid"
        );
        assert_eq!(reader.read_records(2).unwrap(), 2);
        let output = reader.consume_batch().unwrap();
        assert_eq!(
            output
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap()
                .values(),
            &[0, 1, 7, 8]
        );
        assert_eq!(reader.local_cache.len(), 1);
        assert_eq!(reader.read_records(3).unwrap(), 3);
        let output = reader.consume_batch().unwrap();
        assert_eq!(
            output
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap()
                .values(),
            &[9, 10, 11]
        );
    }

    fn assert_sparse_pending_selectors_release_raw_grids(period: usize) {
        const GRID: usize = 8;
        let last_selected = (GRID - 1) * period;
        for role in [CacheRole::Producer, CacheRole::Consumer] {
            // Shared rejection isolates the raw data retained by this reader;
            // a shared-cache byte budget cannot bound local_cache.
            let cache = Arc::new(RwLock::new(RowGroupCache::new(GRID, 0)));
            let mut reader = CachedArrayReader::new(
                Box::new(MockArrayReader::new(
                    (0..(last_selected + GRID + 1) as i32).collect(),
                )),
                cache,
                0,
                role,
                ArrowReaderMetrics::disabled(),
            );
            for selected in 0..GRID {
                assert_eq!(reader.read_records(1).unwrap(), 1);
                if selected + 1 < GRID {
                    assert_eq!(reader.skip_records(period - 1).unwrap(), period - 1);
                }
            }
            assert_eq!(reader.outer_position, last_selected + 1);
            assert!(
                reader.local_cache.len() <= 1 && reader.selections.len() <= GRID,
                "period={period}, role={role:?}: completed grids must retain only selected \
                 pieces; raw local grids={}, pending mask rows={}",
                reader.local_cache.len(),
                reader.selections.len(),
            );
            let output = reader.consume_batch().unwrap();
            let expected: Vec<i32> = (0..GRID).map(|row| (row * period) as i32).collect();
            assert_eq!(
                output
                    .as_any()
                    .downcast_ref::<Int32Array>()
                    .unwrap()
                    .values(),
                &expected
            );
            assert!(reader.local_cache.is_empty());
            assert_eq!(
                reader.read_records(1).unwrap(),
                1,
                "consuming selected pieces is not EOF"
            );
            let output = reader.consume_batch().unwrap();
            assert_eq!(
                output
                    .as_any()
                    .downcast_ref::<Int32Array>()
                    .unwrap()
                    .values(),
                &[(last_selected + 1) as i32]
            );
        }
    }

    #[test]
    fn test_cached_reader_pending_selectors_one_of_9_release_raw_grids() {
        assert_sparse_pending_selectors_release_raw_grids(9);
    }

    #[test]
    fn test_cached_reader_pending_selectors_one_of_17_release_raw_grids() {
        assert_sparse_pending_selectors_release_raw_grids(17);
    }

    #[test]
    fn test_cached_reader_large_values_do_not_pin_completed_raw_grid_payloads() {
        use crate::arrow::array_reader::make_byte_array_reader;
        use crate::arrow::array_reader::test_util::utf8_column;
        use crate::basic::Encoding;
        use crate::data_type::{ByteArray, ByteArrayType};
        use crate::util::InMemoryPageIterator;
        use crate::util::test_common::page_util::{DataPageBuilder, DataPageBuilderImpl};
        use arrow_array::{Array, LargeStringArray};

        const GRID: usize = 8;
        const VALUE_BYTES: usize = 64;
        for (initial_skip, period, rows) in [(7, 8, 32), (0, 17, 64)] {
            let strings: Vec<_> = (0..rows)
                .map(|row| format!("{row:04}{}", "x".repeat(VALUE_BYTES - 4)))
                .collect();
            let values: Vec<ByteArray> = strings
                .iter()
                .map(|s| ByteArray::from(s.as_str()))
                .collect();
            let descriptor = utf8_column();
            let mut page = DataPageBuilderImpl::new(descriptor.clone(), rows as u32, true);
            page.add_def_levels(1, &vec![1; rows]);
            page.add_values::<ByteArrayType>(Encoding::PLAIN, &values);
            let pages = InMemoryPageIterator::new(vec![vec![page.consume()]]);
            let inner = make_byte_array_reader(
                Box::new(pages),
                descriptor,
                Some(ArrowType::LargeUtf8),
                GRID,
                None,
            )
            .unwrap();
            let cache = Arc::new(RwLock::new(RowGroupCache::new(GRID, 0)));
            let mut reader = CachedArrayReader::new(
                inner,
                cache.clone(),
                0,
                CacheRole::Producer,
                ArrowReaderMetrics::disabled(),
            );
            reader.skip_records(initial_skip).unwrap();
            for selected in 0..4 {
                assert_eq!(reader.read_records(1).unwrap(), 1);
                if selected + 1 < 4 {
                    reader.skip_records(period - 1).unwrap();
                }
                assert!(reader.local_cache.len() <= 1);
                assert!(reader.selections.len() <= GRID);
                // The deferred piece is compacted later, or copied by concat.
                let deferred = reader.deferred.as_ref().map(|d| d.piece);
                for (index, piece) in reader.completed_selected.iter().enumerate() {
                    let piece = piece.as_any().downcast_ref::<LargeStringArray>().unwrap();
                    assert_eq!(piece.len(), 1);
                    if Some(index) == deferred {
                        continue;
                    }
                    assert_eq!(
                        piece.value_data().len(),
                        VALUE_BYTES,
                        "a selected slice must not keep a complete raw grid's payload"
                    );
                    assert!(
                        piece.values().capacity() <= 2 * VALUE_BYTES,
                        "selected storage must not pin the raw grid allocation"
                    );
                }
            }
            let output = reader.consume_batch().unwrap();
            let output = output.as_any().downcast_ref::<LargeStringArray>().unwrap();
            let expected: Vec<_> = (0..4).map(|row| initial_skip + row * period).collect();
            assert_eq!(
                output
                    .iter()
                    .map(|s| s.unwrap()[..4].parse::<usize>().unwrap())
                    .collect::<Vec<_>>(),
                expected
            );
            assert_eq!(output.value_data().len(), 4 * VALUE_BYTES);
            let position = initial_skip + 3 * period + 1;
            let remaining = rows - position;
            assert_eq!(reader.read_records(remaining).unwrap(), remaining);
            let tail = reader.consume_batch().unwrap();
            let tail = tail.as_any().downcast_ref::<LargeStringArray>().unwrap();
            assert_eq!(
                tail.iter()
                    .map(|s| s.unwrap()[..4].parse::<usize>().unwrap())
                    .collect::<Vec<_>>(),
                (position..rows).collect::<Vec<_>>()
            );
            assert_eq!(reader.read_records(1).unwrap(), 0);
            assert_eq!(reader.consume_batch().unwrap().len(), 0);
            for id in 0..rows / GRID {
                assert!(cache.read().unwrap().get(0, BatchID { val: id }).is_none());
            }
        }
    }

    #[test]
    fn test_batch_id_calculation_with_incremental_reads() {
        let metrics = ArrowReaderMetrics::disabled();
        let mock_reader = MockArrayReader::new(vec![1, 2, 3, 4, 5, 6, 7, 8, 9]);
        let cache = Arc::new(RwLock::new(RowGroupCache::new(3, usize::MAX))); // Batch size 3

        // Create a producer to populate cache
        let mut producer = CachedArrayReader::new(
            Box::new(MockArrayReader::new(vec![1, 2, 3, 4, 5, 6, 7, 8, 9])),
            cache.clone(),
            0,
            CacheRole::Producer,
            metrics.clone(),
        );

        // Populate cache with first batch (1, 2, 3)
        producer.read_records(3).unwrap();
        producer.consume_batch().unwrap();

        // Now create a consumer that will try to read from cache
        let mut consumer = CachedArrayReader::new(
            Box::new(mock_reader),
            cache.clone(),
            0,
            CacheRole::Consumer,
            metrics,
        );

        // - We want to read 4 records starting from position 0
        // - First 3 records (positions 0-2) should come from cache (batch 0)
        // - The 4th record (position 3) should come from the next batch
        let records_read = consumer.read_records(4).unwrap();
        assert_eq!(records_read, 4);

        let array = consumer.consume_batch().unwrap();
        assert_eq!(array.len(), 4);

        let int32_array = array.as_any().downcast_ref::<Int32Array>().unwrap();
        assert_eq!(int32_array.values(), &[1, 2, 3, 4]);
    }
}

#[cfg(test)]
#[path = "cached_byte_slice_tests.rs"]
mod byte_slice_tests;
