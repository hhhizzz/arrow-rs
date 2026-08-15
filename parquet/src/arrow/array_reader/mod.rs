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

//! Logic for reading into arrow arrays: [`ArrayReader`] and [`RowGroups`]

use crate::errors::Result;
use arrow_array::ArrayRef;
use arrow_schema::DataType as ArrowType;
use std::any::Any;
use std::sync::Arc;

use crate::arrow::arrow_reader::RowSelector;
use crate::arrow::record_reader::GenericRecordReader;
use crate::arrow::record_reader::buffer::ValuesBuffer;
use crate::column::page::PageIterator;
use crate::column::reader::decoder::ColumnValueDecoder;
use crate::file::metadata::ParquetMetaData;
use crate::file::reader::{FilePageIterator, FileReader};

mod builder;
mod byte_array;
mod byte_array_dictionary;
mod byte_view_array;
mod cached_array_reader;
mod empty_array;
mod fixed_len_byte_array;
mod fixed_size_list_array;
mod list_array;
mod list_view_array;
mod map_array;
mod null_array;
mod primitive_array;
mod row_group_cache;
mod row_group_index;
mod row_number;
mod struct_array;

#[cfg(test)]
pub(crate) mod test_util;

// Note that this crate is public under the `experimental` feature flag.
use crate::file::metadata::RowGroupMetaData;
pub use builder::{ArrayReaderBuilder, CacheOptions, CacheOptionsBuilder};
pub use byte_array::make_byte_array_reader;
pub use byte_array_dictionary::make_byte_array_dictionary_reader;
#[allow(unused_imports)] // Only used for benchmarks
pub use byte_view_array::make_byte_view_array_reader;
#[allow(unused_imports)] // Only used for benchmarks
pub use fixed_len_byte_array::make_fixed_len_byte_array_reader;
pub use fixed_size_list_array::FixedSizeListArrayReader;
pub use list_array::ListArrayReader;
pub use list_view_array::ListViewArrayReader;
pub use map_array::MapArrayReader;
pub use null_array::NullArrayReader;
pub use primitive_array::PrimitiveArrayReader;
pub use row_group_cache::RowGroupCache;
pub use struct_array::StructArrayReader;

/// Reads Parquet data into Arrow Arrays.
///
/// This is an internal implementation detail of the Parquet reader, and is not
/// intended for public use.
///
/// This is the core trait for reading encoded Parquet data directly into Arrow
/// Arrays efficiently. There are various specializations of this trait for
/// different combinations of encodings and arrays, such as
/// [`PrimitiveArrayReader`], [`ListArrayReader`], etc.
///
/// Each `ArrayReader` logically contains the following state
/// 1. A handle to the encoded Parquet data
/// 2. An in progress buffered Array
///
/// Data can either be read in batches using [`ArrayReader::next_batch`] or
/// incrementally using [`ArrayReader::read_records`] and [`ArrayReader::skip_records`].
pub trait ArrayReader: Send {
    // TODO: this function is never used, and the trait is not public. Perhaps this should be
    // removed.
    #[allow(dead_code)]
    fn as_any(&self) -> &dyn Any;

    /// Returns the arrow type of this array reader.
    fn get_data_type(&self) -> &ArrowType;

    /// Reads at most `batch_size` records into an arrow array and return it.
    #[cfg(any(feature = "experimental", test))]
    fn next_batch(&mut self, batch_size: usize) -> Result<ArrayRef> {
        self.read_records(batch_size)?;
        self.consume_batch()
    }

    /// Reads at most `batch_size` records' bytes into buffer
    ///
    /// Returns the number of records read, which can be less than `batch_size` if
    /// pages is exhausted.
    fn read_records(&mut self, batch_size: usize) -> Result<usize>;

    /// Consume one output batch's alternating skip/read runs.
    ///
    /// `selection` describes only the physical rows needed for the current
    /// output batch. Implementations may override this method to process the
    /// complete run sequence with reader-local locality. The default preserves
    /// the existing semantics by replaying [`Self::skip_records`] and
    /// [`Self::read_records`] in order.
    fn read_selection(&mut self, selection: &[RowSelector]) -> Result<usize> {
        let mut read = 0;
        for selector in selection {
            if selector.skip {
                let skipped = self.skip_records(selector.row_count)?;
                if skipped != selector.row_count {
                    return Err(general_err!(
                        "failed to skip rows, expected {}, got {}",
                        selector.row_count,
                        skipped
                    ));
                }
            } else {
                match self.read_records(selector.row_count)? {
                    0 => break,
                    records => read += records,
                }
            }
        }
        Ok(read)
    }

    /// Consume all currently stored buffer data
    /// into an arrow array and return it.
    fn consume_batch(&mut self) -> Result<ArrayRef>;

    /// Skips over `num_records` records, returning the number of rows skipped
    ///
    /// Note that calling `skip_records` with large values of `num_records` is
    /// efficient as it avoids decoding data into the the in-progress array.
    /// However, there is overhead to calling this function, so for small values of
    /// `num_records`, it can be more efficient to call read_records and apply
    /// a filter to the resulting array.
    fn skip_records(&mut self, num_records: usize) -> Result<usize>;

    /// If this array has a non-zero definition level, i.e. has a nullable parent
    /// array, returns the definition levels of data from the last call of `next_batch`
    ///
    /// Otherwise returns None
    ///
    /// This is used by parent [`ArrayReader`] to compute their null bitmaps
    fn get_def_levels(&self) -> Option<&[i16]>;

    /// If this array has a non-zero repetition level, i.e. has a repeated parent
    /// array, returns the repetition levels of data from the last call of `next_batch`
    ///
    /// Otherwise returns None
    ///
    /// This is used by parent [`ArrayReader`] to compute their array offsets
    fn get_rep_levels(&self) -> Option<&[i16]>;

    /// Returns the maximum definition level for this reader as defined by
    /// the Parquet schema. For leaf readers this is the column's max def level;
    /// for composite readers it is the level at which the composite itself is
    /// fully defined.
    ///
    /// The default panics. Synthetic readers that are not backed by a Parquet
    /// column (e.g. row-number or empty-struct readers) may rely on this
    /// default because they are never used as children of schema-driven
    /// composite readers (list, map, struct), which are the only callers.
    fn max_def_level(&self) -> i16 {
        panic!("max_def_level called on a reader that does not track definition levels")
    }
}

/// Interface for reading data pages from the columns of one or more RowGroups.
pub trait RowGroups {
    /// Get the number of rows in this collection
    fn num_rows(&self) -> usize;

    /// Returns a [`PageIterator`] for all pages in the specified column chunk
    /// across all row groups in this collection.
    fn column_chunks(&self, i: usize) -> Result<Box<dyn PageIterator>>;

    /// Returns an iterator over the row groups in this collection
    ///
    /// Note this may not include all row groups in [`Self::metadata`].
    fn row_groups(&self) -> Box<dyn Iterator<Item = &RowGroupMetaData> + '_>;

    /// Returns the parquet metadata
    fn metadata(&self) -> &ParquetMetaData;
}

impl RowGroups for Arc<dyn FileReader> {
    fn num_rows(&self) -> usize {
        FileReader::metadata(self.as_ref())
            .file_metadata()
            .num_rows() as usize
    }

    fn column_chunks(&self, column_index: usize) -> Result<Box<dyn PageIterator>> {
        let iterator = FilePageIterator::new(column_index, Arc::clone(self))?;
        Ok(Box::new(iterator))
    }

    fn row_groups(&self) -> Box<dyn Iterator<Item = &RowGroupMetaData> + '_> {
        Box::new(FileReader::metadata(self.as_ref()).row_groups().iter())
    }

    fn metadata(&self) -> &ParquetMetaData {
        FileReader::metadata(self.as_ref())
    }
}

/// Uses `record_reader` to read up to `batch_size` records from `pages`
///
/// Returns the number of records read, which can be less than `batch_size` if
/// pages is exhausted.
fn read_records<V, CV>(
    record_reader: &mut GenericRecordReader<V, CV>,
    pages: &mut dyn PageIterator,
    batch_size: usize,
) -> Result<usize>
where
    V: ValuesBuffer,
    CV: ColumnValueDecoder<Buffer = V>,
{
    let mut records_read = 0usize;
    while records_read < batch_size {
        let records_to_read = batch_size - records_read;

        let records_read_once = record_reader.read_records(records_to_read)?;
        records_read += records_read_once;

        // Record reader exhausted
        if records_read_once < records_to_read {
            if let Some(page_reader) = pages.next() {
                // Read from new page reader (i.e. column chunk)
                record_reader.set_page_reader(page_reader?)?;
            } else {
                // Page reader also exhausted
                break;
            }
        }
    }
    Ok(records_read)
}

/// Uses `record_reader` to skip up to `batch_size` records from `pages`
///
/// Returns the number of records skipped, which can be less than `batch_size` if
/// pages is exhausted
fn skip_records<V, CV>(
    record_reader: &mut GenericRecordReader<V, CV>,
    pages: &mut dyn PageIterator,
    batch_size: usize,
) -> Result<usize>
where
    V: ValuesBuffer,
    CV: ColumnValueDecoder<Buffer = V>,
{
    let mut records_skipped = 0usize;
    while records_skipped < batch_size {
        let records_to_read = batch_size - records_skipped;

        let records_skipped_once = record_reader.skip_records(records_to_read)?;
        records_skipped += records_skipped_once;

        // Record reader exhausted
        if records_skipped_once < records_to_read {
            if let Some(page_reader) = pages.next() {
                // Read from new page reader (i.e. column chunk)
                record_reader.set_page_reader(page_reader?)?;
            } else {
                // Page reader also exhausted
                break;
            }
        }
    }
    Ok(records_skipped)
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::new_empty_array;
    use std::collections::VecDeque;

    #[derive(Debug, Eq, PartialEq)]
    enum Call {
        Read(usize),
        Skip(usize),
    }

    struct ReplayReader {
        data_type: ArrowType,
        reads: VecDeque<usize>,
        skips: VecDeque<usize>,
        calls: Vec<Call>,
    }

    impl ReplayReader {
        fn new(reads: &[usize], skips: &[usize]) -> Self {
            Self {
                data_type: ArrowType::Int32,
                reads: reads.iter().copied().collect(),
                skips: skips.iter().copied().collect(),
                calls: Vec::new(),
            }
        }
    }

    impl ArrayReader for ReplayReader {
        fn as_any(&self) -> &dyn Any {
            self
        }

        fn get_data_type(&self) -> &ArrowType {
            &self.data_type
        }

        fn read_records(&mut self, batch_size: usize) -> Result<usize> {
            self.calls.push(Call::Read(batch_size));
            Ok(self.reads.pop_front().unwrap_or(batch_size))
        }

        fn consume_batch(&mut self) -> Result<ArrayRef> {
            Ok(new_empty_array(&self.data_type))
        }

        fn skip_records(&mut self, num_records: usize) -> Result<usize> {
            self.calls.push(Call::Skip(num_records));
            Ok(self.skips.pop_front().unwrap_or(num_records))
        }

        fn get_def_levels(&self) -> Option<&[i16]> {
            None
        }

        fn get_rep_levels(&self) -> Option<&[i16]> {
            None
        }
    }

    #[test]
    fn read_selection_replays_existing_verbs_in_order() {
        let mut reader = ReplayReader::new(&[], &[]);
        let selection = [
            RowSelector::skip(2),
            RowSelector::select(3),
            RowSelector::skip(1),
            RowSelector::select(4),
        ];

        assert_eq!(reader.read_selection(&selection).unwrap(), 7);
        assert_eq!(
            reader.calls,
            vec![Call::Skip(2), Call::Read(3), Call::Skip(1), Call::Read(4)]
        );
    }

    #[test]
    fn read_selection_stops_replay_on_eof() {
        let mut reader = ReplayReader::new(&[3, 0], &[]);
        let selection = [
            RowSelector::select(3),
            RowSelector::select(4),
            RowSelector::skip(9),
        ];

        assert_eq!(reader.read_selection(&selection).unwrap(), 3);
        assert_eq!(reader.calls, vec![Call::Read(3), Call::Read(4)]);
    }

    #[test]
    fn read_selection_rejects_short_skip() {
        let mut reader = ReplayReader::new(&[], &[2]);
        let error = reader
            .read_selection(&[RowSelector::skip(3)])
            .unwrap_err();

        assert_eq!(
            error.to_string(),
            "Parquet error: failed to skip rows, expected 3, got 2"
        );
    }
}
