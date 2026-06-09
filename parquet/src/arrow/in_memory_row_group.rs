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

use crate::arrow::ProjectionMask;
use crate::arrow::array_reader::RowGroups;
use crate::arrow::arrow_reader::RowSelection;
use crate::column::page::{PageIterator, PageReader};
use crate::errors::ParquetError;
use crate::file::metadata::{ParquetMetaData, RowGroupMetaData};
use crate::file::page_index::offset_index::OffsetIndexMetaData;
use crate::file::reader::{ChunkReader, Length, SerializedPageReader};
use bytes::{Buf, Bytes};
use std::env;
use std::ops::Range;
use std::sync::Arc;

const DENSE_FETCH_MAX_SPARSE_RANGES_ENV: &str =
    "DATAFUSION_PARQUET_DEBUG_DENSE_FETCH_MAX_SPARSE_RANGES";
const DENSE_FETCH_MAX_DENSE_SPARSE_BYTES_RATIO_ENV: &str =
    "DATAFUSION_PARQUET_DEBUG_DENSE_FETCH_MAX_DENSE_SPARSE_BYTES_RATIO";
const DENSE_FETCH_MAX_DENSE_EXTRA_BYTES_ENV: &str =
    "DATAFUSION_PARQUET_DEBUG_DENSE_FETCH_MAX_DENSE_EXTRA_BYTES";

/// An in-memory collection of column chunks
#[derive(Debug)]
pub(crate) struct InMemoryRowGroup<'a> {
    pub(crate) offset_index: Option<&'a [OffsetIndexMetaData]>,
    /// Column chunks for this row group
    pub(crate) column_chunks: Vec<Option<Arc<ColumnChunkData>>>,
    pub(crate) row_count: usize,
    pub(crate) row_group_idx: usize,
    pub(crate) metadata: &'a ParquetMetaData,
}

/// What ranges to fetch for the columns in this row group
#[derive(Debug)]
pub(crate) struct FetchRanges {
    /// The byte ranges to fetch
    pub(crate) ranges: Vec<Range<u64>>,
    /// If `Some`, the start offsets of each page for each column chunk
    pub(crate) page_start_offsets: Option<Vec<Vec<u64>>>,
    /// Debug range fetch metrics for row-selection sparse planning
    pub(crate) metrics: FetchRangeMetrics,
}

/// Debug metrics for selected sparse ranges and dense-fetch fallback.
#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct FetchRangeMetrics {
    pub(crate) request_range_count: usize,
    pub(crate) request_range_bytes: usize,
    pub(crate) sparse_range_count: usize,
    pub(crate) sparse_range_bytes: usize,
    pub(crate) dense_fetch_fallback_count: usize,
    pub(crate) dense_fetch_range_count: usize,
    pub(crate) dense_fetch_range_bytes: usize,
}

impl FetchRangeMetrics {
    fn request(ranges: &[Range<u64>]) -> Self {
        Self {
            request_range_count: ranges.len(),
            request_range_bytes: range_bytes(ranges),
            ..Default::default()
        }
    }

    fn sparse(ranges: &[Range<u64>]) -> Self {
        Self {
            request_range_count: ranges.len(),
            request_range_bytes: range_bytes(ranges),
            sparse_range_count: ranges.len(),
            sparse_range_bytes: range_bytes(ranges),
            ..Default::default()
        }
    }

    fn with_dense_fallback(mut self, dense_ranges: &[Range<u64>]) -> Self {
        self.request_range_count = dense_ranges.len();
        self.request_range_bytes = range_bytes(dense_ranges);
        self.dense_fetch_fallback_count = 1;
        self.dense_fetch_range_count = dense_ranges.len();
        self.dense_fetch_range_bytes = range_bytes(dense_ranges);
        self
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct DenseFetchFallbackConfig {
    max_sparse_ranges: usize,
    max_dense_to_sparse_bytes_ratio: Option<f64>,
    max_dense_extra_bytes: Option<usize>,
}

impl DenseFetchFallbackConfig {
    pub(crate) fn from_env() -> Option<Self> {
        let max_sparse_ranges = parse_env_usize(DENSE_FETCH_MAX_SPARSE_RANGES_ENV)?;
        let max_dense_to_sparse_bytes_ratio =
            parse_env_f64(DENSE_FETCH_MAX_DENSE_SPARSE_BYTES_RATIO_ENV);
        let max_dense_extra_bytes = parse_env_usize(DENSE_FETCH_MAX_DENSE_EXTRA_BYTES_ENV);

        (max_dense_to_sparse_bytes_ratio.is_some() || max_dense_extra_bytes.is_some()).then_some(
            Self {
                max_sparse_ranges,
                max_dense_to_sparse_bytes_ratio,
                max_dense_extra_bytes,
            },
        )
    }

    #[cfg(test)]
    pub(crate) fn new_for_test(
        max_sparse_ranges: usize,
        max_dense_to_sparse_bytes_ratio: Option<f64>,
        max_dense_extra_bytes: Option<usize>,
    ) -> Self {
        Self {
            max_sparse_ranges,
            max_dense_to_sparse_bytes_ratio,
            max_dense_extra_bytes,
        }
    }

    pub(crate) fn max_sparse_ranges(self) -> usize {
        self.max_sparse_ranges
    }

    pub(crate) fn allows_dense_fetch(self, sparse_bytes: usize, dense_bytes: usize) -> bool {
        self.ratio_guard_allows_dense_fetch(sparse_bytes, dense_bytes)
            .unwrap_or(false)
            || self
                .extra_bytes_guard_allows_dense_fetch(sparse_bytes, dense_bytes)
                .unwrap_or(false)
    }

    pub(crate) fn ratio_guard_allows_dense_fetch(
        self,
        sparse_bytes: usize,
        dense_bytes: usize,
    ) -> Option<bool> {
        self.max_dense_to_sparse_bytes_ratio.map(|max_ratio| {
            if sparse_bytes == 0 {
                dense_bytes == 0
            } else {
                (dense_bytes as f64) <= (sparse_bytes as f64 * max_ratio)
            }
        })
    }

    pub(crate) fn extra_bytes_guard_allows_dense_fetch(
        self,
        sparse_bytes: usize,
        dense_bytes: usize,
    ) -> Option<bool> {
        self.max_dense_extra_bytes
            .map(|max_extra| dense_bytes.saturating_sub(sparse_bytes) <= max_extra)
    }
}

impl InMemoryRowGroup<'_> {
    /// Returns the byte ranges to fetch for the columns specified in
    /// `projection` and `selection`.
    ///
    /// `cache_mask` indicates which columns, if any, are being cached by
    /// [`RowGroupCache`](crate::arrow::array_reader::RowGroupCache).
    /// The `selection` for Cached columns is expanded to batch boundaries to simplify
    /// accounting for what data is cached.
    #[allow(dead_code)]
    pub(crate) fn fetch_ranges(
        &self,
        projection: &ProjectionMask,
        selection: Option<&RowSelection>,
        batch_size: usize,
        cache_mask: Option<&ProjectionMask>,
    ) -> FetchRanges {
        self.fetch_ranges_with_dense_fetch_config(
            projection,
            selection,
            batch_size,
            cache_mask,
            DenseFetchFallbackConfig::from_env(),
        )
    }

    pub(crate) fn fetch_ranges_with_dense_fetch_config(
        &self,
        projection: &ProjectionMask,
        selection: Option<&RowSelection>,
        batch_size: usize,
        cache_mask: Option<&ProjectionMask>,
        dense_fetch_config: Option<DenseFetchFallbackConfig>,
    ) -> FetchRanges {
        let metadata = self.metadata.row_group(self.row_group_idx);
        if let Some((selection, offset_index)) = selection.zip(self.offset_index) {
            let expanded_selection =
                selection.expand_to_batch_boundaries(batch_size, self.row_count);

            // If we have a `RowSelection` and an `OffsetIndex` then only fetch
            // pages required for the `RowSelection`
            // Consider preallocating outer vec: https://github.com/apache/arrow-rs/issues/8667
            let mut page_start_offsets: Vec<Vec<u64>> = vec![];

            let ranges: Vec<Range<u64>> = self
                .column_chunks
                .iter()
                .zip(metadata.columns())
                .enumerate()
                .filter(|&(idx, (chunk, _chunk_meta))| {
                    chunk.is_none() && projection.leaf_included(idx)
                })
                .flat_map(|(idx, (_chunk, chunk_meta))| {
                    // If the first page does not start at the beginning of the column,
                    // then we need to also fetch a dictionary page.
                    let mut ranges: Vec<Range<u64>> = vec![];
                    let (start, _len) = chunk_meta.byte_range();
                    match offset_index[idx].page_locations.first() {
                        Some(first) if first.offset as u64 != start => {
                            ranges.push(start..first.offset as u64);
                        }
                        _ => (),
                    }

                    // Expand selection to batch boundaries if needed for caching
                    // (see doc comment for this function for details on `cache_mask`)
                    let use_expanded = cache_mask.map(|m| m.leaf_included(idx)).unwrap_or(false);
                    if use_expanded {
                        ranges.extend(
                            expanded_selection.scan_ranges(&offset_index[idx].page_locations),
                        );
                    } else {
                        ranges.extend(selection.scan_ranges(&offset_index[idx].page_locations));
                    }
                    page_start_offsets.push(ranges.iter().map(|range| range.start).collect());

                    ranges
                })
                .collect();

            let mut metrics = FetchRangeMetrics::sparse(&ranges);
            if dense_fetch_config.is_some_and(|config| ranges.len() > config.max_sparse_ranges) {
                let dense_ranges =
                    dense_ranges_for_projection(metadata, projection, &self.column_chunks);
                if !dense_ranges.is_empty()
                    && dense_fetch_config.is_some_and(|config| {
                        config.allows_dense_fetch(
                            metrics.sparse_range_bytes,
                            range_bytes(&dense_ranges),
                        )
                    })
                {
                    metrics = metrics.with_dense_fallback(&dense_ranges);
                    return FetchRanges {
                        ranges: dense_ranges,
                        page_start_offsets: None,
                        metrics,
                    };
                }
            }

            FetchRanges {
                ranges,
                page_start_offsets: Some(page_start_offsets),
                metrics,
            }
        } else {
            let ranges: Vec<Range<u64>> = self
                .column_chunks
                .iter()
                .enumerate()
                .filter(|&(idx, chunk)| chunk.is_none() && projection.leaf_included(idx))
                .map(|(idx, _chunk)| {
                    let column = metadata.column(idx);
                    let (start, length) = column.byte_range();
                    start..(start + length)
                })
                .collect();
            let metrics = FetchRangeMetrics::request(&ranges);
            FetchRanges {
                ranges,
                page_start_offsets: None,
                metrics,
            }
        }
    }

    /// Fills in `self.column_chunks` with the data fetched from `chunk_data`.
    ///
    /// This function **must** be called with the data from the ranges returned by
    /// `fetch_ranges` and the corresponding page_start_offsets, with the exact same and `selection`.
    pub(crate) fn fill_column_chunks<I>(
        &mut self,
        projection: &ProjectionMask,
        page_start_offsets: Option<Vec<Vec<u64>>>,
        chunk_data: I,
    ) where
        I: IntoIterator<Item = Bytes>,
    {
        let mut chunk_data = chunk_data.into_iter();
        let metadata = self.metadata.row_group(self.row_group_idx);
        if let Some(page_start_offsets) = page_start_offsets {
            // If we have a `RowSelection` and an `OffsetIndex` then only fetch pages required for the
            // `RowSelection`
            let mut page_start_offsets = page_start_offsets.into_iter();

            for (idx, chunk) in self.column_chunks.iter_mut().enumerate() {
                if chunk.is_some() || !projection.leaf_included(idx) {
                    continue;
                }

                if let Some(offsets) = page_start_offsets.next() {
                    let mut chunks = Vec::with_capacity(offsets.len());
                    for _ in 0..offsets.len() {
                        chunks.push(chunk_data.next().unwrap());
                    }

                    *chunk = Some(Arc::new(ColumnChunkData::Sparse {
                        length: metadata.column(idx).byte_range().1 as usize,
                        data: offsets
                            .into_iter()
                            .map(|x| x as usize)
                            .zip(chunks.into_iter())
                            .collect(),
                    }))
                }
            }
        } else {
            for (idx, chunk) in self.column_chunks.iter_mut().enumerate() {
                if chunk.is_some() || !projection.leaf_included(idx) {
                    continue;
                }

                if let Some(data) = chunk_data.next() {
                    *chunk = Some(Arc::new(ColumnChunkData::Dense {
                        offset: metadata.column(idx).byte_range().0 as usize,
                        data,
                    }));
                }
            }
        }
    }
}

fn parse_env_usize(name: &str) -> Option<usize> {
    env::var(name)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
}

fn parse_env_f64(name: &str) -> Option<f64> {
    env::var(name)
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .filter(|value| value.is_finite() && *value > 0.0)
}

fn dense_ranges_for_projection(
    metadata: &RowGroupMetaData,
    projection: &ProjectionMask,
    column_chunks: &[Option<Arc<ColumnChunkData>>],
) -> Vec<Range<u64>> {
    column_chunks
        .iter()
        .enumerate()
        .filter(|&(idx, chunk)| chunk.is_none() && projection.leaf_included(idx))
        .map(|(idx, _chunk)| {
            let column = metadata.column(idx);
            let (start, length) = column.byte_range();
            start..(start + length)
        })
        .collect()
}

fn range_bytes(ranges: &[Range<u64>]) -> usize {
    ranges
        .iter()
        .map(|range| (range.end - range.start) as usize)
        .sum()
}

impl RowGroups for InMemoryRowGroup<'_> {
    fn num_rows(&self) -> usize {
        self.row_count
    }

    /// Return chunks for column i
    fn column_chunks(&self, i: usize) -> crate::errors::Result<Box<dyn PageIterator>> {
        match &self.column_chunks[i] {
            None => Err(ParquetError::General(format!(
                "Invalid column index {i}, column was not fetched"
            ))),
            Some(data) => {
                let page_locations = self
                    .offset_index
                    // filter out empty offset indexes (old versions specified Some(vec![]) when no present)
                    .filter(|index| !index.is_empty())
                    .map(|index| index[i].page_locations.clone());
                let column_chunk_metadata = self.metadata.row_group(self.row_group_idx).column(i);
                let page_reader = SerializedPageReader::new(
                    data.clone(),
                    column_chunk_metadata,
                    self.row_count,
                    page_locations,
                )?;
                let page_reader = page_reader.add_crypto_context(
                    self.row_group_idx,
                    i,
                    self.metadata,
                    column_chunk_metadata,
                )?;

                let page_reader: Box<dyn PageReader> = Box::new(page_reader);

                Ok(Box::new(ColumnChunkIterator {
                    reader: Some(Ok(page_reader)),
                }))
            }
        }
    }

    fn row_groups(&self) -> Box<dyn Iterator<Item = &RowGroupMetaData> + '_> {
        Box::new(std::iter::once(self.metadata.row_group(self.row_group_idx)))
    }

    fn metadata(&self) -> &ParquetMetaData {
        self.metadata
    }
}

/// An in-memory column chunk.
/// This allows us to hold either dense column chunks or sparse column chunks and easily
/// access them by offset.
#[derive(Clone, Debug)]
pub(crate) enum ColumnChunkData {
    /// Column chunk data representing only a subset of data pages.
    /// For example if a row selection (possibly caused by a filter in a query) causes us to read only
    /// a subset of the rows in the column.
    Sparse {
        /// Length of the full column chunk
        length: usize,
        /// Subset of data pages included in this sparse chunk.
        ///
        /// Each element is a tuple of (page offset within file, page data).
        /// Each entry is a complete page and the list is ordered by offset.
        data: Vec<(usize, Bytes)>,
    },
    /// Full column chunk and the offset within the original file
    Dense { offset: usize, data: Bytes },
}

impl ColumnChunkData {
    /// Return the data for this column chunk at the given offset
    fn get(&self, start: u64) -> crate::errors::Result<Bytes> {
        match &self {
            ColumnChunkData::Sparse { data, .. } => data
                .binary_search_by_key(&start, |(offset, _)| *offset as u64)
                .map(|idx| data[idx].1.clone())
                .map_err(|_| {
                    ParquetError::General(format!(
                        "Invalid offset in sparse column chunk data: {start}, no matching page found.\
                         If you are using a `SelectionStrategyPolicy::Mask`, ensure that the OffsetIndex is provided when \
                         creating the InMemoryRowGroup."
                    ))
                }),
            ColumnChunkData::Dense { offset, data } => {
                let start = start as usize - *offset;
                Ok(data.slice(start..))
            }
        }
    }
}

impl Length for ColumnChunkData {
    /// Return the total length of the full column chunk
    fn len(&self) -> u64 {
        match &self {
            ColumnChunkData::Sparse { length, .. } => *length as u64,
            ColumnChunkData::Dense { data, .. } => data.len() as u64,
        }
    }
}

impl ChunkReader for ColumnChunkData {
    type T = bytes::buf::Reader<Bytes>;

    fn get_read(&self, start: u64) -> crate::errors::Result<Self::T> {
        Ok(self.get(start)?.reader())
    }

    fn get_bytes(&self, start: u64, length: usize) -> crate::errors::Result<Bytes> {
        Ok(self.get(start)?.slice(..length))
    }
}

/// Implements [`PageIterator`] for a single column chunk, yielding a single [`PageReader`]
struct ColumnChunkIterator {
    reader: Option<crate::errors::Result<Box<dyn PageReader>>>,
}

impl Iterator for ColumnChunkIterator {
    type Item = crate::errors::Result<Box<dyn PageReader>>;

    fn next(&mut self) -> Option<Self::Item> {
        self.reader.take()
    }
}

impl PageIterator for ColumnChunkIterator {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basic::Type as PhysicalType;
    use crate::file::metadata::{
        ColumnChunkMetaData, FileMetaData, ParquetMetaDataBuilder, RowGroupMetaData,
    };
    use crate::file::page_index::offset_index::{OffsetIndexMetaData, PageLocation};
    use crate::schema::types::{SchemaDescriptor, Type as SchemaType};

    #[test]
    fn dense_fetch_debug_fallback_replaces_sparse_page_ranges() {
        let metadata = test_metadata();
        let row_group = InMemoryRowGroup {
            offset_index: metadata.offset_index().map(|index| index[0].as_slice()),
            column_chunks: vec![None, None],
            row_count: 20,
            row_group_idx: 0,
            metadata: &metadata,
        };
        let projection = ProjectionMask::all();
        let selection = RowSelection::from(vec![
            crate::arrow::arrow_reader::RowSelector::select(5),
            crate::arrow::arrow_reader::RowSelector::skip(5),
            crate::arrow::arrow_reader::RowSelector::select(5),
            crate::arrow::arrow_reader::RowSelector::skip(5),
        ]);

        let fetch = row_group.fetch_ranges_with_dense_fetch_config(
            &projection,
            Some(&selection),
            1024,
            None,
            Some(DenseFetchFallbackConfig {
                max_sparse_ranges: 1,
                max_dense_to_sparse_bytes_ratio: Some(1.0),
                max_dense_extra_bytes: None,
            }),
        );

        assert_eq!(fetch.ranges, vec![100..140, 200..260]);
        assert!(fetch.page_start_offsets.is_none());
    }

    #[test]
    fn dense_fetch_debug_fallback_disabled_keeps_sparse_page_ranges() {
        let metadata = test_metadata();
        let row_group = InMemoryRowGroup {
            offset_index: metadata.offset_index().map(|index| index[0].as_slice()),
            column_chunks: vec![None, None],
            row_count: 20,
            row_group_idx: 0,
            metadata: &metadata,
        };
        let projection = ProjectionMask::all();
        let selection = RowSelection::from(vec![
            crate::arrow::arrow_reader::RowSelector::select(5),
            crate::arrow::arrow_reader::RowSelector::skip(5),
            crate::arrow::arrow_reader::RowSelector::select(5),
            crate::arrow::arrow_reader::RowSelector::skip(5),
        ]);

        let fetch = row_group.fetch_ranges_with_dense_fetch_config(
            &projection,
            Some(&selection),
            1024,
            None,
            None,
        );

        assert_eq!(fetch.ranges, vec![100..120, 120..140, 200..230, 230..260]);
        assert_eq!(
            fetch.page_start_offsets,
            Some(vec![vec![100, 120], vec![200, 230]])
        );
    }

    #[test]
    fn range_metrics_count_full_column_ranges_without_selection() {
        let metadata = test_metadata();
        let row_group = InMemoryRowGroup {
            offset_index: metadata.offset_index().map(|index| index[0].as_slice()),
            column_chunks: vec![None, None],
            row_count: 20,
            row_group_idx: 0,
            metadata: &metadata,
        };
        let projection = ProjectionMask::all();

        let fetch =
            row_group.fetch_ranges_with_dense_fetch_config(&projection, None, 1024, None, None);

        assert_eq!(fetch.ranges, vec![100..140, 200..260]);
        assert_eq!(fetch.metrics.request_range_count, 2);
        assert_eq!(fetch.metrics.request_range_bytes, 100);
        assert_eq!(fetch.metrics.sparse_range_count, 0);
        assert_eq!(fetch.metrics.sparse_range_bytes, 0);
    }

    #[test]
    fn dense_fetch_debug_fallback_requires_byte_guard() {
        let metadata = test_metadata();
        let row_group = InMemoryRowGroup {
            offset_index: metadata.offset_index().map(|index| index[0].as_slice()),
            column_chunks: vec![None, None],
            row_count: 20,
            row_group_idx: 0,
            metadata: &metadata,
        };
        let projection = ProjectionMask::all();
        let selection = RowSelection::from(vec![
            crate::arrow::arrow_reader::RowSelector::select(5),
            crate::arrow::arrow_reader::RowSelector::skip(15),
        ]);

        let fetch = row_group.fetch_ranges_with_dense_fetch_config(
            &projection,
            Some(&selection),
            1024,
            None,
            Some(DenseFetchFallbackConfig {
                max_sparse_ranges: 1,
                max_dense_to_sparse_bytes_ratio: None,
                max_dense_extra_bytes: None,
            }),
        );

        assert_eq!(fetch.ranges, vec![100..120, 200..230]);
        assert_eq!(fetch.metrics.sparse_range_count, 2);
        assert_eq!(fetch.metrics.sparse_range_bytes, 50);
        assert_eq!(fetch.metrics.dense_fetch_fallback_count, 0);
    }

    #[test]
    fn dense_fetch_debug_fallback_allows_extra_byte_guard() {
        let metadata = test_metadata();
        let row_group = InMemoryRowGroup {
            offset_index: metadata.offset_index().map(|index| index[0].as_slice()),
            column_chunks: vec![None, None],
            row_count: 20,
            row_group_idx: 0,
            metadata: &metadata,
        };
        let projection = ProjectionMask::all();
        let selection = RowSelection::from(vec![
            crate::arrow::arrow_reader::RowSelector::select(5),
            crate::arrow::arrow_reader::RowSelector::skip(15),
        ]);

        let fetch = row_group.fetch_ranges_with_dense_fetch_config(
            &projection,
            Some(&selection),
            1024,
            None,
            Some(DenseFetchFallbackConfig {
                max_sparse_ranges: 1,
                max_dense_to_sparse_bytes_ratio: None,
                max_dense_extra_bytes: Some(50),
            }),
        );

        assert_eq!(fetch.ranges, vec![100..140, 200..260]);
        assert_eq!(fetch.metrics.request_range_count, 2);
        assert_eq!(fetch.metrics.request_range_bytes, 100);
        assert_eq!(fetch.metrics.sparse_range_count, 2);
        assert_eq!(fetch.metrics.sparse_range_bytes, 50);
        assert_eq!(fetch.metrics.dense_fetch_fallback_count, 1);
    }

    fn test_metadata() -> ParquetMetaData {
        let schema = SchemaType::group_type_builder("schema")
            .with_fields(vec![
                Arc::new(
                    SchemaType::primitive_type_builder("c0", PhysicalType::INT32)
                        .build()
                        .unwrap(),
                ),
                Arc::new(
                    SchemaType::primitive_type_builder("c1", PhysicalType::INT32)
                        .build()
                        .unwrap(),
                ),
            ])
            .build()
            .unwrap();
        let schema_descr = Arc::new(SchemaDescriptor::new(Arc::new(schema)));
        let columns = vec![
            ColumnChunkMetaData::builder(schema_descr.column(0).clone())
                .set_data_page_offset(100)
                .set_total_compressed_size(40)
                .build()
                .unwrap(),
            ColumnChunkMetaData::builder(schema_descr.column(1).clone())
                .set_data_page_offset(200)
                .set_total_compressed_size(60)
                .build()
                .unwrap(),
        ];
        let row_group = RowGroupMetaData::builder(schema_descr.clone())
            .set_num_rows(20)
            .set_total_byte_size(100)
            .set_column_metadata(columns)
            .build()
            .unwrap();
        let file_metadata = FileMetaData::new(1, 20, None, None, schema_descr, None);
        ParquetMetaDataBuilder::new(file_metadata)
            .add_row_group(row_group)
            .set_offset_index(Some(vec![vec![
                offset_index_column(100, 20),
                offset_index_column(200, 30),
            ]]))
            .build()
    }

    fn offset_index_column(offset: i64, page_size: i32) -> OffsetIndexMetaData {
        OffsetIndexMetaData {
            page_locations: vec![
                PageLocation {
                    offset,
                    compressed_page_size: page_size,
                    first_row_index: 0,
                },
                PageLocation {
                    offset: offset + page_size as i64,
                    compressed_page_size: page_size,
                    first_row_index: 10,
                },
            ],
            unencoded_byte_array_data_bytes: None,
        }
    }
}
