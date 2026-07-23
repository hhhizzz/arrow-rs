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

//! [`DataRequest`] tracks and holds data needed to construct InMemoryRowGroups

use crate::arrow::ProjectionMask;
use crate::arrow::arrow_reader::RowSelection;
use crate::arrow::arrow_reader::metrics::{
    ArrowReaderMetrics, ArrowReaderRangePlanning, PredicateDenseFetchDiagnostics,
};
use crate::arrow::in_memory_row_group::{
    ColumnChunkData, DenseFetchFallbackConfig, FetchRanges, InMemoryRowGroup,
};
use crate::errors::ParquetError;
use crate::file::metadata::ParquetMetaData;
use crate::file::page_index::offset_index::OffsetIndexMetaData;
use crate::file::reader::ChunkReader;
use crate::util::push_buffers::PushBuffers;
use bytes::Bytes;
use std::ops::Range;
use std::sync::Arc;

/// Contains in-progress state to construct InMemoryRowGroups
///
/// See [`DataRequestBuilder`] for creating new requests
#[derive(Debug)]
pub(super) struct DataRequest {
    /// Any previously read column chunk data
    column_chunks: Vec<Option<Arc<ColumnChunkData>>>,
    /// The ranges of data that are needed next from the file
    ranges: Vec<Range<u64>>,
    /// Optional ranges to slice from the fetched data when materializing chunks.
    ///
    /// Predicate dense fallback can fetch a wider envelope while still filling
    /// column chunks from the original per-column or per-page ranges.
    materialize_ranges: Option<Vec<Range<u64>>>,
    /// Optional page start offsets for each requested range. This is used
    /// to create the relevant InMemoryRowGroup
    page_start_offsets: Option<Vec<Vec<u64>>>,
}

impl DataRequest {
    /// return what ranges are still needed to satisfy this request. Returns an empty vec
    /// if all ranges are satisfied
    pub fn needed_ranges(&self, buffers: &PushBuffers) -> Vec<Range<u64>> {
        self.ranges
            .iter()
            .filter(|&range| !buffers.has_range(range))
            .cloned()
            .collect()
    }

    /// Returns the chunks from the buffers that satisfy this request
    fn get_chunks(&self, buffers: &PushBuffers) -> Result<Vec<Bytes>, ParquetError> {
        self.materialize_ranges
            .as_ref()
            .unwrap_or(&self.ranges)
            .iter()
            .map(|range| {
                let length: usize = (range.end - range.start)
                    .try_into()
                    .expect("overflow for offset");
                // should have all the data due to the check above
                buffers.get_bytes(range.start, length).map_err(|e| {
                    ParquetError::General(format!(
                        "Internal Error missing data for range {range:?} in buffers: {e}",
                    ))
                })
            })
            .collect()
    }

    /// Create a new InMemoryRowGroup, and fill it with provided data
    ///
    /// Assumes that all needed data is present in the buffers
    /// and clears any explicitly requested ranges
    pub fn try_into_in_memory_row_group<'a>(
        self,
        row_group_idx: usize,
        row_count: usize,
        parquet_metadata: &'a ParquetMetaData,
        projection: &ProjectionMask,
        buffers: &mut PushBuffers,
    ) -> Result<InMemoryRowGroup<'a>, ParquetError> {
        let chunks = self.get_chunks(buffers)?;

        let Self {
            column_chunks,
            ranges,
            materialize_ranges: _,
            page_start_offsets,
        } = self;

        // Create an InMemoryRowGroup to hold the column chunks, this is a
        // temporary structure used to tell the ArrowReaders what pages are
        // needed for decoding
        let mut in_memory_row_group = InMemoryRowGroup {
            row_count,
            column_chunks,
            offset_index: get_offset_index(parquet_metadata, row_group_idx),
            row_group_idx,
            metadata: parquet_metadata,
        };

        in_memory_row_group.fill_column_chunks(projection, page_start_offsets, chunks);

        // Clear the ranges that were explicitly requested
        buffers.clear_ranges(&ranges);

        Ok(in_memory_row_group)
    }
}

/// Builder for [`DataRequest`]
pub(super) struct DataRequestBuilder<'a> {
    /// The row group index
    row_group_idx: usize,
    /// The number of rows in the row group
    row_count: usize,
    /// The batch size to read
    batch_size: usize,
    /// The parquet metadata
    parquet_metadata: &'a ParquetMetaData,
    /// The projection mask (which columns to read)
    projection: &'a ProjectionMask,
    /// Optional row selection to apply
    selection: Option<&'a RowSelection>,
    /// Optional projection mask if using
    /// [`RowGroupCache`](crate::arrow::array_reader::RowGroupCache)
    /// for caching decoded columns.
    cache_projection: Option<&'a ProjectionMask>,
    /// Any previously read column chunks
    column_chunks: Option<Vec<Option<Arc<ColumnChunkData>>>>,
    /// Optional metrics collector
    metrics: Option<ArrowReaderMetrics>,
    /// Range-planning phase that created this request
    range_planning: ArrowReaderRangePlanning,
    /// Debug dense-fetch fallback guard
    dense_fetch_config: Option<DenseFetchFallbackConfig>,
    /// Whether this request can slice original materialization ranges out of a dense fetch
    dense_fetch_materialization_available: bool,
}

impl<'a> DataRequestBuilder<'a> {
    pub(super) fn new(
        row_group_idx: usize,
        row_count: usize,
        batch_size: usize,
        parquet_metadata: &'a ParquetMetaData,
        projection: &'a ProjectionMask,
    ) -> Self {
        Self {
            row_group_idx,
            row_count,
            batch_size,
            parquet_metadata,
            projection,
            selection: None,
            cache_projection: None,
            column_chunks: None,
            metrics: None,
            range_planning: ArrowReaderRangePlanning::Output,
            dense_fetch_config: DenseFetchFallbackConfig::from_env(),
            dense_fetch_materialization_available: true,
        }
    }

    /// Set an optional row selection to apply
    pub(super) fn with_selection(mut self, selection: Option<&'a RowSelection>) -> Self {
        self.selection = selection;
        self
    }

    /// set columns to cache, if any
    pub(super) fn with_cache_projection(
        mut self,
        cache_projection: Option<&'a ProjectionMask>,
    ) -> Self {
        self.cache_projection = cache_projection;
        self
    }

    /// Provide any previously read column chunks
    pub(super) fn with_column_chunks(
        mut self,
        column_chunks: Option<Vec<Option<Arc<ColumnChunkData>>>>,
    ) -> Self {
        self.column_chunks = column_chunks;
        self
    }

    /// Set the metrics collector
    pub(super) fn with_metrics(mut self, metrics: ArrowReaderMetrics) -> Self {
        self.metrics = Some(metrics);
        self
    }

    /// Classify this request as predicate range planning.
    pub(super) fn with_predicate_range_planning(mut self) -> Self {
        self.range_planning = ArrowReaderRangePlanning::Predicate;
        self
    }

    #[cfg(test)]
    fn with_dense_fetch_config(
        mut self,
        dense_fetch_config: Option<DenseFetchFallbackConfig>,
    ) -> Self {
        self.dense_fetch_config = dense_fetch_config;
        self
    }

    #[cfg(test)]
    fn with_predicate_dense_fetch_materialization_available(mut self, available: bool) -> Self {
        self.dense_fetch_materialization_available = available;
        self
    }

    pub(crate) fn build(self) -> DataRequest {
        let Self {
            row_group_idx,
            row_count,
            batch_size,
            parquet_metadata,
            projection,
            selection,
            cache_projection,
            column_chunks,
            metrics,
            range_planning,
            dense_fetch_config,
            dense_fetch_materialization_available,
        } = self;

        let row_group_meta_data = parquet_metadata.row_group(row_group_idx);

        // If no previously read column chunks are provided, create a new location to hold them
        let column_chunks =
            column_chunks.unwrap_or_else(|| vec![None; row_group_meta_data.columns().len()]);

        // Create an InMemoryRowGroup to hold the column chunks, this is a
        // temporary structure used to tell the ArrowReaders what pages are
        // needed for decoding
        let row_group = InMemoryRowGroup {
            row_count,
            column_chunks,
            offset_index: get_offset_index(parquet_metadata, row_group_idx),
            row_group_idx,
            metadata: parquet_metadata,
        };

        let fetch_ranges = row_group.fetch_ranges_with_dense_fetch_config(
            projection,
            selection,
            batch_size,
            cache_projection,
            dense_fetch_config,
        );

        let FetchRanges {
            ranges: fetch_ranges,
            page_start_offsets,
            metrics: fetch_metrics,
        } = fetch_ranges;

        let PredicateDenseFetch {
            ranges,
            materialize_ranges,
            fallback_count: predicate_dense_fetch_fallback_count,
            range_count: predicate_dense_fetch_range_count,
            range_bytes: predicate_dense_fetch_range_bytes,
            diagnostics: predicate_dense_fetch_diagnostics,
        } = maybe_apply_predicate_dense_fetch_fallback(
            range_planning,
            fetch_ranges,
            dense_fetch_config,
            dense_fetch_materialization_available,
        );

        if let Some(metrics) = metrics {
            metrics.record_range_planning_fetch_ranges(
                range_planning,
                ranges.len(),
                range_bytes(&ranges),
            );
            metrics.record_row_selection_fetch_ranges(
                fetch_metrics.sparse_range_count,
                fetch_metrics.sparse_range_bytes,
                fetch_metrics.dense_fetch_fallback_count,
                fetch_metrics.dense_fetch_range_count,
                fetch_metrics.dense_fetch_range_bytes,
            );
            metrics.record_predicate_dense_fetch_ranges(
                predicate_dense_fetch_fallback_count,
                predicate_dense_fetch_range_count,
                predicate_dense_fetch_range_bytes,
            );
            metrics.record_predicate_dense_fetch_diagnostics(predicate_dense_fetch_diagnostics);
        }

        DataRequest {
            // Save any previously read column chunks
            column_chunks: row_group.column_chunks,
            ranges,
            materialize_ranges,
            page_start_offsets,
        }
    }
}

struct PredicateDenseFetch {
    ranges: Vec<Range<u64>>,
    materialize_ranges: Option<Vec<Range<u64>>>,
    fallback_count: usize,
    range_count: usize,
    range_bytes: usize,
    diagnostics: PredicateDenseFetchDiagnostics,
}

fn maybe_apply_predicate_dense_fetch_fallback(
    range_planning: ArrowReaderRangePlanning,
    ranges: Vec<Range<u64>>,
    dense_fetch_config: Option<DenseFetchFallbackConfig>,
    dense_fetch_materialization_available: bool,
) -> PredicateDenseFetch {
    let mut fallback = PredicateDenseFetch {
        ranges,
        materialize_ranges: None,
        fallback_count: 0,
        range_count: 0,
        range_bytes: 0,
        diagnostics: PredicateDenseFetchDiagnostics::default(),
    };

    if !matches!(range_planning, ArrowReaderRangePlanning::Predicate) {
        fallback.diagnostics.not_predicate_planning_count = 1;
        return fallback;
    }

    let Some(config) = dense_fetch_config else {
        fallback.diagnostics.env_not_parsed_count = 1;
        return fallback;
    };

    if fallback.ranges.len() <= config.max_sparse_ranges() {
        fallback.diagnostics.below_range_threshold_count = 1;
        return fallback;
    }

    let Some(dense_ranges) = dense_envelope_for_ranges(&fallback.ranges) else {
        fallback.diagnostics.no_dense_candidate_count = 1;
        return fallback;
    };

    if dense_ranges.len() >= fallback.ranges.len() {
        fallback.diagnostics.no_dense_candidate_count = 1;
        return fallback;
    }

    let sparse_bytes = range_bytes(&fallback.ranges);
    let dense_bytes = range_bytes(&dense_ranges);
    let ratio_guard_allows = config.ratio_guard_allows_dense_fetch(sparse_bytes, dense_bytes);
    let extra_bytes_guard_allows =
        config.extra_bytes_guard_allows_dense_fetch(sparse_bytes, dense_bytes);
    if !ratio_guard_allows.unwrap_or(false) && !extra_bytes_guard_allows.unwrap_or(false) {
        if ratio_guard_allows == Some(false) {
            fallback.diagnostics.ratio_guard_failed_count = 1;
        }
        if extra_bytes_guard_allows == Some(false) {
            fallback.diagnostics.extra_bytes_guard_failed_count = 1;
        }
        if ratio_guard_allows.is_none() && extra_bytes_guard_allows.is_none() {
            fallback.diagnostics.env_not_parsed_count = 1;
        }
        return fallback;
    }

    if !dense_fetch_materialization_available {
        fallback.diagnostics.materialization_unavailable_count = 1;
        return fallback;
    }

    let diagnostics = fallback.diagnostics;
    PredicateDenseFetch {
        materialize_ranges: Some(fallback.ranges),
        fallback_count: 1,
        range_count: dense_ranges.len(),
        range_bytes: dense_bytes,
        ranges: dense_ranges,
        diagnostics,
    }
}

fn dense_envelope_for_ranges(ranges: &[Range<u64>]) -> Option<Vec<Range<u64>>> {
    let start = ranges.iter().map(|range| range.start).min()?;
    let end = ranges.iter().map(|range| range.end).max()?;
    (start < end).then_some(vec![start..end])
}

fn range_bytes(ranges: &[Range<u64>]) -> usize {
    ranges
        .iter()
        .map(|range| (range.end - range.start) as usize)
        .sum()
}

fn get_offset_index(
    parquet_metadata: &ParquetMetaData,
    row_group_idx: usize,
) -> Option<&[OffsetIndexMetaData]> {
    parquet_metadata
        .offset_index()
        // filter out empty offset indexes (old versions specified Some(vec![]) when no present)
        .filter(|index| !index.is_empty())
        .map(|x| x[row_group_idx].as_slice())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arrow::in_memory_row_group::DenseFetchFallbackConfig;
    use crate::basic::Type as PhysicalType;
    use crate::file::metadata::{
        ColumnChunkMetaData, FileMetaData, ParquetMetaDataBuilder, RowGroupMetaData,
    };
    use crate::schema::types::{SchemaDescriptor, Type as SchemaType};

    #[test]
    fn predicate_dense_fetch_fallback_envelopes_request_ranges() {
        let metadata = test_metadata(&[(100, 40), (150, 40), (260, 30)]);
        let projection = ProjectionMask::leaves(metadata.file_metadata().schema_descr(), [0, 1]);
        let metrics = ArrowReaderMetrics::enabled();

        let request = DataRequestBuilder::new(0, 20, 1024, &metadata, &projection)
            .with_predicate_range_planning()
            .with_metrics(metrics.clone())
            .with_dense_fetch_config(Some(DenseFetchFallbackConfig::new_for_test(
                1,
                None,
                Some(10),
            )))
            .build();

        assert_eq!(
            request.needed_ranges(&PushBuffers::default()),
            vec![100..190]
        );
        assert_eq!(metrics.predicate_fetch_request_count(), Some(1));
        assert_eq!(metrics.predicate_single_range_request_count(), Some(1));
        assert_eq!(metrics.predicate_fetch_range_count(), Some(1));
        assert_eq!(metrics.predicate_fetch_range_bytes(), Some(90));
        assert_eq!(metrics.predicate_dense_fetch_fallback_count(), Some(1));
        assert_eq!(metrics.predicate_dense_fetch_range_count(), Some(1));
        assert_eq!(metrics.predicate_dense_fetch_range_bytes(), Some(90));
        assert_predicate_dense_fetch_diagnostics(&metrics, [0; 7]);

        let data = Bytes::from_iter(0_u8..90);
        let mut buffers = PushBuffers::default();
        buffers.push_range(100..190, data);

        let row_group = request
            .try_into_in_memory_row_group(0, 20, &metadata, &projection, &mut buffers)
            .unwrap();

        assert_dense_column(&row_group.column_chunks[0], 100, 40, 0);
        assert_dense_column(&row_group.column_chunks[1], 150, 40, 50);
        assert!(row_group.column_chunks[2].is_none());
    }

    #[test]
    fn predicate_dense_fetch_fallback_respects_extra_byte_guard() {
        let metadata = test_metadata(&[(100, 40), (150, 40), (260, 30)]);
        let projection = ProjectionMask::leaves(metadata.file_metadata().schema_descr(), [0, 1]);
        let metrics = ArrowReaderMetrics::enabled();

        let request = DataRequestBuilder::new(0, 20, 1024, &metadata, &projection)
            .with_predicate_range_planning()
            .with_metrics(metrics.clone())
            .with_dense_fetch_config(Some(DenseFetchFallbackConfig::new_for_test(
                1,
                None,
                Some(9),
            )))
            .build();

        assert_eq!(
            request.needed_ranges(&PushBuffers::default()),
            vec![100..140, 150..190]
        );
        assert_eq!(metrics.predicate_fetch_request_count(), Some(1));
        assert_eq!(metrics.predicate_single_range_request_count(), Some(0));
        assert_eq!(metrics.predicate_fetch_range_count(), Some(2));
        assert_eq!(metrics.predicate_fetch_range_bytes(), Some(80));
        assert_eq!(metrics.predicate_dense_fetch_fallback_count(), Some(0));
        assert_eq!(metrics.predicate_dense_fetch_range_count(), Some(0));
        assert_eq!(metrics.predicate_dense_fetch_range_bytes(), Some(0));
        assert_predicate_dense_fetch_diagnostics(&metrics, [0, 0, 0, 0, 0, 1, 0]);
    }

    #[test]
    fn predicate_dense_fetch_diagnostics_record_guard_failure_reasons() {
        let metadata = test_metadata(&[(100, 40), (150, 40), (260, 30)]);
        let projection = ProjectionMask::leaves(metadata.file_metadata().schema_descr(), [0, 1]);

        let metrics = ArrowReaderMetrics::enabled();
        DataRequestBuilder::new(0, 20, 1024, &metadata, &projection)
            .with_metrics(metrics.clone())
            .with_dense_fetch_config(Some(DenseFetchFallbackConfig::new_for_test(
                1,
                None,
                Some(10),
            )))
            .build();
        assert_predicate_dense_fetch_diagnostics(&metrics, [1, 0, 0, 0, 0, 0, 0]);

        let metrics = ArrowReaderMetrics::enabled();
        DataRequestBuilder::new(0, 20, 1024, &metadata, &projection)
            .with_predicate_range_planning()
            .with_metrics(metrics.clone())
            .with_dense_fetch_config(None)
            .build();
        assert_predicate_dense_fetch_diagnostics(&metrics, [0, 1, 0, 0, 0, 0, 0]);

        let metrics = ArrowReaderMetrics::enabled();
        DataRequestBuilder::new(0, 20, 1024, &metadata, &projection)
            .with_predicate_range_planning()
            .with_metrics(metrics.clone())
            .with_dense_fetch_config(Some(DenseFetchFallbackConfig::new_for_test(
                2,
                None,
                Some(10),
            )))
            .build();
        assert_predicate_dense_fetch_diagnostics(&metrics, [0, 0, 1, 0, 0, 0, 0]);

        let one_column_projection =
            ProjectionMask::leaves(metadata.file_metadata().schema_descr(), [0]);
        let metrics = ArrowReaderMetrics::enabled();
        DataRequestBuilder::new(0, 20, 1024, &metadata, &one_column_projection)
            .with_predicate_range_planning()
            .with_metrics(metrics.clone())
            .with_dense_fetch_config(Some(DenseFetchFallbackConfig::new_for_test(
                0,
                None,
                Some(10),
            )))
            .build();
        assert_predicate_dense_fetch_diagnostics(&metrics, [0, 0, 0, 1, 0, 0, 0]);

        let metrics = ArrowReaderMetrics::enabled();
        DataRequestBuilder::new(0, 20, 1024, &metadata, &projection)
            .with_predicate_range_planning()
            .with_metrics(metrics.clone())
            .with_dense_fetch_config(Some(DenseFetchFallbackConfig::new_for_test(
                1,
                Some(1.0),
                None,
            )))
            .build();
        assert_predicate_dense_fetch_diagnostics(&metrics, [0, 0, 0, 0, 1, 0, 0]);

        let metrics = ArrowReaderMetrics::enabled();
        DataRequestBuilder::new(0, 20, 1024, &metadata, &projection)
            .with_predicate_range_planning()
            .with_metrics(metrics.clone())
            .with_dense_fetch_config(Some(DenseFetchFallbackConfig::new_for_test(
                1,
                None,
                Some(10),
            )))
            .with_predicate_dense_fetch_materialization_available(false)
            .build();
        assert_predicate_dense_fetch_diagnostics(&metrics, [0, 0, 0, 0, 0, 0, 1]);
    }

    fn assert_dense_column(
        chunk: &Option<Arc<ColumnChunkData>>,
        expected_offset: usize,
        expected_len: usize,
        expected_first_byte: u8,
    ) {
        let Some(chunk) = chunk else {
            panic!("expected dense column chunk");
        };
        let ColumnChunkData::Dense { offset, data } = chunk.as_ref() else {
            panic!("expected dense column chunk");
        };
        assert_eq!(*offset, expected_offset);
        assert_eq!(data.len(), expected_len);
        assert_eq!(data[0], expected_first_byte);
    }

    fn assert_predicate_dense_fetch_diagnostics(
        metrics: &ArrowReaderMetrics,
        [
            not_predicate_planning,
            env_not_parsed,
            below_range_threshold,
            no_dense_candidate,
            ratio_guard_failed,
            extra_bytes_guard_failed,
            materialization_unavailable,
        ]: [usize; 7],
    ) {
        assert_eq!(
            metrics.predicate_dense_fetch_not_predicate_planning_count(),
            Some(not_predicate_planning)
        );
        assert_eq!(
            metrics.predicate_dense_fetch_env_not_parsed_count(),
            Some(env_not_parsed)
        );
        assert_eq!(
            metrics.predicate_dense_fetch_below_range_threshold_count(),
            Some(below_range_threshold)
        );
        assert_eq!(
            metrics.predicate_dense_fetch_no_dense_candidate_count(),
            Some(no_dense_candidate)
        );
        assert_eq!(
            metrics.predicate_dense_fetch_ratio_guard_failed_count(),
            Some(ratio_guard_failed)
        );
        assert_eq!(
            metrics.predicate_dense_fetch_extra_bytes_guard_failed_count(),
            Some(extra_bytes_guard_failed)
        );
        assert_eq!(
            metrics.predicate_dense_fetch_materialization_unavailable_count(),
            Some(materialization_unavailable)
        );
    }

    fn test_metadata(columns: &[(i64, i64)]) -> ParquetMetaData {
        let fields = (0..columns.len())
            .map(|idx| {
                let name = format!("c{idx}");
                Arc::new(
                    SchemaType::primitive_type_builder(&name, PhysicalType::INT32)
                        .build()
                        .unwrap(),
                )
            })
            .collect();
        let schema = SchemaType::group_type_builder("schema")
            .with_fields(fields)
            .build()
            .unwrap();
        let schema_descr = Arc::new(SchemaDescriptor::new(Arc::new(schema)));
        let columns = columns
            .iter()
            .enumerate()
            .map(|(idx, (start, len))| {
                ColumnChunkMetaData::builder(schema_descr.column(idx).clone())
                    .set_data_page_offset(*start)
                    .set_total_compressed_size(*len)
                    .build()
                    .unwrap()
            })
            .collect();
        let row_group = RowGroupMetaData::builder(schema_descr.clone())
            .set_num_rows(20)
            .set_total_byte_size(100)
            .set_column_metadata(columns)
            .build()
            .unwrap();
        let file_metadata = FileMetaData::new(1, 20, None, None, schema_descr, None);
        ParquetMetaDataBuilder::new(file_metadata)
            .add_row_group(row_group)
            .build()
    }
}
