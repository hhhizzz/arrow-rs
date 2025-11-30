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

//! Test validating predicate pushdown for a Parquet file that mimics the
//! `inv_quantity_on_hand` column from TPC-DS. The test compares the number of
//! rows returned when applying the predicate via `RowFilter` against scanning
//! all rows and evaluating the predicate in memory for the range `[100, 500]`.

use std::fs::File;
use std::path::Path;

use arrow::array::{Array, BooleanArray, Int32Array, Int64Array, UInt32Array, UInt64Array};
use arrow::datatypes::DataType;
use arrow::error::ArrowError;
use arrow::record_batch::RecordBatch;
use parquet::arrow::ProjectionMask;
use parquet::arrow::arrow_reader::{
    ArrowPredicateFn, ArrowReaderOptions, ParquetRecordBatchReaderBuilder, RowFilter,
};

use futures::TryStreamExt;

use object_store::local::LocalFileSystem;

use object_store::path::Path as ObjectStorePath;

use object_store::ObjectStore;

use parquet::arrow::async_reader::ParquetObjectReader;
use parquet::arrow::async_reader::ParquetRecordBatchStreamBuilder;

use std::sync::Arc;

const PARQUET_PATH: &str =
    "/Users/xunixhuang/RustroverProjects/parquet-learn/inventory_small.parquet";
// const PARQUET_PATH: &str = "/Users/xunixhuang/Downloads/tpcds/sf10/inventory/part-00000-daf09f9c-3890-4658-8160-1d1d3111a03a-c000.snappy.parquet";
const COLUMN_NAME: &str = "inv_quantity_on_hand";
const LOWER_BOUND: i64 = 100;
const UPPER_BOUND: i64 = 500;

struct ColumnInfo {
    index: usize,
    data_type: DataType,
}

// #[test]
// fn predicate_pushdown_matches_manual_scan() {
//     let path = Path::new(PARQUET_PATH);
//     if !path.exists() {
//         panic!(
//             "{PARQUET_PATH} not found. Update PARQUET_PATH to point to the desired file before running this test."
//         );
//     }
//
//     let ColumnInfo {
//         index: column_index,
//         data_type,
//     } = resolve_column_info(path);
//
//     let manual_count = read_without_pushdown(path, column_index, &data_type);
//     let pushdown_count = read_with_pushdown(path, column_index, &data_type);
//
//     assert_eq!(pushdown_count, manual_count);
// }

// fn read_with_pushdown(path: &Path, column_index: usize, data_type: &DataType) -> usize {
//     let file =
//         File::open(path).unwrap_or_else(|err| panic!("Failed to open {}: {err}", path.display()));
//     let options = ArrowReaderOptions::new().with_page_index(true);
//     let builder = ParquetRecordBatchReaderBuilder::try_new_with_options(file, options)
//         .unwrap_or_else(|err| panic!("Failed to build reader for {}: {err}", path.display()));
//
//     let schema_descr = builder.parquet_schema();
//     let predicate_type = data_type.clone();
//     let filter_mask = ProjectionMask::roots(schema_descr, [column_index]);
//     let predicate = ArrowPredicateFn::new(filter_mask, move |batch: RecordBatch| {
//         let array = batch.column(0).as_ref();
//         build_between_mask(array, &predicate_type, LOWER_BOUND, UPPER_BOUND)
//     });
//     let row_filter = RowFilter::new(vec![Box::new(predicate)]);
//
//     let projection = ProjectionMask::all();
//     let mut reader = builder
//         .with_projection(projection)
//         .with_row_filter(row_filter)
//         .with_batch_size(1024 * 1024)
//         .build()
//         .expect("Failed to build record batch reader");
//
//     let mut total_rows = 0usize;
//     while let Some(batch) = reader.next() {
//         let batch = batch.expect("Failed to read record batch");
//         total_rows += batch.num_rows();
//     }
//     total_rows
// }
//
// fn read_without_pushdown(path: &Path, column_index: usize, data_type: &DataType) -> usize {
//     let file =
//         File::open(path).unwrap_or_else(|err| panic!("Failed to open {}: {err}", path.display()));
//     let options = ArrowReaderOptions::new().with_page_index(true);
//     let mut reader = ParquetRecordBatchReaderBuilder::try_new_with_options(file, options)
//         .unwrap_or_else(|err| panic!("Failed to build reader for {}: {err}", path.display()))
//         .with_projection(ProjectionMask::all())
//         .with_batch_size(1024 * 1024)
//         .build()
//         .expect("Failed to build record batch reader");
//
//     let mut total_rows = 0usize;
//     while let Some(batch) = reader.next() {
//         let batch = batch.expect("Failed to read record batch");
//         let array = batch.column(column_index).as_ref();
//         let mask = build_between_mask(array, data_type, LOWER_BOUND, UPPER_BOUND)
//             .expect("Failed to evaluate predicate");
//         total_rows += mask.iter().filter(|value| value.unwrap_or(false)).count();
//     }
//     total_rows
// }

async fn create_object_reader(path: &Path) -> ParquetObjectReader {
    let directory = path
        .parent()
        .unwrap_or_else(|| panic!("Parent directory missing for {}", path.display()));
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or_else(|| panic!("Non UTF-8 file name in {}", path.display()));

    let store =
        Arc::new(LocalFileSystem::new_with_prefix(directory).unwrap()) as Arc<dyn ObjectStore>;
    let location = ObjectStorePath::from(file_name);
    let metadata = store
        .head(&location)
        .await
        .unwrap_or_else(|err| panic!("Failed to load metadata for {}: {err}", path.display()));

    ParquetObjectReader::new(store, metadata.location).with_file_size(metadata.size)
}

async fn read_with_pushdown_object_reader(
    path: &Path,
    column_index: usize,
    data_type: &DataType,
) -> usize {
    let reader = create_object_reader(path).await;
    let options = ArrowReaderOptions::new().with_page_index(true);
    let builder = ParquetRecordBatchStreamBuilder::new_with_options(reader, options)
        .await
        .unwrap_or_else(|err| panic!("Failed to build async reader for {}: {err}", path.display()));

    let predicate_type = data_type.clone();
    let filter_mask = ProjectionMask::roots(builder.parquet_schema(), [column_index]);
    let predicate = ArrowPredicateFn::new(filter_mask, move |batch: RecordBatch| {
        let array = batch.column(0).as_ref();
        build_between_mask(array, &predicate_type, LOWER_BOUND, UPPER_BOUND)
    });
    let row_filter = RowFilter::new(vec![Box::new(predicate)]);

    let projection = ProjectionMask::all();
    let mut stream = builder
        .with_projection(projection)
        .with_row_filter(row_filter)
        .with_batch_size(1024*1024)
        .build()
        .expect("Failed to build record batch stream");

    let mut total_rows = 0usize;
    while let Some(batch) = stream
        .try_next()
        .await
        .expect("Failed to read record batch with predicate pushdown")
    {
        total_rows += batch.num_rows();
    }
    total_rows
}

async fn read_without_pushdown_object_reader(
    path: &Path,
    column_index: usize,
    data_type: &DataType,
) -> usize {
    let reader = create_object_reader(path).await;
    let options = ArrowReaderOptions::new().with_page_index(true);
    let mut stream = ParquetRecordBatchStreamBuilder::new_with_options(reader, options)
        .await
        .unwrap_or_else(|err| panic!("Failed to build async reader for {}: {err}", path.display()))
        .with_projection(ProjectionMask::all())
        .with_batch_size(1024*1024)
        .build()
        .expect("Failed to build record batch stream");

    let mut total_rows = 0usize;
    while let Some(batch) = stream
        .try_next()
        .await
        .expect("Failed to read record batch without predicate pushdown")
    {
        let array = batch.column(column_index).as_ref();
        let mask = build_between_mask(array, data_type, LOWER_BOUND, UPPER_BOUND)
            .expect("Failed to evaluate predicate");
        total_rows += mask.iter().filter(|value| value.unwrap_or(false)).count();
    }
    total_rows
}

#[tokio::test]
async fn predicate_pushdown_matches_manual_scan_object_reader() {
    let path = Path::new(PARQUET_PATH);
    if !path.exists() {
        panic!(
            "{PARQUET_PATH} not found. Update PARQUET_PATH to point to the desired file before running this test."
        );
    }

    let ColumnInfo {
        index: column_index,
        data_type,
    } = resolve_column_info(path);

    let start = std::time::Instant::now();
    let manual_count = read_without_pushdown_object_reader(path, column_index, &data_type).await;
    let manual_duration = start.elapsed();

    let start = std::time::Instant::now();
    let pushdown_count = read_with_pushdown_object_reader(path, column_index, &data_type).await;
    let pushdown_duration = start.elapsed();

    println!(
        "Manual scan: {} rows in {:?}, Pushdown: {} rows in {:?}",
        manual_count, manual_duration, pushdown_count, pushdown_duration
    );

    assert_eq!(pushdown_count, manual_count);
}

fn build_between_mask(
    array: &dyn Array,
    data_type: &DataType,
    lower: i64,
    upper: i64,
) -> Result<BooleanArray, ArrowError> {
    match data_type {
        DataType::Int32 => {
            let values = array
                .as_any()
                .downcast_ref::<Int32Array>()
                .ok_or_else(|| ArrowError::ComputeError("Expected Int32Array".to_string()))?;
            Ok(BooleanArray::from_iter((0..values.len()).map(|i| {
                Some(
                    values.is_valid(i) && {
                        let v = values.value(i) as i64;
                        v >= lower && v <= upper
                    },
                )
            })))
        }
        DataType::Int64 => {
            let values = array
                .as_any()
                .downcast_ref::<Int64Array>()
                .ok_or_else(|| ArrowError::ComputeError("Expected Int64Array".to_string()))?;
            Ok(BooleanArray::from_iter((0..values.len()).map(|i| {
                Some(
                    values.is_valid(i) && {
                        let v = values.value(i);
                        v >= lower && v <= upper
                    },
                )
            })))
        }
        DataType::UInt32 => {
            let values = array
                .as_any()
                .downcast_ref::<UInt32Array>()
                .ok_or_else(|| ArrowError::ComputeError("Expected UInt32Array".to_string()))?;
            Ok(BooleanArray::from_iter((0..values.len()).map(|i| {
                Some(
                    values.is_valid(i) && {
                        let v = values.value(i) as i64;
                        v >= lower && v <= upper
                    },
                )
            })))
        }
        DataType::UInt64 => {
            let values = array
                .as_any()
                .downcast_ref::<UInt64Array>()
                .ok_or_else(|| ArrowError::ComputeError("Expected UInt64Array".to_string()))?;
            Ok(BooleanArray::from_iter((0..values.len()).map(|i| {
                Some(
                    values.is_valid(i) && {
                        let v = values.value(i) as i128;
                        v >= lower as i128 && v <= upper as i128
                    },
                )
            })))
        }
        _ => Err(ArrowError::ComputeError(format!(
            "Unsupported data type {data_type:?} for predicate pushdown benchmark"
        ))),
    }
}

fn resolve_column_info(path: &Path) -> ColumnInfo {
    let file =
        File::open(path).unwrap_or_else(|err| panic!("Failed to open {}: {err}", path.display()));
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .unwrap_or_else(|err| panic!("Failed to read schema from {}: {err}", path.display()));
    let schema = builder.schema();

    let column_index = schema
        .fields()
        .iter()
        .position(|field| field.name() == COLUMN_NAME)
        .unwrap_or_else(|| panic!("Column '{COLUMN_NAME}' not found in {}", path.display()));

    let data_type = schema.field(column_index).data_type().clone();

    ColumnInfo {
        index: column_index,
        data_type,
    }
}
