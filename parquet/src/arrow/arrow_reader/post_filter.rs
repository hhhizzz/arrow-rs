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

//! Applies an Arrow row filter after decoding its predicate and output columns.

use super::RowFilter;
use crate::arrow::ProjectionMask;
use crate::errors::{ParquetError, Result};
use crate::schema::types::SchemaDescriptor;
use arrow_array::RecordBatch;
use arrow_schema::{ArrowError, Schema, SchemaRef};
use arrow_select::filter::filter_record_batch;
use std::sync::{Arc, Mutex};

#[derive(Debug)]
pub(super) struct PostFilter {
    filter: Arc<Mutex<RowFilter>>,
    predicate_indices: Vec<Vec<usize>>,
    predicate_schemas: Vec<SchemaRef>,
    output_indices: Vec<usize>,
    output_schema: SchemaRef,
}

impl PostFilter {
    pub(super) fn try_new(
        filter: Arc<Mutex<RowFilter>>,
        parquet_schema: &SchemaDescriptor,
        read_schema: &Schema,
        read_projection: &ProjectionMask,
        output_projection: &ProjectionMask,
    ) -> Result<Self> {
        let filter_guard = filter.lock().map_err(|_| poisoned_filter())?;
        let predicate_indices = filter_guard
            .predicates
            .iter()
            .map(|predicate| {
                projection_indices(parquet_schema, read_projection, predicate.projection())
            })
            .collect::<Result<Vec<_>>>()?;
        drop(filter_guard);

        let predicate_schemas = predicate_indices
            .iter()
            .map(|indices| read_schema.project(indices).map(SchemaRef::new))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let output_indices =
            projection_indices(parquet_schema, read_projection, output_projection)?;
        let output_schema = SchemaRef::new(read_schema.project(&output_indices)?);

        Ok(Self {
            filter,
            predicate_indices,
            predicate_schemas,
            output_indices,
            output_schema,
        })
    }

    pub(super) fn output_schema(&self) -> SchemaRef {
        Arc::clone(&self.output_schema)
    }

    pub(super) fn apply(&mut self, mut batch: RecordBatch) -> Result<RecordBatch> {
        let mut filter = self.filter.lock().map_err(|_| poisoned_filter())?;

        for (predicate_idx, (predicate, indices)) in filter
            .predicates
            .iter_mut()
            .zip(&self.predicate_indices)
            .enumerate()
        {
            let input_rows = batch.num_rows();
            let predicate_batch = project_batch(
                &batch,
                indices,
                Arc::clone(&self.predicate_schemas[predicate_idx]),
            )?;
            let mask = predicate.evaluate(predicate_batch)?;
            if mask.len() != input_rows {
                return Err(general_err!(
                    "ArrowPredicate predicate returned {} rows, expected {input_rows}",
                    mask.len()
                ));
            }
            batch = filter_record_batch(&batch, &mask)?;
            if batch.num_rows() == 0 {
                break;
            }
        }

        Ok(project_batch(
            &batch,
            &self.output_indices,
            Arc::clone(&self.output_schema),
        )?)
    }
}

fn poisoned_filter() -> ParquetError {
    ParquetError::General("post-filter predicate state was poisoned".to_string())
}

fn projection_indices(
    parquet_schema: &SchemaDescriptor,
    read_projection: &ProjectionMask,
    target_projection: &ProjectionMask,
) -> Result<Vec<usize>> {
    let mut indices = Vec::new();
    let mut read_idx = 0;

    for leaf_idx in 0..parquet_schema.num_columns() {
        if !read_projection.leaf_included(leaf_idx) {
            continue;
        }
        let root = parquet_schema.get_column_root(leaf_idx);
        if !root.is_primitive() {
            return Err(general_err!(
                "post-filter fallback does not support nested projections"
            ));
        }
        if target_projection.leaf_included(leaf_idx) {
            indices.push(read_idx);
        }
        read_idx += 1;
    }
    Ok(indices)
}

#[inline]
fn project_batch(
    batch: &RecordBatch,
    indices: &[usize],
    schema: SchemaRef,
) -> std::result::Result<RecordBatch, ArrowError> {
    if indices.len() == batch.num_columns() && indices.iter().copied().eq(0..batch.num_columns()) {
        return Ok(batch.clone());
    }

    let columns = indices
        .iter()
        .map(|idx| {
            batch.columns().get(*idx).cloned().ok_or_else(|| {
                ArrowError::SchemaError(format!(
                    "project index {idx} out of bounds, max field {}",
                    batch.num_columns()
                ))
            })
        })
        .collect::<std::result::Result<Vec<_>, _>>()?;

    RecordBatch::try_new_with_options(
        schema,
        columns,
        &arrow_array::RecordBatchOptions::new().with_row_count(Some(batch.num_rows())),
    )
}
