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
use crate::basic::Type as PhysicalType;
use crate::file::metadata::RowGroupMetaData;

#[derive(Debug, Clone, Copy, Default)]
pub(super) struct ProjectionReadProfile {
    row_count: i64,
    leaf_count: usize,
    variable_width_leaf_count: usize,
    uncompressed_bytes: u64,
}

impl ProjectionReadProfile {
    pub(super) fn from_projection(
        row_group: &RowGroupMetaData,
        projection: &ProjectionMask,
    ) -> Self {
        Self::from_leaves(row_group, |leaf_idx| projection.leaf_included(leaf_idx))
    }

    pub(super) fn from_leaves(
        row_group: &RowGroupMetaData,
        mut leaf_included: impl FnMut(usize) -> bool,
    ) -> Self {
        let mut profile = Self {
            row_count: row_group.num_rows(),
            ..Self::default()
        };

        for leaf_idx in 0..row_group.num_columns() {
            if !leaf_included(leaf_idx) {
                continue;
            }

            profile.leaf_count += 1;
            let column = row_group.column(leaf_idx);
            if column.column_type() == PhysicalType::BYTE_ARRAY {
                profile.variable_width_leaf_count += 1;
            }
            profile.uncompressed_bytes += column.uncompressed_size().max(0) as u64;
        }

        profile
    }

    pub(super) fn row_count(self) -> i64 {
        self.row_count
    }

    pub(super) fn variable_width_leaf_count(self) -> usize {
        self.variable_width_leaf_count
    }

    pub(super) fn has_variable_width_leaf(self) -> bool {
        self.variable_width_leaf_count > 0
    }

    pub(super) fn uncompressed_bytes_per_row(self) -> f64 {
        if self.row_count <= 0 {
            0.0
        } else {
            self.uncompressed_bytes as f64 / self.row_count as f64
        }
    }

    pub(super) fn is_cheap_fixed_width_read(self, max_bytes_per_row: f64) -> bool {
        self.row_count > 0
            && self.leaf_count > 0
            && !self.has_variable_width_leaf()
            && self.uncompressed_bytes_per_row() <= max_bytes_per_row
    }
}
