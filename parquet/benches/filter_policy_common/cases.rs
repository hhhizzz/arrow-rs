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

use super::model::{CaseSpec, RowGroupPattern};
use super::shapes::{
    BURSTY_50_SAME_SUMMARY, CLUSTERED_50_RUN128, FRAGMENTED_50_RUN1, MODERATE_12_5_RUN32,
    REGULAR_50_RUN32, SPARSE_1_56_RUN32,
};

const FOUR_SPARSE: &[RowGroupPattern] = &[RowGroupPattern::Cycle(SPARSE_1_56_RUN32); 4];

const FOUR_MODERATE: &[RowGroupPattern] = &[RowGroupPattern::Cycle(MODERATE_12_5_RUN32); 4];

const FOUR_FRAGMENTED: &[RowGroupPattern] = &[RowGroupPattern::Cycle(FRAGMENTED_50_RUN1); 4];

const FOUR_CLUSTERED: &[RowGroupPattern] = &[RowGroupPattern::Cycle(CLUSTERED_50_RUN128); 4];

const FOUR_REGULAR: &[RowGroupPattern] = &[RowGroupPattern::Cycle(REGULAR_50_RUN32); 4];

const FOUR_BURSTY: &[RowGroupPattern] = &[RowGroupPattern::Cycle(BURSTY_50_SAME_SUMMARY); 4];

const FOUR_ALL_SELECTED: &[RowGroupPattern] = &[RowGroupPattern::AllSelected; 4];

const SPARSE_TO_DENSE: &[RowGroupPattern] = &[
    RowGroupPattern::Cycle(SPARSE_1_56_RUN32),
    RowGroupPattern::Cycle(FRAGMENTED_50_RUN1),
    RowGroupPattern::Cycle(FRAGMENTED_50_RUN1),
    RowGroupPattern::Cycle(FRAGMENTED_50_RUN1),
];

const DENSE_TO_SPARSE: &[RowGroupPattern] = &[
    RowGroupPattern::Cycle(FRAGMENTED_50_RUN1),
    RowGroupPattern::Cycle(SPARSE_1_56_RUN32),
    RowGroupPattern::Cycle(SPARSE_1_56_RUN32),
    RowGroupPattern::Cycle(SPARSE_1_56_RUN32),
];

const ONE_FRAGMENTED: &[RowGroupPattern] = &[RowGroupPattern::Cycle(FRAGMENTED_50_RUN1)];

const TWO_FRAGMENTED: &[RowGroupPattern] = &[RowGroupPattern::Cycle(FRAGMENTED_50_RUN1); 2];

const EIGHT_FRAGMENTED: &[RowGroupPattern] = &[RowGroupPattern::Cycle(FRAGMENTED_50_RUN1); 8];

pub(crate) const CORE_CASES: &[CaseSpec] = &[
    CaseSpec {
        name: "sparse_1_56_run32",
        row_groups: FOUR_SPARSE,
    },
    CaseSpec {
        name: "moderate_12_5_run32",
        row_groups: FOUR_MODERATE,
    },
    CaseSpec {
        name: "fragmented_50_run1",
        row_groups: FOUR_FRAGMENTED,
    },
    CaseSpec {
        name: "clustered_50_run128",
        row_groups: FOUR_CLUSTERED,
    },
    CaseSpec {
        name: "regular_50_run32",
        row_groups: FOUR_REGULAR,
    },
    CaseSpec {
        name: "bursty_50_same_summary",
        row_groups: FOUR_BURSTY,
    },
    CaseSpec {
        name: "all_selected",
        row_groups: FOUR_ALL_SELECTED,
    },
];

pub(crate) const DRIFT_CASES: &[CaseSpec] = &[
    CaseSpec {
        name: "all_sparse",
        row_groups: FOUR_SPARSE,
    },
    CaseSpec {
        name: "all_dense",
        row_groups: FOUR_FRAGMENTED,
    },
    CaseSpec {
        name: "sparse_to_dense",
        row_groups: SPARSE_TO_DENSE,
    },
    CaseSpec {
        name: "dense_to_sparse",
        row_groups: DENSE_TO_SPARSE,
    },
];

pub(crate) const AMORTIZATION_CASES: &[CaseSpec] = &[
    CaseSpec {
        name: "fragmented_50_run1/rg01",
        row_groups: ONE_FRAGMENTED,
    },
    CaseSpec {
        name: "fragmented_50_run1/rg02",
        row_groups: TWO_FRAGMENTED,
    },
    CaseSpec {
        name: "fragmented_50_run1/rg04",
        row_groups: FOUR_FRAGMENTED,
    },
    CaseSpec {
        name: "fragmented_50_run1/rg08",
        row_groups: EIGHT_FRAGMENTED,
    },
];
