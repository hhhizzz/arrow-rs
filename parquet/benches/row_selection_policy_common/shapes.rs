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

use super::model::ROWS_PER_GROUP;
use super::model::{RowGroupPattern, SelectionRun};
use parquet::arrow::arrow_reader::{RowSelection, RowSelector};

use super::fixture::ORACLE_ROW_GROUPS;

pub(crate) const ORACLE_L_SWEEP: &[usize] = &[1, 2, 4, 8, 16, 32, 64, 128, 512, 2_048];
pub(crate) const ORACLE_SELECTIVITY_PERCENT: &[usize] = &[2, 10, 90, 98];
pub(crate) const ORACLE_SELECTIVITY_L: &[usize] = &[8, 64, 512];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum OracleAutoChoice {
    Selectors,
    Mask,
}

impl OracleAutoChoice {
    pub(crate) const fn label(self) -> &'static str {
        match self {
            Self::Selectors => "selectors",
            Self::Mask => "mask",
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct OracleShape {
    pub(crate) name: String,
    pub(crate) nominal_skip: Option<usize>,
    pub(crate) nominal_select: Option<usize>,
    selectors: Vec<RowSelector>,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct OracleShapeSummary {
    pub(crate) selected_rows: usize,
    pub(crate) skipped_rows: usize,
    pub(crate) run_count: usize,
    pub(crate) avg_run_len: f64,
    pub(crate) selected_fraction: f64,
    pub(crate) long_skip_share_1024: f64,
    pub(crate) long_skip_share_4096: f64,
}

impl OracleShape {
    pub(crate) fn periodic(name: impl Into<String>, skip: usize, select: usize) -> Self {
        assert!(skip > 0 || select > 0, "periodic shape must make progress");
        assert!(
            select > 0,
            "oracle shapes must select at least one row per cycle"
        );
        let cycle = skip.saturating_add(select);
        let mut selectors = Vec::new();
        let mut rows = 0usize;
        while rows.saturating_add(cycle) <= ROWS_PER_GROUP {
            push_selector(&mut selectors, RowSelector::skip(skip));
            push_selector(&mut selectors, RowSelector::select(select));
            rows += cycle;
        }
        // Do not partially execute a final cycle: the unfilled row-group tail
        // is deliberately one skip, matching the oracle design contract.
        push_selector(
            &mut selectors,
            RowSelector::skip(ROWS_PER_GROUP.saturating_sub(rows)),
        );
        assert_eq!(selector_rows(&selectors), ROWS_PER_GROUP);
        Self {
            name: name.into(),
            nominal_skip: Some(skip),
            nominal_select: Some(select),
            selectors,
        }
    }

    pub(crate) fn l_sweep(run_len: usize) -> Self {
        Self::periodic(format!("f50_l{run_len}"), run_len, run_len)
    }

    pub(crate) fn selectivity(percent: usize, run_len: usize) -> Self {
        assert!(percent > 0 && percent < 100);
        let cycle = run_len * 2;
        let selected = ((cycle * percent + 50) / 100).max(1);
        Self::periodic(
            format!("f{percent:02}_l{run_len}"),
            cycle.saturating_sub(selected),
            selected,
        )
    }

    pub(crate) fn bursty(long_skip_percent: usize) -> Self {
        assert!(matches!(long_skip_percent, 30 | 70));
        const CYCLE_ROWS: usize = 16_384;
        let skipped = CYCLE_ROWS / 2;
        let long_skip = (skipped * long_skip_percent + 50) / 100;
        let short_rows = skipped - long_skip;
        let mut cycle = Vec::with_capacity(short_rows * 2 + 2);
        push_selector(&mut cycle, RowSelector::skip(long_skip));
        push_selector(&mut cycle, RowSelector::select(long_skip));
        for _ in 0..short_rows {
            push_selector(&mut cycle, RowSelector::skip(1));
            push_selector(&mut cycle, RowSelector::select(1));
        }
        assert_eq!(selector_rows(&cycle), CYCLE_ROWS);

        let mut selectors = Vec::new();
        for _ in 0..ROWS_PER_GROUP / CYCLE_ROWS {
            for selector in &cycle {
                push_selector(&mut selectors, *selector);
            }
        }
        Self {
            name: format!("bursty_50_longskip{long_skip_percent}"),
            nominal_skip: None,
            nominal_select: None,
            selectors,
        }
    }

    pub(crate) fn sparse_cluster() -> Self {
        Self::periodic("sparse_cluster_f1_56_k32", 2_016, 32)
    }

    pub(crate) fn dense() -> Self {
        Self::periodic("dense_f98_44_skip1", 1, 63)
    }

    pub(crate) fn all_selected() -> Self {
        Self::periodic("all_selected", 0, ROWS_PER_GROUP)
    }

    /// Tier-B page control with the same f50, average run length, and number
    /// of selected/skipped runs as `f50_l64`, but with four internal 4096-row
    /// skips. The long skips begin after 2048 output rows in each 16K block so
    /// they cannot be discarded as a free leading skip at an output boundary.
    pub(crate) fn page_matched_bursty() -> Self {
        let mut selectors = Vec::with_capacity(1_024);
        for _ in 0..4 {
            for _ in 0..32 {
                push_selector(&mut selectors, RowSelector::skip(64));
                push_selector(&mut selectors, RowSelector::select(64));
            }
            push_selector(&mut selectors, RowSelector::skip(4_096));
            push_selector(&mut selectors, RowSelector::select(64));
            for short_skip_idx in 0..95 {
                let skip = if short_skip_idx < 53 { 22 } else { 21 };
                push_selector(&mut selectors, RowSelector::skip(skip));
                push_selector(&mut selectors, RowSelector::select(64));
            }
        }
        assert_eq!(selector_rows(&selectors), ROWS_PER_GROUP);
        let shape = Self {
            name: "page_matched_bursty_f50_l64".to_string(),
            nominal_skip: None,
            nominal_select: None,
            selectors,
        };
        let summary = shape.summary();
        assert_eq!(summary.selected_rows, ROWS_PER_GROUP / 2);
        assert_eq!(summary.skipped_rows, ROWS_PER_GROUP / 2);
        assert_eq!(summary.run_count, 1_024);
        assert_eq!(summary.avg_run_len, 64.0);
        shape
    }

    pub(crate) fn selection(&self) -> RowSelection {
        let selectors = (0..ORACLE_ROW_GROUPS)
            .flat_map(|_| self.selectors.iter().copied())
            .collect::<Vec<_>>();
        RowSelection::from(selectors)
    }

    pub(crate) fn selection_for_row_group(&self) -> RowSelection {
        RowSelection::from(self.selectors.clone())
    }

    pub(crate) fn selected_mask(&self) -> Vec<bool> {
        let mut mask = Vec::with_capacity(ROWS_PER_GROUP);
        for selector in &self.selectors {
            mask.extend(std::iter::repeat_n(!selector.skip, selector.row_count));
        }
        assert_eq!(mask.len(), ROWS_PER_GROUP);
        mask
    }

    pub(crate) fn predicate_values(&self) -> Vec<i32> {
        let mut values = Vec::with_capacity(ORACLE_ROW_GROUPS * ROWS_PER_GROUP);
        for _ in 0..ORACLE_ROW_GROUPS {
            for selector in &self.selectors {
                values.extend(std::iter::repeat_n(
                    i32::from(!selector.skip),
                    selector.row_count,
                ));
            }
        }
        assert_eq!(values.len(), ORACLE_ROW_GROUPS * ROWS_PER_GROUP);
        values
    }

    pub(crate) fn summary(&self) -> OracleShapeSummary {
        let selected_rows = self
            .selectors
            .iter()
            .filter(|selector| !selector.skip)
            .map(|selector| selector.row_count)
            .sum::<usize>();
        let skipped_rows = ROWS_PER_GROUP - selected_rows;
        let run_count = self.selectors.len();
        OracleShapeSummary {
            selected_rows,
            skipped_rows,
            run_count,
            avg_run_len: ROWS_PER_GROUP as f64 / run_count as f64,
            selected_fraction: selected_rows as f64 / ROWS_PER_GROUP as f64,
            long_skip_share_1024: long_skip_share(&self.selectors, skipped_rows, 1_024),
            long_skip_share_4096: long_skip_share(&self.selectors, skipped_rows, 4_096),
        }
    }

    pub(crate) fn total_selected_rows(&self) -> usize {
        self.summary().selected_rows * ORACLE_ROW_GROUPS
    }

    pub(crate) fn auto_choice(&self) -> OracleAutoChoice {
        let summary = self.summary();
        if ROWS_PER_GROUP < summary.run_count.saturating_mul(32) {
            OracleAutoChoice::Mask
        } else {
            OracleAutoChoice::Selectors
        }
    }
}

fn push_selector(selectors: &mut Vec<RowSelector>, selector: RowSelector) {
    if selector.row_count == 0 {
        return;
    }
    if let Some(last) = selectors.last_mut()
        && last.skip == selector.skip
    {
        last.row_count += selector.row_count;
    } else {
        selectors.push(selector);
    }
}

fn selector_rows(selectors: &[RowSelector]) -> usize {
    selectors.iter().map(|selector| selector.row_count).sum()
}

fn long_skip_share(selectors: &[RowSelector], skipped_rows: usize, threshold: usize) -> f64 {
    if skipped_rows == 0 {
        return 0.0;
    }
    selectors
        .iter()
        .filter(|selector| selector.skip && selector.row_count >= threshold)
        .map(|selector| selector.row_count)
        .sum::<usize>() as f64
        / skipped_rows as f64
}

pub(crate) const SPARSE_1_56_RUN32: &[SelectionRun] =
    &[SelectionRun::skip(2_016), SelectionRun::select(32)];

pub(crate) const MODERATE_12_5_RUN32: &[SelectionRun] =
    &[SelectionRun::skip(224), SelectionRun::select(32)];

pub(crate) const FRAGMENTED_50_RUN1: &[SelectionRun] =
    &[SelectionRun::skip(1), SelectionRun::select(1)];

pub(crate) const CLUSTERED_50_RUN128: &[SelectionRun] =
    &[SelectionRun::skip(128), SelectionRun::select(128)];

pub(crate) const REGULAR_50_RUN32: &[SelectionRun] =
    &[SelectionRun::skip(32), SelectionRun::select(32)];

pub(crate) const DENSE_98_44_SKIP1_SELECT63: &[SelectionRun] =
    &[SelectionRun::skip(1), SelectionRun::select(63)];

/// Has the same selectivity, run density, and mean selected/skipped run lengths
/// as [`REGULAR_50_RUN32`], but a different run variance and ordering.
pub(crate) const BURSTY_50_SAME_SUMMARY: &[SelectionRun] = &[
    SelectionRun::skip(1),
    SelectionRun::select(1),
    SelectionRun::skip(1),
    SelectionRun::select(1),
    SelectionRun::skip(1),
    SelectionRun::select(1),
    SelectionRun::skip(125),
    SelectionRun::select(125),
];

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct ShapeSummary {
    selected_rows: usize,
    skipped_rows: usize,
    selector_count: usize,
    selected_run_count: usize,
    skipped_run_count: usize,
}

impl ShapeSummary {
    fn from_cycle(cycle: &[SelectionRun]) -> Self {
        cycle.iter().fold(Self::default(), |mut summary, run| {
            assert!(run.len > 0, "selection run must be non-empty");
            summary.selector_count += 1;
            if run.selected {
                summary.selected_rows += run.len;
                summary.selected_run_count += 1;
            } else {
                summary.skipped_rows += run.len;
                summary.skipped_run_count += 1;
            }
            summary
        })
    }

    fn total_rows(self) -> usize {
        self.selected_rows + self.skipped_rows
    }

    fn selected_ratio(self) -> f64 {
        self.selected_rows as f64 / self.total_rows() as f64
    }

    fn run_density(self) -> f64 {
        self.selector_count as f64 / self.total_rows() as f64
    }

    fn average_selected_run(self) -> f64 {
        self.selected_rows as f64 / self.selected_run_count as f64
    }

    fn average_skipped_run(self) -> f64 {
        self.skipped_rows as f64 / self.skipped_run_count as f64
    }
}

pub(crate) fn assert_shape_contracts() {
    let regular = ShapeSummary::from_cycle(REGULAR_50_RUN32);
    let bursty = ShapeSummary::from_cycle(BURSTY_50_SAME_SUMMARY);

    assert_eq!(regular.selected_ratio(), bursty.selected_ratio());
    assert_eq!(regular.run_density(), bursty.run_density());
    assert_eq!(
        regular.average_selected_run(),
        bursty.average_selected_run()
    );
    assert_eq!(regular.average_skipped_run(), bursty.average_skipped_run());

    let dense = ShapeSummary::from_cycle(DENSE_98_44_SKIP1_SELECT63);
    assert_eq!(dense.selected_rows, 63);
    assert_eq!(dense.skipped_rows, 1);
    assert_eq!(dense.selector_count, 2);
    assert_eq!(dense.average_selected_run(), 63.0);
    assert_eq!(dense.average_skipped_run(), 1.0);
}

pub(crate) fn expand_pattern(pattern: RowGroupPattern, row_count: usize) -> Vec<i32> {
    match pattern {
        RowGroupPattern::AllSelected => vec![1; row_count],
        RowGroupPattern::Cycle(cycle) => {
            assert!(!cycle.is_empty(), "selection cycle must not be empty");
            let summary = ShapeSummary::from_cycle(cycle);
            assert_eq!(
                row_count % summary.total_rows(),
                0,
                "row group size must be divisible by cycle size"
            );

            let mut values = Vec::with_capacity(row_count);
            while values.len() < row_count {
                for run in cycle {
                    values.extend(std::iter::repeat_n(i32::from(run.selected), run.len));
                }
            }
            assert_eq!(values.len(), row_count);
            values
        }
    }
}

pub(crate) fn selected_rows(pattern: RowGroupPattern, row_count: usize) -> usize {
    match pattern {
        RowGroupPattern::AllSelected => row_count,
        RowGroupPattern::Cycle(cycle) => {
            let summary = ShapeSummary::from_cycle(cycle);
            assert_eq!(row_count % summary.total_rows(), 0);
            summary.selected_rows * (row_count / summary.total_rows())
        }
    }
}

pub(crate) fn assert_oracle_shape_contracts() {
    for run_len in ORACLE_L_SWEEP {
        let shape = OracleShape::l_sweep(*run_len);
        let summary = shape.summary();
        assert_eq!(summary.selected_rows, ROWS_PER_GROUP / 2);
        assert_eq!(summary.skipped_rows, ROWS_PER_GROUP / 2);
        assert_eq!(summary.avg_run_len, *run_len as f64);
        assert_eq!(selector_rows(&shape.selectors), ROWS_PER_GROUP);
    }

    for percent in ORACLE_SELECTIVITY_PERCENT {
        for run_len in ORACLE_SELECTIVITY_L {
            let shape = OracleShape::selectivity(*percent, *run_len);
            assert_eq!(selector_rows(&shape.selectors), ROWS_PER_GROUP);
            assert!(shape.summary().selected_rows > 0);
        }
    }

    for percent in [30, 70] {
        let shape = OracleShape::bursty(percent);
        let summary = shape.summary();
        assert_eq!(summary.selected_rows, ROWS_PER_GROUP / 2);
        assert_eq!(summary.skipped_rows, ROWS_PER_GROUP / 2);
        assert_eq!(selector_rows(&shape.selectors), ROWS_PER_GROUP);
    }

    assert_eq!(
        OracleShape::sparse_cluster().summary().selected_rows,
        ROWS_PER_GROUP / 64
    );
    assert_eq!(
        OracleShape::dense().summary().selected_rows,
        ROWS_PER_GROUP * 63 / 64
    );
    assert_eq!(
        OracleShape::all_selected().summary().selected_rows,
        ROWS_PER_GROUP
    );

    let regular = OracleShape::l_sweep(64).summary();
    let page_bursty = OracleShape::page_matched_bursty().summary();
    assert_eq!(regular.selected_rows, page_bursty.selected_rows);
    assert_eq!(regular.skipped_rows, page_bursty.skipped_rows);
    assert_eq!(regular.run_count, page_bursty.run_count);
    assert_eq!(regular.avg_run_len, page_bursty.avg_run_len);
}
