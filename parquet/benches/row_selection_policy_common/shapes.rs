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
    pub(crate) selected_run_count: usize,
    pub(crate) skipped_run_count: usize,
    pub(crate) avg_run_len: f64,
    pub(crate) selected_fraction: f64,
    pub(crate) first_selected_row: usize,
    pub(crate) last_selected_row_exclusive: usize,
    pub(crate) max_skip_run: usize,
    pub(crate) long_skip_rows_1024: usize,
    pub(crate) long_skip_count_1024: usize,
    pub(crate) long_skip_rows_4096: usize,
    pub(crate) long_skip_count_4096: usize,
    pub(crate) long_skip_share_1024: f64,
    pub(crate) long_skip_share_4096: f64,
}

impl OracleShapeSummary {
    pub(crate) fn leading_skip_present(self) -> bool {
        self.first_selected_row > 0
    }

    pub(crate) fn trailing_skip_present(self) -> bool {
        self.last_selected_row_exclusive < ROWS_PER_GROUP
    }

    pub(crate) fn internal_skip_run_count(self) -> usize {
        self.skipped_run_count
            .saturating_sub(usize::from(self.leading_skip_present()))
            .saturating_sub(usize::from(self.trailing_skip_present()))
    }

    pub(crate) fn internal_transition_count(self) -> usize {
        self.selected_run_count
            .saturating_add(self.internal_skip_run_count())
            .saturating_sub(1)
    }
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

    /// PC-1 spelling for the frozen 30%-long-skip bursty control already used
    /// by the CAL and R series. The runbook's `l4` suffix is retained verbatim
    /// for cross-series joins; the selector bytes come from the established
    /// bursty generator rather than introducing a post-registration variant.
    pub(crate) fn pc_bursty03_l4() -> Self {
        let mut shape = Self::bursty(30);
        shape.name = "bursty03_l4".to_string();
        shape
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

    pub(crate) fn leading_only() -> Self {
        Self::periodic("leading_only_select1", ROWS_PER_GROUP - 1, 1)
    }

    pub(crate) fn internal_bookend(gap: usize) -> Self {
        assert!(gap <= ROWS_PER_GROUP - 2);
        let mut selectors = Vec::with_capacity(4);
        push_selector(&mut selectors, RowSelector::select(1));
        push_selector(&mut selectors, RowSelector::skip(gap));
        push_selector(&mut selectors, RowSelector::select(1));
        push_selector(&mut selectors, RowSelector::skip(ROWS_PER_GROUP - gap - 2));
        assert_eq!(selector_rows(&selectors), ROWS_PER_GROUP);
        Self {
            name: format!("internal_bookend_gap{gap}"),
            nominal_skip: None,
            nominal_select: None,
            selectors,
        }
    }

    pub(crate) fn multi_gap64() -> Self {
        let mut selectors = Vec::with_capacity(130);
        push_selector(&mut selectors, RowSelector::select(1));
        for _ in 0..64 {
            push_selector(&mut selectors, RowSelector::skip(64));
            push_selector(&mut selectors, RowSelector::select(1));
        }
        let tail = ROWS_PER_GROUP - selector_rows(&selectors);
        push_selector(&mut selectors, RowSelector::skip(tail));
        assert_eq!(selector_rows(&selectors), ROWS_PER_GROUP);
        Self {
            name: "multi_gap64_count64".to_string(),
            nominal_skip: None,
            nominal_select: None,
            selectors,
        }
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

    /// Tier-C R-family: hold selected rows, selected span, and selectivity
    /// fixed while changing only the number of internal alternating runs.
    pub(crate) fn tier_c_transitions(selected_runs: usize) -> Self {
        assert!(matches!(selected_runs, 2 | 8 | 32 | 128 | 512));
        let skipped_runs = selected_runs - 1;
        let mut selectors = Vec::with_capacity(selected_runs + skipped_runs);
        for index in 0..selected_runs {
            push_selector(
                &mut selectors,
                RowSelector::select(distributed_len(ROWS_PER_GROUP / 2, selected_runs, index)),
            );
            if index < skipped_runs {
                push_selector(
                    &mut selectors,
                    RowSelector::skip(distributed_len(ROWS_PER_GROUP / 2, skipped_runs, index)),
                );
            }
        }
        assert_eq!(selector_rows(&selectors), ROWS_PER_GROUP);
        let shape = Self {
            name: format!("tc_r_selruns{selected_runs}_f50"),
            nominal_skip: None,
            nominal_select: None,
            selectors,
        };
        let summary = shape.summary();
        assert_eq!(summary.selected_rows, ROWS_PER_GROUP / 2);
        assert_eq!(summary.skipped_rows, ROWS_PER_GROUP / 2);
        assert_eq!(summary.first_selected_row, 0);
        assert_eq!(summary.last_selected_row_exclusive, ROWS_PER_GROUP);
        assert_eq!(summary.selected_run_count, selected_runs);
        assert_eq!(summary.internal_skip_run_count(), skipped_runs);
        assert_eq!(summary.internal_transition_count(), 2 * selected_runs - 2);
        shape
    }

    /// Tier-C W-family: hold two selected blocks and one internal skip fixed,
    /// varying the selected span. The remaining row-group tail is one trimmed
    /// skip and is not an internal transition.
    pub(crate) fn tier_c_gap(gap_rows: usize) -> Self {
        assert!(matches!(gap_rows, 64 | 512 | 4_096 | 16_384 | 63_488));
        const SELECT_BLOCK: usize = 1_024;
        let selected_span = SELECT_BLOCK * 2 + gap_rows;
        assert!(selected_span <= ROWS_PER_GROUP);
        let mut selectors = Vec::with_capacity(4);
        push_selector(&mut selectors, RowSelector::select(SELECT_BLOCK));
        push_selector(&mut selectors, RowSelector::skip(gap_rows));
        push_selector(&mut selectors, RowSelector::select(SELECT_BLOCK));
        push_selector(
            &mut selectors,
            RowSelector::skip(ROWS_PER_GROUP - selected_span),
        );
        assert_eq!(selector_rows(&selectors), ROWS_PER_GROUP);
        let shape = Self {
            name: format!("tc_w_gap{gap_rows}_s2048"),
            nominal_skip: None,
            nominal_select: None,
            selectors,
        };
        let summary = shape.summary();
        assert_eq!(summary.selected_rows, SELECT_BLOCK * 2);
        assert_eq!(summary.first_selected_row, 0);
        assert_eq!(summary.last_selected_row_exclusive, selected_span);
        assert_eq!(summary.internal_skip_run_count(), 1);
        assert_eq!(summary.internal_transition_count(), 2);
        shape
    }

    /// Tier-C F-family: hold the selected span and internal transition count
    /// fixed while changing exact selected rows.
    pub(crate) fn tier_c_selectivity(selected_rows: usize) -> Self {
        assert!(matches!(
            selected_rows,
            1_024 | 8_192 | 32_768 | 57_344 | 64_512
        ));
        const SELECTED_RUNS: usize = 64;
        const SKIPPED_RUNS: usize = 63;
        let skipped_rows = ROWS_PER_GROUP - selected_rows;
        assert!(selected_rows >= SELECTED_RUNS && skipped_rows >= SKIPPED_RUNS);
        let mut selectors = Vec::with_capacity(SELECTED_RUNS + SKIPPED_RUNS);
        for index in 0..SELECTED_RUNS {
            push_selector(
                &mut selectors,
                RowSelector::select(distributed_len(selected_rows, SELECTED_RUNS, index)),
            );
            if index < SKIPPED_RUNS {
                push_selector(
                    &mut selectors,
                    RowSelector::skip(distributed_len(skipped_rows, SKIPPED_RUNS, index)),
                );
            }
        }
        assert_eq!(selector_rows(&selectors), ROWS_PER_GROUP);
        let shape = Self {
            name: format!("tc_f_s{selected_rows}_t126"),
            nominal_skip: None,
            nominal_select: None,
            selectors,
        };
        let summary = shape.summary();
        assert_eq!(summary.selected_rows, selected_rows);
        assert_eq!(summary.first_selected_row, 0);
        assert_eq!(summary.last_selected_row_exclusive, ROWS_PER_GROUP);
        assert_eq!(summary.selected_run_count, SELECTED_RUNS);
        assert_eq!(summary.internal_skip_run_count(), SKIPPED_RUNS);
        assert_eq!(summary.internal_transition_count(), 126);
        shape
    }

    /// Tier-D high-transition guard surface. Both axes are explicit so the
    /// experiment can distinguish transition and selectivity boundaries while
    /// keeping the full selected span and exact run topology fixed.
    pub(crate) fn tier_d_transition_selectivity(
        selected_runs: usize,
        selected_rows: usize,
    ) -> Self {
        assert!(matches!(
            selected_runs,
            128 | 256 | 384 | 512 | 640 | 768 | 1_024
        ));
        assert!(matches!(
            selected_rows,
            1_024 | 8_192 | 32_768 | 57_344 | 64_512
        ));
        let skipped_runs = selected_runs - 1;
        let skipped_rows = ROWS_PER_GROUP - selected_rows;
        assert!(selected_rows >= selected_runs);
        assert!(skipped_rows >= skipped_runs);
        let mut selectors = Vec::with_capacity(selected_runs + skipped_runs);
        for index in 0..selected_runs {
            push_selector(
                &mut selectors,
                RowSelector::select(distributed_len(selected_rows, selected_runs, index)),
            );
            if index < skipped_runs {
                push_selector(
                    &mut selectors,
                    RowSelector::skip(distributed_len(skipped_rows, skipped_runs, index)),
                );
            }
        }
        assert_eq!(selector_rows(&selectors), ROWS_PER_GROUP);
        let shape = Self {
            name: format!("td_r{selected_runs}_s{selected_rows}"),
            nominal_skip: None,
            nominal_select: None,
            selectors,
        };
        let summary = shape.summary();
        assert_eq!(summary.selected_rows, selected_rows);
        assert_eq!(summary.first_selected_row, 0);
        assert_eq!(summary.last_selected_row_exclusive, ROWS_PER_GROUP);
        assert_eq!(summary.selected_run_count, selected_runs);
        assert_eq!(summary.internal_skip_run_count(), skipped_runs);
        assert_eq!(summary.internal_transition_count(), 2 * selected_runs - 2);
        shape
    }

    pub(crate) fn invariant_material(&self) -> String {
        let summary = self.summary();
        let runs = self
            .selectors
            .iter()
            .map(|selector| {
                format!(
                    "{}{}",
                    if selector.skip { 'K' } else { 'S' },
                    selector.row_count
                )
            })
            .collect::<Vec<_>>()
            .join("|");
        format!(
            "arrow-row-selection-shape-v1;N={ROWS_PER_GROUP};name={};M={};S={};selected_runs={};internal_skip_runs={};internal_transitions={};runs={runs}",
            self.name,
            summary.last_selected_row_exclusive - summary.first_selected_row,
            summary.selected_rows,
            summary.selected_run_count,
            summary.internal_skip_run_count(),
            summary.internal_transition_count(),
        )
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
        let selected_run_count = self
            .selectors
            .iter()
            .filter(|selector| !selector.skip)
            .count();
        let skipped_run_count = run_count - selected_run_count;
        let first_selected_row = self
            .selectors
            .iter()
            .scan(0usize, |position, selector| {
                let start = *position;
                *position += selector.row_count;
                Some((start, selector))
            })
            .find(|(_, selector)| !selector.skip)
            .map(|(start, _)| start)
            .expect("oracle shape must select at least one row");
        let last_selected_row_exclusive = self
            .selectors
            .iter()
            .scan(0usize, |position, selector| {
                *position += selector.row_count;
                Some((*position, selector))
            })
            .filter(|(_, selector)| !selector.skip)
            .map(|(end, _)| end)
            .last()
            .expect("oracle shape must select at least one row");
        let max_skip_run = self
            .selectors
            .iter()
            .filter(|selector| selector.skip)
            .map(|selector| selector.row_count)
            .max()
            .unwrap_or(0);
        let (long_skip_rows_1024, long_skip_count_1024) = long_skip_stats(&self.selectors, 1_024);
        let (long_skip_rows_4096, long_skip_count_4096) = long_skip_stats(&self.selectors, 4_096);
        OracleShapeSummary {
            selected_rows,
            skipped_rows,
            run_count,
            selected_run_count,
            skipped_run_count,
            avg_run_len: ROWS_PER_GROUP as f64 / run_count as f64,
            selected_fraction: selected_rows as f64 / ROWS_PER_GROUP as f64,
            first_selected_row,
            last_selected_row_exclusive,
            max_skip_run,
            long_skip_rows_1024,
            long_skip_count_1024,
            long_skip_rows_4096,
            long_skip_count_4096,
            long_skip_share_1024: long_skip_share(long_skip_rows_1024, skipped_rows),
            long_skip_share_4096: long_skip_share(long_skip_rows_4096, skipped_rows),
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

fn distributed_len(total: usize, parts: usize, index: usize) -> usize {
    assert!(parts > 0 && index < parts && total >= parts);
    total / parts + usize::from(index < total % parts)
}

fn long_skip_stats(selectors: &[RowSelector], threshold: usize) -> (usize, usize) {
    let matching = selectors
        .iter()
        .filter(|selector| selector.skip && selector.row_count >= threshold);
    (
        matching.clone().map(|selector| selector.row_count).sum(),
        matching.count(),
    )
}

fn long_skip_share(long_skip_rows: usize, skipped_rows: usize) -> f64 {
    if skipped_rows == 0 {
        return 0.0;
    }
    long_skip_rows as f64 / skipped_rows as f64
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

    for selected_runs in [2, 8, 32, 128, 512] {
        let summary = OracleShape::tier_c_transitions(selected_runs).summary();
        assert_eq!(summary.selected_rows, ROWS_PER_GROUP / 2);
        assert_eq!(summary.internal_transition_count(), 2 * selected_runs - 2);
    }
    for gap_rows in [64, 512, 4_096, 16_384, 63_488] {
        let summary = OracleShape::tier_c_gap(gap_rows).summary();
        assert_eq!(summary.selected_rows, 2_048);
        assert_eq!(summary.internal_transition_count(), 2);
        assert_eq!(summary.last_selected_row_exclusive, gap_rows + 2_048);
    }
    for selected_rows in [1_024, 8_192, 32_768, 57_344, 64_512] {
        let summary = OracleShape::tier_c_selectivity(selected_rows).summary();
        assert_eq!(summary.selected_rows, selected_rows);
        assert_eq!(summary.internal_transition_count(), 126);
    }
}
