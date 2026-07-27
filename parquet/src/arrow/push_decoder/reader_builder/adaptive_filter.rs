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

use crate::arrow::arrow_reader::RowSelection;

/// One-shot execution choice for a scan's row filter.
///
/// The first eligible row group uses predicate pushdown and supplies the only
/// observation. The choice is then fixed so later row groups do not repeatedly
/// classify selections in the decode path.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) enum AdaptiveFilterMode {
    #[default]
    Observe,
    Pushdown,
    PostFilter,
}

#[derive(Debug, Default)]
pub(super) struct AdaptiveFilter {
    mode: AdaptiveFilterMode,
}

impl AdaptiveFilter {
    pub(super) fn mode(&self) -> AdaptiveFilterMode {
        self.mode
    }

    pub(super) fn observe(&mut self, selection: Option<&RowSelection>, row_count: usize) {
        debug_assert_eq!(self.mode, AdaptiveFilterMode::Observe);
        self.mode = if SelectionShape::new(selection, row_count).should_post_filter() {
            AdaptiveFilterMode::PostFilter
        } else {
            AdaptiveFilterMode::Pushdown
        };
    }

    pub(super) fn force_pushdown(&mut self) {
        self.mode = AdaptiveFilterMode::Pushdown;
    }
}

#[derive(Debug, Default)]
struct SelectionShape {
    selected_rows: usize,
    total_rows: usize,
    selector_count: usize,
    selected_run_count: usize,
}

impl SelectionShape {
    const MIN_SELECTED_PERCENT: u128 = 8;
    const MAX_SELECTED_PERCENT: u128 = 50;
    const MAX_SELECTED_RUN_LENGTH: u128 = 4;
    const MIN_SELECTORS_PER_HUNDRED_ROWS: u128 = 1;

    fn new(selection: Option<&RowSelection>, row_count: usize) -> Self {
        let Some(selection) = selection else {
            return Self {
                selected_rows: row_count,
                total_rows: row_count,
                selector_count: usize::from(row_count != 0),
                selected_run_count: usize::from(row_count != 0),
            };
        };

        selection
            .iter()
            .fold(Self::default(), |mut shape, selector| {
                if selector.row_count != 0 {
                    shape.total_rows += selector.row_count;
                    shape.selector_count += 1;
                    if !selector.skip {
                        shape.selected_rows += selector.row_count;
                        shape.selected_run_count += 1;
                    }
                }
                shape
            })
    }

    fn should_post_filter(&self) -> bool {
        let selected = self.selected_rows as u128;
        let total = self.total_rows as u128;
        let selectors = self.selector_count as u128;
        let selected_runs = self.selected_run_count as u128;

        total != 0
            && selected_runs != 0
            // Limit fallback to the measured moderate-selectivity region.
            && selected * 100 >= total * Self::MIN_SELECTED_PERCENT
            && selected * 100 < total * Self::MAX_SELECTED_PERCENT
            // Dense selector boundaries make skip/read execution expensive.
            && selected <= selected_runs * Self::MAX_SELECTED_RUN_LENGTH
            && selectors * 100 >= total * Self::MIN_SELECTORS_PER_HUNDRED_ROWS
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arrow::arrow_reader::RowSelector;

    #[test]
    fn fragmented_moderate_selection_activates_post_filter() {
        let selection = RowSelection::from(
            (0..10)
                .flat_map(|_| [RowSelector::select(1), RowSelector::skip(9)])
                .collect::<Vec<_>>(),
        );
        let mut adaptive = AdaptiveFilter::default();

        assert_eq!(adaptive.mode(), AdaptiveFilterMode::Observe);
        adaptive.observe(Some(&selection), 100);
        assert_eq!(adaptive.mode(), AdaptiveFilterMode::PostFilter);
    }

    #[test]
    fn unproven_shapes_stay_on_pushdown() {
        let cases = [
            // Too sparse: 5% selected.
            (1, 19),
            // Too dense: exactly 50% selected.
            (1, 1),
            // Moderate but clustered into long selected runs.
            (10, 10),
        ];

        for (selected, skipped) in cases {
            let selection = RowSelection::from(
                (0..10)
                    .flat_map(|_| [RowSelector::select(selected), RowSelector::skip(skipped)])
                    .collect::<Vec<_>>(),
            );
            let mut adaptive = AdaptiveFilter::default();
            adaptive.observe(Some(&selection), (selected + skipped) * 10);
            assert_eq!(adaptive.mode(), AdaptiveFilterMode::Pushdown);
        }
    }
}
