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

use std::env;
use std::ops::Range;

const MAX_REQUESTS_ENV: &str = "DATAFUSION_PARQUET_DEBUG_BATCH_PREDICATE_REQUESTS_MAX_BATCHES";
const MAX_GAP_BYTES_ENV: &str = "DATAFUSION_PARQUET_DEBUG_BATCH_PREDICATE_REQUESTS_MAX_GAP_BYTES";
const MAX_BYTES_ENV: &str = "DATAFUSION_PARQUET_DEBUG_BATCH_PREDICATE_REQUESTS_MAX_BYTES";
const MAX_EXTRA_BYTES_ENV: &str =
    "DATAFUSION_PARQUET_DEBUG_BATCH_PREDICATE_REQUESTS_MAX_EXTRA_BYTES";
const MAX_OVERREAD_RATIO_ENV: &str =
    "DATAFUSION_PARQUET_DEBUG_BATCH_PREDICATE_REQUESTS_MAX_OVERREAD_RATIO";

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct PredicateRequestBatchConfig {
    max_requests: usize,
    max_gap_bytes: u64,
    max_bytes: u64,
    max_extra_bytes: u64,
    max_overread_ratio_basis_points: Option<u64>,
}

impl PredicateRequestBatchConfig {
    pub(crate) fn from_env() -> Option<Self> {
        let max_requests = parse_env_usize(MAX_REQUESTS_ENV)?;
        if max_requests <= 1 {
            return None;
        }

        let max_gap_bytes = parse_env_u64(MAX_GAP_BYTES_ENV).unwrap_or(0);
        let max_bytes = parse_env_u64(MAX_BYTES_ENV).unwrap_or(u64::MAX);
        let max_extra_bytes = parse_env_u64(MAX_EXTRA_BYTES_ENV).unwrap_or(u64::MAX);
        let max_overread_ratio_basis_points =
            parse_env_overread_ratio_basis_points(MAX_OVERREAD_RATIO_ENV);
        if max_bytes == 0 {
            return None;
        }

        Some(Self {
            max_requests,
            max_gap_bytes,
            max_bytes,
            max_extra_bytes,
            max_overread_ratio_basis_points,
        })
    }

    #[cfg(test)]
    pub(crate) fn new_for_test(max_requests: usize, max_gap_bytes: u64, max_bytes: u64) -> Self {
        Self {
            max_requests,
            max_gap_bytes,
            max_bytes,
            max_extra_bytes: u64::MAX,
            max_overread_ratio_basis_points: None,
        }
    }

    #[cfg(test)]
    pub(crate) fn with_max_extra_bytes_for_test(mut self, max_extra_bytes: u64) -> Self {
        self.max_extra_bytes = max_extra_bytes;
        self
    }

    #[cfg(test)]
    pub(crate) fn with_max_overread_ratio_for_test(mut self, max_overread_ratio: f64) -> Self {
        self.max_overread_ratio_basis_points =
            overread_ratio_basis_points(max_overread_ratio).filter(|ratio| *ratio >= 10_000);
        self
    }

    pub(crate) fn max_requests(self) -> usize {
        self.max_requests
    }

    fn max_gap_bytes(self) -> u64 {
        self.max_gap_bytes
    }

    fn max_bytes(self) -> u64 {
        self.max_bytes
    }

    fn max_extra_bytes(self) -> u64 {
        self.max_extra_bytes
    }

    fn max_overread_ratio_basis_points(self) -> Option<u64> {
        self.max_overread_ratio_basis_points
    }
}

#[derive(Debug)]
pub(crate) struct PredicateRequestBatch {
    config: PredicateRequestBatchConfig,
    request_count: usize,
    requested_bytes: u64,
    requested_ranges: Vec<Range<u64>>,
    ranges: Vec<Range<u64>>,
}

impl PredicateRequestBatch {
    pub(crate) fn new(config: PredicateRequestBatchConfig, ranges: Vec<Range<u64>>) -> Self {
        let requested_bytes = range_bytes(&ranges);
        let request_count = usize::from(requested_bytes != 0);
        let requested_ranges = ranges;
        let ranges =
            coalesce_predicate_request_ranges(requested_ranges.clone(), config.max_gap_bytes());
        Self {
            config,
            request_count,
            requested_bytes,
            requested_ranges,
            ranges,
        }
    }

    pub(crate) fn try_push(&mut self, ranges: Vec<Range<u64>>) -> bool {
        if ranges.is_empty() {
            return true;
        }
        if self.request_count >= self.config.max_requests() {
            return false;
        }

        let mut candidate_requested_ranges = self.requested_ranges.clone();
        let candidate_requested_bytes = self.requested_bytes.saturating_add(range_bytes(&ranges));
        candidate_requested_ranges.extend(ranges);

        let coalesced_candidate = coalesce_predicate_request_ranges(
            candidate_requested_ranges.clone(),
            self.config.max_gap_bytes(),
        );
        let coalesced_candidate_bytes = range_bytes(&coalesced_candidate);
        let candidate = if self
            .config
            .allows_fetch_bytes(candidate_requested_bytes, coalesced_candidate_bytes)
        {
            coalesced_candidate
        } else {
            let mut uncoalesced_candidate = self.ranges.clone();
            uncoalesced_candidate.extend(
                candidate_requested_ranges[self.requested_ranges.len()..]
                    .iter()
                    .cloned(),
            );
            let uncoalesced_candidate = coalesce_predicate_request_ranges(uncoalesced_candidate, 0);
            let uncoalesced_candidate_bytes = range_bytes(&uncoalesced_candidate);
            if !self
                .config
                .allows_fetch_bytes(candidate_requested_bytes, uncoalesced_candidate_bytes)
            {
                return false;
            }
            uncoalesced_candidate
        };

        self.requested_ranges = candidate_requested_ranges;
        self.ranges = candidate;
        self.requested_bytes = candidate_requested_bytes;
        self.request_count += 1;
        true
    }

    pub(crate) fn request_count(&self) -> usize {
        self.request_count
    }

    pub(crate) fn ranges(&self) -> &[Range<u64>] {
        &self.ranges
    }

    pub(crate) fn extra_bytes(&self) -> u64 {
        range_bytes(&self.ranges).saturating_sub(self.requested_bytes)
    }

    pub(crate) fn into_ranges(self) -> Vec<Range<u64>> {
        self.ranges
    }
}

impl PredicateRequestBatchConfig {
    fn allows_fetch_bytes(self, requested_bytes: u64, fetch_bytes: u64) -> bool {
        if fetch_bytes > self.max_bytes() {
            return false;
        }
        if fetch_bytes.saturating_sub(requested_bytes) > self.max_extra_bytes() {
            return false;
        }
        let Some(max_overread_ratio_basis_points) = self.max_overread_ratio_basis_points() else {
            return true;
        };
        if requested_bytes == 0 {
            return false;
        }
        u128::from(fetch_bytes) * 10_000
            <= u128::from(requested_bytes) * u128::from(max_overread_ratio_basis_points)
    }
}

pub(crate) fn coalesce_predicate_request_ranges(
    mut ranges: Vec<Range<u64>>,
    max_gap_bytes: u64,
) -> Vec<Range<u64>> {
    ranges.retain(|range| range.start < range.end);
    ranges.sort_by_key(|range| (range.start, range.end));

    let mut coalesced: Vec<Range<u64>> = Vec::with_capacity(ranges.len());
    for range in ranges {
        if let Some(last) = coalesced.last_mut()
            && range.start <= last.end.saturating_add(max_gap_bytes)
        {
            last.end = last.end.max(range.end);
            continue;
        }
        coalesced.push(range);
    }
    coalesced
}

pub(crate) fn range_bytes(ranges: &[Range<u64>]) -> u64 {
    ranges
        .iter()
        .map(|range| range.end.saturating_sub(range.start))
        .sum()
}

fn parse_env_usize(name: &str) -> Option<usize> {
    env::var(name).ok()?.parse().ok()
}

fn parse_env_u64(name: &str) -> Option<u64> {
    env::var(name).ok()?.parse().ok()
}

fn parse_env_overread_ratio_basis_points(name: &str) -> Option<u64> {
    overread_ratio_basis_points(env::var(name).ok()?.parse().ok()?)
}

fn overread_ratio_basis_points(ratio: f64) -> Option<u64> {
    ratio
        .is_finite()
        .then_some(ratio)
        .filter(|ratio| *ratio >= 0.0)
        .map(|ratio| (ratio * 10_000.0) as u64)
}
