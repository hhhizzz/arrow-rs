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

use std::collections::VecDeque;
use std::hint;

use arrow_buffer::{BooleanBuffer, BooleanBufferBuilder};
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use parquet::arrow::arrow_reader::{RowSelection, RowSelector};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

const TOTAL_ROWS: usize = 1 << 20;
const BATCH_SIZE: usize = 1 << 10;
const BASE_SEED: u64 = 0xA55AA55A;

fn criterion_benchmark(c: &mut Criterion) {
    let avg_selector_lengths: &[usize] = &[2, 4, 8, 16, 32, 64, 128];

    let scenarios = [
        Scenario {
            name: "uniform50",
            select_ratio: 0.5,
            start_with_select: false,
            distribution: RunDistribution::Constant,
        },
        Scenario {
            name: "spread50",
            select_ratio: 0.5,
            start_with_select: false,
            distribution: RunDistribution::Uniform { spread: 0.9 },
        },
        Scenario {
            name: "sparse20",
            select_ratio: 0.2,
            start_with_select: false,
            distribution: RunDistribution::Bimodal {
                long_factor: 6.0,
                long_prob: 0.1,
            },
        },
        Scenario {
            name: "dense80",
            select_ratio: 0.8,
            start_with_select: true,
            distribution: RunDistribution::Bimodal {
                long_factor: 4.0,
                long_prob: 0.05,
            },
        },
    ];

    for scenario in scenarios.iter() {
        for (offset, &avg_len) in avg_selector_lengths.iter().enumerate() {
            let selectors =
                generate_selectors(avg_len, TOTAL_ROWS, scenario, BASE_SEED + offset as u64);
            let stats = SelectorStats::new(&selectors);
            let suffix = format!(
                "{}-avg{:.1}-sel{:02}",
                scenario.name,
                stats.average_selector_len,
                (stats.select_ratio * 100.0).round() as u32
            );

            c.bench_with_input(
                BenchmarkId::new("mask_build", &suffix),
                &selectors,
                |b, selectors| {
                    b.iter(|| {
                        let mask = MaskState::build_mask(selectors);
                        hint::black_box(mask);
                    });
                },
            );

            c.bench_with_input(
                BenchmarkId::new("selector_build", &suffix),
                &selectors,
                |b, selectors| {
                    b.iter(|| {
                        let queue = SelectorState::build_queue(selectors);
                        hint::black_box(queue);
                    });
                },
            );

            let mask_state = MaskState::new(&selectors);
            c.bench_with_input(
                BenchmarkId::new("mask_scan", &suffix),
                &mask_state,
                |b, state| {
                    b.iter(|| {
                        let mut run = state.clone();
                        let total = run.consume_all(BATCH_SIZE);
                        hint::black_box(total);
                    });
                },
            );

            let selector_state = SelectorState::new(&selectors);
            c.bench_with_input(
                BenchmarkId::new("selector_scan", &suffix),
                &selector_state,
                |b, state| {
                    b.iter(|| {
                        let mut run = state.clone();
                        let total = run.consume_all(BATCH_SIZE);
                        hint::black_box(total);
                    });
                },
            );
        }
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);

#[derive(Clone)]
struct Scenario {
    name: &'static str,
    select_ratio: f64,
    start_with_select: bool,
    distribution: RunDistribution,
}

#[derive(Clone)]
enum RunDistribution {
    Constant,
    Uniform { spread: f64 },
    Bimodal { long_factor: f64, long_prob: f64 },
}

fn generate_selectors(
    avg_selector_len: usize,
    total_rows: usize,
    scenario: &Scenario,
    seed: u64,
) -> Vec<RowSelector> {
    assert!(
        (0.0..=1.0).contains(&scenario.select_ratio),
        "select_ratio must be in [0, 1]"
    );

    let mut select_mean = scenario.select_ratio * 2.0 * avg_selector_len as f64;
    let mut skip_mean = (1.0 - scenario.select_ratio) * 2.0 * avg_selector_len as f64;

    select_mean = select_mean.max(1.0);
    skip_mean = skip_mean.max(1.0);

    let sum = select_mean + skip_mean;
    // Rebalance the sampled select/skip run lengths so their sum matches the requested
    // average selector length while respecting the configured selectivity ratio.
    let scale = if sum == 0.0 {
        1.0
    } else {
        (2.0 * avg_selector_len as f64) / sum
    };
    select_mean *= scale;
    skip_mean *= scale;

    let mut rng = StdRng::seed_from_u64(seed ^ (avg_selector_len as u64).wrapping_mul(0x9E3779B1));
    let mut selectors = Vec::with_capacity(total_rows / avg_selector_len.max(1));
    let mut remaining = total_rows;
    let mut is_select = scenario.start_with_select;

    while remaining > 0 {
        let mean = if is_select { select_mean } else { skip_mean };
        let len = sample_length(mean, &scenario.distribution, &mut rng).max(1);
        let len = len.min(remaining);
        selectors.push(if is_select {
            RowSelector::select(len)
        } else {
            RowSelector::skip(len)
        });
        remaining -= len;
        if remaining == 0 {
            break;
        }
        is_select = !is_select;
    }

    let selection: RowSelection = selectors.into();
    selection.into()
}

fn sample_length(mean: f64, distribution: &RunDistribution, rng: &mut StdRng) -> usize {
    match distribution {
        RunDistribution::Constant => mean.round().max(1.0) as usize,
        RunDistribution::Uniform { spread } => {
            let spread = spread.clamp(0.0, 0.99);
            let lower = (mean * (1.0 - spread)).max(1.0);
            let upper = (mean * (1.0 + spread)).max(lower + f64::EPSILON);
            if (upper - lower) < 1.0 {
                lower.round().max(1.0) as usize
            } else {
                let low = lower.floor() as usize;
                let high = upper.ceil() as usize;
                rng.random_range(low..=high).max(1)
            }
        }
        RunDistribution::Bimodal {
            long_factor,
            long_prob,
        } => {
            let long_prob = long_prob.clamp(0.0, 0.5);
            let short_prob = 1.0 - long_prob;
            let short_factor = if short_prob == 0.0 {
                1.0 / long_factor.max(f64::EPSILON)
            } else {
                (1.0 - long_prob * long_factor).max(0.0) / short_prob
            };
            let use_long = rng.random_bool(long_prob);
            let factor = if use_long {
                *long_factor
            } else {
                short_factor.max(0.1)
            };
            (mean * factor).round().max(1.0) as usize
        }
    }
}

#[derive(Clone)]
struct MaskState {
    mask: BooleanBuffer,
    position: usize,
}

impl MaskState {
    fn new(selectors: &[RowSelector]) -> Self {
        Self {
            mask: Self::build_mask(selectors),
            position: 0,
        }
    }

    fn build_mask(selectors: &[RowSelector]) -> BooleanBuffer {
        let total_rows: usize = selectors.iter().map(|s| s.row_count).sum();
        let mut builder = BooleanBufferBuilder::new(total_rows);
        for selector in selectors {
            builder.append_n(selector.row_count, !selector.skip);
        }
        builder.finish()
    }

    fn consume_all(&mut self, batch_size: usize) -> usize {
        let mut selected = 0;
        while let Some(batch) = self.next_mask_batch(batch_size) {
            hint::black_box(batch.initial_skip);
            hint::black_box(batch.chunk_rows);
            hint::black_box(batch.mask_start);
            selected += batch.selected_rows;
        }
        selected
    }

    fn next_mask_batch(&mut self, batch_size: usize) -> Option<MaskBatch> {
        let mask = &self.mask;
        if self.position >= mask.len() {
            return None;
        }

        let start_position = self.position;
        let mut cursor = start_position;
        let mut initial_skip = 0usize;

        while cursor < mask.len() && !mask.value(cursor) {
            initial_skip += 1;
            cursor += 1;
        }

        let mask_start = cursor;
        let mut chunk_rows = 0usize;
        let mut selected_rows = 0usize;

        while cursor < mask.len() && selected_rows < batch_size {
            chunk_rows += 1;
            if mask.value(cursor) {
                selected_rows += 1;
            }
            cursor += 1;
        }

        self.position = cursor;

        Some(MaskBatch {
            initial_skip,
            chunk_rows,
            selected_rows,
            mask_start,
        })
    }
}

#[derive(Clone)]
struct MaskBatch {
    initial_skip: usize,
    chunk_rows: usize,
    selected_rows: usize,
    mask_start: usize,
}

#[derive(Clone)]
struct SelectorState {
    selectors: VecDeque<RowSelector>,
}

impl SelectorState {
    fn new(selectors: &[RowSelector]) -> Self {
        Self {
            selectors: Self::build_queue(selectors),
        }
    }

    fn build_queue(selectors: &[RowSelector]) -> VecDeque<RowSelector> {
        selectors.iter().copied().collect()
    }

    fn consume_all(&mut self, batch_size: usize) -> usize {
        // Drain the selector queue in `batch_size` chunks, splitting SELECT runs
        // when they span across multiple batches.
        let mut selected_total = 0usize;

        while !self.selectors.is_empty() {
            let mut selected_rows = 0usize;

            while selected_rows < batch_size {
                let front = match self.selectors.pop_front() {
                    Some(selector) => selector,
                    None => break,
                };

                hint::black_box(front.row_count);

                if front.skip {
                    continue;
                }

                let need = (batch_size - selected_rows).min(front.row_count);
                selected_rows += need;
                if front.row_count > need {
                    self.selectors
                        .push_front(RowSelector::select(front.row_count - need));
                    break;
                }
            }

            hint::black_box(selected_rows);
            selected_total += selected_rows;

            if selected_rows == 0 && self.selectors.is_empty() {
                break;
            }
        }

        selected_total
    }
}

struct SelectorStats {
    average_selector_len: f64,
    select_ratio: f64,
}

impl SelectorStats {
    fn new(selectors: &[RowSelector]) -> Self {
        if selectors.is_empty() {
            return Self {
                average_selector_len: 0.0,
                select_ratio: 0.0,
            };
        }

        let total_rows: usize = selectors.iter().map(|s| s.row_count).sum();
        let selected_rows: usize = selectors
            .iter()
            .filter(|s| !s.skip)
            .map(|s| s.row_count)
            .sum();

        Self {
            average_selector_len: total_rows as f64 / selectors.len() as f64,
            select_ratio: if total_rows == 0 {
                0.0
            } else {
                selected_rows as f64 / total_rows as f64
            },
        }
    }
}
