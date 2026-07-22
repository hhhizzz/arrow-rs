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

//! Auto-policy benchmark for async Parquet row filtering.
//!
//! This benchmark is intended for direct comparisons between `main` and
//! candidate branches. Forced row-selection policies and decode-then-filter
//! diagnostics intentionally belong in separate oracle benchmarks.

mod filter_policy_common;

use criterion::{Criterion, criterion_group, criterion_main};
use filter_policy_common::cases::{AMORTIZATION_CASES, CORE_CASES, DRIFT_CASES};
use filter_policy_common::register::register_auto_group;
use filter_policy_common::shapes::assert_regular_and_bursty_have_same_summary;

fn benchmark_auto(c: &mut Criterion) {
    assert_regular_and_bursty_have_same_summary();

    register_auto_group(c, "arrow_reader_filter_policy/auto/core", CORE_CASES);
    register_auto_group(c, "arrow_reader_filter_policy/auto/drift", DRIFT_CASES);
    register_auto_group(
        c,
        "arrow_reader_filter_policy/auto/amortization",
        AMORTIZATION_CASES,
    );
}

criterion_group!(benches, benchmark_auto);
criterion_main!(benches);
