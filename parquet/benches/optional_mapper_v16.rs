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

use criterion::{Criterion, SamplingMode, Throughput, criterion_group, criterion_main};
use parquet::bench_support::optional_mapper_v16::{
    OPTIONAL_MAPPER_V16_ARM_ENV, OPTIONAL_MAPPER_V16_CASE_ENV, OPTIONAL_MAPPER_V16_CONTRACT_ENV,
    OPTIONAL_MAPPER_V16_CONTRACT_ID, OPTIONAL_MAPPER_V16_CRITERION_PREFIX,
    OPTIONAL_MAPPER_V16_ORACLE_PREFIX, OptionalMapperV16Arm, OptionalMapperV16Case,
    PreparedOptionalMapperV16, exact_optional_mapper_v16_binding,
    expected_optional_mapper_v16_route, observe_optional_mapper_v16_route,
    optional_mapper_v16_exact_env_request,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Binding {
    case: OptionalMapperV16Case,
    arm: OptionalMapperV16Arm,
}

pub(crate) fn requested() -> bool {
    let contract = optional_env(OPTIONAL_MAPPER_V16_CONTRACT_ENV);
    let case = optional_env(OPTIONAL_MAPPER_V16_CASE_ENV);
    let arm = optional_env(OPTIONAL_MAPPER_V16_ARM_ENV);
    optional_mapper_v16_exact_env_request(contract.as_deref(), case.as_deref(), arm.as_deref())
        .unwrap_or_else(|error| panic!("invalid optional-mapper-v16 route: {error}"))
}

fn optional_env(name: &str) -> Option<String> {
    match env::var(name) {
        Ok(value) => Some(value),
        Err(env::VarError::NotPresent) => None,
        Err(env::VarError::NotUnicode(_)) => panic!("{name} must contain valid UTF-8"),
    }
}

pub(crate) fn timing_binding(
    case: Option<&str>,
    arm: Option<&str>,
) -> Result<Binding, &'static str> {
    let (case, arm) = exact_optional_mapper_v16_binding(case, arm)?;
    Ok(Binding { case, arm })
}

pub(crate) fn benchmark_id(binding: Binding) -> String {
    format!(
        "{}/{}/{}",
        OPTIONAL_MAPPER_V16_CRITERION_PREFIX,
        binding.case.id(),
        binding.arm.id()
    )
}

fn criterion_benchmark(c: &mut Criterion) {
    assert!(
        requested(),
        "set PARQUET_OPTIONAL_MAPPER_V16_CONTRACT, CASE, and ARM to one exact benchmark binding"
    );
    let case = env::var(OPTIONAL_MAPPER_V16_CASE_ENV).ok();
    let arm = env::var(OPTIONAL_MAPPER_V16_ARM_ENV).ok();
    let binding = timing_binding(case.as_deref(), arm.as_deref())
        .unwrap_or_else(|error| panic!("invalid optional-mapper-v16 binding: {error}"));
    let full_id = benchmark_id(binding);
    let mut prepared = PreparedOptionalMapperV16::try_new(binding.case, binding.arm)
        .unwrap_or_else(|error| panic!("failed to bind optional-mapper-v16 fixture: {error}"));
    let metadata = prepared.metadata();

    let route = observe_optional_mapper_v16_route(binding.case, binding.arm)
        .unwrap_or_else(|error| panic!("optional-mapper-v16 route oracle failed: {error}"));
    assert_eq!(
        route,
        expected_optional_mapper_v16_route(binding.case, binding.arm),
        "forced mapper route did not execute exactly"
    );
    let before = prepared
        .invoke_oracle()
        .unwrap_or_else(|error| panic!("pre-timing optional-mapper-v16 oracle failed: {error}"));
    assert!(before.workspace_unchanged());
    assert_eq!(before.semantic_digest, metadata.semantic_digest);
    eprintln!(
        "{} phase=pre contract={} benchmark_id={} case={} arm={} fixture_digest={:016x} semantic_digest={:016x} logical_rows={} frames={} selected_rows={} present_rows={} selected_present_rows={} selected_null_rows={} physical_compression_calls={} output_compression_calls={} morphology_distributed={} morphology_clustered={} morphology_edge_heavy={} current_backend_fragments={} adaptive_backend_fragments={} bmi2_backend_fragments={} current_scalar_compression_calls={} adaptive_physical_sparse_calls={} adaptive_physical_fallback_calls={} adaptive_output_sparse_calls={} adaptive_output_fallback_calls={} bmi2_compression_calls={} physical_pointer={} physical_capacity={} validity_pointer={:?} validity_capacity={:?}",
        OPTIONAL_MAPPER_V16_ORACLE_PREFIX,
        OPTIONAL_MAPPER_V16_CONTRACT_ID,
        full_id,
        metadata.id,
        binding.arm.id(),
        metadata.fixture_digest,
        before.semantic_digest,
        metadata.logical_rows,
        metadata.frames,
        metadata.selected_rows,
        metadata.present_rows,
        metadata.selected_present_rows,
        metadata.selected_null_rows,
        metadata.physical_compression_calls,
        metadata.output_compression_calls,
        metadata.morphology_counts[0],
        metadata.morphology_counts[1],
        metadata.morphology_counts[2],
        route.current_backend_fragments,
        route.adaptive_backend_fragments,
        route.bmi2_backend_fragments,
        route.current_scalar_compression_calls,
        route.adaptive_physical_sparse_calls,
        route.adaptive_physical_fallback_calls,
        route.adaptive_output_sparse_calls,
        route.adaptive_output_fallback_calls,
        route.bmi2_compression_calls,
        before.workspace_after.physical_pointer,
        before.workspace_after.physical_capacity,
        before.workspace_after.validity_pointer,
        before.workspace_after.validity_capacity,
    );

    let mut group = c.benchmark_group(format!(
        "{}/{}",
        OPTIONAL_MAPPER_V16_CRITERION_PREFIX, metadata.id
    ));
    group.sampling_mode(SamplingMode::Flat);
    group.throughput(Throughput::Elements(metadata.logical_rows as u64));
    group.bench_function(binding.arm.id(), |bencher| {
        bencher.iter(|| {
            let outcome = prepared
                .invoke_lean()
                .unwrap_or_else(|error| panic!("timed optional-mapper-v16 call failed: {error}"));
            std::hint::black_box(outcome)
        });
    });
    group.finish();

    let after = prepared
        .invoke_oracle()
        .unwrap_or_else(|error| panic!("post-timing optional-mapper-v16 oracle failed: {error}"));
    assert!(after.workspace_unchanged());
    assert_eq!(after.semantic_digest, before.semantic_digest);
    assert_eq!(after.timed, before.timed);
    assert_eq!(after.workspace_before, before.workspace_after);
    eprintln!(
        "{} phase=post contract={} benchmark_id={} fixture_digest={:016x} semantic_digest={:016x} physical_pointer={} physical_capacity={} validity_pointer={:?} validity_capacity={:?}",
        OPTIONAL_MAPPER_V16_ORACLE_PREFIX,
        OPTIONAL_MAPPER_V16_CONTRACT_ID,
        full_id,
        metadata.fixture_digest,
        after.semantic_digest,
        after.workspace_after.physical_pointer,
        after.workspace_after.physical_capacity,
        after.workspace_after.validity_pointer,
        after.workspace_after.validity_capacity,
    );
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
