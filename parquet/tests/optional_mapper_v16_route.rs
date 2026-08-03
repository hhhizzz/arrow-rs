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

#![cfg(all(feature = "experimental", feature = "arrow"))]

use parquet::bench_support::optional_mapper_v16::{
    OPTIONAL_MAPPER_V16_ARM_ENV, OPTIONAL_MAPPER_V16_CASE_ENV, OPTIONAL_MAPPER_V16_CONTRACT_ENV,
    OPTIONAL_MAPPER_V16_CONTRACT_ID, OPTIONAL_MAPPER_V16_CRITERION_PREFIX,
    OPTIONAL_MAPPER_V16_ORACLE_PREFIX, OptionalMapperV16Arm, OptionalMapperV16Case,
    PreparedOptionalMapperV16, exact_optional_mapper_v16_binding,
    expected_optional_mapper_v16_route, observe_optional_mapper_v16_route,
    optional_mapper_v16_arm_supported, optional_mapper_v16_exact_env_request,
    optional_mapper_v16_route_requested_value, optional_mapper_v16_routes_are_exclusive,
};

#[test]
fn exact_contract_case_and_arm_bindings_are_frozen() {
    assert_eq!(
        OPTIONAL_MAPPER_V16_CONTRACT_ID,
        "arrow_unified_frame_observability.optional_mapper.v1_6.mapper_leaf"
    );
    assert_eq!(
        OPTIONAL_MAPPER_V16_CONTRACT_ENV,
        "PARQUET_OPTIONAL_MAPPER_V16_CONTRACT"
    );
    assert_eq!(
        OPTIONAL_MAPPER_V16_CASE_ENV,
        "PARQUET_OPTIONAL_MAPPER_V16_CASE"
    );
    assert_eq!(
        OPTIONAL_MAPPER_V16_ARM_ENV,
        "PARQUET_OPTIONAL_MAPPER_V16_ARM"
    );
    assert_eq!(OPTIONAL_MAPPER_V16_CRITERION_PREFIX, "optional_mapper_v16");
    assert_eq!(
        OPTIONAL_MAPPER_V16_ORACLE_PREFIX,
        "optional_mapper_v16_oracle"
    );
    assert_eq!(
        OptionalMapperV16Arm::ALL.map(OptionalMapperV16Arm::id),
        ["current_set_bit_scalar", "adaptive_scalar_v1", "bmi2_pext"]
    );

    for case in OptionalMapperV16Case::ALL {
        for arm in OptionalMapperV16Arm::ALL {
            assert_eq!(
                exact_optional_mapper_v16_binding(Some(case.id()), Some(arm.id())).unwrap(),
                (case, arm)
            );
        }
    }
    assert!(exact_optional_mapper_v16_binding(None, Some("adaptive_scalar_v1")).is_err());
    assert!(exact_optional_mapper_v16_binding(Some("av_s50_n8"), None).is_err());
    assert!(
        exact_optional_mapper_v16_binding(Some("unknown"), Some("adaptive_scalar_v1")).is_err()
    );
    assert!(exact_optional_mapper_v16_binding(Some("av_s50_n8"), Some("adaptive_scalar")).is_err());
}

#[test]
fn contract_value_and_route_exclusivity_fail_closed() {
    assert!(!optional_mapper_v16_route_requested_value(None).unwrap());
    assert!(
        optional_mapper_v16_route_requested_value(Some(OPTIONAL_MAPPER_V16_CONTRACT_ID)).unwrap()
    );
    for invalid in ["", "1", "true", "v1_6_a", "optional_mapper_v16"] {
        assert!(optional_mapper_v16_route_requested_value(Some(invalid)).is_err());
    }

    assert!(optional_mapper_v16_exact_env_request(None, None, None).is_ok_and(|value| !value));
    assert!(optional_mapper_v16_exact_env_request(None, Some("av_s50_n8"), None).is_err());
    assert!(optional_mapper_v16_exact_env_request(None, None, Some("adaptive_scalar_v1")).is_err());
    assert!(
        optional_mapper_v16_exact_env_request(
            Some(OPTIONAL_MAPPER_V16_CONTRACT_ID),
            None,
            Some("adaptive_scalar_v1")
        )
        .is_err()
    );
    assert!(
        optional_mapper_v16_exact_env_request(
            Some(OPTIONAL_MAPPER_V16_CONTRACT_ID),
            Some("av_s50_n8"),
            None
        )
        .is_err()
    );
    assert!(
        optional_mapper_v16_exact_env_request(
            Some(OPTIONAL_MAPPER_V16_CONTRACT_ID),
            Some("av_s50_n8"),
            Some("adaptive_scalar_v1")
        )
        .unwrap()
    );
    assert!(
        optional_mapper_v16_exact_env_request(
            Some(OPTIONAL_MAPPER_V16_CONTRACT_ID),
            Some("unknown"),
            Some("adaptive_scalar_v1")
        )
        .is_err()
    );
    assert!(
        optional_mapper_v16_exact_env_request(
            Some(OPTIONAL_MAPPER_V16_CONTRACT_ID),
            Some("av_s50_n8"),
            Some("adaptive_scalar")
        )
        .is_err()
    );

    assert!(optional_mapper_v16_routes_are_exclusive(false, &[]).is_ok());
    assert!(optional_mapper_v16_routes_are_exclusive(true, &[]).is_ok());
    assert!(optional_mapper_v16_routes_are_exclusive(false, &[true]).is_ok());
    assert!(optional_mapper_v16_routes_are_exclusive(true, &[false, false]).is_ok());
    assert!(optional_mapper_v16_routes_are_exclusive(true, &[true, false]).is_err());
    assert!(optional_mapper_v16_routes_are_exclusive(false, &[true, true]).is_err());
}

#[test]
fn untimed_observation_proves_exactly_one_forced_outer_route() {
    for case in OptionalMapperV16Case::ALL {
        for arm in OptionalMapperV16Arm::ALL {
            if !optional_mapper_v16_arm_supported(arm) {
                assert_eq!(arm, OptionalMapperV16Arm::Bmi2Pext);
                assert!(PreparedOptionalMapperV16::try_new(case, arm).is_err());
                assert!(observe_optional_mapper_v16_route(case, arm).is_err());
                continue;
            }
            let observed = observe_optional_mapper_v16_route(case, arm).unwrap();
            let expected = expected_optional_mapper_v16_route(case, arm);
            assert_eq!(observed, expected, "case={} arm={}", case.id(), arm.id());
            assert_eq!(
                observed.current_backend_fragments
                    + observed.adaptive_backend_fragments
                    + observed.bmi2_backend_fragments,
                1
            );
            assert_eq!(
                observed.current_scalar_compression_calls
                    + observed.adaptive_physical_sparse_calls
                    + observed.adaptive_physical_fallback_calls
                    + observed.adaptive_output_sparse_calls
                    + observed.adaptive_output_fallback_calls
                    + observed.bmi2_compression_calls,
                observed.physical_compression_calls + observed.output_compression_calls
            );
        }
    }
}

#[test]
fn adaptive_threshold_boundary_cases_take_the_expected_subroutes() {
    let n8 = expected_optional_mapper_v16_route(
        OptionalMapperV16Case::AllValidS50N8,
        OptionalMapperV16Arm::AdaptiveScalar,
    );
    assert_eq!(n8.adaptive_physical_sparse_calls, 128);
    assert_eq!(n8.adaptive_physical_fallback_calls, 0);

    let n9 = expected_optional_mapper_v16_route(
        OptionalMapperV16Case::AllValidS50N9,
        OptionalMapperV16Arm::AdaptiveScalar,
    );
    assert_eq!(n9.adaptive_physical_sparse_calls, 0);
    assert_eq!(n9.adaptive_physical_fallback_calls, 128);

    let sn4 = expected_optional_mapper_v16_route(
        OptionalMapperV16Case::GeneralS50N9Sn4,
        OptionalMapperV16Arm::AdaptiveScalar,
    );
    assert_eq!(sn4.adaptive_output_sparse_calls, 128);
    assert_eq!(sn4.adaptive_output_fallback_calls, 0);

    let sn5 = expected_optional_mapper_v16_route(
        OptionalMapperV16Case::GeneralS50N16Sn5,
        OptionalMapperV16Arm::AdaptiveScalar,
    );
    assert_eq!(sn5.adaptive_output_sparse_calls, 0);
    assert_eq!(sn5.adaptive_output_fallback_calls, 128);
}
