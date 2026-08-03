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

use std::collections::HashSet;

use parquet::bench_support::optional_mapper_v16::{
    OPTIONAL_MAPPER_V16_FRAMES, OPTIONAL_MAPPER_V16_LOGICAL_ROWS, OptionalMapperV16Arm,
    OptionalMapperV16Case, OptionalMapperV16FrameClass, OptionalMapperV16Morphology,
    PreparedOptionalMapperV16, optional_mapper_v16_arm_supported,
    optional_mapper_v16_case_metadata, optional_mapper_v16_case_spec,
    optional_mapper_v16_frame_facts,
};

#[test]
fn frozen_matrix_has_18_exact_per_frame_cases() {
    // `s90` is a density label, frozen at 56/64 (87.5%) so the n16/sn8
    // orthogonal case remains mathematically feasible with exact frame counts.
    let expected = [
        (
            "identity_s05_n0",
            OptionalMapperV16FrameClass::AllPresentIdentity,
            3,
            0,
            0,
        ),
        (
            "identity_s90_n0",
            OptionalMapperV16FrameClass::AllPresentIdentity,
            56,
            0,
            0,
        ),
        (
            "full_s100_n4",
            OptionalMapperV16FrameClass::FullSelection,
            64,
            4,
            4,
        ),
        (
            "empty_n16",
            OptionalMapperV16FrameClass::EmptySelection,
            0,
            16,
            0,
        ),
        (
            "av_s25_n1",
            OptionalMapperV16FrameClass::AllValidSelected,
            16,
            1,
            0,
        ),
        (
            "av_s25_n4",
            OptionalMapperV16FrameClass::AllValidSelected,
            16,
            4,
            0,
        ),
        (
            "av_s50_n8",
            OptionalMapperV16FrameClass::AllValidSelected,
            32,
            8,
            0,
        ),
        (
            "av_s50_n9",
            OptionalMapperV16FrameClass::AllValidSelected,
            32,
            9,
            0,
        ),
        (
            "av_s50_n16",
            OptionalMapperV16FrameClass::AllValidSelected,
            32,
            16,
            0,
        ),
        (
            "av_s90_n4",
            OptionalMapperV16FrameClass::AllValidSelected,
            56,
            4,
            0,
        ),
        (
            "g_s05_n1_sn1",
            OptionalMapperV16FrameClass::General,
            3,
            1,
            1,
        ),
        (
            "g_s25_n2_sn1",
            OptionalMapperV16FrameClass::General,
            16,
            2,
            1,
        ),
        (
            "g_s25_n4_sn2",
            OptionalMapperV16FrameClass::General,
            16,
            4,
            2,
        ),
        (
            "g_s50_n8_sn4",
            OptionalMapperV16FrameClass::General,
            32,
            8,
            4,
        ),
        (
            "g_s50_n9_sn4",
            OptionalMapperV16FrameClass::General,
            32,
            9,
            4,
        ),
        (
            "g_s50_n16_sn5",
            OptionalMapperV16FrameClass::General,
            32,
            16,
            5,
        ),
        (
            "g_s90_n8_sn4",
            OptionalMapperV16FrameClass::General,
            56,
            8,
            4,
        ),
        (
            "g_s90_n16_sn8",
            OptionalMapperV16FrameClass::General,
            56,
            16,
            8,
        ),
    ];
    assert_eq!(OptionalMapperV16Case::ALL.len(), expected.len());

    let mut fixture_digests = HashSet::new();
    for (case, (id, frame_class, selected, nulls, selected_nulls)) in
        OptionalMapperV16Case::ALL.into_iter().zip(expected)
    {
        let spec = optional_mapper_v16_case_spec(case);
        assert_eq!(
            (
                spec.id,
                spec.frame_class,
                spec.selected_per_frame,
                spec.null_per_frame,
                spec.selected_null_per_frame,
            ),
            (id, frame_class, selected, nulls, selected_nulls)
        );
        let metadata = optional_mapper_v16_case_metadata(case);
        assert_eq!(metadata.logical_rows, OPTIONAL_MAPPER_V16_LOGICAL_ROWS);
        assert_eq!(metadata.frames, OPTIONAL_MAPPER_V16_FRAMES);
        assert_eq!(
            metadata.selected_rows,
            selected * OPTIONAL_MAPPER_V16_FRAMES
        );
        assert_eq!(
            metadata.present_rows,
            (64 - nulls) * OPTIONAL_MAPPER_V16_FRAMES
        );
        assert_eq!(
            metadata.selected_null_rows,
            selected_nulls * OPTIONAL_MAPPER_V16_FRAMES
        );
        assert_eq!(
            metadata.selected_present_rows,
            (selected - selected_nulls) * OPTIONAL_MAPPER_V16_FRAMES
        );
        assert_eq!(metadata.morphology_counts.iter().sum::<usize>(), 128);
        let min = *metadata.morphology_counts.iter().min().unwrap();
        let max = *metadata.morphology_counts.iter().max().unwrap();
        assert_eq!((min, max), (42, 43));
        assert!(fixture_digests.insert(metadata.fixture_digest));

        let expected_physical = usize::from(matches!(
            frame_class,
            OptionalMapperV16FrameClass::AllValidSelected | OptionalMapperV16FrameClass::General
        )) * 128;
        let expected_output =
            usize::from(frame_class == OptionalMapperV16FrameClass::General) * 128;
        assert_eq!(metadata.physical_compression_calls, expected_physical);
        assert_eq!(metadata.output_compression_calls, expected_output);

        let frames = optional_mapper_v16_frame_facts(case);
        assert_eq!(frames.len(), 128);
        for frame in &frames {
            assert_eq!(frame.selected_count, selected);
            assert_eq!(frame.null_count, nulls);
            assert_eq!(frame.selected_null_count, selected_nulls);
            assert_eq!(frame.frame_class, frame_class);
            assert_eq!(frame.selected_mask.count_ones() as usize, selected);
            assert_eq!((!frame.present_mask).count_ones() as usize, nulls);
            assert_eq!(
                (frame.selected_mask & !frame.present_mask).count_ones() as usize,
                selected_nulls
            );
        }
        for window in frames.windows(3) {
            assert_eq!(
                window
                    .iter()
                    .map(|frame| frame.morphology)
                    .collect::<HashSet<_>>(),
                HashSet::from([
                    OptionalMapperV16Morphology::Distributed,
                    OptionalMapperV16Morphology::Clustered,
                    OptionalMapperV16Morphology::EdgeHeavy,
                ])
            );
        }
    }
}

#[test]
fn all_supported_arms_match_the_independent_semantic_oracle() {
    for case in OptionalMapperV16Case::ALL {
        let metadata = optional_mapper_v16_case_metadata(case);
        let mut current =
            PreparedOptionalMapperV16::try_new(case, OptionalMapperV16Arm::CurrentSetBitScalar)
                .unwrap();
        let current_before = current.invoke_oracle().unwrap();
        let current_timed = current.invoke_lean().unwrap();
        let current_after = current.invoke_oracle().unwrap();
        assert!(current_before.workspace_unchanged());
        assert!(current_after.workspace_unchanged());
        assert_eq!(current_before.semantic_digest, metadata.semantic_digest);
        assert_eq!(current_after.semantic_digest, metadata.semantic_digest);
        assert_eq!(current_before.timed, current_timed);
        assert_eq!(current_before.timed, current_after.timed);
        assert_eq!(
            current_before.workspace_after,
            current_after.workspace_after
        );

        for arm in [
            OptionalMapperV16Arm::AdaptiveScalar,
            OptionalMapperV16Arm::Bmi2Pext,
        ] {
            if !optional_mapper_v16_arm_supported(arm) {
                assert!(PreparedOptionalMapperV16::try_new(case, arm).is_err());
                continue;
            }
            let mut prepared = PreparedOptionalMapperV16::try_new(case, arm).unwrap();
            let before = prepared.invoke_oracle().unwrap();
            let timed = prepared.invoke_lean().unwrap();
            let after = prepared.invoke_oracle().unwrap();
            assert!(before.workspace_unchanged());
            assert!(after.workspace_unchanged());
            assert_eq!(before.semantic_digest, current_before.semantic_digest);
            assert_eq!(after.semantic_digest, current_after.semantic_digest);
            assert_eq!(before.timed, current_before.timed);
            assert_eq!(timed, current_timed);
            assert_eq!(after.timed, current_after.timed);
            assert_eq!(before.workspace_after, after.workspace_after);
        }
    }
}
