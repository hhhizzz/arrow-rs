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

//! Frozen V1.6-A three-arm optional mapper attribution experiment.
//!
//! Fixture construction, exact-stat discovery, semantic oracles, and route
//! observation stay outside the timed call. [`PreparedOptionalMapperV16::invoke_lean`]
//! enters the same whole-fragment `OptionalSelectionMapper::map_into` seam used
//! by the reader, with already-warm physical and validity workspaces.

use arrow_buffer::BooleanBufferBuilder;

use crate::arrow::record_reader::optional_selection::{
    ForcedOptionalMapBackend, OptionalFrameCounters, OptionalSelectionMapper, OptionalSelectionView,
};
use crate::errors::{ParquetError, Result};

pub const OPTIONAL_MAPPER_V16_CONTRACT_ID: &str =
    "arrow_unified_frame_observability.optional_mapper.v1_6.mapper_leaf";
pub const OPTIONAL_MAPPER_V16_CONTRACT_ENV: &str = "PARQUET_OPTIONAL_MAPPER_V16_CONTRACT";
pub const OPTIONAL_MAPPER_V16_CASE_ENV: &str = "PARQUET_OPTIONAL_MAPPER_V16_CASE";
pub const OPTIONAL_MAPPER_V16_ARM_ENV: &str = "PARQUET_OPTIONAL_MAPPER_V16_ARM";
pub const OPTIONAL_MAPPER_V16_CRITERION_PREFIX: &str = "optional_mapper_v16";
pub const OPTIONAL_MAPPER_V16_ORACLE_PREFIX: &str = "optional_mapper_v16_oracle";

pub const OPTIONAL_MAPPER_V16_LOGICAL_ROWS: usize = 8192;
pub const OPTIONAL_MAPPER_V16_FRAME_ROWS: usize = 64;
pub const OPTIONAL_MAPPER_V16_FRAMES: usize =
    OPTIONAL_MAPPER_V16_LOGICAL_ROWS / OPTIONAL_MAPPER_V16_FRAME_ROWS;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptionalMapperV16Case {
    IdentityS05N0,
    IdentityS90N0,
    FullS100N4,
    EmptyN16,
    AllValidS25N1,
    AllValidS25N4,
    AllValidS50N8,
    AllValidS50N9,
    AllValidS50N16,
    AllValidS90N4,
    GeneralS05N1Sn1,
    GeneralS25N2Sn1,
    GeneralS25N4Sn2,
    GeneralS50N8Sn4,
    GeneralS50N9Sn4,
    GeneralS50N16Sn5,
    GeneralS90N8Sn4,
    GeneralS90N16Sn8,
}

impl OptionalMapperV16Case {
    pub const ALL: [Self; 18] = [
        Self::IdentityS05N0,
        Self::IdentityS90N0,
        Self::FullS100N4,
        Self::EmptyN16,
        Self::AllValidS25N1,
        Self::AllValidS25N4,
        Self::AllValidS50N8,
        Self::AllValidS50N9,
        Self::AllValidS50N16,
        Self::AllValidS90N4,
        Self::GeneralS05N1Sn1,
        Self::GeneralS25N2Sn1,
        Self::GeneralS25N4Sn2,
        Self::GeneralS50N8Sn4,
        Self::GeneralS50N9Sn4,
        Self::GeneralS50N16Sn5,
        Self::GeneralS90N8Sn4,
        Self::GeneralS90N16Sn8,
    ];

    pub fn id(self) -> &'static str {
        case_spec(self).id
    }

    pub fn from_id(id: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|case| case.id() == id)
    }

    fn ordinal(self) -> usize {
        Self::ALL
            .iter()
            .position(|candidate| *candidate == self)
            .unwrap()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptionalMapperV16Arm {
    CurrentSetBitScalar,
    AdaptiveScalar,
    Bmi2Pext,
}

impl OptionalMapperV16Arm {
    pub const ALL: [Self; 3] = [
        Self::CurrentSetBitScalar,
        Self::AdaptiveScalar,
        Self::Bmi2Pext,
    ];

    pub const fn id(self) -> &'static str {
        match self {
            Self::CurrentSetBitScalar => "current_set_bit_scalar",
            Self::AdaptiveScalar => "adaptive_scalar_v1",
            Self::Bmi2Pext => "bmi2_pext",
        }
    }

    pub fn from_id(id: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|arm| arm.id() == id)
    }

    fn backend(self) -> ForcedOptionalMapBackend {
        match self {
            Self::CurrentSetBitScalar => ForcedOptionalMapBackend::CurrentSetBitScalar,
            Self::AdaptiveScalar => ForcedOptionalMapBackend::AdaptiveScalar,
            Self::Bmi2Pext => ForcedOptionalMapBackend::Bmi2Pext,
        }
    }
}

pub fn optional_mapper_v16_arm_supported(arm: OptionalMapperV16Arm) -> bool {
    OptionalSelectionMapper::try_new_forced(arm.backend()).is_ok()
}

pub fn exact_optional_mapper_v16_binding(
    case_id: Option<&str>,
    arm_id: Option<&str>,
) -> std::result::Result<(OptionalMapperV16Case, OptionalMapperV16Arm), &'static str> {
    let case_id = case_id.ok_or("exact optional-mapper-v16 CASE id is required")?;
    let arm_id = arm_id.ok_or("exact optional-mapper-v16 ARM id is required")?;
    let case =
        OptionalMapperV16Case::from_id(case_id).ok_or("unknown optional-mapper-v16 CASE id")?;
    let arm = OptionalMapperV16Arm::from_id(arm_id).ok_or("unknown optional-mapper-v16 ARM id")?;
    Ok((case, arm))
}

pub fn optional_mapper_v16_route_requested_value(
    value: Option<&str>,
) -> std::result::Result<bool, &'static str> {
    match value {
        None => Ok(false),
        Some(OPTIONAL_MAPPER_V16_CONTRACT_ID) => Ok(true),
        Some(_) => Err("optional-mapper-v16 contract must be unset or equal the exact contract id"),
    }
}

pub fn optional_mapper_v16_exact_env_request(
    contract: Option<&str>,
    case: Option<&str>,
    arm: Option<&str>,
) -> std::result::Result<bool, &'static str> {
    let requested = optional_mapper_v16_route_requested_value(contract)?;
    match (requested, case.is_some(), arm.is_some()) {
        (false, false, false) => Ok(false),
        (false, _, _) => Err(
            "optional-mapper-v16 CASE/ARM must be unset when the exact contract is not requested",
        ),
        (true, true, true) => {
            exact_optional_mapper_v16_binding(case, arm)?;
            Ok(true)
        }
        (true, _, _) => Err("optional-mapper-v16 exact contract requires both CASE and ARM"),
    }
}

pub fn optional_mapper_v16_routes_are_exclusive(
    mapper_requested: bool,
    other_routes: &[bool],
) -> std::result::Result<(), &'static str> {
    let requested = usize::from(mapper_requested)
        + other_routes.iter().copied().map(usize::from).sum::<usize>();
    if requested > 1 {
        Err("optional-mapper-v16 and all other row-selection benchmark routes are exclusive")
    } else {
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptionalMapperV16FrameClass {
    EmptySelection,
    AllPresentIdentity,
    FullSelection,
    AllValidSelected,
    General,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OptionalMapperV16Morphology {
    Distributed,
    Clustered,
    EdgeHeavy,
}

impl OptionalMapperV16Morphology {
    pub const ALL: [Self; 3] = [Self::Distributed, Self::Clustered, Self::EdgeHeavy];
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OptionalMapperV16CaseSpec {
    pub case: OptionalMapperV16Case,
    pub id: &'static str,
    pub frame_class: OptionalMapperV16FrameClass,
    pub selected_per_frame: usize,
    pub null_per_frame: usize,
    pub selected_null_per_frame: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OptionalMapperV16FrameFacts {
    pub frame_ordinal: usize,
    pub morphology: OptionalMapperV16Morphology,
    pub selected_count: usize,
    pub null_count: usize,
    pub selected_null_count: usize,
    pub selected_mask: u64,
    pub present_mask: u64,
    pub frame_class: OptionalMapperV16FrameClass,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OptionalMapperV16CaseMetadata {
    pub id: &'static str,
    pub logical_rows: usize,
    pub frames: usize,
    pub selected_rows: usize,
    pub present_rows: usize,
    pub selected_present_rows: usize,
    pub selected_null_rows: usize,
    pub physical_compression_calls: usize,
    pub output_compression_calls: usize,
    pub morphology_counts: [usize; 3],
    pub fixture_digest: u64,
    pub semantic_digest: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OptionalMapperV16RouteObservation {
    pub current_backend_fragments: usize,
    pub adaptive_backend_fragments: usize,
    pub bmi2_backend_fragments: usize,
    pub physical_compression_calls: usize,
    pub output_compression_calls: usize,
    pub current_scalar_compression_calls: usize,
    pub adaptive_physical_sparse_calls: usize,
    pub adaptive_physical_fallback_calls: usize,
    pub adaptive_output_sparse_calls: usize,
    pub adaptive_output_fallback_calls: usize,
    pub bmi2_compression_calls: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OptionalMapperV16WorkspaceIdentity {
    pub physical_pointer: usize,
    pub physical_capacity: usize,
    pub validity_pointer: Option<usize>,
    pub validity_capacity: Option<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OptionalMapperV16TimedOutcome {
    pub logical_rows: usize,
    pub selected_rows: usize,
    pub present_rows: usize,
    pub selected_present_rows: usize,
    pub output_validity_materialized: bool,
    pub output_validity_len: usize,
    pub physical_len: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OptionalMapperV16OracleOutcome {
    pub timed: OptionalMapperV16TimedOutcome,
    pub semantic_digest: u64,
    pub workspace_before: OptionalMapperV16WorkspaceIdentity,
    pub workspace_after: OptionalMapperV16WorkspaceIdentity,
}

impl OptionalMapperV16OracleOutcome {
    pub fn workspace_unchanged(self) -> bool {
        self.workspace_before == self.workspace_after
    }
}

#[derive(Debug)]
pub struct PreparedOptionalMapperV16 {
    arm: OptionalMapperV16Arm,
    metadata: OptionalMapperV16CaseMetadata,
    fixture: Fixture,
    mapper: OptionalSelectionMapper,
    output_validity: Option<BooleanBufferBuilder>,
}

impl PreparedOptionalMapperV16 {
    pub fn try_new(
        case: OptionalMapperV16Case,
        arm: OptionalMapperV16Arm,
    ) -> std::result::Result<Self, String> {
        let fixture = Fixture::new(case);
        let metadata = metadata_from_fixture(case_spec(case), &fixture);
        let mapper = OptionalSelectionMapper::try_new_forced(arm.backend())
            .map_err(|error| error.to_string())?;
        let mut prepared = Self {
            arm,
            metadata,
            fixture,
            mapper,
            output_validity: None,
        };

        // One untimed invocation establishes the high-water marks for both
        // reusable buffers before the prepared value can enter Criterion.
        let warm = prepared
            .invoke_core::<false>()
            .map_err(|error| error.to_string())?;
        prepared
            .validate_semantics(warm)
            .map_err(|error| error.to_string())?;
        Ok(prepared)
    }

    pub fn arm(&self) -> OptionalMapperV16Arm {
        self.arm
    }

    pub fn metadata(&self) -> OptionalMapperV16CaseMetadata {
        self.metadata
    }

    pub fn workspace_identity(&self) -> OptionalMapperV16WorkspaceIdentity {
        OptionalMapperV16WorkspaceIdentity {
            physical_pointer: self.mapper.physical_selection().as_ptr() as usize,
            physical_capacity: self.mapper.physical_capacity(),
            validity_pointer: self
                .output_validity
                .as_ref()
                .map(|builder| builder.as_slice().as_ptr() as usize),
            validity_capacity: self
                .output_validity
                .as_ref()
                .map(BooleanBufferBuilder::capacity),
        }
    }

    /// The timing entry point. Setup-time route observation and digest work do
    /// not share this call graph; this method always reaches lean `map_into`.
    #[inline(never)]
    pub fn invoke_lean(&mut self) -> Result<OptionalMapperV16TimedOutcome> {
        let outcome = self.invoke_core::<false>()?;
        std::hint::black_box(self.mapper.physical_selection());
        if let Some(validity) = &self.output_validity {
            std::hint::black_box(validity.as_slice());
        }
        Ok(outcome)
    }

    pub fn invoke_oracle(&mut self) -> Result<OptionalMapperV16OracleOutcome> {
        let workspace_before = self.workspace_identity();
        let timed = self.invoke_core::<false>()?;
        self.validate_semantics(timed)?;
        let semantic_digest = self.semantic_digest(timed);
        let workspace_after = self.workspace_identity();
        Ok(OptionalMapperV16OracleOutcome {
            timed,
            semantic_digest,
            workspace_before,
            workspace_after,
        })
    }

    pub fn invoke_observed(&mut self) -> Result<OptionalMapperV16RouteObservation> {
        self.reset_output();
        let selection = OptionalSelectionView::new(
            &self.fixture.selection,
            0,
            OPTIONAL_MAPPER_V16_LOGICAL_ROWS,
        )?;
        let counters = self.mapper.map_into_observed(
            selection,
            &self.fixture.validity,
            0,
            &mut self.output_validity,
            0,
        )?;
        let timed = timed_outcome(counters, &self.mapper, &self.output_validity);
        self.validate_semantics(timed)?;
        Ok(route_observation(counters))
    }

    fn invoke_core<const OBSERVE: bool>(&mut self) -> Result<OptionalMapperV16TimedOutcome> {
        debug_assert!(!OBSERVE, "observed calls use invoke_observed");
        self.reset_output();
        let selection = OptionalSelectionView::new(
            &self.fixture.selection,
            0,
            OPTIONAL_MAPPER_V16_LOGICAL_ROWS,
        )?;
        let counters = self.mapper.map_into(
            selection,
            &self.fixture.validity,
            0,
            &mut self.output_validity,
            0,
        )?;
        Ok(timed_outcome(counters, &self.mapper, &self.output_validity))
    }

    fn reset_output(&mut self) {
        if let Some(validity) = &mut self.output_validity {
            validity.truncate(0);
        }
    }

    fn validate_semantics(&self, timed: OptionalMapperV16TimedOutcome) -> Result<()> {
        if timed.logical_rows != self.metadata.logical_rows
            || timed.selected_rows != self.metadata.selected_rows
            || timed.present_rows != self.metadata.present_rows
            || timed.selected_present_rows != self.metadata.selected_present_rows
        {
            return Err(general_err!(
                "optional-mapper-v16 counters disagree with frozen fixture metadata"
            ));
        }
        if timed.physical_len != self.metadata.present_rows {
            return Err(general_err!(
                "optional-mapper-v16 physical length {} differs from expected {}",
                timed.physical_len,
                self.metadata.present_rows
            ));
        }
        let expected_materialized = self.metadata.selected_null_rows != 0;
        if timed.output_validity_materialized != expected_materialized {
            return Err(general_err!(
                "optional-mapper-v16 validity materialization differs from frozen fixture"
            ));
        }
        let expected_validity_len = if expected_materialized {
            self.metadata.selected_rows
        } else {
            0
        };
        if timed.output_validity_len != expected_validity_len {
            return Err(general_err!(
                "optional-mapper-v16 validity length {} differs from expected {expected_validity_len}",
                timed.output_validity_len
            ));
        }
        let actual_digest = self.semantic_digest(timed);
        if actual_digest != self.metadata.semantic_digest {
            return Err(general_err!(
                "optional-mapper-v16 semantic digest {actual_digest:016x} differs from expected {:016x}",
                self.metadata.semantic_digest
            ));
        }
        Ok(())
    }

    fn semantic_digest(&self, timed: OptionalMapperV16TimedOutcome) -> u64 {
        semantic_digest(
            self.mapper.physical_selection(),
            timed.physical_len,
            self.output_validity.as_ref(),
            timed.selected_rows,
        )
    }
}

pub fn optional_mapper_v16_case_spec(case: OptionalMapperV16Case) -> OptionalMapperV16CaseSpec {
    case_spec(case)
}

pub fn optional_mapper_v16_case_metadata(
    case: OptionalMapperV16Case,
) -> OptionalMapperV16CaseMetadata {
    let fixture = Fixture::new(case);
    metadata_from_fixture(case_spec(case), &fixture)
}

pub fn optional_mapper_v16_frame_facts(
    case: OptionalMapperV16Case,
) -> Vec<OptionalMapperV16FrameFacts> {
    Fixture::new(case).frames
}

/// Observe backend routing on a throwaway prepared mapper. The instance used
/// for route proof is intentionally distinct from the warm state timed by
/// Criterion.
pub fn observe_optional_mapper_v16_route(
    case: OptionalMapperV16Case,
    arm: OptionalMapperV16Arm,
) -> std::result::Result<OptionalMapperV16RouteObservation, String> {
    let mut observed = PreparedOptionalMapperV16::try_new(case, arm)?;
    observed
        .invoke_observed()
        .map_err(|error| error.to_string())
}

pub fn expected_optional_mapper_v16_route(
    case: OptionalMapperV16Case,
    arm: OptionalMapperV16Arm,
) -> OptionalMapperV16RouteObservation {
    let spec = case_spec(case);
    let physical = usize::from(matches!(
        spec.frame_class,
        OptionalMapperV16FrameClass::AllValidSelected | OptionalMapperV16FrameClass::General
    )) * OPTIONAL_MAPPER_V16_FRAMES;
    let output = usize::from(spec.frame_class == OptionalMapperV16FrameClass::General)
        * OPTIONAL_MAPPER_V16_FRAMES;
    let mut route = OptionalMapperV16RouteObservation {
        current_backend_fragments: 0,
        adaptive_backend_fragments: 0,
        bmi2_backend_fragments: 0,
        physical_compression_calls: physical,
        output_compression_calls: output,
        current_scalar_compression_calls: 0,
        adaptive_physical_sparse_calls: 0,
        adaptive_physical_fallback_calls: 0,
        adaptive_output_sparse_calls: 0,
        adaptive_output_fallback_calls: 0,
        bmi2_compression_calls: 0,
    };
    match arm {
        OptionalMapperV16Arm::CurrentSetBitScalar => {
            route.current_backend_fragments = 1;
            route.current_scalar_compression_calls = physical + output;
        }
        OptionalMapperV16Arm::AdaptiveScalar => {
            route.adaptive_backend_fragments = 1;
            if physical != 0 {
                if physical_sparse_route(spec.null_per_frame) {
                    route.adaptive_physical_sparse_calls = physical;
                } else {
                    route.adaptive_physical_fallback_calls = physical;
                }
            }
            if output != 0 {
                if output_sparse_route(spec.selected_per_frame, spec.selected_null_per_frame) {
                    route.adaptive_output_sparse_calls = output;
                } else {
                    route.adaptive_output_fallback_calls = output;
                }
            }
        }
        OptionalMapperV16Arm::Bmi2Pext => {
            route.bmi2_backend_fragments = 1;
            route.bmi2_compression_calls = physical + output;
        }
    }
    route
}

fn timed_outcome(
    counters: OptionalFrameCounters,
    mapper: &OptionalSelectionMapper,
    output_validity: &Option<BooleanBufferBuilder>,
) -> OptionalMapperV16TimedOutcome {
    OptionalMapperV16TimedOutcome {
        logical_rows: counters.logical_rows,
        selected_rows: counters.selected_logical_rows,
        present_rows: counters.present_rows,
        selected_present_rows: counters.selected_present_rows,
        output_validity_materialized: output_validity.is_some(),
        output_validity_len: output_validity
            .as_ref()
            .map_or(0, BooleanBufferBuilder::len),
        physical_len: mapper.physical_len(),
    }
}

fn route_observation(counters: OptionalFrameCounters) -> OptionalMapperV16RouteObservation {
    OptionalMapperV16RouteObservation {
        current_backend_fragments: counters.current_backend_fragments,
        adaptive_backend_fragments: counters.adaptive_backend_fragments,
        bmi2_backend_fragments: counters.bmi2_backend_fragments,
        physical_compression_calls: counters.physical_compression_calls,
        output_compression_calls: counters.output_compression_calls,
        current_scalar_compression_calls: counters.current_scalar_compression_calls,
        adaptive_physical_sparse_calls: counters.adaptive_physical_sparse_calls,
        adaptive_physical_fallback_calls: counters.adaptive_physical_fallback_calls,
        adaptive_output_sparse_calls: counters.adaptive_output_sparse_calls,
        adaptive_output_fallback_calls: counters.adaptive_output_fallback_calls,
        bmi2_compression_calls: counters.bmi2_compression_calls,
    }
}

fn case_spec(case: OptionalMapperV16Case) -> OptionalMapperV16CaseSpec {
    use OptionalMapperV16Case::*;
    use OptionalMapperV16FrameClass::*;
    // `s90` is the experiment's density bucket label. Its frozen 64-row
    // discretization is 56 selected rows (87.5%): this leaves eight unselected
    // rows so `g_s90_n16_sn8` can independently hold exactly eight selected
    // nulls and eight unselected nulls in every frame.
    let (id, frame_class, selected_per_frame, null_per_frame, selected_null_per_frame) = match case
    {
        IdentityS05N0 => ("identity_s05_n0", AllPresentIdentity, 3, 0, 0),
        IdentityS90N0 => ("identity_s90_n0", AllPresentIdentity, 56, 0, 0),
        FullS100N4 => ("full_s100_n4", FullSelection, 64, 4, 4),
        EmptyN16 => ("empty_n16", EmptySelection, 0, 16, 0),
        AllValidS25N1 => ("av_s25_n1", AllValidSelected, 16, 1, 0),
        AllValidS25N4 => ("av_s25_n4", AllValidSelected, 16, 4, 0),
        AllValidS50N8 => ("av_s50_n8", AllValidSelected, 32, 8, 0),
        AllValidS50N9 => ("av_s50_n9", AllValidSelected, 32, 9, 0),
        AllValidS50N16 => ("av_s50_n16", AllValidSelected, 32, 16, 0),
        AllValidS90N4 => ("av_s90_n4", AllValidSelected, 56, 4, 0),
        GeneralS05N1Sn1 => ("g_s05_n1_sn1", General, 3, 1, 1),
        GeneralS25N2Sn1 => ("g_s25_n2_sn1", General, 16, 2, 1),
        GeneralS25N4Sn2 => ("g_s25_n4_sn2", General, 16, 4, 2),
        GeneralS50N8Sn4 => ("g_s50_n8_sn4", General, 32, 8, 4),
        GeneralS50N9Sn4 => ("g_s50_n9_sn4", General, 32, 9, 4),
        GeneralS50N16Sn5 => ("g_s50_n16_sn5", General, 32, 16, 5),
        GeneralS90N8Sn4 => ("g_s90_n8_sn4", General, 56, 8, 4),
        GeneralS90N16Sn8 => ("g_s90_n16_sn8", General, 56, 16, 8),
    };
    OptionalMapperV16CaseSpec {
        case,
        id,
        frame_class,
        selected_per_frame,
        null_per_frame,
        selected_null_per_frame,
    }
}

#[derive(Debug)]
struct Fixture {
    selection: Vec<u8>,
    validity: Vec<u8>,
    frames: Vec<OptionalMapperV16FrameFacts>,
    expected_physical: Vec<u8>,
    expected_validity: Vec<u8>,
}

impl Fixture {
    fn new(case: OptionalMapperV16Case) -> Self {
        let spec = case_spec(case);
        let mut selection = Vec::with_capacity(OPTIONAL_MAPPER_V16_LOGICAL_ROWS / 8);
        let mut validity = Vec::with_capacity(OPTIONAL_MAPPER_V16_LOGICAL_ROWS / 8);
        let mut frames = Vec::with_capacity(OPTIONAL_MAPPER_V16_FRAMES);
        for frame_ordinal in 0..OPTIONAL_MAPPER_V16_FRAMES {
            let morphology = OptionalMapperV16Morphology::ALL
                [(frame_ordinal + case.ordinal()) % OptionalMapperV16Morphology::ALL.len()];
            let phase = (case.ordinal() * 17 + frame_ordinal * 11) % 64;
            let order = ordered_positions(morphology, phase);
            let (selected_mask, present_mask) = frame_masks(spec, &order);
            selection.extend_from_slice(&selected_mask.to_le_bytes());
            validity.extend_from_slice(&present_mask.to_le_bytes());
            let frame_class = classify_frame(selected_mask, present_mask);
            let facts = OptionalMapperV16FrameFacts {
                frame_ordinal,
                morphology,
                selected_count: selected_mask.count_ones() as usize,
                null_count: (!present_mask).count_ones() as usize,
                selected_null_count: (selected_mask & !present_mask).count_ones() as usize,
                selected_mask,
                present_mask,
                frame_class,
            };
            assert_eq!(facts.selected_count, spec.selected_per_frame);
            assert_eq!(facts.null_count, spec.null_per_frame);
            assert_eq!(facts.selected_null_count, spec.selected_null_per_frame);
            assert_eq!(facts.frame_class, spec.frame_class);
            frames.push(facts);
        }

        let mut expected_physical = vec![0_u8; count_bits(&validity).div_ceil(8)];
        let mut expected_validity = vec![0_u8; count_bits(&selection).div_ceil(8)];
        let mut physical_output = 0;
        let mut validity_output = 0;
        for logical in 0..OPTIONAL_MAPPER_V16_LOGICAL_ROWS {
            let selected = bit_is_set(&selection, logical);
            let present = bit_is_set(&validity, logical);
            if present {
                if selected {
                    set_bit(&mut expected_physical, physical_output);
                }
                physical_output += 1;
            }
            if selected {
                if present {
                    set_bit(&mut expected_validity, validity_output);
                }
                validity_output += 1;
            }
        }
        assert_eq!(physical_output, count_bits(&validity));
        assert_eq!(validity_output, count_bits(&selection));
        Self {
            selection,
            validity,
            frames,
            expected_physical,
            expected_validity,
        }
    }
}

fn frame_masks(spec: OptionalMapperV16CaseSpec, order: &[usize; 64]) -> (u64, u64) {
    let mut selected = 0_u64;
    let mut nulls = 0_u64;
    match spec.frame_class {
        OptionalMapperV16FrameClass::EmptySelection => {
            add_positions(&mut nulls, &order[..spec.null_per_frame]);
        }
        OptionalMapperV16FrameClass::AllPresentIdentity => {
            add_positions(&mut selected, &order[..spec.selected_per_frame]);
        }
        OptionalMapperV16FrameClass::FullSelection => {
            selected = u64::MAX;
            add_positions(&mut nulls, &order[..spec.null_per_frame]);
        }
        OptionalMapperV16FrameClass::AllValidSelected => {
            add_positions(&mut selected, &order[..spec.selected_per_frame]);
            let null_start = spec.selected_per_frame;
            let null_end = null_start + spec.null_per_frame;
            add_positions(&mut nulls, &order[null_start..null_end]);
        }
        OptionalMapperV16FrameClass::General => {
            add_positions(&mut selected, &order[..spec.selected_per_frame]);
            add_positions(&mut nulls, &order[..spec.selected_null_per_frame]);
            let null_only = spec.null_per_frame - spec.selected_null_per_frame;
            let null_start = spec.selected_per_frame;
            add_positions(&mut nulls, &order[null_start..null_start + null_only]);
        }
    }
    (selected, !nulls)
}

fn ordered_positions(morphology: OptionalMapperV16Morphology, phase: usize) -> [usize; 64] {
    let mut positions = [0_usize; 64];
    for (rank, position) in positions.iter_mut().enumerate() {
        let base = match morphology {
            OptionalMapperV16Morphology::Distributed => (rank * 37) % 64,
            OptionalMapperV16Morphology::Clustered => rank,
            OptionalMapperV16Morphology::EdgeHeavy => {
                if rank % 2 == 0 {
                    rank / 2
                } else {
                    63 - rank / 2
                }
            }
        };
        *position = (base + phase) % 64;
    }
    debug_assert!((0..64).all(|position| positions.contains(&position)));
    positions
}

fn add_positions(mask: &mut u64, positions: &[usize]) {
    for &position in positions {
        *mask |= 1_u64 << position;
    }
}

fn classify_frame(selected: u64, present: u64) -> OptionalMapperV16FrameClass {
    if selected == 0 {
        OptionalMapperV16FrameClass::EmptySelection
    } else if present == u64::MAX {
        OptionalMapperV16FrameClass::AllPresentIdentity
    } else if selected == u64::MAX {
        OptionalMapperV16FrameClass::FullSelection
    } else if selected & !present == 0 {
        OptionalMapperV16FrameClass::AllValidSelected
    } else {
        OptionalMapperV16FrameClass::General
    }
}

fn metadata_from_fixture(
    spec: OptionalMapperV16CaseSpec,
    fixture: &Fixture,
) -> OptionalMapperV16CaseMetadata {
    let selected_rows = count_bits(&fixture.selection);
    let present_rows = count_bits(&fixture.validity);
    let selected_present_rows = (0..OPTIONAL_MAPPER_V16_LOGICAL_ROWS)
        .filter(|&idx| bit_is_set(&fixture.selection, idx) && bit_is_set(&fixture.validity, idx))
        .count();
    let physical_compression_calls = usize::from(matches!(
        spec.frame_class,
        OptionalMapperV16FrameClass::AllValidSelected | OptionalMapperV16FrameClass::General
    )) * OPTIONAL_MAPPER_V16_FRAMES;
    let output_compression_calls =
        usize::from(spec.frame_class == OptionalMapperV16FrameClass::General)
            * OPTIONAL_MAPPER_V16_FRAMES;
    let mut morphology_counts = [0_usize; 3];
    for frame in &fixture.frames {
        let idx = match frame.morphology {
            OptionalMapperV16Morphology::Distributed => 0,
            OptionalMapperV16Morphology::Clustered => 1,
            OptionalMapperV16Morphology::EdgeHeavy => 2,
        };
        morphology_counts[idx] += 1;
    }
    let fixture_digest = fixture_digest(spec, fixture);
    let semantic_digest = semantic_digest_from_bytes(
        &fixture.expected_physical,
        present_rows,
        &fixture.expected_validity,
        selected_rows,
    );
    OptionalMapperV16CaseMetadata {
        id: spec.id,
        logical_rows: OPTIONAL_MAPPER_V16_LOGICAL_ROWS,
        frames: OPTIONAL_MAPPER_V16_FRAMES,
        selected_rows,
        present_rows,
        selected_present_rows,
        selected_null_rows: selected_rows - selected_present_rows,
        physical_compression_calls,
        output_compression_calls,
        morphology_counts,
        fixture_digest,
        semantic_digest,
    }
}

fn physical_sparse_route(null_count: usize) -> bool {
    null_count != 0 && null_count <= 8 && 2 * null_count <= 64 - null_count
}

fn output_sparse_route(selected_count: usize, selected_null_count: usize) -> bool {
    selected_null_count != 0
        && selected_null_count <= 4
        && 2 * selected_null_count <= selected_count
}

fn semantic_digest(
    physical: &[u8],
    physical_len: usize,
    validity: Option<&BooleanBufferBuilder>,
    selected_len: usize,
) -> u64 {
    let mut digest = 0xcbf29ce484222325_u64;
    digest = digest_usize(digest, physical_len);
    digest = digest_bytes(digest, physical);
    digest = digest_usize(digest, selected_len);
    match validity {
        Some(validity) => digest_bytes(digest, validity.as_slice()),
        None => {
            for byte_idx in 0..selected_len.div_ceil(8) {
                let remaining = selected_len - byte_idx * 8;
                let byte = if remaining >= 8 {
                    u8::MAX
                } else {
                    (1_u8 << remaining) - 1
                };
                digest = fnv1a(digest, byte);
            }
            digest
        }
    }
}

fn semantic_digest_from_bytes(
    physical: &[u8],
    physical_len: usize,
    validity: &[u8],
    selected_len: usize,
) -> u64 {
    let mut digest = 0xcbf29ce484222325_u64;
    digest = digest_usize(digest, physical_len);
    digest = digest_bytes(digest, physical);
    digest = digest_usize(digest, selected_len);
    digest_bytes(digest, validity)
}

fn fixture_digest(spec: OptionalMapperV16CaseSpec, fixture: &Fixture) -> u64 {
    let mut digest = 0xcbf29ce484222325_u64;
    for value in [
        spec.case.ordinal(),
        spec.selected_per_frame,
        spec.null_per_frame,
        spec.selected_null_per_frame,
    ] {
        digest = digest_usize(digest, value);
    }
    digest = digest_bytes(digest, &fixture.selection);
    digest_bytes(digest, &fixture.validity)
}

fn count_bits(bitmap: &[u8]) -> usize {
    bitmap.iter().map(|byte| byte.count_ones() as usize).sum()
}

fn bit_is_set(bitmap: &[u8], idx: usize) -> bool {
    bitmap[idx / 8] & (1 << (idx % 8)) != 0
}

fn set_bit(bitmap: &mut [u8], idx: usize) {
    bitmap[idx / 8] |= 1 << (idx % 8);
}

fn digest_usize(mut digest: u64, value: usize) -> u64 {
    for byte in (value as u64).to_le_bytes() {
        digest = fnv1a(digest, byte);
    }
    digest
}

fn digest_bytes(mut digest: u64, bytes: &[u8]) -> u64 {
    for &byte in bytes {
        digest = fnv1a(digest, byte);
    }
    digest
}

#[inline]
fn fnv1a(digest: u64, byte: u8) -> u64 {
    (digest ^ u64::from(byte)).wrapping_mul(0x100000001b3)
}
