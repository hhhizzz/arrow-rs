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

//! PROTOTYPE — allocation-free logical-to-physical mapping for flat optional columns.
//!
//! This module answers one question: can a reusable, lazy-validity mapper with
//! an optional BMI2 backend materially reduce the nullable selection-mapping leaf
//! when it is reached through the real `GenericRecordReader` compact lane?

use std::mem::size_of;
use std::sync::atomic::{AtomicU64, Ordering};

use arrow_buffer::BooleanBufferBuilder;

use crate::errors::{ParquetError, Result};

/// Borrowed, LSB-first logical selection bitmap used by the prototype.
#[derive(Debug, Clone, Copy)]
pub(crate) struct OptionalSelectionView<'a> {
    data: &'a [u8],
    bit_offset: usize,
    len: usize,
}

impl<'a> OptionalSelectionView<'a> {
    pub(crate) fn new(data: &'a [u8], bit_offset: usize, len: usize) -> Result<Self> {
        let end = bit_offset
            .checked_add(len)
            .ok_or_else(|| general_err!("optional selection bit range overflows usize"))?;
        let available = data
            .len()
            .checked_mul(u8::BITS as usize)
            .ok_or_else(|| general_err!("optional selection bitmap size overflows usize"))?;
        if end > available {
            return Err(general_err!(
                "optional selection range ends at {end}, but bitmap contains {available} bits"
            ));
        }
        Ok(Self {
            data,
            bit_offset,
            len,
        })
    }

    #[inline]
    pub(crate) const fn len(self) -> usize {
        self.len
    }

    #[cfg(test)]
    #[inline]
    pub(crate) fn slice(self, offset: usize, len: usize) -> Self {
        assert!(offset + len <= self.len);
        Self {
            data: self.data,
            bit_offset: self.bit_offset + offset,
            len,
        }
    }

    #[inline]
    fn bits_u64(self, start: usize, len: usize) -> u64 {
        debug_assert!(len <= 64);
        debug_assert!(start + len <= self.len);
        load_bits(self.data, self.bit_offset, start, len)
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct OptionalFrameCounters {
    pub logical_rows: usize,
    pub present_rows: usize,
    pub selected_logical_rows: usize,
    pub selected_present_rows: usize,
    pub all_present_frames: usize,
    pub all_null_frames: usize,
    pub mixed_present_frames: usize,
    pub selection_empty_frames: usize,
    pub selection_full_frames: usize,
    pub selection_mixed_frames: usize,
    pub output_empty_frames: usize,
    pub output_all_valid_frames: usize,
    pub output_all_null_frames: usize,
    pub output_mixed_frames: usize,
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
pub(crate) struct OptionalFrameFacts {
    pub logical_len: u8,
    pub selected_count: u8,
    pub present_count: u8,
    pub selected_present_count: u8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum OptionalFrameClass {
    EmptySelection,
    AllNull,
    AllPresentIdentity,
    FullSelection,
    AllValidSelected,
    General,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MappedOptionalFrame {
    pub facts: OptionalFrameFacts,
    pub class: OptionalFrameClass,
    /// Selected rows in logical-row coordinates.
    pub selected_mask: u64,
    /// Present rows in logical-row coordinates.
    pub present_mask: u64,
}

/// Yields one semantic mapping frame for at most 64 logical rows.
pub(crate) struct OptionalFrameCursor<'a> {
    selection: OptionalSelectionView<'a>,
    validity: &'a [u8],
    validity_offset: usize,
    logical_offset: usize,
}

impl<'a> OptionalFrameCursor<'a> {
    pub(crate) fn new(
        selection: OptionalSelectionView<'a>,
        validity: &'a [u8],
        validity_offset: usize,
    ) -> Result<Self> {
        let validity_end = validity_offset
            .checked_add(selection.len())
            .ok_or_else(|| general_err!("optional validity bit range overflows usize"))?;
        let available = validity
            .len()
            .checked_mul(u8::BITS as usize)
            .ok_or_else(|| general_err!("optional validity bitmap size overflows usize"))?;
        if validity_end > available {
            return Err(general_err!(
                "optional validity range ends at {validity_end}, but bitmap contains {available} bits"
            ));
        }
        Ok(Self {
            selection,
            validity,
            validity_offset,
            logical_offset: 0,
        })
    }

    #[cfg(test)]
    fn logical_offset(&self) -> usize {
        self.logical_offset
    }
}

impl Iterator for OptionalFrameCursor<'_> {
    type Item = MappedOptionalFrame;

    fn next(&mut self) -> Option<Self::Item> {
        if self.logical_offset == self.selection.len() {
            return None;
        }
        let frame_len = (self.selection.len() - self.logical_offset).min(u64::BITS as usize);
        let full_mask = trailing_mask(frame_len);
        let selected_mask = self.selection.bits_u64(self.logical_offset, frame_len);
        let present_mask = load_bits(
            self.validity,
            self.validity_offset,
            self.logical_offset,
            frame_len,
        );
        let selected_count = selected_mask.count_ones() as u8;
        let present_count = present_mask.count_ones() as u8;
        let selected_present_count = (selected_mask & present_mask).count_ones() as u8;
        let class = if selected_mask == 0 {
            OptionalFrameClass::EmptySelection
        } else if present_mask == 0 {
            OptionalFrameClass::AllNull
        } else if present_mask == full_mask {
            OptionalFrameClass::AllPresentIdentity
        } else if selected_mask == full_mask {
            OptionalFrameClass::FullSelection
        } else if selected_present_count == selected_count {
            OptionalFrameClass::AllValidSelected
        } else {
            OptionalFrameClass::General
        };
        self.logical_offset += frame_len;
        Some(MappedOptionalFrame {
            facts: OptionalFrameFacts {
                logical_len: frame_len as u8,
                selected_count,
                present_count,
                selected_present_count,
            },
            class,
            selected_mask,
            present_mask,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(not(any(test, feature = "experimental")), allow(dead_code))]
enum OptionalMapBackend {
    CurrentSetBitScalar,
    AdaptiveScalar,
    Bmi2Pext,
}

// Each benchmark arm changes only this source-bound default. BMI2 capability
// is resolved once when the RecordReader constructs its mapper, never inside
// the per-frame loop.
const DEFAULT_OPTIONAL_MAP_BACKEND: OptionalMapBackend = OptionalMapBackend::AdaptiveScalar;
const OBSERVE_OPTIONAL_SELECTION_ROUTES: bool = true;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
/// Process-wide diagnostic snapshot for the reader-integrated prototype.
pub struct OptionalSelectionRouteCounters {
    /// Optional page fragments mapped by this process.
    pub mapped_fragments: u64,
    /// Logical rows covered by mapped fragments.
    pub logical_rows: u64,
    /// Present physical values covered by mapped fragments.
    pub present_rows: u64,
    /// Logical rows selected for compact output.
    pub selected_logical_rows: u64,
    /// Selected rows that contain a physical value.
    pub selected_present_rows: u64,
    /// Fragments dispatched to the current set-bit Scalar backend.
    pub current_scalar_fragments: u64,
    /// Fragments dispatched to the adaptive portable Scalar backend.
    pub adaptive_scalar_fragments: u64,
    /// Fragments dispatched to the BMI2 PEXT backend.
    pub bmi2_pext_fragments: u64,
    /// Non-empty compact output batches consumed by Arrow readers.
    pub compact_output_leaf_batches: u64,
    /// Compact batches whose output validity stayed all-valid and lazy.
    pub lazy_validity_omitted_leaf_batches: u64,
    /// Compact batches that materialized output validity.
    pub materialized_validity_leaf_batches: u64,
}

static MAPPED_FRAGMENTS: AtomicU64 = AtomicU64::new(0);
static LOGICAL_ROWS: AtomicU64 = AtomicU64::new(0);
static PRESENT_ROWS: AtomicU64 = AtomicU64::new(0);
static SELECTED_LOGICAL_ROWS: AtomicU64 = AtomicU64::new(0);
static SELECTED_PRESENT_ROWS: AtomicU64 = AtomicU64::new(0);
static CURRENT_SCALAR_FRAGMENTS: AtomicU64 = AtomicU64::new(0);
static ADAPTIVE_SCALAR_FRAGMENTS: AtomicU64 = AtomicU64::new(0);
static BMI2_PEXT_FRAGMENTS: AtomicU64 = AtomicU64::new(0);
static COMPACT_OUTPUT_BATCHES: AtomicU64 = AtomicU64::new(0);
static LAZY_VALIDITY_OMITTED_BATCHES: AtomicU64 = AtomicU64::new(0);
static MATERIALIZED_VALIDITY_BATCHES: AtomicU64 = AtomicU64::new(0);

/// Reset process-wide prototype route counters before an attributed run.
pub fn reset_optional_selection_route_counters() {
    for counter in [
        &MAPPED_FRAGMENTS,
        &LOGICAL_ROWS,
        &PRESENT_ROWS,
        &SELECTED_LOGICAL_ROWS,
        &SELECTED_PRESENT_ROWS,
        &CURRENT_SCALAR_FRAGMENTS,
        &ADAPTIVE_SCALAR_FRAGMENTS,
        &BMI2_PEXT_FRAGMENTS,
        &COMPACT_OUTPUT_BATCHES,
        &LAZY_VALIDITY_OMITTED_BATCHES,
        &MATERIALIZED_VALIDITY_BATCHES,
    ] {
        counter.store(0, Ordering::Relaxed);
    }
}

/// Snapshot process-wide prototype route counters after an attributed run.
pub fn optional_selection_route_counters() -> OptionalSelectionRouteCounters {
    OptionalSelectionRouteCounters {
        mapped_fragments: MAPPED_FRAGMENTS.load(Ordering::Relaxed),
        logical_rows: LOGICAL_ROWS.load(Ordering::Relaxed),
        present_rows: PRESENT_ROWS.load(Ordering::Relaxed),
        selected_logical_rows: SELECTED_LOGICAL_ROWS.load(Ordering::Relaxed),
        selected_present_rows: SELECTED_PRESENT_ROWS.load(Ordering::Relaxed),
        current_scalar_fragments: CURRENT_SCALAR_FRAGMENTS.load(Ordering::Relaxed),
        adaptive_scalar_fragments: ADAPTIVE_SCALAR_FRAGMENTS.load(Ordering::Relaxed),
        bmi2_pext_fragments: BMI2_PEXT_FRAGMENTS.load(Ordering::Relaxed),
        compact_output_leaf_batches: COMPACT_OUTPUT_BATCHES.load(Ordering::Relaxed),
        lazy_validity_omitted_leaf_batches: LAZY_VALIDITY_OMITTED_BATCHES.load(Ordering::Relaxed),
        materialized_validity_leaf_batches: MATERIALIZED_VALIDITY_BATCHES.load(Ordering::Relaxed),
    }
}

pub(crate) fn observe_compact_output_validity(materialized: bool) {
    COMPACT_OUTPUT_BATCHES.fetch_add(1, Ordering::Relaxed);
    if materialized {
        MATERIALIZED_VALIDITY_BATCHES.fetch_add(1, Ordering::Relaxed);
    } else {
        LAZY_VALIDITY_OMITTED_BATCHES.fetch_add(1, Ordering::Relaxed);
    }
}

fn observe_route(backend: OptionalMapBackend, counters: &OptionalFrameCounters) {
    MAPPED_FRAGMENTS.fetch_add(1, Ordering::Relaxed);
    LOGICAL_ROWS.fetch_add(counters.logical_rows as u64, Ordering::Relaxed);
    PRESENT_ROWS.fetch_add(counters.present_rows as u64, Ordering::Relaxed);
    SELECTED_LOGICAL_ROWS.fetch_add(counters.selected_logical_rows as u64, Ordering::Relaxed);
    SELECTED_PRESENT_ROWS.fetch_add(counters.selected_present_rows as u64, Ordering::Relaxed);
    match backend {
        OptionalMapBackend::CurrentSetBitScalar => {
            CURRENT_SCALAR_FRAGMENTS.fetch_add(1, Ordering::Relaxed);
        }
        OptionalMapBackend::AdaptiveScalar => {
            ADAPTIVE_SCALAR_FRAGMENTS.fetch_add(1, Ordering::Relaxed);
        }
        OptionalMapBackend::Bmi2Pext => {
            BMI2_PEXT_FRAGMENTS.fetch_add(1, Ordering::Relaxed);
        }
    }
}

/// Forced mapper backends for isolated attribution benchmarks. Reader builds
/// use the source-bound [`DEFAULT_OPTIONAL_MAP_BACKEND`] instead.
#[cfg(any(test, feature = "experimental"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ForcedOptionalMapBackend {
    CurrentSetBitScalar,
    AdaptiveScalar,
    Bmi2Pext,
}

/// Reusable mapper workspace. After its packed buffer reaches the largest
/// fragment capacity, mapping equal or smaller fragments does not allocate.
#[derive(Debug)]
pub(crate) struct OptionalSelectionMapper {
    physical_selection: BooleanBufferBuilder,
    backend: OptionalMapBackend,
}

impl Default for OptionalSelectionMapper {
    fn default() -> Self {
        let backend = match DEFAULT_OPTIONAL_MAP_BACKEND {
            OptionalMapBackend::Bmi2Pext if !bmi2_supported() => OptionalMapBackend::AdaptiveScalar,
            backend => backend,
        };
        Self {
            physical_selection: BooleanBufferBuilder::new(0),
            backend,
        }
    }
}

impl OptionalSelectionMapper {
    pub(crate) fn map_into(
        &mut self,
        selection: OptionalSelectionView<'_>,
        validity: &[u8],
        validity_offset: usize,
        output_validity: &mut Option<BooleanBufferBuilder>,
        output_prefix_len: usize,
    ) -> Result<OptionalFrameCounters> {
        let counters = self.map_into_impl::<false>(
            selection,
            validity,
            validity_offset,
            output_validity,
            output_prefix_len,
        )?;
        if OBSERVE_OPTIONAL_SELECTION_ROUTES {
            observe_route(self.backend, &counters);
        }
        Ok(counters)
    }

    /// Route-observing oracle for correctness and attribution setup. Timed
    /// mapper loops use [`Self::map_into`], whose `OBSERVE = false` static
    /// instance contains no backend-specific telemetry increments.
    #[cfg(any(test, feature = "experimental"))]
    pub(crate) fn map_into_observed(
        &mut self,
        selection: OptionalSelectionView<'_>,
        validity: &[u8],
        validity_offset: usize,
        output_validity: &mut Option<BooleanBufferBuilder>,
        output_prefix_len: usize,
    ) -> Result<OptionalFrameCounters> {
        self.map_into_impl::<true>(
            selection,
            validity,
            validity_offset,
            output_validity,
            output_prefix_len,
        )
    }

    fn map_into_impl<const OBSERVE: bool>(
        &mut self,
        selection: OptionalSelectionView<'_>,
        validity: &[u8],
        validity_offset: usize,
        output_validity: &mut Option<BooleanBufferBuilder>,
        output_prefix_len: usize,
    ) -> Result<OptionalFrameCounters> {
        if output_validity
            .as_ref()
            .is_some_and(|builder| builder.len() != output_prefix_len)
        {
            return Err(general_err!(
                "optional output validity has length {}, expected prefix length {output_prefix_len}",
                output_validity.as_ref().unwrap().len()
            ));
        }
        // Validate the full input range before resetting reusable output state.
        let frames = OptionalFrameCursor::new(selection, validity, validity_offset)?;
        self.physical_selection.truncate(0);
        self.physical_selection.reserve(selection.len());

        // This is the only backend dispatch in a fragment. Each arm owns the
        // complete frame loop, with no per-word function pointer or CPU probe.
        match self.backend {
            OptionalMapBackend::CurrentSetBitScalar => map_frames::<CurrentSetBitScalar, OBSERVE>(
                frames,
                &mut self.physical_selection,
                output_validity,
                output_prefix_len,
            ),
            OptionalMapBackend::AdaptiveScalar => map_frames::<AdaptiveScalar, OBSERVE>(
                frames,
                &mut self.physical_selection,
                output_validity,
                output_prefix_len,
            ),
            OptionalMapBackend::Bmi2Pext => {
                // SAFETY: construction resolves an unsupported source-bound
                // BMI2 default to AdaptiveScalar, forced construction rejects
                // unsupported BMI2, and `backend` is private.
                unsafe {
                    map_frames_bmi2::<OBSERVE>(
                        frames,
                        &mut self.physical_selection,
                        output_validity,
                        output_prefix_len,
                    )
                }
            }
        }
    }

    #[cfg(any(test, feature = "experimental"))]
    pub(crate) fn try_new_forced(backend: ForcedOptionalMapBackend) -> Result<Self> {
        Self::try_new_forced_with_bmi2_support(backend, bmi2_supported())
    }

    #[cfg(any(test, feature = "experimental"))]
    fn try_new_forced_with_bmi2_support(
        backend: ForcedOptionalMapBackend,
        bmi2_supported: bool,
    ) -> Result<Self> {
        let backend = match backend {
            ForcedOptionalMapBackend::CurrentSetBitScalar => {
                OptionalMapBackend::CurrentSetBitScalar
            }
            ForcedOptionalMapBackend::AdaptiveScalar => OptionalMapBackend::AdaptiveScalar,
            ForcedOptionalMapBackend::Bmi2Pext if bmi2_supported => OptionalMapBackend::Bmi2Pext,
            ForcedOptionalMapBackend::Bmi2Pext => {
                return Err(general_err!(
                    "forced optional mapper backend Bmi2Pext requires x86_64 BMI2 support"
                ));
            }
        };
        Ok(Self {
            physical_selection: BooleanBufferBuilder::new(0),
            backend,
        })
    }

    pub(crate) fn physical_selection(&self) -> &[u8] {
        self.physical_selection.as_slice()
    }

    pub(crate) fn physical_len(&self) -> usize {
        self.physical_selection.len()
    }

    #[cfg(any(test, feature = "experimental"))]
    pub(crate) fn physical_capacity(&self) -> usize {
        self.physical_selection.capacity()
    }
}

trait OptionalFrameKernel {
    fn map<const OBSERVE: bool>(
        frame: &MappedOptionalFrame,
        counters: &mut OptionalFrameCounters,
    ) -> (u64, u64);

    fn observe_fragment(counters: &mut OptionalFrameCounters);
}

struct CurrentSetBitScalar;

impl OptionalFrameKernel for CurrentSetBitScalar {
    #[inline(always)]
    fn map<const OBSERVE: bool>(
        frame: &MappedOptionalFrame,
        _counters: &mut OptionalFrameCounters,
    ) -> (u64, u64) {
        if OBSERVE {
            observe_current_scalar_compressions(frame, _counters);
        }
        map_frame(frame, compress_scalar, compress_scalar)
    }

    #[inline(always)]
    fn observe_fragment(counters: &mut OptionalFrameCounters) {
        counters.current_backend_fragments += 1;
    }
}

struct AdaptiveScalar;

impl OptionalFrameKernel for AdaptiveScalar {
    #[inline(always)]
    fn map<const OBSERVE: bool>(
        frame: &MappedOptionalFrame,
        counters: &mut OptionalFrameCounters,
    ) -> (u64, u64) {
        let facts = frame.facts;
        let selected_count = facts.selected_count as usize;
        let present_count = facts.present_count as usize;
        match frame.class {
            OptionalFrameClass::EmptySelection | OptionalFrameClass::AllNull => (0, 0),
            OptionalFrameClass::AllPresentIdentity => {
                (frame.selected_mask, trailing_mask(selected_count))
            }
            OptionalFrameClass::FullSelection => (trailing_mask(present_count), frame.present_mask),
            OptionalFrameClass::AllValidSelected => {
                let (physical, sparse) = compress_physical_adaptive(
                    frame.selected_mask,
                    frame.present_mask,
                    facts.logical_len as usize,
                    present_count,
                );
                if OBSERVE {
                    counters.physical_compression_calls += 1;
                    if sparse {
                        counters.adaptive_physical_sparse_calls += 1;
                    } else {
                        counters.adaptive_physical_fallback_calls += 1;
                    }
                }
                (physical, trailing_mask(selected_count))
            }
            OptionalFrameClass::General => {
                let (physical, physical_sparse) = compress_physical_adaptive(
                    frame.selected_mask,
                    frame.present_mask,
                    facts.logical_len as usize,
                    present_count,
                );
                let (validity, output_sparse) = compress_output_validity_adaptive(
                    frame.present_mask,
                    frame.selected_mask,
                    selected_count,
                    facts.selected_present_count as usize,
                );
                if OBSERVE {
                    counters.physical_compression_calls += 1;
                    counters.output_compression_calls += 1;
                    if physical_sparse {
                        counters.adaptive_physical_sparse_calls += 1;
                    } else {
                        counters.adaptive_physical_fallback_calls += 1;
                    }
                    if output_sparse {
                        counters.adaptive_output_sparse_calls += 1;
                    } else {
                        counters.adaptive_output_fallback_calls += 1;
                    }
                }
                (physical, validity)
            }
        }
    }

    #[inline(always)]
    fn observe_fragment(counters: &mut OptionalFrameCounters) {
        counters.adaptive_backend_fragments += 1;
    }
}

#[cfg(target_arch = "x86_64")]
struct Bmi2Pext;

#[cfg(target_arch = "x86_64")]
impl OptionalFrameKernel for Bmi2Pext {
    #[inline(always)]
    fn map<const OBSERVE: bool>(
        frame: &MappedOptionalFrame,
        counters: &mut OptionalFrameCounters,
    ) -> (u64, u64) {
        if OBSERVE {
            observe_bmi2_compressions(frame, counters);
        }
        map_frame(frame, compress_bmi2, compress_bmi2)
    }

    #[inline(always)]
    fn observe_fragment(counters: &mut OptionalFrameCounters) {
        counters.bmi2_backend_fragments += 1;
    }
}

#[inline(always)]
fn compression_shape(frame: &MappedOptionalFrame) -> (usize, usize) {
    match frame.class {
        OptionalFrameClass::AllValidSelected => (1, 0),
        OptionalFrameClass::General => (1, 1),
        _ => (0, 0),
    }
}

#[inline(always)]
fn observe_current_scalar_compressions(
    frame: &MappedOptionalFrame,
    counters: &mut OptionalFrameCounters,
) {
    let (physical, output) = compression_shape(frame);
    counters.physical_compression_calls += physical;
    counters.output_compression_calls += output;
    counters.current_scalar_compression_calls += physical + output;
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn observe_bmi2_compressions(frame: &MappedOptionalFrame, counters: &mut OptionalFrameCounters) {
    let (physical, output) = compression_shape(frame);
    counters.physical_compression_calls += physical;
    counters.output_compression_calls += output;
    counters.bmi2_compression_calls += physical + output;
}

#[inline(always)]
fn map_frame(
    frame: &MappedOptionalFrame,
    physical_compress: impl Fn(u64, u64) -> u64,
    validity_compress: impl Fn(u64, u64) -> u64,
) -> (u64, u64) {
    let facts = frame.facts;
    let selected_count = facts.selected_count as usize;
    let present_count = facts.present_count as usize;
    match frame.class {
        OptionalFrameClass::EmptySelection | OptionalFrameClass::AllNull => (0, 0),
        OptionalFrameClass::AllPresentIdentity => {
            (frame.selected_mask, trailing_mask(selected_count))
        }
        OptionalFrameClass::FullSelection => (trailing_mask(present_count), frame.present_mask),
        OptionalFrameClass::AllValidSelected => (
            physical_compress(frame.selected_mask, frame.present_mask),
            trailing_mask(selected_count),
        ),
        OptionalFrameClass::General => (
            physical_compress(frame.selected_mask, frame.present_mask),
            validity_compress(frame.present_mask, frame.selected_mask),
        ),
    }
}

#[inline(always)]
fn map_frames<K: OptionalFrameKernel, const OBSERVE: bool>(
    frames: OptionalFrameCursor<'_>,
    physical_selection: &mut BooleanBufferBuilder,
    output_validity: &mut Option<BooleanBufferBuilder>,
    output_prefix_len: usize,
) -> Result<OptionalFrameCounters> {
    let mut counters = OptionalFrameCounters::default();
    if OBSERVE {
        K::observe_fragment(&mut counters);
    }
    let mut output_rows = output_prefix_len;
    for frame in frames {
        let facts = frame.facts;
        let logical_len = facts.logical_len as usize;
        let selected_count = facts.selected_count as usize;
        let present_count = facts.present_count as usize;
        let selected_present_count = facts.selected_present_count as usize;

        counters.logical_rows += logical_len;
        counters.present_rows += present_count;
        counters.selected_logical_rows += selected_count;
        counters.selected_present_rows += selected_present_count;

        match present_count {
            0 => counters.all_null_frames += 1,
            count if count == logical_len => counters.all_present_frames += 1,
            _ => counters.mixed_present_frames += 1,
        }
        match selected_count {
            0 => counters.selection_empty_frames += 1,
            count if count == logical_len => counters.selection_full_frames += 1,
            _ => counters.selection_mixed_frames += 1,
        }
        match (selected_count, selected_present_count) {
            (0, _) => counters.output_empty_frames += 1,
            (_, 0) => counters.output_all_null_frames += 1,
            (selected, present) if selected == present => counters.output_all_valid_frames += 1,
            _ => counters.output_mixed_frames += 1,
        }

        let (mapped_physical, mapped_validity) = K::map::<OBSERVE>(&frame, &mut counters);
        // Even an empty logical selection must append `present_count` zero
        // bits: the physical cursor still advances across every present row.
        physical_selection.append_word(mapped_physical, present_count);
        match output_validity {
            Some(builder) => builder.append_word(mapped_validity, selected_count),
            None if selected_present_count != selected_count => {
                let mut builder = BooleanBufferBuilder::new(0);
                builder.append_n(output_rows, true);
                builder.append_word(mapped_validity, selected_count);
                *output_validity = Some(builder);
            }
            None => {}
        }
        output_rows = output_rows
            .checked_add(selected_count)
            .ok_or_else(|| general_err!("optional output row count overflowed usize"))?;
    }
    debug_assert!(
        output_validity
            .as_ref()
            .is_none_or(|builder| builder.len() == output_rows)
    );
    Ok(counters)
}

/// The BMI2 target feature covers the whole fragment loop, rather than one
/// indirect call per 64-row word.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "bmi2")]
unsafe fn map_frames_bmi2<const OBSERVE: bool>(
    frames: OptionalFrameCursor<'_>,
    physical_selection: &mut BooleanBufferBuilder,
    output_validity: &mut Option<BooleanBufferBuilder>,
    output_prefix_len: usize,
) -> Result<OptionalFrameCounters> {
    map_frames::<Bmi2Pext, OBSERVE>(
        frames,
        physical_selection,
        output_validity,
        output_prefix_len,
    )
}

#[cfg(not(target_arch = "x86_64"))]
unsafe fn map_frames_bmi2<const OBSERVE: bool>(
    _frames: OptionalFrameCursor<'_>,
    _physical_selection: &mut BooleanBufferBuilder,
    _output_validity: &mut Option<BooleanBufferBuilder>,
    _output_prefix_len: usize,
) -> Result<OptionalFrameCounters> {
    Err(general_err!(
        "forced optional mapper backend Bmi2Pext requires x86_64 BMI2 support"
    ))
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn compress_bmi2(value: u64, mask: u64) -> u64 {
    // SAFETY: this function is reachable only from `map_frames_bmi2`, whose
    // caller has already established CPU support during forced construction.
    unsafe { std::arch::x86_64::_pext_u64(value, mask) }
}

#[inline]
fn bmi2_supported() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::arch::is_x86_feature_detected!("bmi2")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

const PHYSICAL_SPARSE_NULL_MAX: u32 = 8;
const OUTPUT_SPARSE_NULL_MAX: u32 = 4;

/// Compresses a logical selection into physical coordinates by deleting null
/// positions from high to low. Descending order keeps the remaining lower
/// logical positions stable while each deletion shifts the higher suffix once.
#[inline]
fn compress_physical_sparse_null(
    selected: u64,
    present: u64,
    logical_len: usize,
    present_count: usize,
) -> u64 {
    let full_mask = trailing_mask(logical_len);
    let mut output = selected & full_mask;
    let mut nulls = !present & full_mask;
    while nulls != 0 {
        let position = u64::BITS as usize - 1 - nulls.leading_zeros() as usize;
        let lower = trailing_mask(position);
        output = (output & lower) | ((output >> 1) & !lower);
        nulls &= !(1_u64 << position);
    }
    output & trailing_mask(present_count)
}

/// Starts with an all-valid compact output and clears only selected-null ranks.
#[inline]
fn compress_output_validity_sparse_null(present: u64, selected: u64, selected_count: usize) -> u64 {
    let mut output = trailing_mask(selected_count);
    let mut selected_nulls = selected & !present;
    while selected_nulls != 0 {
        let null = selected_nulls & selected_nulls.wrapping_neg();
        let lower = null - 1;
        let rank = (selected & lower).count_ones();
        output &= !(1_u64 << rank);
        selected_nulls ^= null;
    }
    output
}

#[inline]
fn compress_physical_adaptive(
    selected: u64,
    present: u64,
    logical_len: usize,
    present_count: usize,
) -> (u64, bool) {
    let null_count = logical_len - present_count;
    if null_count != 0
        && null_count <= PHYSICAL_SPARSE_NULL_MAX as usize
        && 2 * null_count <= present_count
    {
        (
            compress_physical_sparse_null(selected, present, logical_len, present_count),
            true,
        )
    } else {
        (compress_scalar(selected, present), false)
    }
}

#[inline]
fn compress_output_validity_adaptive(
    present: u64,
    selected: u64,
    selected_count: usize,
    selected_present_count: usize,
) -> (u64, bool) {
    let selected_null_count = selected_count - selected_present_count;
    if selected_null_count != 0
        && selected_null_count <= OUTPUT_SPARSE_NULL_MAX as usize
        && 2 * selected_null_count <= selected_count
    {
        (
            compress_output_validity_sparse_null(present, selected, selected_count),
            true,
        )
    } else {
        (compress_scalar(present, selected), false)
    }
}

/// Portable ScalarWordCompress reference backend. P4 may add an independently
/// selected BMI2 adapter, but ISA capability never selects the value strategy.
#[inline]
fn compress_scalar(value: u64, mut mask: u64) -> u64 {
    let mut result = 0_u64;
    let mut destination = 1_u64;
    while mask != 0 {
        let lowest = mask & mask.wrapping_neg();
        if value & lowest != 0 {
            result |= destination;
        }
        destination <<= 1;
        mask ^= lowest;
    }
    result
}

#[inline]
fn trailing_mask(len: usize) -> u64 {
    (u64::MAX >> ((64 - len) & 63)) * u64::from(len != 0)
}

#[inline]
fn load_bits(data: &[u8], bit_offset: usize, start: usize, len: usize) -> u64 {
    debug_assert!(len <= 64);
    if len == 0 {
        return 0;
    }
    let absolute = bit_offset + start;
    let byte_offset = absolute / u8::BITS as usize;
    let shift = absolute % u8::BITS as usize;
    let mut word = load_u64_le(data, byte_offset) >> shift;
    if shift != 0 && byte_offset + size_of::<u64>() < data.len() {
        word |= (data[byte_offset + size_of::<u64>()] as u64) << (64 - shift);
    }
    word & trailing_mask(len)
}

#[inline]
fn load_u64_le(data: &[u8], byte_offset: usize) -> u64 {
    let remaining = &data[byte_offset..];
    if let Some(bytes) = remaining.get(..size_of::<u64>()) {
        return u64::from_le_bytes(bytes.try_into().unwrap());
    }
    let mut padded = [0_u8; size_of::<u64>()];
    padded[..remaining.len()].copy_from_slice(remaining);
    u64::from_le_bytes(padded)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn supported_forced_backends() -> Vec<ForcedOptionalMapBackend> {
        let mut backends = vec![
            ForcedOptionalMapBackend::CurrentSetBitScalar,
            ForcedOptionalMapBackend::AdaptiveScalar,
        ];
        if bmi2_supported() {
            backends.push(ForcedOptionalMapBackend::Bmi2Pext);
        }
        backends
    }

    fn forced_mapper(backend: ForcedOptionalMapBackend) -> OptionalSelectionMapper {
        OptionalSelectionMapper::try_new_forced(backend).unwrap()
    }

    fn packed_with_offset(bits: &[bool], offset: usize) -> Vec<u8> {
        let mut out = vec![0; (offset + bits.len()).div_ceil(8)];
        for (idx, selected) in bits.iter().copied().enumerate() {
            if selected {
                let bit = offset + idx;
                out[bit / 8] |= 1 << (bit % 8);
            }
        }
        out
    }

    fn packed_window_with_poison(bits: &[bool], offset: usize) -> Vec<u8> {
        let mut out = vec![u8::MAX; (offset + bits.len() + 8).div_ceil(8)];
        for (idx, value) in bits.iter().copied().enumerate() {
            let bit = offset + idx;
            if value {
                out[bit / 8] |= 1 << (bit % 8);
            } else {
                out[bit / 8] &= !(1 << (bit % 8));
            }
        }
        out
    }

    fn unpack(bytes: &[u8], len: usize) -> Vec<bool> {
        (0..len)
            .map(|idx| bytes[idx / 8] & (1 << (idx % 8)) != 0)
            .collect()
    }

    fn unpack_lazy_validity(validity: &Option<BooleanBufferBuilder>, len: usize) -> Vec<bool> {
        validity
            .as_ref()
            .map(|builder| unpack(builder.as_slice(), len))
            .unwrap_or_else(|| vec![true; len])
    }

    fn assert_unused_tail_is_zero(bytes: &[u8], len: usize) {
        if len % 8 != 0 {
            let used_mask = (1_u8 << (len % 8)) - 1;
            assert_eq!(bytes.last().copied().unwrap_or_default() & !used_mask, 0);
        }
    }

    fn reference(present: &[bool], selected: &[bool]) -> (Vec<bool>, Vec<bool>) {
        let physical = present
            .iter()
            .zip(selected)
            .filter_map(|(&present, &selected)| present.then_some(selected))
            .collect();
        let validity = present
            .iter()
            .zip(selected)
            .filter_map(|(&present, &selected)| selected.then_some(present))
            .collect();
        (physical, validity)
    }

    #[test]
    fn exhaustive_forced_mappers_match_reference_through_eight_rows() {
        for len in 0..=8 {
            let variants = 1usize << len;
            for present_bits in 0..variants {
                for selected_bits in 0..variants {
                    let present = (0..len)
                        .map(|idx| present_bits & (1 << idx) != 0)
                        .collect::<Vec<_>>();
                    let selected = (0..len)
                        .map(|idx| selected_bits & (1 << idx) != 0)
                        .collect::<Vec<_>>();
                    let selected_bytes = packed_with_offset(&selected, 0);
                    let present_bytes = packed_with_offset(&present, 0);
                    let selection = OptionalSelectionView::new(&selected_bytes, 0, len).unwrap();
                    let (expected_physical, expected_validity) = reference(&present, &selected);
                    for backend in supported_forced_backends() {
                        let mut mapper = forced_mapper(backend);
                        let mut validity = None;
                        let counters = mapper
                            .map_into(selection, &present_bytes, 0, &mut validity, 0)
                            .unwrap();
                        assert_eq!(
                            mapper.physical_len(),
                            expected_physical.len(),
                            "{backend:?}"
                        );
                        assert_eq!(
                            unpack(mapper.physical_selection(), expected_physical.len()),
                            expected_physical,
                            "{backend:?}"
                        );
                        assert_eq!(
                            unpack_lazy_validity(&validity, expected_validity.len()),
                            expected_validity,
                            "{backend:?}"
                        );
                        assert_unused_tail_is_zero(
                            mapper.physical_selection(),
                            expected_physical.len(),
                        );
                        if let Some(validity) = validity.as_ref() {
                            assert_eq!(validity.len(), expected_validity.len(), "{backend:?}");
                            assert_unused_tail_is_zero(
                                validity.as_slice(),
                                expected_validity.len(),
                            );
                        }
                        assert_eq!(counters.logical_rows, len, "{backend:?}");
                        assert_eq!(
                            counters.present_rows,
                            present_bits.count_ones() as usize,
                            "{backend:?}"
                        );
                        assert_eq!(
                            counters.selected_logical_rows,
                            selected_bits.count_ones() as usize,
                            "{backend:?}"
                        );
                        assert_eq!(
                            counters.selected_present_rows,
                            (present_bits & selected_bits).count_ones() as usize,
                            "{backend:?}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn unaligned_multiword_offsets_and_frame_classes_match_reference() {
        for len in [63, 64, 65, 127, 128, 129] {
            for selection_offset in 0..8 {
                for validity_offset in 0..8 {
                    let present = (0..len)
                        .map(|idx| idx % 11 != 3 && idx % 17 != 5)
                        .collect::<Vec<_>>();
                    let selected = (0..len)
                        .map(|idx| idx % 7 != 1 && idx % 19 != 2)
                        .collect::<Vec<_>>();
                    let selected_bytes = packed_with_offset(&selected, selection_offset);
                    let present_bytes = packed_with_offset(&present, validity_offset);
                    let selection =
                        OptionalSelectionView::new(&selected_bytes, selection_offset, len).unwrap();
                    let (expected_physical, expected_validity) = reference(&present, &selected);
                    for backend in supported_forced_backends() {
                        let mut mapper = forced_mapper(backend);
                        let mut validity = None;
                        mapper
                            .map_into(selection, &present_bytes, validity_offset, &mut validity, 0)
                            .unwrap();
                        assert_eq!(
                            mapper.physical_len(),
                            expected_physical.len(),
                            "{backend:?}"
                        );
                        assert_eq!(
                            unpack(mapper.physical_selection(), expected_physical.len()),
                            expected_physical,
                            "{backend:?}"
                        );
                        assert_eq!(
                            unpack_lazy_validity(&validity, expected_validity.len()),
                            expected_validity,
                            "{backend:?}"
                        );
                        assert_unused_tail_is_zero(
                            mapper.physical_selection(),
                            expected_physical.len(),
                        );
                        if let Some(validity) = validity.as_ref() {
                            assert_unused_tail_is_zero(
                                validity.as_slice(),
                                expected_validity.len(),
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn source_bits_outside_unaligned_views_do_not_leak() {
        for len in [0, 1, 7, 8, 9, 63, 64, 65, 127, 128, 129] {
            for selection_offset in 0..8 {
                for validity_offset in 0..8 {
                    let present = (0..len)
                        .map(|idx| idx % 3 != 0 && idx % 17 != 4)
                        .collect::<Vec<_>>();
                    let selected = (0..len)
                        .map(|idx| idx % 5 == 0 || idx % 11 == 7)
                        .collect::<Vec<_>>();
                    let selected_bytes = packed_window_with_poison(&selected, selection_offset);
                    let present_bytes = packed_window_with_poison(&present, validity_offset);
                    let logical =
                        OptionalSelectionView::new(&selected_bytes, selection_offset, len).unwrap();
                    let (expected_physical, expected_validity) = reference(&present, &selected);
                    for backend in supported_forced_backends() {
                        let mut mapper = forced_mapper(backend);
                        let mut validity = None;
                        mapper
                            .map_into(logical, &present_bytes, validity_offset, &mut validity, 0)
                            .unwrap();
                        assert_eq!(
                            mapper.physical_len(),
                            expected_physical.len(),
                            "{backend:?}"
                        );
                        assert_eq!(
                            unpack(mapper.physical_selection(), expected_physical.len()),
                            expected_physical,
                            "{backend:?}"
                        );
                        assert_eq!(
                            unpack_lazy_validity(&validity, expected_validity.len()),
                            expected_validity,
                            "{backend:?}"
                        );
                        assert_unused_tail_is_zero(
                            mapper.physical_selection(),
                            expected_physical.len(),
                        );
                        if let Some(validity) = validity.as_ref() {
                            assert_unused_tail_is_zero(
                                validity.as_slice(),
                                expected_validity.len(),
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn semantic_frame_classes_are_explicit() {
        let cases = [
            (0b1111, 0b0000, OptionalFrameClass::EmptySelection),
            (0b0000, 0b0011, OptionalFrameClass::AllNull),
            (0b1111, 0b0101, OptionalFrameClass::AllPresentIdentity),
            (0b0101, 0b1111, OptionalFrameClass::FullSelection),
            (0b0101, 0b0101, OptionalFrameClass::AllValidSelected),
            (0b0101, 0b0110, OptionalFrameClass::General),
        ];
        for (present, selected, expected) in cases {
            let present = [present as u8];
            let selected = [selected as u8];
            let logical = OptionalSelectionView::new(&selected, 0, 4).unwrap();
            let mut cursor = OptionalFrameCursor::new(logical, &present, 0).unwrap();
            assert_eq!(cursor.next().unwrap().class, expected);
            assert_eq!(cursor.logical_offset(), 4);
            assert!(cursor.next().is_none());
        }
    }

    #[test]
    fn lazy_validity_backfills_an_all_valid_prefix_on_first_selected_null() {
        for backend in supported_forced_backends() {
            let mut mapper = forced_mapper(backend);
            let mut validity = None;

            let selected = [0b0010_1101];
            let all_present = [u8::MAX];
            let logical = OptionalSelectionView::new(&selected, 0, 6).unwrap();
            let first = mapper
                .map_into(logical, &all_present, 0, &mut validity, 0)
                .unwrap();
            assert_eq!(first.selected_logical_rows, 4, "{backend:?}");
            assert!(validity.is_none(), "{backend:?}");

            let selected = [0b0000_1111];
            let present = [0b0000_1011];
            let logical = OptionalSelectionView::new(&selected, 0, 4).unwrap();
            let second = mapper
                .map_into(logical, &present, 0, &mut validity, 4)
                .unwrap();
            assert_eq!(second.selected_logical_rows, 4, "{backend:?}");
            let validity = validity.as_ref().unwrap();
            assert_eq!(validity.len(), 8, "{backend:?}");
            assert_eq!(
                unpack(validity.as_slice(), validity.len()),
                [true, true, true, true, true, true, false, true],
                "{backend:?}"
            );
        }
    }

    #[test]
    fn lazy_validity_materializes_immediately_and_appends_later_valid_rows() {
        for backend in supported_forced_backends() {
            let mut mapper = forced_mapper(backend);
            let mut validity = None;

            let selected = [0b0000_1111];
            let present = [0b0000_1101];
            let logical = OptionalSelectionView::new(&selected, 0, 4).unwrap();
            mapper
                .map_into(logical, &present, 0, &mut validity, 0)
                .unwrap();
            assert_eq!(
                unpack_lazy_validity(&validity, 4),
                [true, false, true, true],
                "{backend:?}"
            );

            let selected = [0b0010_1101];
            let all_present = [u8::MAX];
            let logical = OptionalSelectionView::new(&selected, 0, 6).unwrap();
            mapper
                .map_into(logical, &all_present, 0, &mut validity, 4)
                .unwrap();
            assert_eq!(
                unpack_lazy_validity(&validity, 8),
                [true, false, true, true, true, true, true, true],
                "{backend:?}"
            );
        }
    }

    #[test]
    fn forced_bmi2_rejects_an_unsupported_cpu_before_mapping() {
        let err = OptionalSelectionMapper::try_new_forced_with_bmi2_support(
            ForcedOptionalMapBackend::Bmi2Pext,
            false,
        )
        .unwrap_err();
        assert!(err.to_string().contains("requires x86_64 BMI2 support"));

        assert!(
            OptionalSelectionMapper::try_new_forced_with_bmi2_support(
                ForcedOptionalMapBackend::CurrentSetBitScalar,
                false,
            )
            .is_ok()
        );
        assert!(
            OptionalSelectionMapper::try_new_forced_with_bmi2_support(
                ForcedOptionalMapBackend::AdaptiveScalar,
                false,
            )
            .is_ok()
        );
    }

    #[test]
    fn forced_backend_route_counters_bind_one_fragment_and_exact_compressions() {
        let selected = [0x55; 8];
        let present_sparse_null = [0xef, 0xfd, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff];
        let logical = OptionalSelectionView::new(&selected, 0, 64).unwrap();

        for backend in supported_forced_backends() {
            let mut lean = forced_mapper(backend);
            let counters = lean
                .map_into(logical, &present_sparse_null, 0, &mut None, 0)
                .unwrap();
            assert_eq!(counters.current_backend_fragments, 0, "{backend:?}");
            assert_eq!(counters.adaptive_backend_fragments, 0, "{backend:?}");
            assert_eq!(counters.bmi2_backend_fragments, 0, "{backend:?}");
            assert_eq!(counters.physical_compression_calls, 0, "{backend:?}");
            assert_eq!(counters.output_compression_calls, 0, "{backend:?}");
            assert_eq!(counters.current_scalar_compression_calls, 0, "{backend:?}");
            assert_eq!(counters.adaptive_physical_sparse_calls, 0, "{backend:?}");
            assert_eq!(counters.adaptive_physical_fallback_calls, 0, "{backend:?}");
            assert_eq!(counters.adaptive_output_sparse_calls, 0, "{backend:?}");
            assert_eq!(counters.adaptive_output_fallback_calls, 0, "{backend:?}");
            assert_eq!(counters.bmi2_compression_calls, 0, "{backend:?}");
        }

        let mut current = forced_mapper(ForcedOptionalMapBackend::CurrentSetBitScalar);
        let counters = current
            .map_into_observed(logical, &present_sparse_null, 0, &mut None, 0)
            .unwrap();
        assert_eq!(counters.current_backend_fragments, 1);
        assert_eq!(counters.physical_compression_calls, 1);
        assert_eq!(counters.output_compression_calls, 1);
        assert_eq!(counters.current_scalar_compression_calls, 2);

        let mut adaptive = forced_mapper(ForcedOptionalMapBackend::AdaptiveScalar);
        let counters = adaptive
            .map_into_observed(logical, &present_sparse_null, 0, &mut None, 0)
            .unwrap();
        assert_eq!(counters.adaptive_backend_fragments, 1);
        assert_eq!(counters.physical_compression_calls, 1);
        assert_eq!(counters.output_compression_calls, 1);
        assert_eq!(counters.adaptive_physical_sparse_calls, 1);
        assert_eq!(counters.adaptive_output_sparse_calls, 1);

        let present_many_nulls = [0x0f; 8];
        let counters = adaptive
            .map_into_observed(logical, &present_many_nulls, 0, &mut None, 0)
            .unwrap();
        assert_eq!(counters.adaptive_backend_fragments, 1);
        assert_eq!(counters.adaptive_physical_fallback_calls, 1);
        assert_eq!(counters.adaptive_output_fallback_calls, 1);

        if bmi2_supported() {
            let mut bmi2 = forced_mapper(ForcedOptionalMapBackend::Bmi2Pext);
            let counters = bmi2
                .map_into_observed(logical, &present_sparse_null, 0, &mut None, 0)
                .unwrap();
            assert_eq!(counters.bmi2_backend_fragments, 1);
            assert_eq!(counters.physical_compression_calls, 1);
            assert_eq!(counters.output_compression_calls, 1);
            assert_eq!(counters.bmi2_compression_calls, 2);
        }
    }

    #[test]
    fn adaptive_sparse_null_thresholds_are_inclusive_and_fall_back_above_them() {
        let selected = [0x55; 8];
        let logical = OptionalSelectionView::new(&selected, 0, 64).unwrap();
        let mut adaptive = forced_mapper(ForcedOptionalMapBackend::AdaptiveScalar);

        // Eight logical nulls include exactly four selected nulls, so both
        // conservative sparse-null thresholds are still admitted.
        let present = (!trailing_mask(8)).to_le_bytes();
        let counters = adaptive
            .map_into_observed(logical, &present, 0, &mut None, 0)
            .unwrap();
        assert_eq!(counters.adaptive_physical_sparse_calls, 1);
        assert_eq!(counters.adaptive_physical_fallback_calls, 0);
        assert_eq!(counters.adaptive_output_sparse_calls, 1);
        assert_eq!(counters.adaptive_output_fallback_calls, 0);

        // The ninth logical null is the fifth selected null for this mask,
        // placing both dimensions one step beyond their respective limits.
        let present = (!trailing_mask(9)).to_le_bytes();
        let counters = adaptive
            .map_into_observed(logical, &present, 0, &mut None, 0)
            .unwrap();
        assert_eq!(counters.adaptive_physical_sparse_calls, 0);
        assert_eq!(counters.adaptive_physical_fallback_calls, 1);
        assert_eq!(counters.adaptive_output_sparse_calls, 0);
        assert_eq!(counters.adaptive_output_fallback_calls, 1);
    }

    #[test]
    fn non_aligned_materialized_validity_prefix_is_preserved() {
        let mut prefix = BooleanBufferBuilder::new(0);
        for bit in [true, false, true, true, false] {
            prefix.append(bit);
        }
        let mut validity = Some(prefix);
        let mut mapper = OptionalSelectionMapper::default();
        let selected = [0b0011_1111];
        let present = [0b0010_1101];
        let logical = OptionalSelectionView::new(&selected, 0, 6).unwrap();
        mapper
            .map_into(logical, &present, 0, &mut validity, 5)
            .unwrap();

        let validity = validity.as_ref().unwrap();
        assert_eq!(validity.len(), 11);
        assert_eq!(
            unpack(validity.as_slice(), validity.len()),
            [
                true, false, true, true, false, true, false, true, true, false, true
            ]
        );
    }

    #[test]
    fn invalid_ranges_fail_without_mutating_output_and_mapper_remains_reusable() {
        let mut prefix = BooleanBufferBuilder::new(0);
        prefix.append_n(3, true);
        let before = prefix.as_slice().to_vec();
        let mut validity = Some(prefix);
        let selected = [0b0000_0011];
        let logical = OptionalSelectionView::new(&selected, 0, 2).unwrap();
        let mut mapper = OptionalSelectionMapper::default();

        assert!(mapper.map_into(logical, &[], 0, &mut validity, 3).is_err());
        let output = validity.as_ref().unwrap();
        assert_eq!(output.len(), 3);
        assert_eq!(output.as_slice(), before);

        let present = [0b0000_0011];
        mapper
            .map_into(logical, &present, 0, &mut validity, 3)
            .unwrap();
        assert_eq!(mapper.physical_len(), 2);
        assert_eq!(validity.as_ref().unwrap().len(), 5);

        let empty_selection = OptionalSelectionView::new(&selected, 8, 0).unwrap();
        mapper
            .map_into(empty_selection, &present, 8, &mut validity, 5)
            .unwrap();
        assert!(
            mapper
                .map_into(logical, &present, usize::MAX, &mut validity, 5)
                .is_err()
        );
    }

    #[test]
    fn mapping_is_equivalent_across_fragment_partitions() {
        let len = 129;
        let present = (0..len)
            .map(|idx| idx % 13 != 2 && idx % 29 != 7)
            .collect::<Vec<_>>();
        let selected = (0..len)
            .map(|idx| idx % 5 != 1 && idx % 23 != 4)
            .collect::<Vec<_>>();
        let present_bytes = packed_with_offset(&present, 3);
        let selected_bytes = packed_with_offset(&selected, 5);
        let whole_selection = OptionalSelectionView::new(&selected_bytes, 5, len).unwrap();

        let mut whole_mapper = OptionalSelectionMapper::default();
        let mut whole_validity = None;
        whole_mapper
            .map_into(whole_selection, &present_bytes, 3, &mut whole_validity, 0)
            .unwrap();
        let whole_physical_len = whole_mapper.physical_len();
        let whole_physical = whole_mapper.physical_selection().to_vec();

        let mut partitioned_mapper = OptionalSelectionMapper::default();
        let mut partitioned_physical = BooleanBufferBuilder::new(0);
        let mut partitioned_validity = None;
        let mut logical_start = 0;
        let mut output_prefix = 0;
        for fragment_len in [0, 1, 63, 0, 64, 1] {
            let logical = whole_selection.slice(logical_start, fragment_len);
            let counters = partitioned_mapper
                .map_into(
                    logical,
                    &present_bytes,
                    3 + logical_start,
                    &mut partitioned_validity,
                    output_prefix,
                )
                .unwrap();
            partitioned_physical.append_packed_range(
                0..partitioned_mapper.physical_len(),
                partitioned_mapper.physical_selection(),
            );
            logical_start += fragment_len;
            output_prefix += counters.selected_logical_rows;
        }

        assert_eq!(logical_start, len);
        assert_eq!(partitioned_physical.len(), whole_physical_len);
        assert_eq!(partitioned_physical.as_slice(), whole_physical);
        assert_eq!(
            unpack_lazy_validity(&partitioned_validity, output_prefix),
            unpack_lazy_validity(&whole_validity, output_prefix)
        );
    }

    #[test]
    fn reusable_physical_scratch_keeps_warm_capacity() {
        let len = 4096;
        let present = vec![0xff; len / 8];
        let selected = vec![0x55; len / 8];
        let selection = OptionalSelectionView::new(&selected, 0, len).unwrap();
        let mut mapper = OptionalSelectionMapper::default();
        let mut validity = None;
        mapper
            .map_into(selection, &present, 0, &mut validity, 0)
            .unwrap();
        let capacity = mapper.physical_capacity();
        let allocation = mapper.physical_selection().as_ptr();

        let selection = OptionalSelectionView::new(&selected, 0, len / 2).unwrap();
        mapper
            .map_into(selection, &present, 0, &mut validity, 0)
            .unwrap();
        assert_eq!(mapper.physical_capacity(), capacity);
        assert_eq!(mapper.physical_selection().as_ptr(), allocation);
        assert!(validity.is_none());
    }
}
