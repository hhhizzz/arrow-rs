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
//! an optional BMI2 backend materially reduce the nullable selection-mapping leaf?
//! It is intentionally not wired into `GenericRecordReader` on this branch.

use std::mem::size_of;

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
    /// Low `present_count` bits in physical-value coordinates.
    pub physical_selection: u64,
    /// Low `selected_count` bits in selected-output coordinates.
    pub output_validity: u64,
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
        let (physical_selection, output_validity) = match class {
            OptionalFrameClass::EmptySelection | OptionalFrameClass::AllNull => (0, 0),
            OptionalFrameClass::AllPresentIdentity => {
                (selected_mask, trailing_mask(selected_count as usize))
            }
            OptionalFrameClass::FullSelection => {
                (trailing_mask(present_count as usize), present_mask)
            }
            OptionalFrameClass::AllValidSelected => (
                compress_scalar(selected_mask, present_mask),
                trailing_mask(selected_count as usize),
            ),
            OptionalFrameClass::General => (
                compress_scalar(selected_mask, present_mask),
                compress_scalar(present_mask, selected_mask),
            ),
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
            physical_selection,
            output_validity,
        })
    }
}

/// Reusable mapper workspace. After its packed buffer reaches the largest
/// fragment capacity, mapping equal or smaller fragments does not allocate.
#[derive(Debug)]
pub(crate) struct OptionalSelectionMapper {
    physical_selection: BooleanBufferBuilder,
}

impl Default for OptionalSelectionMapper {
    fn default() -> Self {
        Self {
            physical_selection: BooleanBufferBuilder::new(0),
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
        if output_validity
            .as_ref()
            .is_some_and(|builder| builder.len() != output_prefix_len)
        {
            return Err(general_err!(
                "optional output validity has length {}, expected prefix length {output_prefix_len}",
                output_validity.as_ref().unwrap().len()
            ));
        }
        self.physical_selection.truncate(0);
        self.physical_selection.reserve(selection.len());

        let mut counters = OptionalFrameCounters::default();
        let mut output_rows = output_prefix_len;
        for frame in OptionalFrameCursor::new(selection, validity, validity_offset)? {
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

            // Even an empty logical selection must append `present_count` zero
            // bits: the physical cursor still advances across every present row.
            self.physical_selection
                .append_word(frame.physical_selection, present_count);
            match output_validity {
                Some(builder) => builder.append_word(frame.output_validity, selected_count),
                None if selected_present_count != selected_count => {
                    let mut builder = BooleanBufferBuilder::new(0);
                    builder.append_n(output_rows, true);
                    builder.append_word(frame.output_validity, selected_count);
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

    pub(crate) fn physical_selection(&self) -> &[u8] {
        self.physical_selection.as_slice()
    }

    pub(crate) fn physical_len(&self) -> usize {
        self.physical_selection.len()
    }

    #[cfg(test)]
    fn physical_capacity(&self) -> usize {
        self.physical_selection.capacity()
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
    fn exhaustive_scalar_mapper_matches_reference_through_eight_rows() {
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
                    let mut mapper = OptionalSelectionMapper::default();
                    let mut validity = None;
                    let counters = mapper
                        .map_into(selection, &present_bytes, 0, &mut validity, 0)
                        .unwrap();
                    let (expected_physical, expected_validity) = reference(&present, &selected);
                    assert_eq!(mapper.physical_len(), expected_physical.len());
                    assert_eq!(
                        unpack(mapper.physical_selection(), expected_physical.len()),
                        expected_physical
                    );
                    assert_eq!(
                        unpack_lazy_validity(&validity, expected_validity.len()),
                        expected_validity
                    );
                    assert_unused_tail_is_zero(
                        mapper.physical_selection(),
                        expected_physical.len(),
                    );
                    if let Some(validity) = validity.as_ref() {
                        assert_eq!(validity.len(), expected_validity.len());
                        assert_unused_tail_is_zero(validity.as_slice(), expected_validity.len());
                    }
                    assert_eq!(counters.logical_rows, len);
                    assert_eq!(counters.present_rows, present_bits.count_ones() as usize);
                    assert_eq!(
                        counters.selected_logical_rows,
                        selected_bits.count_ones() as usize
                    );
                    assert_eq!(
                        counters.selected_present_rows,
                        (present_bits & selected_bits).count_ones() as usize
                    );
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
                    let mut mapper = OptionalSelectionMapper::default();
                    let mut validity = None;
                    mapper
                        .map_into(selection, &present_bytes, validity_offset, &mut validity, 0)
                        .unwrap();
                    let (expected_physical, expected_validity) = reference(&present, &selected);
                    assert_eq!(mapper.physical_len(), expected_physical.len());
                    assert_eq!(
                        unpack(mapper.physical_selection(), expected_physical.len()),
                        expected_physical
                    );
                    assert_eq!(
                        unpack_lazy_validity(&validity, expected_validity.len()),
                        expected_validity
                    );
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
                    let mut mapper = OptionalSelectionMapper::default();
                    let mut validity = None;
                    mapper
                        .map_into(logical, &present_bytes, validity_offset, &mut validity, 0)
                        .unwrap();
                    let (expected_physical, expected_validity) = reference(&present, &selected);
                    assert_eq!(mapper.physical_len(), expected_physical.len());
                    assert_eq!(
                        unpack(mapper.physical_selection(), expected_physical.len()),
                        expected_physical
                    );
                    assert_eq!(
                        unpack_lazy_validity(&validity, expected_validity.len()),
                        expected_validity
                    );
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
        let mut mapper = OptionalSelectionMapper::default();
        let mut validity = None;

        let selected = [0b0010_1101];
        let all_present = [u8::MAX];
        let logical = OptionalSelectionView::new(&selected, 0, 6).unwrap();
        let first = mapper
            .map_into(logical, &all_present, 0, &mut validity, 0)
            .unwrap();
        assert_eq!(first.selected_logical_rows, 4);
        assert!(validity.is_none());

        let selected = [0b0000_1111];
        let present = [0b0000_1011];
        let logical = OptionalSelectionView::new(&selected, 0, 4).unwrap();
        let second = mapper
            .map_into(logical, &present, 0, &mut validity, 4)
            .unwrap();
        assert_eq!(second.selected_logical_rows, 4);
        let validity = validity.as_ref().unwrap();
        assert_eq!(validity.len(), 8);
        assert_eq!(
            unpack(validity.as_slice(), validity.len()),
            [true, true, true, true, true, true, false, true]
        );
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
