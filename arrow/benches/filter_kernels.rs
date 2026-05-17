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
extern crate arrow;

use std::mem::size_of;
use std::sync::Arc;

use arrow::buffer::{MutableBuffer, ScalarBuffer};
use arrow::compute::{FilterBuilder, FilterPredicate, filter_record_batch};
use arrow::util::bench_util::*;

use arrow::array::*;
use arrow::compute::filter;
use arrow::datatypes::{Field, Float32Type, Int32Type, Int64Type, Schema, UInt8Type};

use arrow_array::types::Decimal128Type;
use criterion::{Criterion, criterion_group, criterion_main};
use std::hint;

fn bench_filter(data_array: &dyn Array, filter_array: &BooleanArray) {
    hint::black_box(filter(data_array, filter_array).unwrap());
}

fn bench_built_filter(filter: &FilterPredicate, array: &dyn Array) {
    hint::black_box(filter.filter(array).unwrap());
}

#[cfg(target_arch = "x86_64")]
fn avx512_i32_filter_supported() -> bool {
    std::is_x86_feature_detected!("avx512f")
}

#[cfg(target_arch = "x86_64")]
fn bench_filter_i32_avx512(data_array: &Int32Array, filter_array: &BooleanArray) {
    // SAFETY: The caller only invokes this when AVX512F is available.
    hint::black_box(unsafe { filter_i32_avx512(data_array, filter_array) });
}

#[cfg(target_arch = "x86_64")]
fn assert_filter_i32_avx512_matches_baseline(data_array: &Int32Array, filter_array: &BooleanArray) {
    let baseline = filter(data_array, filter_array).unwrap();
    let baseline = baseline.as_any().downcast_ref::<Int32Array>().unwrap();

    // SAFETY: The caller only invokes this when AVX512F is available.
    let actual = unsafe { filter_i32_avx512(data_array, filter_array) };
    assert_eq!(baseline, &actual);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn filter_i32_avx512(data_array: &Int32Array, filter_array: &BooleanArray) -> Int32Array {
    use std::arch::x86_64::{
        _mm512_loadu_epi32, _mm512_mask_compressstoreu_epi32, _mm512_maskz_loadu_epi32,
    };

    assert_eq!(data_array.len(), filter_array.len());
    assert_eq!(data_array.null_count(), 0);
    assert_eq!(filter_array.null_count(), 0);

    let selected = filter_array.values().count_set_bits();
    let mut output = MutableBuffer::with_capacity(selected * size_of::<i32>());

    let values = data_array.values().as_ptr();
    let out = output.as_mut_ptr() as *mut i32;
    let chunks = filter_array.values().bit_chunks();

    let mut input_offset = 0;
    let mut output_offset = 0;

    for filter_chunk in chunks.iter() {
        if filter_chunk != 0 {
            for i in 0..4 {
                let mask = ((filter_chunk >> (i * 16)) & 0xFFFF) as u16;
                if mask != 0 {
                    let data_chunk = unsafe { _mm512_loadu_epi32(values.add(input_offset)) };
                    unsafe {
                        _mm512_mask_compressstoreu_epi32(out.add(output_offset), mask, data_chunk);
                    }
                }
                output_offset += mask.count_ones() as usize;
                input_offset += 16;
            }
        } else {
            input_offset += 64;
        }
    }

    let remainder = chunks.remainder_bits();
    let mut remaining = chunks.remainder_len();
    for i in 0..4 {
        if remaining == 0 {
            break;
        }

        let lane_count = remaining.min(16);
        let active_mask = if lane_count == 16 {
            u16::MAX
        } else {
            (1_u16 << lane_count) - 1
        };
        let mask = ((remainder >> (i * 16)) as u16) & active_mask;

        if mask != 0 {
            let data_chunk =
                unsafe { _mm512_maskz_loadu_epi32(active_mask, values.add(input_offset)) };
            unsafe {
                _mm512_mask_compressstoreu_epi32(out.add(output_offset), mask, data_chunk);
            }
        }

        output_offset += mask.count_ones() as usize;
        input_offset += lane_count;
        remaining -= lane_count;
    }

    debug_assert_eq!(output_offset, selected);
    unsafe { output.set_len(selected * size_of::<i32>()) };
    Int32Array::new(ScalarBuffer::from(output), None)
}

fn add_benchmark(c: &mut Criterion) {
    let size = 65536;
    let filter_array = create_boolean_array(size, 0.0, 0.5);
    let dense_filter_array = create_boolean_array(size, 0.0, 1.0 - 1.0 / 1024.0);
    let sparse_filter_array = create_boolean_array(size, 0.0, 1.0 / 1024.0);
    let issue_1949_dense_filter_array = create_boolean_array(size, 0.0, 0.95);
    let issue_1949_sparse_filter_array = create_boolean_array(size, 0.0, 0.05);

    let filter = FilterBuilder::new(&filter_array).optimize().build();
    let dense_filter = FilterBuilder::new(&dense_filter_array).optimize().build();
    let sparse_filter = FilterBuilder::new(&sparse_filter_array).optimize().build();

    let data_array = create_primitive_array::<UInt8Type>(size, 0.0);

    c.bench_function("filter optimize (kept 1/2)", |b| {
        b.iter(|| FilterBuilder::new(&filter_array).optimize().build())
    });

    c.bench_function("filter optimize high selectivity (kept 1023/1024)", |b| {
        b.iter(|| FilterBuilder::new(&dense_filter_array).optimize().build())
    });

    c.bench_function("filter optimize low selectivity (kept 1/1024)", |b| {
        b.iter(|| FilterBuilder::new(&sparse_filter_array).optimize().build())
    });

    c.bench_function("filter u8 (kept 1/2)", |b| {
        b.iter(|| bench_filter(&data_array, &filter_array))
    });
    c.bench_function("filter u8 high selectivity (kept 1023/1024)", |b| {
        b.iter(|| bench_filter(&data_array, &dense_filter_array))
    });
    c.bench_function("filter u8 low selectivity (kept 1/1024)", |b| {
        b.iter(|| bench_filter(&data_array, &sparse_filter_array))
    });

    c.bench_function("filter context u8 (kept 1/2)", |b| {
        b.iter(|| bench_built_filter(&filter, &data_array))
    });
    c.bench_function("filter context u8 high selectivity (kept 1023/1024)", |b| {
        b.iter(|| bench_built_filter(&dense_filter, &data_array))
    });
    c.bench_function("filter context u8 low selectivity (kept 1/1024)", |b| {
        b.iter(|| bench_built_filter(&sparse_filter, &data_array))
    });

    let data_array = create_primitive_array::<Int32Type>(size, 0.0);
    c.bench_function("filter i32 (kept 1/2)", |b| {
        b.iter(|| bench_filter(&data_array, &filter_array))
    });
    c.bench_function("filter i32 high selectivity (kept 1023/1024)", |b| {
        b.iter(|| bench_filter(&data_array, &dense_filter_array))
    });
    c.bench_function("filter i32 low selectivity (kept 1/1024)", |b| {
        b.iter(|| bench_filter(&data_array, &sparse_filter_array))
    });
    c.bench_function("filter i32 issue1949 high selectivity (kept 95%)", |b| {
        b.iter(|| bench_filter(&data_array, &issue_1949_dense_filter_array))
    });
    c.bench_function("filter i32 issue1949 low selectivity (kept 5%)", |b| {
        b.iter(|| bench_filter(&data_array, &issue_1949_sparse_filter_array))
    });

    #[cfg(target_arch = "x86_64")]
    if avx512_i32_filter_supported() {
        assert_filter_i32_avx512_matches_baseline(&data_array, &filter_array);
        assert_filter_i32_avx512_matches_baseline(&data_array, &dense_filter_array);
        assert_filter_i32_avx512_matches_baseline(&data_array, &sparse_filter_array);
        assert_filter_i32_avx512_matches_baseline(&data_array, &issue_1949_dense_filter_array);
        assert_filter_i32_avx512_matches_baseline(&data_array, &issue_1949_sparse_filter_array);

        c.bench_function("filter i32 avx512 issue1949 (kept 1/2)", |b| {
            b.iter(|| bench_filter_i32_avx512(&data_array, &filter_array))
        });
        c.bench_function("filter i32 avx512 high selectivity (kept 1023/1024)", |b| {
            b.iter(|| bench_filter_i32_avx512(&data_array, &dense_filter_array))
        });
        c.bench_function("filter i32 avx512 low selectivity (kept 1/1024)", |b| {
            b.iter(|| bench_filter_i32_avx512(&data_array, &sparse_filter_array))
        });
        c.bench_function(
            "filter i32 avx512 issue1949 high selectivity (kept 95%)",
            |b| b.iter(|| bench_filter_i32_avx512(&data_array, &issue_1949_dense_filter_array)),
        );
        c.bench_function(
            "filter i32 avx512 issue1949 low selectivity (kept 5%)",
            |b| b.iter(|| bench_filter_i32_avx512(&data_array, &issue_1949_sparse_filter_array)),
        );
    }

    c.bench_function("filter context i32 (kept 1/2)", |b| {
        b.iter(|| bench_built_filter(&filter, &data_array))
    });
    c.bench_function(
        "filter context i32 high selectivity (kept 1023/1024)",
        |b| b.iter(|| bench_built_filter(&dense_filter, &data_array)),
    );
    c.bench_function("filter context i32 low selectivity (kept 1/1024)", |b| {
        b.iter(|| bench_built_filter(&sparse_filter, &data_array))
    });

    let data_array = create_primitive_array::<Int32Type>(size, 0.5);
    c.bench_function("filter context i32 w NULLs (kept 1/2)", |b| {
        b.iter(|| bench_built_filter(&filter, &data_array))
    });
    c.bench_function(
        "filter context i32 w NULLs high selectivity (kept 1023/1024)",
        |b| b.iter(|| bench_built_filter(&dense_filter, &data_array)),
    );
    c.bench_function(
        "filter context i32 w NULLs low selectivity (kept 1/1024)",
        |b| b.iter(|| bench_built_filter(&sparse_filter, &data_array)),
    );

    let data_array = create_primitive_array::<UInt8Type>(size, 0.5);
    c.bench_function("filter context u8 w NULLs (kept 1/2)", |b| {
        b.iter(|| bench_built_filter(&filter, &data_array))
    });
    c.bench_function(
        "filter context u8 w NULLs high selectivity (kept 1023/1024)",
        |b| b.iter(|| bench_built_filter(&dense_filter, &data_array)),
    );
    c.bench_function(
        "filter context u8 w NULLs low selectivity (kept 1/1024)",
        |b| b.iter(|| bench_built_filter(&sparse_filter, &data_array)),
    );

    let data_array = create_primitive_array::<Float32Type>(size, 0.5);
    c.bench_function("filter f32 (kept 1/2)", |b| {
        b.iter(|| bench_filter(&data_array, &filter_array))
    });
    c.bench_function("filter context f32 (kept 1/2)", |b| {
        b.iter(|| bench_built_filter(&filter, &data_array))
    });
    c.bench_function(
        "filter context f32 high selectivity (kept 1023/1024)",
        |b| b.iter(|| bench_built_filter(&dense_filter, &data_array)),
    );
    c.bench_function("filter context f32 low selectivity (kept 1/1024)", |b| {
        b.iter(|| bench_built_filter(&sparse_filter, &data_array))
    });

    let data_array = create_primitive_array::<Decimal128Type>(size, 0.0);
    c.bench_function("filter decimal128 (kept 1/2)", |b| {
        b.iter(|| bench_filter(&data_array, &filter_array))
    });
    c.bench_function("filter decimal128 high selectivity (kept 1023/1024)", |b| {
        b.iter(|| bench_filter(&data_array, &dense_filter_array))
    });
    c.bench_function("filter decimal128 low selectivity (kept 1/1024)", |b| {
        b.iter(|| bench_filter(&data_array, &sparse_filter_array))
    });

    c.bench_function("filter context decimal128 (kept 1/2)", |b| {
        b.iter(|| bench_built_filter(&filter, &data_array))
    });
    c.bench_function(
        "filter context decimal128 high selectivity (kept 1023/1024)",
        |b| b.iter(|| bench_built_filter(&dense_filter, &data_array)),
    );
    c.bench_function(
        "filter context decimal128 low selectivity (kept 1/1024)",
        |b| b.iter(|| bench_built_filter(&sparse_filter, &data_array)),
    );

    let data_array = create_string_array::<i32>(size, 0.5);
    c.bench_function("filter context string (kept 1/2)", |b| {
        b.iter(|| bench_built_filter(&filter, &data_array))
    });
    c.bench_function(
        "filter context string high selectivity (kept 1023/1024)",
        |b| b.iter(|| bench_built_filter(&dense_filter, &data_array)),
    );
    c.bench_function("filter context string low selectivity (kept 1/1024)", |b| {
        b.iter(|| bench_built_filter(&sparse_filter, &data_array))
    });

    let data_array = create_string_dict_array::<Int32Type>(size, 0.0, 4);
    c.bench_function("filter context string dictionary (kept 1/2)", |b| {
        b.iter(|| bench_built_filter(&filter, &data_array))
    });
    c.bench_function(
        "filter context string dictionary high selectivity (kept 1023/1024)",
        |b| b.iter(|| bench_built_filter(&dense_filter, &data_array)),
    );
    c.bench_function(
        "filter context string dictionary low selectivity (kept 1/1024)",
        |b| b.iter(|| bench_built_filter(&sparse_filter, &data_array)),
    );

    let data_array = create_string_dict_array::<Int32Type>(size, 0.5, 4);
    c.bench_function("filter context string dictionary w NULLs (kept 1/2)", |b| {
        b.iter(|| bench_built_filter(&filter, &data_array))
    });
    c.bench_function(
        "filter context string dictionary w NULLs high selectivity (kept 1023/1024)",
        |b| b.iter(|| bench_built_filter(&dense_filter, &data_array)),
    );
    c.bench_function(
        "filter context string dictionary w NULLs low selectivity (kept 1/1024)",
        |b| b.iter(|| bench_built_filter(&sparse_filter, &data_array)),
    );

    let mut add_benchmark_for_fsb_with_length = |value_length: usize| {
        let data_array = create_fsb_array(size, 0.0, value_length);
        c.bench_function(
            format!("filter fsb with value length {value_length} (kept 1/2)").as_str(),
            |b| b.iter(|| bench_filter(&data_array, &filter_array)),
        );
        c.bench_function(
            format!(
                "filter fsb with value length {value_length} high selectivity (kept 1023/1024)"
            )
            .as_str(),
            |b| b.iter(|| bench_filter(&data_array, &dense_filter_array)),
        );
        c.bench_function(
            format!("filter fsb with value length {value_length} low selectivity (kept 1/1024)")
                .as_str(),
            |b| b.iter(|| bench_filter(&data_array, &sparse_filter_array)),
        );

        c.bench_function(
            format!("filter context fsb with value length {value_length} (kept 1/2)").as_str(),
            |b| b.iter(|| bench_built_filter(&filter, &filter_array)),
        );
        c.bench_function(
            format!(
                "filter context fsb with value length {value_length} high selectivity (kept 1023/1024)"
            )
            .as_str(),
            |b| b.iter(|| bench_built_filter(&filter, &dense_filter_array)),
        );
        c.bench_function(
            format!(
                "filter context fsb with value length {value_length} low selectivity (kept 1/1024)"
            )
            .as_str(),
            |b| b.iter(|| bench_built_filter(&filter, &sparse_filter_array)),
        );
    };

    add_benchmark_for_fsb_with_length(5);
    add_benchmark_for_fsb_with_length(20);
    add_benchmark_for_fsb_with_length(50);

    let data_array = create_primitive_array::<Float32Type>(size, 0.0);

    let field = Field::new("c1", data_array.data_type().clone(), true);
    let schema = Schema::new(vec![field]);

    let batch = RecordBatch::try_new(Arc::new(schema), vec![Arc::new(data_array)]).unwrap();

    c.bench_function("filter single record batch", |b| {
        b.iter(|| filter_record_batch(&batch, &filter_array))
    });

    let data_array = create_string_view_array_with_len(size, 0.5, 4, false);
    c.bench_function("filter context short string view (kept 1/2)", |b| {
        b.iter(|| bench_built_filter(&filter, &data_array))
    });
    c.bench_function(
        "filter context short string view high selectivity (kept 1023/1024)",
        |b| b.iter(|| bench_built_filter(&dense_filter, &data_array)),
    );
    c.bench_function(
        "filter context short string view low selectivity (kept 1/1024)",
        |b| b.iter(|| bench_built_filter(&sparse_filter, &data_array)),
    );

    let data_array = create_string_view_array_with_len(size, 0.5, 4, true);
    c.bench_function("filter context mixed string view (kept 1/2)", |b| {
        b.iter(|| bench_built_filter(&filter, &data_array))
    });
    c.bench_function(
        "filter context mixed string view high selectivity (kept 1023/1024)",
        |b| b.iter(|| bench_built_filter(&dense_filter, &data_array)),
    );
    c.bench_function(
        "filter context mixed string view low selectivity (kept 1/1024)",
        |b| b.iter(|| bench_built_filter(&sparse_filter, &data_array)),
    );

    let data_array = create_primitive_run_array::<Int32Type, Int64Type>(size, size);
    c.bench_function("filter run array (kept 1/2)", |b| {
        b.iter(|| bench_built_filter(&filter, &data_array))
    });
    c.bench_function("filter run array high selectivity (kept 1023/1024)", |b| {
        b.iter(|| bench_built_filter(&dense_filter, &data_array))
    });
    c.bench_function("filter run array low selectivity (kept 1/1024)", |b| {
        b.iter(|| bench_built_filter(&sparse_filter, &data_array))
    });
}

criterion_group!(benches, add_benchmark);
criterion_main!(benches);
