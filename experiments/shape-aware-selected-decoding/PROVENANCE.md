# Provenance and reproduction

Companion to [README.md](./README.md). Everything here is stated so a reader can
check a number rather than take it on trust.

---

## 1. Commit pins

### arrow-rs (this repository)

Branch `exp/v21-rle-selected-fill-20260807`.

| SHA | What it is |
|---|---|
| `5b748e140` | merge base with upstream `main` at the time the wiring was designed |
| `bbe630194` | Step 1 — five-layer selected-decode wiring, default off |
| `4122ce07a` | Step 2 — flag threaded through the async / push-decoder path |
| `d7ab900a7a` | fixes for both correctness traps found by the differential gate |
| `45b1cc34a` | coverage counters (`parquet::arrow::selected_decode_metrics`) — **the commit all reported end-to-end numbers were measured on** |

### DataFusion (separate repository, same experiment)

| Branch | SHA | Role |
|---|---|---|
| `exp/v21-rle-selected-fill-20260807` | `2004a140c1` | shared base: session option `datafusion.execution.parquet.selected_decode` (**default false**), plumbing, benchmark instrumentation |
| `exp/v26-gw2-selected-decode-on-20260809` | `a5765d32c8` | measurement-only arm: identical to the base except the option defaults to **true**. Never intended for merge |
| `exp/v26-gw1-selected-decode-on-20260809` | `ef358c22d0` | earlier measurement-only arm, retained as the evidence pin for the correctness gate's first (failing) run |

**The two end-to-end arms differ in exactly one file** (`datafusion/common/src/config.rs`,
the default value). Verified with `git diff --stat`.

### Why the flag default is flipped in a branch instead of passed as a runner flag

The benchmark cluster enforces a `ValidatingAdmissionPolicy` that pins the
runner container's argument count *and* the value at each position. Adding a new
CLI flag would have caused the cluster to reject every job submitted by anyone,
so the A/B varies the DataFusion commit instead. An initial attempt to extend the
shared tooling was fully reverted and re-verified against its own test baseline.

---

## 2. Measurement matrix

All end-to-end runs: one Kubernetes job per (suite × arm × round), 5 iterations
per query, 24 partitions, batch size 8192, `pushdown_filters` enabled.

| Purpose | Suite | Arms | Rounds |
|---|---|---|---|
| Correctness gate, first run | ClickBench + TPC-DS | off vs on | 1 |
| Correctness gate, after fixes | ClickBench + TPC-DS | off vs on | 1 |
| Digest-stability control | TPC-DS (q31/q65/q71), ClickBench (q17/q31/q32/q39) | off vs off, same commit | 2 |
| Coverage + benefit | ClickBench + TPC-DS | off vs on | 1 (+1 ClickBench confirmation) |
| Leaf, production-shape comparator | 4 captured replay fixtures + synthetic grid | 6–7 arms | 2 |

Datasets: ClickBench `hits` single-file (100M rows) and TPC-DS SF10, both pinned
by content digest in the run records.

---

## 3. Data files

| File | Contents |
|---|---|
| [`data/per-query-coverage-and-timing.csv`](./data/per-query-coverage-and-timing.csv) | Every query of both suites: selected/fallback rows, selected-row fraction, selected/fallback batches, and per-round off/on timings (141 rows) |
| [`data/leaf-production-shape-comparison.csv`](./data/leaf-production-shape-comparison.csv) | Leaf-level timings per captured fixture per round: `tiered`, `production_shape`, `decode_all_indices_compact`, `materialize_then_filter`, and the speedup ratio |
| [`data/tpch-sf1-coverage-v27.csv`](./data/tpch-sf1-coverage-v27.csv) | **Added 2026-08-10.** TPC-H SF1, all 22 queries × 3 arms (flag off / flag on / flag on with the predicate cache disabled). Coverage counters only |

### The TPC-H run is not comparable to the other end-to-end runs

It was executed **locally, not on the benchmark cluster, and reports no
timings**. That is deliberate rather than a shortcut: admission is decided by
`supports_selected_decode()`, which reads only `max_def_level`, `max_rep_level`
and the reader type — **encoding is explicitly not consulted**. Coverage is
therefore a pure function of schema and projection, invariant to scale factor and
to hardware, so a local SF1 run answers the coverage question exactly. **No
timing claim is made from it, and none should be read into it** — this program
has previously measured 19% swings between rounds on identical binaries on
dedicated hardware, and a laptop is worse.

Fixture: `tpchgen-cli` 2.0.2, `--scale-factor 1 --format parquet
--parquet-compression ZSTD(1)`, reorganised into the `tpch-table-directory-v1`
layout the runner expects. Same generator and version as the reviewed
`tpch-sf10-v1` dataset recipe, so the schema — every leaf column `REQUIRED` — is
the same one the cluster dataset would present.

Two local-only build affordances were used and **reverted afterwards**: a
`[patch.crates-io]` block pointing DataFusion at the arrow-rs worktree, and an
env-gated trace in `read_mask_batch` that separates "the flag never arrived" from
"the reader declined". Neither is committed. The reported numbers come from the
counters, not the trace.

### A caveat you will see in the leaf data

`clickbench/id26/hits.WatchID/q27` round 1 reports `production_shape` at
1012.2 µs, giving an implausible 6.12x. **This is an outlier and is not used in
any claim.** It is a 6x deviation from every other measurement in the dataset
(84 arm/cell/round combinations otherwise agree to within 0.1–9.4% round to
round); round 2 reads 170.36 µs, which matches an independent local measurement
and is corroborated by `tiered`'s own tight round-to-round agreement at the same
cell (165.29 vs 166.31 µs, ruling out a whole-round slowdown). The row is left in
the CSV rather than deleted so the anomaly is visible; §2 of the README uses the
round-2 value, and this cell sits in the near-full-survival regime where nothing
is claimed either way.

---

## 4. Fixture provenance

The leaf fixtures are not synthetic. They are real dictionary pages plus the real
row-selection masks captured from instrumented ClickBench and TPC-DS runs on the
pinned datasets, sealed as `dict.bin` / `page1.bin` / mask pairs with the source
file fingerprint, row-group and column-chunk identity recorded alongside. Each
fixture is identified in the CSV as
`<suite>/<capture-id>/<table>.<column>/<query>`.

Every benchmark arm — including the production-shape comparator — is
digest-verified to produce byte-identical output to every other arm on every
fixture *before* any timing is recorded, in an untimed phase. A timing number
is never reported for an arm that does not compute the same answer.

---

## 5. Reproducing

Inside this repository:

```bash
# Correctness: differential + fuzz + coverage-counter tests
cargo test -p parquet --lib --all-features selected_decode

# Leaf benchmarks (fixtures live in the DataFusion-side bench crate)
cargo bench -p v21-rle-selected-fill-bench --bench bitpacked_direct_gather_replay
cargo bench -p v21-rle-selected-fill-bench --bench tiered_rle_admission_grid
```

End-to-end runs used an internal Kubernetes benchmark harness, so they are not
directly reproducible outside that environment. What *is* portable: both commit
pins, the per-query CSVs above, and the fact that the two arms differ only in one
default value — so an equivalent A/B is straightforward to reconstruct with any
runner by building the two DataFusion commits and comparing per-query output and
timing.

---

## 6. Key tests to look at

| Test | Guards |
|---|---|
| `test_selected_decode_survives_mid_chunk_encoding_change` | the silent wrong-row trap (README §3, Trap 2) |
| `test_selected_decode_differential_fuzz` | randomized single-column equivalence across run structure, dictionary width, page/row-group boundaries |
| `test_selected_decode_differential_fuzz_multi_column` | randomized *mixed-eligibility* projections — the admission-ordering trap (Trap 1) |
| `test_selected_decode_coverage_counter_attributes_rows` | the counter itself, for eligible+on / eligible+off / ineligible+on |
| `test_async_selected_decode_matches_default` | the flag actually reaches the async / push-decoder path |
| `probe_tpch_lineitem_admission_by_type` (`parquet/tests/tpch_selected_decode_probe.rs`) | **Added 2026-08-10.** Which projected shapes the admission rule accepts, measured directly against a real TPC-H `lineitem` file: plain `INT32`/`INT64`, `DECIMAL`-on-`INT64` and `DATE`-on-`INT32` all admitted; `BYTE_ARRAY` correctly rejected. Skips when the fixture is absent |

All tests exercising this path force `RowSelectionPolicy::Mask` explicitly. Under
the default `Auto` policy, long selector spans resolve to `Selectors` and
`read_mask_batch` never executes — two earlier fuzz tests passed for exactly that
reason and measured nothing. Each gate-bearing test was verified to **fail on a
deliberately broken build** before its pass was trusted.
