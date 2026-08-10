# Shape-aware selected decoding for Parquet dictionary streams — a completed investigation with a negative result

**Status: closed. Not proposed as an upstream feature.**
Branch: `exp/v21-rle-selected-fill-20260807` (arrow-rs) — this file lives on that branch.
Date: 2026-08-09

---

## TL;DR

I built an opt-in Parquet reader path that decodes **only the selected dictionary
indices** for flat, required, primitive leaves, wired it through both the
synchronous reader and the async/push-decoder path, and ran it end-to-end on
ClickBench and TPC-DS SF10.

- **The kernel is fast.** Against the *production-shaped* comparator — stock
  `get_batch_with_dict` full decode followed by Arrow's real
  `arrow::compute::filter` — it is **2.6x–4.6x faster** on captured
  low-survival pages.
- **The integration is correct**, but only after a correctness gate caught two
  real traps, one of which is a *silent wrong-row* bug (right row count, wrong
  data) that no row-count-only benchmark would detect.
- **Almost nothing is eligible to use it.** Under the conservative v0 admission
  rule, **0 of 99** TPC-DS queries and **4 of 42** ClickBench queries ever enter
  the path.
- **No query-level benefit was established**, and the reachable portion is small
  enough that the Amdahl ceiling is low regardless (see the explicit bound
  below).

I am **not** proposing the wiring as a production feature. This is a negative
result about *applicability under the current reader architecture on the
evaluated workloads* — not about the isolated decoder kernel.

The parts I think are worth keeping are the **correctness traps**, the
**methodology**, and the **captured workload traces** — see
[Reusable artifacts](#reusable-artifacts).

---

## 1. What was built

Five additive layers, each an optional method defaulting to today's behaviour so
every existing implementor compiles untouched:

| Layer | File | Addition |
|---|---|---|
| `Decoder<T>` | `parquet/src/encodings/decoding.rs` | `get_selected(...) -> Result<Option<(consumed, written)>>`, implemented by `DictDecoder` |
| `ColumnValueDecoder` | `parquet/src/column/reader/decoder.rs` | `read_selected(...)`, never declines for a structurally eligible column |
| `GenericColumnReader` | `parquet/src/column/reader.rs` | `read_selected(...)`, bounded by the current page's remainder |
| `GenericRecordReader` / `ArrayReader` | `parquet/src/arrow/{record_reader,array_reader}/` | `read_records_selected(...)` + `supports_selected_decode()` |
| `read_mask_batch` | `parquet/src/arrow/arrow_reader/mod.rs` | dispatch, decided once up front for the whole subtree |

Plus a decode kernel family in `parquet/src/encodings/rle.rs` and a
`BooleanBuffer → PackedSelection` handoff that is zero-copy (the Mask path
already holds exactly the `(bytes, bit_offset, len)` triple the kernel wants).

Enabled by `ArrowReaderBuilder::with_selected_decode(bool)`, **default off**.
The DataFusion side (separate branch, see
[Provenance](./PROVENANCE.md)) adds
`datafusion.execution.parquet.selected_decode`.

**v0 admission rule (deliberately conservative):** every projected column must be
flat (`max_rep_level == 0`), required (`max_def_level == 0`) and primitive;
execution must be on the row-selection `Mask` path. If any projected column is
ineligible, the whole batch falls back to today's unmodified code. Nullable
columns need a value-index↔row-index mapping through definition levels; that was
scoped as v1 and never built.

---

## 2. The kernel is fast — measured against what production actually runs

A late but decisive correction to this investigation: for most of its life it
compared against *hand-written* scalar baselines. Those are not what a reader
executes. A `production_shape` arm was added — stock full decode plus Arrow's
real SIMD `filter` kernel, with the predicate `BooleanArray` precomputed once per
page so the comparison is fair — and re-run on real hardware, two rounds,
digest-verified to produce identical output to every other arm.

| Fixture (captured real pages) | survival | selected vs production |
|---|---|---|
| ClickBench `hits.WatchID` @ q1 mask | 2.78% | **4.55x** |
| TPC-DS `catalog_returns.cr_returned_date_sk` @ q49 | 0.88% | 3.75x |
| TPC-DS `catalog_returns.cr_refunded_hdemo_sk` @ q5 | 5.11% | 3.28x |
| TPC-DS `catalog_returns.cr_returned_date_sk` @ q77 | 10.93% | 2.63x |
| synthetic grid, RLE runs ≥ 4096 | 1.6–25% | 1.73–2.31x (geomean 2.01x) |
| near-full survival (99.5%+) | ~100% | inverts — production 15–20% faster |

Production turned out to be **slower** than this project's own synthetic
baselines at low survival, i.e. the real leaf-level advantage was *larger* than
previously believed. The near-full-survival inversion is expected: at ~100%
survival the selected path degenerates into the same full decode, so there is
nothing to win.

---

## 3. Correctness: two integration traps worth knowing about

An end-to-end differential gate (per-query output digests, feature-off vs
feature-on, full ClickBench + TPC-DS) **failed on first run** and surfaced two
defects. Both are architectural rather than incidental, and I think they are the
most transferable output of this work.

### Trap 1 — subtree admission must be decided before any child produces output

`StructArrayReader` originally walked children in order and discovered a mixed
projection only *after* an earlier child had already written compact
(already-filtered) data. That cannot be unwound: you now hold one compact column
and one full column destined for the same `RecordBatch`.

It surfaced on ClickBench q10 (`SELECT "MobilePhoneModel", COUNT(DISTINCT
"UserID") ... WHERE "MobilePhoneModel" <> ''`) — a BYTE_ARRAY column projected
alongside an eligible integer one. The guard fired loudly instead of producing
wrong data, but the query aborted.

**Fix:** a side-effect-free `supports_selected_decode()` answered for the entire
reader subtree *before* the first byte is decoded. A child declining afterwards
is now an internal error, not a fallback.

### Trap 2 — a column chunk can change encoding mid-chunk, and a short read must not be read as "chunk exhausted"

This one is the reason I would write this up even if everything else had gone
well.

Real writers **abandon dictionary encoding partway through a column chunk** once
the dictionary outgrows its page-size budget. So a single column chunk can hold
`RLE_DICTIONARY` pages followed by `PLAIN` pages.

The selected decoder *declined* at the first `PLAIN` page and returned
`consumed < requested`. The layer above interpreted that short return as
"this column chunk is exhausted" and **advanced to the next row group**, taking
the remaining rows from the wrong place.

**The row count still came out exactly right.** Only a content digest caught it.
Any benchmark or test that validates row counts — or that compares only
aggregate results with a tolerance — would have passed this silently.

**Fix:** a structurally eligible column never declines. An unsupported encoding
falls through to ordinary decode-then-filter *internally*, producing identical
compact output. Regression test:
`test_selected_decode_survives_mid_chunk_encoding_change`.

### After the fixes

Every query with a **control-stable** output digest matched byte-for-byte
between feature-off and feature-on, across both suites. See
[§6](#6-what-the-digest-oracle-can-and-cannot-tell-you) for the important
caveat about what "control-stable" means and what it does not.

---

## 4. Reachability: the actual reason this stops

Gate rules were frozen before any number was read, and required coverage to be
demonstrated by a **counter**, not inferred. A counter that silently reports zero
is indistinguishable from "the path is never reached" — which is itself the kill
condition — so the counter is unit-tested to attribute rows correctly for
eligible+on, eligible+off, and ineligible+on.

| Suite | Queries entering the selected path | Share of decoded rows |
|---|---|---|
| **TPC-DS SF10** | **0 / 99** | **0.00%** (of 2.41B rows) |
| **ClickBench** | **4 / 42** (q30, q31, q41, q42) | **12.44%** |

Per-query, ClickBench:

| query | selected rows | fallback rows | coverage |
|---|---|---|---|
| q30 | 65,491,130 | 0 | 100% |
| q31 | 65,491,130 | 0 | 100% |
| q42 | 3,357,595 | 2,856,535 | 54.0% |
| q41 | 513,380 | 6,214,130 | 7.6% |

TPC-DS never qualifies because v0 requires *every* projected column to be flat,
required and primitive, and TPC-DS's schema is nullable-heavy. Its end-to-end
timing is correspondingly an exact null (geomean 0.9954 across 99 queries),
which is a useful validity check on the whole apparatus: zero coverage predicted
zero effect, and zero effect is what appeared.

---

## 5. Query-level effect: none established, and the ceiling is low anyway

### 5.1 No direction established

On the four covered ClickBench queries, two rounds disagree:

| query | coverage | round 1 (off/on) | round 2 (off/on) |
|---|---|---|---|
| q30 | 100% | 0.923 | 0.986 |
| q31 | 100% | 0.944 | **1.125** |
| q41 | 8% | 0.928 | 1.030 |
| q42 | 54% | 0.784 | 1.041 |

Round 1 alone would have supported a confident "measurably harmful" claim.
Round 2 refuted it on three of four queries — q31 swung 19% on identical
binaries. Pooled across both rounds, covered queries sit at geomean **0.9654**
against a zero-coverage control of **0.9875** (n=76), inside a per-query noise
band spanning **0.673–1.372**.

**The defensible statement is that no direction was established — neither
benefit nor harm.** I am recording the misleading single-round reading
explicitly because it is exactly the kind of result that gets published as a
regression.

### 5.2 The Amdahl bound, stated strictly

The four covered ClickBench queries are **3.02%–3.59%** of total suite runtime;
coverage-weighted (scaling each by its measured selected-row fraction),
**f = 2.76%–3.25%**.

For a decode speedup `S` applied to a fraction `f` of runtime of which a
fraction `d` is actually dictionary decode:

```
Δ ≤ f · d · (1 − 1/S)
```

Taking the **unrealistic** best case `d = 1` (i.e. pretending the entire
coverage-weighted portion is dictionary decode):

| S | bound on total suite improvement |
|---|---|
| 2x | **≤ 1.38% – 1.63%** |
| 4x | **≤ 2.07% – 2.44%** |
| ∞ | ≤ 2.76% – 3.25% |

So the strictly defensible claim is: **bounded below roughly 2.5% even under an
assumption known to be too generous.** The true bound is lower because
dictionary decode is only part of that time — but **`d` was not separately
measured**, so I do not claim a specific smaller number.

This matters more than the noise question: even a clean, reproducible 4x leaf
win could not have produced a maintainer-visible workload improvement here.

### 5.3 What this does *not* yet separate — an honest gap

The leaf fixtures that produced 2.6–4.6x were captured from ClickBench q1 and
TPC-DS q49/q5/q77. The queries that actually **reach** the path are ClickBench
q30/q31/q41/q42. **These are not the same populations.**

I have not captured the page/mask/run morphology of the four *reachable*
queries, so I cannot yet distinguish:

- **(a)** the reachable queries do sit in the leaf-winning regime, and the win is
  simply drowned by the small runtime share (§5.2) and other reader CPU; from
- **(b)** the reachable queries sit *outside* the leaf-winning regime (near-full
  survival, short runs, unfavourable bit widths, cheap columns), in which case the
  admission rule is also selecting unprofitable shapes.

So the precise conclusion is: **reachability is very small, and no reader-level
benefit was observed on the limited reachable set — with the "unprofitable
shape" and "insufficient CPU share" explanations not yet fully separated.** The
kill verdict does not depend on which it is, but the causal story is incomplete
and I would rather say so than overstate it.

The experiment that would close this is small and does not require implementing
anything: capture the pages and masks that q30/q31/q41/q42 actually feed to the
selected path, and report survival, run morphology, bit width, RLE/bit-packed
share, and selected-vs-production timing per column.

---

## 6. What the digest oracle can and cannot tell you

Five queries produced **different output digests between two runs of the *same*
commit**: TPC-DS q31, and ClickBench q17, q31, q32, q39. Across four independent
feature-**off** ClickBench runs, q17/q31/q32 each produced **four distinct
digests in four runs**.

They were excluded from the byte-identity requirement, with a control run
identifying them, and reported as excluded rather than as passes. Of the 38
ClickBench queries demonstrably stable across independent feature-off runs, the
feature-on arm mismatched **none**.

**Important caveat on wording and on what this evidence supports.** These digests
are computed over raw output bytes. Instability therefore shows the *raw output
byte sequence* is unstable — **not** that the query is semantically
non-deterministic. Benign causes include partition completion order, batch
boundary/merge order, absent or tie-broken `ORDER BY`, and floating-point
reduction order under parallel aggregation.

So the correct claim is: **these queries produced control-unstable raw output
digests**, and this project's oracle is a raw-byte oracle with that limitation.
I am deliberately **not** filing these as DataFusion non-determinism bugs on this
evidence. Establishing that would need a query-aware canonicalisation — sort by
full row key when `ORDER BY` is absent or partial, multiset semantics preserving
duplicates, float comparison at the benchmark's tolerance, and normalisation of
NaN, dictionary representation and batch boundaries.

The practical takeaway that *does* hold for anyone building a similar gate: a
raw-digest equality gate over these suites will produce false kills unless it is
screened by a same-commit control run.

---

## 7. On not continuing to v1 (nullable support)

The obvious next move would be to lift v0's admission to nullable columns. Two
pieces of evidence argue against it, and one methodological correction argues
against the cheap version of the argument.

**Evidence from an earlier page-level census on this same frozen workload.**
Across captured accesses: **103,271** actual dictionary page accesses, of which
**19,163** involved optional (nullable) mapping — so the nullable dictionary
population genuinely exists and is not a microbenchmark fiction. But the
population that would actually be *selected* is far narrower (**1,916** accesses),
and crucially it is **dense**: `bitpacked_selected_values / bitpacked_values`
averages **86.07%**, and only **129 / 1,916 (6.73%)** of accesses have overall
physical density ≤ 25%. Dense selection is precisely the regime where selected
decoding has nothing to win (cf. the near-full-survival inversion in §2).

**These numbers are access-weighted and are not an Amdahl fraction.** They say
nothing directly about scan/decode CPU share, and the source report says so
itself. They bound *incidence*, not *benefit*.

**A correction to the cheap reopen test.** I previously suggested a
metadata-only eligibility census would settle whether v1 is worth it. Trap 2
above shows that is not sufficient: a column chunk's encoding set tells you an
encoding *appeared*, not which encoding the *selected* pages use, and metadata
carries no selection or run morphology at all. A metadata-only census gives a
**structural upper bound** and nothing more.

---

## 8. Reopen conditions

The closure stands unless one of these holds:

1. **A two-stage eligibility census** shows a materially large admissible share
   on a workload that matters:
   - *stage 1*: metadata-only structural upper bound (def/rep levels, types,
     encodings — no decoding needed);
   - *stage 2*: only if stage 1 is large enough — an actual-page,
     row/byte-weighted census plus a measurement of scan/decode CPU share
     (i.e. the `d` in §5.2).
   Neither stage requires implementing nullable support.
2. **An upstream reader architecture change** that threads selection natively
   through `ArrayReader`, which would make these kernels and — more importantly —
   the two traps in §3 directly relevant prior art.
3. **A concrete flat-required, dictionary-heavy workload** (telemetry/log shapes)
   where v0's admission is commonly satisfied, making this branch usable as-is
   behind its flag.

---

## Reusable artifacts

Ranked by what I think is actually worth someone's time:

1. **The mixed dictionary/`PLAIN` correctness case** (§3, Trap 2) — a regression
   test for reading across a mid-chunk encoding change under `RowSelection`,
   validating the full value sequence rather than row counts. Potentially a
   small, independent upstream PR. *Before proposing it I would need to confirm
   whether upstream `main` already covers this, and I would not claim it fixes
   any existing corruption unless it reproduces on unmodified upstream.*
2. **Captured workload selection traces** from real ClickBench/TPC-DS runs, with
   provenance — potentially useful to the existing captured-trace benchmark work,
   independent of anything in this investigation.
3. **The coverage-counter methodology** — proving an experimental path is
   actually executed, with the counter itself tested.
4. **The exact-output correctness oracle** and its documented limitation (§6).

**Not proposed for upstream:** the nullable mapper (never built), the
selected-decode public API, the DataFusion session option, the production
coverage metrics, and the five-layer wiring itself.

---

## Methodology notes

Four things I would do again, recorded because each caught a real error:

- **Freeze the gates before reading any number.** Every kill/verdict threshold
  here was written down first. The final stop was a rule firing, not a judgement
  call.
- **Compare against what production actually executes**, not a hand-rolled
  imitation of it. Correcting this changed the leaf result materially — and in
  the candidate's favour.
- **Verify that a passing test can fail.** Two differential fuzz tests passed and
  were worthless: with long selector spans, the default `RowSelectionPolicy::Auto`
  resolves to `Selectors`, so `read_mask_batch` never executed and the tests
  measured nothing. Caught only by deliberately breaking the code and observing
  the tests stay green. Everything gate-bearing here now forces
  `RowSelectionPolicy::Mask` and was sabotage-checked.
- **Test the instrument, not just the subject.** A dead coverage counter reads
  exactly like the kill condition it exists to detect.

---

## Provenance and reproduction

See [PROVENANCE.md](./PROVENANCE.md) for exact commit SHAs on both repositories,
the measurement matrix, per-query data as CSV, and the commands used.
