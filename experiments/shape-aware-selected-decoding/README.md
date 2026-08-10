# Shape-aware selected decoding for Parquet dictionary streams — a completed investigation with a negative result

**Status: closed. Not proposed as an upstream feature.**
Branch: `exp/v21-rle-selected-fill-20260807` (arrow-rs) — this file lives on that branch.
Date: 2026-08-09, **amended 2026-08-10**

> **Amendment (2026-08-10).** A follow-up asked a question this report could not
> answer — *was TPC-H ever measured?* It was not, and chasing that turned up two
> architectural facts that materially narrow the original reachability claim.
> **The v0 wiring reaches far less of the available opportunity than the first
> version of this report implied, and the reason is implementation scope rather
> than workload shape.** §4.2 and §4.3 are new; §4.1, the TL;DR, §5.3, §7 and §8
> are corrected. The closure still stands, but on a different and narrower
> argument — see §4.4.

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
- **Very little of the evaluated workload reaches it**: **0 of 99** TPC-DS
  queries and **4 of 42** ClickBench queries. But *why* is not what the first
  version of this report said — see the next two bullets, which are the 2026-08-10
  amendment.
- **Two suppressors are architectural, not workload properties.** (1) The
  predicate cache's `CachedArrayReader` does not implement the admission
  predicate, so **any column that is both a filter column and an output column
  disqualifies the entire scan**. (2) The filter chain in
  `arrow_reader/read_plan.rs` pins `selected_decode = false`, which excludes the
  place where the original paper's own TPC-H Q6 result comes from. On Q6 the
  shipped wiring addresses **under 2%** of the maskable rows (§4.3).
- **TPC-H was never measured** although the program's own scope named it. It is
  the most favourable suite available (every column `REQUIRED`), and it reaches
  the path on **10 of 22** queries — **14 of 22** with the predicate cache
  disabled (§4.2).
- **No query-level benefit was established.** This survives the amendment intact
  and is now the load-bearing argument: the two fully-covered queries
  (ClickBench q30/q31, **100%** coverage, 65M rows each) showed **no reproducible
  direction** across two rounds.

I am **not** proposing the wiring as a production feature. This is a negative
result about *what this v0 wiring achieved on the evaluated workloads* — not
about the decoder kernel, and (after the amendment) **not** a general claim that
selection pushdown is unreachable in arrow-rs.

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

## 4. Reachability

### 4.1 What the counter measured

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

Because the counters live inside `read_mask_batch`, they distinguish two things
that both look like zero: *never arriving at the decision point* versus
*arriving and being rejected*. **84 of 99 TPC-DS queries reached the decision
point and were rejected**; the zero is a real rejection, not a dead path. The
end-to-end timing is correspondingly an exact null (geomean 0.9954 across 99
queries) — a useful validity check: zero coverage predicted zero effect, and
zero effect is what appeared.

**Correction (2026-08-10).** The first version of this report explained the
TPC-DS zero as *"TPC-DS's schema is nullable-heavy."* That was an assertion, not
a measurement, and it sits badly beside this report's own §7 census, in which
only **19,163 of 103,271** dictionary page accesses (**18.6%**) involved optional
mapping. The measured discriminator is different. Four ClickBench queries form a
controlled comparison — same table, same predicate `WHERE "SearchPhrase" <> ''`,
same 65,491,130 rows, differing **only** in the output projection:

| query | projection | coverage |
|---|---|---|
| q30 | SearchEngineID, ClientIP, IsRefresh, ResolutionWidth | **100%** |
| q31 | WatchID, ClientIP, IsRefresh, ResolutionWidth | **100%** |
| q12 | **SearchPhrase** | 0% |
| q13 | **SearchPhrase**, UserID | 0% |
| q14 | SearchEngineID, **SearchPhrase** | 0% |

What decides admission here is a `BYTE_ARRAY` column in the projection, under an
**all-or-nothing** rule — not nullability. And all-or-nothing amplifies a rare
property into a near-universal one: if an ineligible attribute occurs
independently on 18.6% of columns, a scan projecting *k* of them admits with
probability 0.814^k — about 19% at k=8, 5% at k=15.

### 4.2 TPC-H — the suite that was never measured

The program's own scope (ticket 18, statement b) named "real TPC-DS/**TPC-H**
selection masks". TPC-H then dropped out without adjudication: it was never
instrumented, never appears in the coverage data, and the page census ran on
`tpcds-sf10-v1` only. It is also the *most favourable* suite available — in
`tpchgen-cli` output **every leaf column of every table is `REQUIRED`**, so the
nullability clause cannot reject anything.

Measured after the fact (TPC-H SF1, local, three arms, **coverage only — no
timing claim is made from this run**; coverage is a pure function of schema and
projection, so it is scale- and hardware-independent):

| arm | queries reaching the path | selected rows |
|---|---|---|
| flag off (control) | 0 / 22 | 0 |
| flag on, as shipped | **10 / 22** | 1,053,250 |
| flag on, predicate cache disabled | **14 / 22** | 4,074,629 (**3.9x**) |

So the headline "0/99 and 4/42" was never a general statement about
reachability. On the suite the program had itself scoped and then dropped, the
same rule admits 45% of queries.

### 4.3 Two architectural suppressors, and why Q6 is the sharpest case

Chasing why TPC-H **q6** reported 0% — every column `REQUIRED`, every projected
column `INT32`/`INT64` — produced the two findings that most narrow this
report's original claim. An isolation probe
(`parquet/tests/tpch_selected_decode_probe.rs`) confirms arrow-rs admits q6's
exact projection at **100%**, including `DECIMAL`-on-`INT64` and `DATE`-on-`INT32`,
and correctly rejects `BYTE_ARRAY`. So the rejection came from above the leaves.

**Suppressor 1 — the predicate cache.** `CachedArrayReader`
(`array_reader/cached_array_reader.rs`) does not implement
`supports_selected_decode()`, so it inherits the trait default `false`. When a
column is *both* a filter column and an output column, arrow-rs wraps its reader
to avoid decoding it twice — and under all-or-nothing that one wrapper
disqualifies the whole scan. Confirmed by disabling the cache: q6 goes from
**0 → 114,160** selected rows, and the suite from 10/22 to 14/22. This also
explains precisely *which* ClickBench queries succeeded: q30/q31 filter on
`SearchPhrase`, which is **not** in their output projection, so nothing is
cached and nothing is disqualified.

**Suppressor 2 — the filter chain is excluded entirely.**
`ReadPlanBuilder::with_predicate_options` builds its reader with
`ParquetRecordBatchReader::new`, which pins `selected_decode = false`. That
reader is not incidental: predicate *N* reads under the selection accumulated
from predicates *1..N-1*, so **the filter chain is exactly where selection
pushdown does its work**. Tracing every `read_mask_batch` call for q6:

| call class | calls | rows | flag |
|---|---|---|---|
| filter chain, eligible columns | **711** | ~6,000,000 | **hardcoded false** |
| output phase | 106 | 114,160 | true (reader declined — suppressor 1) |
| filter chain, ineligible readers | 371 | — | false |

The shipped wiring therefore addresses **under 2%** of the maskable rows in this
query. That matters more than it first appears, because **q6 is the original
paper's own headline end-to-end result**: SIGMOD'23 §7.3 reports **3.1x** for
preloaded non-null TPC-H SF10 Q6 (13.7x nullable, 21.1x for a modified repeated
projection), and §7.4 reports 1.1x–5.5x across ten Spark TPC-H queries. The
paper runs **no TPC-DS experiment at all**. This program evaluated end-to-end on
TPC-DS and ClickBench — neither of which the paper used — and left the paper's
own workload unmeasured.

One corroboration that the two are looking at the same thing: the paper states
Q6's three filters leave ~**1.9%** of rows; the counter here measured 114,160 of
~6.0M = **1.90%**.

### 4.4 What the closure now rests on

Not "almost nothing is eligible" — that claim is now known to be an artifact of
implementation scope as much as of workload shape. It rests instead on the one
result the amendment does not touch:

**ClickBench q30 and q31 ran at 100% coverage over 65M rows each and produced no
reproducible direction across two rounds** (§5.1). Raising coverage — by fixing
either suppressor, by adding nullable support, or by choosing TPC-H — adds
*opportunity*, not *benefit*. The fully-covered case was already measured, and
nothing happened there.

That also puts a weak empirical bound on the unmeasured `d` of §5.2: at 100%
coverage with a leaf speedup of 2.6x–4.6x, `Δ ≈ 0` implies `d · (1 − 1/S) ≈ 0`.
The noise band is far too wide (0.673–1.372) to make this rigorous, but the
direction is that reader-level time is not going where this optimisation acts.

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

**Amendment (2026-08-10).** §4.3 adds a third candidate explanation that this
report originally could not see: **(c)** the reachable set is not merely small
but *systematically biased* by the two suppressors. Admission survives only when
a scan's filter columns and output columns are disjoint (otherwise the predicate
cache disqualifies it), which selects for exactly the queries where the
filter-chain opportunity — the larger one — is absent. q30/q31 are that shape.
So the queries that reached the path are close to a worst case for observing
benefit, and (a)/(b)/(c) remain unseparated.

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

> **Amendment (2026-08-10).** This section was written on the premise that
> nullable support was the natural next lever. §4.3 shows it is not even the
> largest one: two suppressors that have nothing to do with nullability — the
> predicate-cache wrapper and the excluded filter chain — gate more opportunity,
> and both are cheaper to address. The evidence below still stands on its own
> terms (it bounds the *nullable* population), but it should no longer be read
> as "there is nothing left to reach."

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

**Amended 2026-08-10 — the condition that is now closest to met.** §4.3 supplies
a fourth, and it is more specific than the three above:

4. **The filter chain is wired and re-measured.** Selection pushdown through
   predicates *1..N-1* is the mechanism behind the original paper's TPC-H Q6
   result, and it is currently excluded by one hardcoded `false`. Wiring it is a
   plumbing change — thread `selected_decode` through `ReadPlanBuilder` — not new
   kernel work, and it would move q6 from under 2% of maskable rows to most of
   them. Two caveats keep this from being an automatic reopen:
   - The correctness gates (§3) must be re-run in full. Both traps were found in
     the *output* phase; the filter chain has not been differentially tested.
   - This program's own earlier finding is that modern arrow-rs
     (`DecodeAllIndicesCompact`) already internalised most of the paper's
     headroom — a 2.82x geomean against the paper's own baseline shape. The
     paper's 3.1x is measured against a 2023 C++ baseline, so the residual
     against today's arrow-rs is expected to be substantially smaller, and
     possibly inside noise. **Do not treat 3.1x as a target.**

   A reopen on this basis should be pre-registered as a fresh gate, not framed as
   a rescue of the closed one.

---

## Reusable artifacts

Ranked by what I think is actually worth someone's time:

1. **The two admission suppressors** (§4.3) — the predicate-cache wrapper not
   implementing the admission predicate, and the filter chain being excluded by a
   hardcoded `false`. Anyone implementing selection pushdown on this reader will
   hit both, and neither is visible from coverage numbers alone: both look
   identical to "the workload is not eligible."
2. **The mixed dictionary/`PLAIN` correctness case** (§3, Trap 2) — a regression
   test for reading across a mid-chunk encoding change under `RowSelection`,
   validating the full value sequence rather than row counts. Potentially a
   small, independent upstream PR. *Before proposing it I would need to confirm
   whether upstream `main` already covers this, and I would not claim it fixes
   any existing corruption unless it reproduces on unmodified upstream.*
3. **Captured workload selection traces** from real ClickBench/TPC-DS runs, with
   provenance — potentially useful to the existing captured-trace benchmark work,
   independent of anything in this investigation.
4. **The coverage-counter methodology** — proving an experimental path is
   actually executed, with the counter itself tested. With the §4.3 caveat: a
   counter proves *whether* a path ran, never *why* it did not. Attributing a
   zero needs a separate instrument, and this report initially attributed one
   wrongly for want of it.
5. **The exact-output correctness oracle** and its documented limitation (§6).
6. **The type-isolation probe** (`parquet/tests/tpch_selected_decode_probe.rs`) —
   answers "which projected shapes does the admission rule actually accept" in
   two seconds against a real file, independent of any query engine.

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
- **A fifth, learned the hard way on 2026-08-10: never explain a zero you did not
  instrument.** This report shipped with "TPC-DS's schema is nullable-heavy" as
  the reason for 0/99. It was a plausible story written to fill a gap, it
  contradicted this report's own census, and it was wrong about the dominant
  mechanism. The counter could say *whether* the path ran; nothing in the
  apparatus could say *why not*. A measurement programme that gates on a number
  should treat the explanation of that number as needing its own evidence — a
  wrong attribution survives peer reading far more easily than a wrong
  measurement, because it is the part nobody can check against the data files.

---

## Provenance and reproduction

See [PROVENANCE.md](./PROVENANCE.md) for exact commit SHAs on both repositories,
the measurement matrix, per-query data as CSV, and the commands used.
