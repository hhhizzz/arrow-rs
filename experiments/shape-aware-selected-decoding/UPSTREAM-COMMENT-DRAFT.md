# Draft comments for upstream

Rewritten 2026-08-10 after the v27 amendment. The earlier drafts led with
"0/99 TPC-DS and 4/42 ClickBench queries were eligible" as a workload finding.
That framing is now known to be substantially wrong — most of the shortfall was
implementation scope, not workload shape — and these replace it.

**Accuracy note that governs all of the text below.** `supports_selected_decode()`
is an addition on my branch, not an upstream API. So neither finding is an
upstream bug report. They are both of the form *"if you add an admission
predicate to `ArrayReader`, here is where it silently loses coverage"* — which is
useful precisely because both failures are invisible: they look identical to
"the workload is not eligible."

---

## A. Reply to `alamb` on `apache/arrow-rs#10141`

Context: alamb linked https://dl.acm.org/doi/10.1145/3589323 and wrote *"We
probably already do it for unnested (leaf) nodes -- but need a bit more
cleverness for structured nodes."* Send as a reply in that thread.

> This comment prompted me to go and try it — I implemented the paper's approach
> on top of this `Mask` infrastructure and ran it end to end. The structured-node
> problem you point at is real, but it turned out not to be the binding one, so
> let me give you the three things that cost me the most time.
>
> **1. Subtree admission has to be decided before any child produces output.**
> This is the structured-node bit. My first cut checked children in order and
> only discovered a mixed projection after an earlier child had already written
> compact (already-filtered) data — at which point you hold one filtered column
> and one unfiltered column destined for the same `RecordBatch`, and it cannot be
> unwound. It aborted a real ClickBench query (a `BYTE_ARRAY` column projected
> alongside an eligible integer one). The fix was a side-effect-free predicate
> answered for the whole subtree before the first byte is decoded, plus a
> contract that a leaf may never decline mid-stream for anything it could have
> known statically.
>
> **2. The predicate cache silently disqualifies whole scans.**
> `CachedArrayReader` wraps a column that is read both by a filter and by the
> output projection. It forwards `read_records`, but an admission predicate added
> to `ArrayReader` is not something it forwards — so it inherits the trait
> default, and under an all-or-nothing subtree rule that one wrapper takes out
> the entire scan. This is the single largest coverage loss I measured, and it is
> completely invisible: the counter just reads zero.
>
> It also predicts exactly which queries survive. On ClickBench my only
> 100%-covered queries were q30/q31, whose filter column (`SearchPhrase`) is not
> in their output projection — nothing gets cached, so nothing is disqualified.
> On TPC-H, disabling the predicate cache moved coverage from 10/22 to 14/22
> queries and 3.9x the rows.
>
> **3. The filter chain is where the paper's own result lives, and it is the
> easiest place to leave it out.** `ReadPlanBuilder::with_predicate_options`
> builds its reader with `ParquetRecordBatchReader::new`. Predicate *N* there
> reads under the selection accumulated from predicates *1..N-1* — which is
> precisely the selection pushdown the paper is about. I threaded my flag through
> the output path and not through that constructor, and the result is that on
> **TPC-H Q6 — the paper's own headline query** — my wiring touched **under 2%**
> of the maskable rows: 711 filter-chain calls over ~6.0M rows with the option
> pinned off, against 106 output-phase calls over 114K rows. If someone builds
> this, that constructor is the one to thread first, not last.
>
> **One correctness trap worth knowing regardless of any of the above.** A column
> chunk can change encoding mid-chunk — writers abandon dictionary encoding once
> the dictionary outgrows its page budget, so `RLE_DICTIONARY` pages can be
> followed by `PLAIN` ones in the same chunk. My decoder declined at the first
> `PLAIN` page and returned a short read; the layer above read that as "chunk
> exhausted" and advanced to the next row group. **The row count still came out
> exactly right, with values sourced from the wrong place.** Only a full-value
> content check caught it — row-count assertions and tolerance-based aggregate
> comparisons all passed.
>
> **On whether it pays off, I have to be honest and unhelpful:** I did not
> establish a query-level benefit, and I stopped at a pre-registered gate. The
> leaf kernel is genuinely fast — 2.6x–4.6x against the production-shaped
> baseline (stock `get_batch_with_dict` + `arrow::compute::filter`) on captured
> low-survival pages. But my two fully-covered queries ran at 100% coverage over
> 65M rows each and showed no reproducible direction across two rounds. I would
> also temper expectations against the paper's numbers specifically: its 3.1x for
> TPC-H Q6 is measured against a 2023 C++ baseline, and a separate arm of this
> investigation found that modern arrow-rs's existing full-decode path has already
> internalised most of that headroom (~2.8x geomean against the paper's own
> baseline shape). The residual against today's `main` should be expected to be
> much smaller.
>
> Full write-up with the data, commit pins and the caveats:
> https://github.com/hhhizzz/arrow-rs/blob/exp/v21-rle-selected-fill-20260807/experiments/shape-aware-selected-decoding/README.md

---

## B. Primary write-up — `apache/arrow-rs#7456`

Longer form, for the issue that tracks this area.

> I ran a full investigation into shape-aware selected decoding for Parquet
> dictionary streams — an opt-in reader path that decodes only the selected
> dictionary indices for flat, required, primitive leaves, wired through the
> sync and async/push-decoder paths and evaluated end to end. The result is
> negative, but the parts I think are worth your time are two architectural
> findings and two correctness traps, not the verdict.
>
> **The kernel is fast.** 2.6x–4.6x against the production-shaped comparator —
> stock `get_batch_with_dict` full decode followed by `arrow::compute::filter` —
> on captured low-survival pages. (For most of this investigation I had been
> comparing against hand-written baselines; correcting that changed the result
> materially, in the candidate's favour.)
>
> **Two places an admission predicate silently loses coverage.** Both look
> identical to "the workload is not eligible", which is how I initially
> misdiagnosed my own result.
>
> 1. **The predicate cache.** `CachedArrayReader` wraps any column read by both a
>    filter and the output projection. It does not forward an admission predicate
>    added to `ArrayReader`, so it inherits the default, and under an
>    all-or-nothing subtree rule it disqualifies the entire scan. Disabling the
>    cache took TPC-H coverage from 10/22 to 14/22 queries and 3.9x the rows.
> 2. **The filter chain.** `ReadPlanBuilder::with_predicate_options` builds its
>    reader via `ParquetRecordBatchReader::new`; predicate *N* reads under the
>    selection accumulated from predicates *1..N-1*, so that is where selection
>    pushdown actually does its work. Threading the option through the output
>    path only left **under 2%** of the maskable rows addressed on TPC-H Q6.
>
> **Two correctness traps.**
>
> 1. Admission must be decided for the complete reader subtree *before* any child
>    produces output. Discovering a mixed projection after one child has written
>    compact data cannot be unwound.
> 2. A column chunk can change encoding mid-chunk (`RLE_DICTIONARY` pages then
>    `PLAIN`, once the dictionary outgrows its page budget). Treating an
>    unsupported page as end-of-chunk let the reader advance to the next row group
>    and return **the correct row count with values sourced from the wrong
>    place**. Only a full-value content check catches this.
>
> After fixing both, every query with a control-stable output digest matched
> between feature-off and feature-on across both suites.
>
> **What I did not establish.** Any query-level benefit. The two queries that ran
> at 100% coverage (65M rows each) showed no reproducible direction across two
> rounds — one swung 19% on identical binaries. That is the result the rest of
> the amendment does not touch, and it is why I am **not** proposing the wiring
> as a production feature.
>
> **What I got wrong, since it may matter to anyone reading the earlier version
> of my report.** I first reported this as "almost nothing is eligible: 0/99
> TPC-DS and 4/42 ClickBench queries", and attributed it to TPC-DS's schema being
> nullable-heavy. That attribution was an assertion I never instrumented, and it
> was wrong about the dominant mechanism — it is the two suppressors above plus a
> `BYTE_ARRAY` column in the projection under all-or-nothing, not nullability. I
> had also never measured TPC-H, which is both the paper's own workload and the
> most favourable one available (every column `REQUIRED`); it reaches the path on
> 10/22 queries. A counter can prove *whether* a path ran, never *why* it did
> not.
>
> Possibly reusable as small independent PRs, if any of it is wanted:
> - a regression case for reading across a mid-chunk dictionary→`PLAIN` change
>   under `RowSelection`, validating the full value sequence;
> - captured real-workload selection traces with provenance;
> - the coverage-counter methodology, with the counter itself unit-tested.
>
> Report, data and commit pins: https://github.com/hhhizzz/arrow-rs/blob/exp/v21-rle-selected-fill-20260807/experiments/shape-aware-selected-decoding/README.md
>
> I would want to check whether upstream already covers the mixed-encoding case
> before proposing anything — I am not claiming it reproduces on unmodified
> `main`.

---

## C. Secondary cross-link — `apache/arrow-rs#8846` (adaptive Mask/Selectors)

> Related data point from a completed selected-decoding investigation
> ([report](https://github.com/hhhizzz/arrow-rs/blob/exp/v21-rle-selected-fill-20260807/experiments/shape-aware-selected-decoding/README.md)):
> when I pushed a row selection down into dictionary decoding, selector
> morphology alone did not predict where the optimisation could even *run*. Two
> structural facts dominated instead — whether a column was wrapped by the
> predicate cache (any column read by both a filter and the projection), and
> whether the option reached the filter-chain reader at all. Shape mattered, but
> it was third. Offered because it suggests an adaptive policy needs to consider
> what the reader tree looks like, not only what the selection looks like.

---

## D. Limited relation — `apache/arrow-rs#10136` (BMI2/PEXT bit filtering)

Deliberately narrow; different subject (general bit filtering vs Parquet
dictionary selected decoding). Methodology only.

> Two methodology notes from a related investigation
> ([report](https://github.com/hhhizzz/arrow-rs/blob/exp/v21-rle-selected-fill-20260807/experiments/shape-aware-selected-decoding/README.md)),
> offered only as process input since the subject differs:
> - the comparator should be the production implementation, not a hand-written
>   equivalent — correcting this materially changed our leaf-level result;
> - report a leaf speedup alongside evidence that the path is actually reached.
>   Ours was 2.6x–4.6x at the leaf, and a coverage counter later showed most of
>   the intended opportunity was never entered for reasons that had nothing to do
>   with the kernel.

---

## Blog post (separate track)

Working titles:

- *A 4.6× Parquet Decode Kernel That Reached 2% of Its Own Benchmark*
- *Explaining a Zero You Did Not Instrument*

The contribution is not "this failed" but:

- why a coverage counter cannot tell you why a path was not taken, and how that
  produced a confidently-worded wrong attribution in the first version;
- how the predicate cache and the filter-chain constructor each silently remove
  coverage from a reader-level optimisation;
- why mixed encoding within one column chunk creates a silent wrong-row risk;
- how much of the original paper's headroom modern arrow-rs baselines have
  already absorbed;
- why row coverage is not the Amdahl fraction.
