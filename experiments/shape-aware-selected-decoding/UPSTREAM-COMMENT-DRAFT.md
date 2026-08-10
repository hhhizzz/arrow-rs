# Draft comments for upstream

Ready to send. Links below point at the published report on this branch.

---

## A. Primary — `apache/arrow-rs#7456`

Framed as a completed investigation with a negative result, not a feature
proposal.

> I completed an investigation into shape-aware selected decoding for Parquet
> dictionary streams, and I think the negative result plus two correctness traps
> are worth writing up here.
>
> The experiment added an opt-in reader path that decodes only the selected
> dictionary indices for flat, required, primitive leaves, wired through both the
> synchronous reader and the async/push-decoder path, and evaluated end to end on
> ClickBench and TPC-DS SF10.
>
> **The kernel is fast.** On captured low-survival pages it is 2.6x–4.6x faster
> than the production-shaped baseline — stock `get_batch_with_dict` full decode
> followed by `arrow::compute::filter`. (For much of this investigation I had been
> comparing against hand-written baselines instead; correcting that changed the
> result materially, in the candidate's favour.)
>
> **Two integration traps showed up, and I think these are the most transferable
> part.**
>
> 1. Admission must be decided for the complete reader subtree *before* any child
>    produces output. Discovering a mixed projection after one child has already
>    written compact (already-filtered) data cannot be unwound.
> 2. A column chunk can change encoding mid-chunk — writers abandon dictionary
>    encoding once the dictionary outgrows its page budget, so `RLE_DICTIONARY`
>    pages can be followed by `PLAIN` ones in the same chunk. If an unsupported
>    page is treated as end-of-chunk, the reader can advance to the next row group
>    and return **the correct row count with values sourced from the wrong place**.
>    Only a full-value content check catches this; row-count assertions and
>    tolerance-based aggregate comparisons do not.
>
> After fixing both, every query with a control-stable output digest matched
> between feature-off and feature-on across both suites.
>
> **Reachability was the limiting factor.** Under the conservative v0 admission
> rule (all projected columns flat, required, primitive; Mask execution path),
> a coverage counter showed **0 of 99** TPC-DS queries and **4 of 42** ClickBench
> queries entering the path — 12.4% of decoded ClickBench rows. The four covered
> queries are 3.0–3.6% of total suite runtime.
>
> **No repeatable query-level direction was established.** Two timing rounds
> disagreed (one query swung 19% on identical binaries), so I am claiming neither
> benefit nor regression. Independently of the noise, the Amdahl ceiling is low:
> with a coverage-weighted share of 2.8–3.3%, even assuming the entire covered
> portion were dictionary decode, a 2x–4x kernel speedup bounds total suite
> improvement at roughly 1.4%–2.5%. The real bound is lower, since dictionary
> decode is only part of that time — I did not measure that fraction separately.
>
> One gap I want to be explicit about: the fixtures that produced 2.6x–4.6x were
> captured from different queries than the four that actually reach the path, so
> I have not yet separated "the reachable shapes are unprofitable" from "the
> reachable shapes are profitable but too small a share of runtime". The stop
> decision does not depend on which it is, but the causal story is incomplete.
>
> I am therefore **not** proposing the selected-decoding wiring as a production
> feature. This is a negative result about applicability under the current reader
> architecture on the evaluated workloads, not about the decoder kernel.
>
> Possibly reusable, independent of the above:
> - a regression case for reading across a mid-chunk dictionary→PLAIN change
>   under `RowSelection`, validating the full value sequence;
> - captured real workload selection traces with provenance;
> - the coverage-counter methodology for proving an experimental path is actually
>   executed (the counter is itself unit-tested, since a dead counter looks
>   exactly like "never reached").
>
> Full report, data and commit pins: https://github.com/hhhizzz/arrow-rs/blob/exp/v21-rle-selected-fill-20260807/experiments/shape-aware-selected-decoding/README.md
>
> Would either the mixed-encoding correctness test or one or two captured
> workload traces be useful as small, independent follow-up PRs? I would want to
> check first whether upstream already covers the mixed-encoding case — I am not
> claiming it reproduces on unmodified `main`.

---

## B. Secondary — `apache/arrow-rs#8846` (adaptive Mask/Selectors execution)

Short cross-link only.

> Related data point from a completed selected-decoding investigation ([report](https://github.com/hhhizzz/arrow-rs/blob/exp/v21-rle-selected-fill-20260807/experiments/shape-aware-selected-decoding/README.md)):
> when I pushed a row selection down into dictionary decoding, selector
> morphology alone did not predict query-level benefit. Two further things had to
> hold — the optimization had to be *reachable* under a correctness-preserving
> admission rule (0/99 TPC-DS and 4/42 ClickBench queries qualified), and the
> affected work had to be a meaningful share of query CPU (3.0–3.6% of suite
> runtime for the covered queries). This seems consistent with the multi-level
> adaptivity direction in this issue: shape is a necessary input to the decision
> but not a sufficient one.

---

## C. Limited relation — `apache/arrow-rs#10136` (BMI2/PEXT bit filtering)

Deliberately narrow. This investigation does **not** refute that work; the
subject is different (general bit filtering vs Parquet dictionary selected
decoding). Share only the methodology points, if anything:

> Two methodology notes from a related investigation ([report](https://github.com/hhhizzz/arrow-rs/blob/exp/v21-rle-selected-fill-20260807/experiments/shape-aware-selected-decoding/README.md)), offered only as
> process input since the subject differs:
> - the comparator should be the production implementation, not a hand-written
>   equivalent — correcting this materially changed our leaf-level result;
> - a leaf speedup is worth reporting alongside its end-to-end applicability;
>   ours was 2.6x–4.6x at the leaf and reached 0% of one benchmark suite.

---

## Blog post (separate track)

Working titles:

- *When a 4.6× Parquet Decode Kernel Does Not Speed Up the Query*
- *From Leaf Benchmarks to Reader Reachability: A Negative Result in Parquet Selected Decoding*

The contribution is not "this failed" but:

- how much of the original paper's headroom modern arrow-rs baselines have
  already absorbed;
- how to prove an experimental path is actually executed (and why a dead counter
  is indistinguishable from a dead path);
- why mixed encoding within one column chunk creates a silent wrong-row risk;
- why row coverage is not the Amdahl fraction;
- why a negative result still constrains reader architecture decisions.
