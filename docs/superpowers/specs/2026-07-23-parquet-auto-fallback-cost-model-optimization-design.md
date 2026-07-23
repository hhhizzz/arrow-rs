# Parquet Auto-Fallback Cost Model Optimization Design

## Objective

Improve the async Parquet `RowSelectionPolicy::Auto` benchmark without giving up the two proven wins of the current PR:

- fragmented selections amortized across several row groups
- very dense selections that benefit from mask materialization

The work will be developed on the isolated branch `codex/exp/auto-fallback-cost-model-staged-20260723`, based on PR commit `7c6be704c886894ccfe911c0e6e9a2339c4da40d`. It will not be pushed into `codex/parquet-reader-auto-fallback-pr` until the measured result is reviewed separately.

## Baseline Evidence

The latest page-index-free benchmark baseline is `eaec65a202e27759a69c70e194ea27426f55d327`. Against it, PR `7c6be704c8` measured:

- fragmented 50%, four row groups: `-38.84%`
- fragmented 50%, eight row groups: `-47.06%`
- dense 98.44%: `-49.03%`
- fragmented 50%, one row group: `+16.01%`
- all selected: `+7.48%`
- sparse two groups followed by fragmented two groups: `+7.47%`

Disabling the post-filter cost model recovered about 7% on the one-row-group case and 5.5% on all-selected, but removed the fragmented multi-row-group wins. It did not materially change sparse-first ordering. A separate 3-8% pushdown-path overhead remained in most non-dense cases.

## Constraints

- Preserve output row count and payload ordering for every benchmark case.
- Preserve explicit `Mask` and `Selectors` behavior; changes apply only to `Auto` cost-model execution.
- Preserve limit, offset, virtual-column, nested-projection, and base-selection safety gates.
- Do not use page indexes in this benchmark fixture.
- Do not change the benchmark cases while comparing optimization stages.
- Keep each stage independently revertible and benchmarkable.
- Do not push experimental commits to the PR branch during this work.

## Chosen Approach

Use a conservative, staged state-machine refinement. Each stage addresses one measured cause and is validated before the next stage is retained.

### Stage 1: Switch Only Future Row Groups

The observing row group has already evaluated predicates and built its `RowSelection`. If that observation prefers post-filter, the current implementation redirects the same row group through `StartPostSelection`, decoding dense output and applying the selection after decode.

Change the transition so the observed row group completes its existing pushdown output plan. Install reusable post-filter state and set `UsePostFilter`, but apply it only when planning a later row group.

Expected effects:

- remove the unamortized transition penalty from the one-row-group fragmented case
- reduce the first-row-group tax in fragmented multi-row-group cases
- preserve later-row-group post-filter gains
- avoid evaluating predicates twice

The obsolete `StartPostSelection` transition/state/helper should be removed if no other path uses it after the change.

### Stage 2: Keep Non-Triggering Observations Re-Evaluable

Today, one non-triggering observation sets the terminal `UsePushdown` state. This makes row-group order determine the entire scan policy.

When an eligible observation says `PushdownStillPreferred`, retain the accumulated `Observing` state instead of making pushdown terminal. Re-evaluate after each later eligible pushdown row group. Once an observation chooses `UsePostFilter`, keep that choice terminal for the remainder of the reader. A support denial may remain terminal because its safety condition is reader-scoped.

This is accumulated observation, not a sliding window. For `sparse2_then_fragmented2`, the third group should make the aggregate fragmented enough to enable post-filter for the fourth group. It avoids oscillation and keeps the implementation small.

Metrics must continue to report each observed row group and each decision reason. `adaptive_kept_pushdown` may increment more than once while observing; tests will document that behavior.

### Stage 3: Gate High-Selectivity No-Pruning

`HighSelectivityNoPruning` currently selects post-filter without considering whether any future row group exists or how much deferred output must be decoded together with the predicate.

For this reason only, require both:

1. at least one row group remains after the observed group
2. deferred fixed-width output is cheap, using the existing 24 uncompressed bytes-per-row threshold; variable-width deferred output is not cheap

Fragmentation-triggered post-filter decisions are not subject to this gate because the benchmark proves that avoiding fragmented selection execution can dominate wider decode cost.

If the high-selectivity gate rejects post-filter, continue observing rather than permanently disabling it. A later shape may still justify post-filter for fragmentation.

### Stage 4: Reuse RowSelection Shape Work

The observation path and final read-plan path can classify the same `RowSelection` separately. The richer classifier counts selected/skipped rows, selector count, and both run counts, which is measurable for run-1 fragmentation.

Compute `RowSelectionShape` once for a completed predicate selection and carry it with row-group planning state until that exact selection is consumed by output planning. Final Auto strategy resolution accepts the precomputed shape while still applying projection, loaded-page, and expensive-output safety rules independently.

The cached shape must be invalidated whenever the underlying selection changes, including predicate intersection, limit/offset application, or base-selection composition. If a safe same-selection seam cannot be established without broad API changes, this stage will be dropped rather than introducing stale-shape risk.

## Alternatives Considered

### Minimal Guard-Only Change

Only remove current-row-group post-selection and gate all-selected. This is low risk but leaves sparse-prefix order sensitivity and repeated shape work unresolved.

### Fully Rolling or Bidirectional Policy

Use a sliding observation window and allow transitions in both directions. This adapts to arbitrary drift but introduces hysteresis, filter-state restoration, and metric semantics that are disproportionate to the current evidence.

### Static Post-Filter Admission

Predict fragmented scans from schema or metadata before evaluating predicates. The actual selection shape is not available statically, so this would trade measured adaptation for fragile heuristics.

## Testing

Each stage adds or updates focused unit tests before implementation behavior is considered complete.

Required state-machine tests:

- the observed row group remains pushdown and only the next group uses post-filter
- a one-row-group fragmented scan never executes post-filter
- sparse observations remain re-evaluable
- sparse-to-fragmented eventually switches; fragmented-to-sparse does not oscillate back
- all-selected with wide deferred output stays pushdown
- all-selected with cheap/no deferred output may switch when a later row group exists
- base selection, limit/offset, virtual columns, and unsupported projections retain existing behavior
- metrics counts match the revised transition semantics

Required checks after each retained stage:

- focused Parquet push-decoder and read-plan tests
- `cargo check -p parquet --bench arrow_reader_row_selection_policy --features arrow,async`
- `cargo fmt --all -- --check`
- relevant clippy/check if the touched seam requires it

The final combined candidate receives the broader relevant Parquet test suite and code review.

## Benchmark Protocol

Each stage is compared against the immediately preceding retained stage using the unchanged benchmark:

- host `sz-data-b-1`
- CPU affinity `0-23`
- benchmark `arrow_reader_row_selection_policy`
- filter `arrow_reader_row_selection_policy/auto`
- three alternating rounds
- one-second warm-up, two-second measurement, 20 samples
- 12 results per log, with Criterion estimates required
- zero sample-time warnings required for a clean comparison

The final retained candidate is also compared directly against:

- PR `7c6be704c8`, to measure optimization gain
- baseline `eaec65a202`, to show the final user-visible balance

Primary acceptance criteria:

- one-row-group fragmented regression materially reduced from `+16.01%`
- all-selected regression materially reduced from `+7.48%`
- sparse-first ordering regression materially reduced from `+7.47%`
- four/eight-row-group fragmented wins remain substantial
- dense 98.44% remains close to its roughly 49% win
- no correctness or safety regression

If a stage improves its target but causes a material regression elsewhere, retain the previous stage and report the rejected experiment separately.

## Deliverables

- isolated experimental commits for retained stages
- local and remote test evidence
- per-stage benchmark logs and summaries under `codex/logs/`
- final recommendation on whether and which commits should be applied to the PR branch
