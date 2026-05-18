# Phase 3: O(T log T) BPE on GPU via rank-bucket scheduling

Scope: a second kernel that replaces Phase 2's "scan all positions every
iteration" loop with a rank-bucket queue + doubly-linked-list of live
positions. Phase 2 stays in place (unmodified) for short inputs; Phase 3
takes over for long inputs (the regime where Phase 2's tail is O(T²)).

---

## What Phase 2 does today

Per outer iteration (current kernel):

1. Parallel scan all T positions → compute pair rank at each.
2. Block-wide min reduction → find global min rank.
3. Single-thread walk → mark non-overlapping leftmost positions of that rank.
4. Multi-tile BlockScan compaction → emit merged sequence.

Each iteration is O(T). Number of iterations is bounded by the number of
distinct ranks visited. For short sequences many merges fire per iteration
and total work is roughly O(T). For long sequences (>2 kbp) the tail of
the schedule has only one merge per iter, so total work degrades to O(T²).

## What Phase 3 changes

Phase 3 only touches positions that have an *applicable* merge in the
current rank bucket. It maintains:

- `tokens[T]`        — current token IDs (length T₀ = input bytes; never grows).
- `next[T]`, `prev[T]` — doubly-linked list over **live** positions.
  - `next[i] = j` where j is the next live position after i.
  - `next[i] = -1` if i is the last live position.
  - `prev[i] = j` symmetric. `prev[head] = -1`.
  - When a position is merged-away (becomes the right operand of a merge), we
    set `next[i] = -2` (sentinel: dead).
- `bucket_head[R]`   — for each rank R ∈ [0, num_merges), head of a linked
  list of positions whose pair `(tokens[i], tokens[next[i]])` may have rank R.
- `bucket_next[T]`   — next-pointer inside a bucket linked list.
- `current_rank`     — the lowest rank we have not yet drained.

Entries inside a bucket can be **stale** (the position has been merged away,
or its right neighbor changed and its pair rank is no longer R). We
re-validate on pop instead of trying to remove eagerly. Stale entries are
cheap to detect — one cuCollections lookup per entry — and we never visit
the same stale entry twice because each iteration *drains* the bucket.

### Outer loop

```
init: for i in 0..T-1: insert position i into bucket of its initial pair
      current_rank = 0

while true:
    # Advance current_rank to the next non-empty bucket
    while current_rank < num_merges and bucket_head[current_rank] == NULL:
        current_rank += 1
    if current_rank >= num_merges: break    # converged

    R = current_rank

    # Gather valid entries in left-to-right order (single pass)
    #   walk bucket_head[R] -> follow bucket_next
    #   for each entry p:
    #     valid iff next[p] != -2 and lookup(tokens[p], tokens[next[p]]).rank == R
    # Output: list of valid positions, sorted by p.
    valid = collect_valid_entries(R)

    # Non-overlap filter (leftmost): walk valid in order; if last selected
    # position s has next[s] == p, skip p (would overlap on the shared
    # right operand). Otherwise select p.
    selected = non_overlap_filter(valid)

    # Apply all selected merges in parallel
    parallel for p in selected:
        new_tok = merge_map[(tokens[p], tokens[next[p]])].new_token
        r = next[p]      # the right operand position (will be killed)
        # Splice r out of the DLL
        tokens[p] = new_tok
        next[p] = next[r]
        if next[r] != -1: prev[next[r]] = p
        next[r] = -2     # mark dead

    # Update bucket entries for the NEW pairs created
    parallel for p in selected:
        # Pair to the left: (tokens[prev[p]], tokens[p])
        if prev[p] != -1:
            insert_into_bucket(prev[p])    # uses new rank from merge_map
        # Pair to the right: (tokens[p], tokens[next[p]])
        if next[p] != -1:
            insert_into_bucket(p)
```

`bucket_head[R]` is cleared after we finish iterating it. We never revisit
a rank-R bucket because the BPE training invariant guarantees no new
rank-R entry can be created after rank R is drained (see below).

### Why `current_rank` only ever advances

The BPE training procedure builds the merge table bottom-up: rank 0 is
learned first (only over the byte alphabet); rank R is learned over the
vocabulary that includes ranks 0..R-1. Therefore any token of rank R
**cannot appear as an operand** in a rank-R'<R merge. Once we drain
bucket R and all rank-R merge results enter the sequence, the only
new pairs they form are with their immediate neighbors, and those pairs
involve at least one token whose ID ≥ rank-R+vocab_byte_offset (or the
pair simply isn't in the merge table). So the new pair's rank, if it has
one, is necessarily ≥ R+1. The `current_rank` pointer never moves
backward.

Empirically this is the same property Phase 2's "rank-batched leftmost
non-overlap" trick relies on, just lifted from "within one outer
iteration" to "across the entire schedule".

### Why within-bucket order doesn't matter (modulo overlap)

Within a single rank-R bucket, HF Algorithm 1 says: process the leftmost
applicable pair, then re-check. After the leftmost merge fires, the
positions to its right are unchanged in their pair identities EXCEPT
those that overlap with it. Therefore for a left-to-right walk where we
keep only non-overlapping selections, we obtain the same set of merges HF
would have applied at rank R (before moving to rank R+1).

Overlapping entries that we didn't select aren't lost — their pair
identities haven't been disturbed by the merges we did apply, so they
remain valid bucket entries and will be processed when the same rank R
becomes current again. *But by the training invariant, R can only
*increase*.* So in fact, those overlapping unselected entries become
stale *only* if a left-neighbor merge clobbered their pair, in which
case they're re-validated as stale on pop. The simplest correct rule is:
process bucket R until it's empty (drain), where "drain" means walk it
once and reinsert anything that's still valid but couldn't be selected.

Concretely: when we filter for non-overlap, the unselected-because-of-
overlap entries get re-inserted into bucket R (their pair rank hasn't
changed). On the next pass of the outer loop, `current_rank` is still R
(we haven't moved on yet), so we walk those re-inserted entries again.
This iterates O(k) times within bucket R where k is the chain length of
the longest overlapping run — bounded by T total across all iterations.

### Complexity

- Each position is merged away exactly once → O(T) parallel merge work.
- Each position is inserted into a bucket at most O(1) times per merge in
  its neighborhood → O(T) total bucket inserts.
- `current_rank` advances monotonically through O(num_merges) buckets →
  O(num_merges) bucket-empty checks total.
- Re-validation on pop: each bucket entry is popped at most twice (once
  with valid state, once when stale; the stale one is detected immediately
  and discarded) → O(T) total cuCollections lookups.

**Total: O(T + num_merges) work per sequence.** Compare with Phase 2's
O(T²) tail and HF's O(T log T).

### Where the parallelism lives

- Initial bucket insertion: parallel over T positions.
- `collect_valid_entries`: per-bucket walk is sequential, but we walk
  multiple buckets across the outer loop, and the inner re-validation is
  block-parallel using one thread per entry over a small bounded window.
- `non_overlap_filter`: one-pass scan — sequential but O(k) where k is
  the bucket size for this rank, usually small.
- Apply selected merges: parallel across selected positions.
- Update bucket entries: parallel across the same selected set.

The sequential-walk-per-bucket is the only serial bottleneck per iter.
Bucket sizes drop fast (each rank is drained in 1-2 outer-loop passes),
so per-iter serial work is small. Total serial work across the whole
schedule remains O(T).

---

## What we keep from Phase 2

- The same merge-table data structure (cuCollections static_map keyed by
  packed (a,b) → (rank, new_token)).
- The same correctness gate (bit-identical to HF, 256 random sequences,
  4 bp – 2 kbp; we add a long-sequence extension up to 16 kbp).
- The same Python wrapper surface — Phase 3 is a new method on the same
  class.

## What's new in the .cu file

- `bpe_algorithm1_v2_kernel` — the rank-bucket scheduling kernel.
- `DNATokBPE::tokenize_batch_v2` — host launcher; allocates the extra
  DLL/bucket buffers (`next`, `prev`, `bucket_head`, `bucket_next`).
- Workspace caching grown to include the new buffers.

## Roll-out

1. Implement v2 kernel.
2. Pass bit-identical gate on every BPE model in the registry.
3. Benchmark crossover length where v2 starts beating v1+HF-fallback.
4. Update routing in the Python wrapper: short → v1, long → v2.
5. If v2 wins everywhere, retire v1 (later PR).

---

## Status — bucket-scheduling design BROKEN (2026-05-19)

**The "current_rank only advances upward" invariant is wrong.**

Counterexample: tokenizer with merges `(T,T) → TT` at rank 0 and
`(G,TT) → GTT` at rank 1, while `(G,T) → GT` at rank 5 (perfectly
plausible — once `TT` exists after rank 0, the pair `(G,TT)` becomes a
training candidate and may be more frequent than `(G,T)`).

Input `CGTT` traces:
- Init: positions 0,1,2 in buckets R(C,G)=?, R(G,T)=5, R(T,T)=0.
- Drain bucket 0 (only position 2): merge `(T,T)` → `TT`. Tokens are
  now `[C, G, TT, _]`. Position 1's pair is now `(G, TT)` at rank 1.
- Step 3e tries to insert position 1 into bucket 1 — but `bucket_st[1]`
  is already 5 (stale). The atomicCAS-claim fails. Position 1 stays in
  bucket 5 only.
- `current_rank` advances from 0 looking for next non-empty bucket.
  Bucket 1 is empty (insert failed). Bucket 5 has position 1.
- Drain bucket 5: validate position 1 → rank is now 1, not 5. Re-route
  to bucket 1.
- Advance `current_rank` past 5 to find next non-empty. **But bucket 1
  was just re-inserted at rank 1 < 5.** We never visit it again.

The merge `(G, TT) → GTT` is never fired. v2 outputs `[C, G, TT]`
instead of HF's `[C, GTT]`.

### Why this isn't fixable with small tweaks

Going back to a lower rank when we discover a stale entry's new pair has
lower rank would be wrong: we already processed merges at the higher
current rank, in the wrong order.

The clean fix is to abandon the "monotone advance" assumption and use a
true priority queue (min-heap on GPU) — but heap-on-GPU is non-trivial.

### Path forward — DLL-Phase-2

Drop the rank-bucket design entirely. Instead, **keep Phase 2's
algorithm and just add a doubly-linked list to skip dead positions in
the per-iteration scan.** This is the smallest correct change that
addresses Phase 2's "O(T²) tail" pathology.

Per-iter cost in DLL-Phase-2 is O(live_count), summed over all iters
is O(T log T) under typical workloads. Same asymptotic as HF, with
GPU-parallel constant factors.

The buggy v2 kernel in `dnatok_bpe.cu` is preserved but unused; its
header docs and this file mark it as broken. DLL-Phase-2 will become
the new v2 in a follow-up commit.

---

## v3 — entry-pool bucket scheduling (CORRECT, landed 2026-05-19)

Instead of switching to DLL-Phase-2, we kept rank-bucket scheduling and
fixed the correctness issue by **allowing a position to be in multiple
buckets simultaneously**. Each bucket entry is a separate node in a
shared per-sequence entry pool; positions are no longer their own
linked-list nodes. New pairs always insert a fresh entry at their actual
rank, and stale entries are discarded by re-validation at drain time.

Capacity: each position generates ≤ 3 entries across the whole schedule
(initial + left-insert + right-insert). The pool is sized at
`8 × T_max` entries per sequence with a runtime overflow check.

Correctness: validated bit-identical against the Phase 2 kernel (which
is bit-identical to HF) and against the Python reference on **264/264
sequences up to 16 kbp** across two real merge tables. Full pytest
suite passes **119/119 cases** with engine ∈ {`gputok`, `dnatok`,
`dnatok_v3`}.

### Length sweep — DNABERT-2 on GB10, B=32 (median ms of 30 iters)

After the parallelization pass (CUB `BlockMergeSort`, parallel
validation, fused dedup+filter):

| len (bp) | HF native | v1 (Phase 2 + HF fallback) | v3 (entry-pool, parallelised) | v3 vs HF | v3 vs v1 |
|---:|---:|---:|---:|---:|---:|
| 32    | 0.63  | 0.14 | 0.19  | 3.3× | 0.70× |
| 128   | 0.82  | 0.34 | 0.43  | 1.9× | 0.81× |
| 512   | 2.49  | 1.34 | 1.32  | 1.9× | 1.02× |
| 1024  | 3.97  | 3.37 | 2.53  | 1.6× | 1.33× |
| 2048  | 6.86  | 9.70 | 3.66  | **1.9×** | 2.65× |
| 4096  | 12.30 | 27.86 | 6.36  | **1.9×** | 4.38× |
| 8192  | 23.01 | 58.02 | 11.41 | **2.0×** | 5.09× |
| 16384 | 44.50 | 117.51 | 19.53 | **2.3×** | 6.02× |
| 32768 | 92.31 | 240.92 | 39.31 | **2.3×** | 6.13× |

Takeaways:
- **v3 strictly beats HF from ~512 bp upward** and matches HF at 32-128 bp
  (where v1's lower kernel overhead pulls ahead).
- **v3 beats v1 from 1 kbp upward** and dominates above 2 kbp (where v1
  routes to HF). At 32 kbp v3 is **6.1× faster than v1**.
- The asymptotic O(T log T) cost is realised; the parallelisation
  removed the constant-factor bottleneck that hurt v3 at 16 kbp+ in
  the first iteration.

### Audit + parallelisation pass — 2026-05-19

What was wrong with the initial v3 implementation:
1. **Step 3a (drain)**: a single thread walked the linked list AND did
   the expensive `map.find()` validation. Split into two passes — a
   single-thread linked-list walk (cheap pointer-follows) followed by
   parallel validation over all threads.
2. **Step 3b (sort)**: single-thread O(n²) insertion sort. Replaced
   with CUB `BlockMergeSort` for n ≥ 96 candidates, with single-thread
   fallback for very small (< 96) and very large (> 4096) buckets. The
   threshold was tuned to the GB10 crossover point.
3. **Step 3c (filter)**: fused dedup + sentinel-skip + non-overlap
   filter into a single sequential pass. Stays single-threaded (the
   non-overlap decision is fundamentally sequential, but each pass is
   now O(unique candidates) with no extra dedup phase).

What I found and fixed in the audit (correctness-side):
1. **Latent bug**: the scratch buffer was sized at `T_max`, but the
   linked-list drain in step 3a can — on pathological inputs — produce
   up to ~3T entries for a single bucket. Resized scratch to the entry
   pool capacity (`8 × T_max`) and added an overflow check that sets a
   sentinel rather than silently writing past the buffer.
2. **Stride consistency**: after the resize, the scratch tensor's
   row-stride became `entry_pool_size`, not `T_max`. Updated the kernel
   pointer arithmetic to use the correct stride parameter.

### Recommended routing

For the publication benchmark:
- BPE inputs of any length → `engine="dnatok_v3"` (now strictly faster
  than HF from ~256 bp upward).
- `engine="dnatok"` (v1 + HF fallback at 2 kbp) remains available for
  comparison.

### Future work

- The 32-128 bp regime where v1 still edges out v3 by ~1.5× could be
  closed by short-circuiting the entry-pool scaffolding entirely on
  tiny inputs (effectively running v1's algorithm when n_raw < ~16).
- BlockMergeSort `ITEMS_PER_THREAD = 16` caps the parallel sort at 4096
  items. For ultra-long inputs (T > 128 kbp) the initial bucket can
  exceed this. Bumping to 32 (max 8192) would help but increases
  shared-memory pressure.
