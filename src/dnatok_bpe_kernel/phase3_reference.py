"""Phase 3 reference implementation — pure Python, no perf goal.

Purpose:
    Implement the rank-bucket + DLL algorithm exactly as described in
    PHASE3_DESIGN.md, against a small Python data structure that is easy
    to inspect and debug. Then validate it produces output bit-identical
    to HF's tokenizer on real BPE merges (DNABERT-2 / GENA-LM / METAGENE-1).

    Only when this reference passes the bit-identical gate do we port the
    algorithm to CUDA. Catches design bugs (especially the within-bucket
    leftmost-order requirement) cheaply.

Algorithm reference: see PHASE3_DESIGN.md.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# A "dead" sentinel for next[]; matches what the CUDA kernel will use.
DEAD = -2
NIL = -1


@dataclass
class BPEEncoder:
    """Reference BPE encoder using Phase 3's rank-bucket + DLL algorithm.

    Inputs (constructor):
        merges: list of (a_id, b_id, rank, new_id) tuples. Each row defines
            "pair (a_id, b_id) has rank R and merges to new_id". Ranks must
            be 0..len(merges)-1 (monotone).
        byte_to_id: list of 256 int → initial-token IDs (identity in our
            DNA case).

    Run:
        encode(bytes_seq) -> list[int] of merged token IDs.
    """
    merges: Dict[Tuple[int, int], Tuple[int, int]] = field(default_factory=dict)
    # (a, b) -> (rank, new_id)
    byte_to_id: List[int] = field(default_factory=lambda: list(range(256)))
    num_merges: int = 0

    @classmethod
    def from_merges_file(cls, merges_path: str) -> "BPEEncoder":
        """Build a vocab+merges table by replaying merges.txt, mirroring the
        CUDA kernel's build_vocab_from_merges() exactly."""
        token_to_id: Dict[str, int] = {}
        for b in range(256):
            token_to_id[chr(b)] = b
        merges: Dict[Tuple[int, int], Tuple[int, int]] = {}

        with open(merges_path, "r", encoding="utf-8") as f:
            first = f.readline()  # skip "#version: ..." line
            rank = 0
            for line in f:
                line = line.rstrip("\n")
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) != 2:
                    continue
                a, b = parts
                if a not in token_to_id:
                    token_to_id[a] = len(token_to_id)
                if b not in token_to_id:
                    token_to_id[b] = len(token_to_id)
                merged = a + b
                if merged not in token_to_id:
                    token_to_id[merged] = len(token_to_id)
                merges[(token_to_id[a], token_to_id[b])] = (
                    rank,
                    token_to_id[merged],
                )
                rank += 1

        enc = cls(merges=merges, num_merges=len(merges))
        enc._token_to_id = token_to_id  # for debugging only
        return enc

    def encode(self, text: str) -> List[int]:
        """Phase 3 algorithm: rank-bucket scheduling + DLL.

        For the reference we use a min-heap as the bucket queue. Each entry
        is (rank, insertion_order, position). insertion_order is a monotone
        counter — within a rank, ties are broken by smallest insertion_order
        which we will THEN resolve by smallest position via a sort at drain
        time. The heap is the CUDA-equivalent of "find the lowest non-empty
        bucket"; the sort at drain time is the CUDA-equivalent of "sort
        within bucket by position."

        Stale-on-pop: an entry (rank, _, pos) is stale if pos is dead OR if
        the current pair at pos doesn't have this rank. Stale entries are
        discarded and we re-validate the next pop.
        """
        bs = text.encode("latin-1")  # raw byte sequence
        n = len(bs)
        if n == 0:
            return []

        tokens: List[int] = [self.byte_to_id[b] for b in bs]
        # DLL: live linked list over positions [0..n).
        next_: List[int] = list(range(1, n)) + [NIL]
        prev_: List[int] = [NIL] + list(range(0, n - 1))

        # Bucket queue as min-heap. We push (rank, order, pos). Stale check
        # happens on pop.
        heap: List[Tuple[int, int, int]] = []
        order = 0  # monotone insertion counter (kept for stable popping)

        def pair_rank(p: int) -> Optional[Tuple[int, int]]:
            """Return (rank, new_id) if (tokens[p], tokens[next[p]]) is in
            the merge table; None otherwise."""
            nb = next_[p]
            if nb == NIL or nb == DEAD or next_[p] == DEAD:
                return None
            key = (tokens[p], tokens[nb])
            return self.merges.get(key)

        # Initial bucket fill: every position with a known pair.
        for p in range(n - 1):
            r = pair_rank(p)
            if r is not None:
                heapq.heappush(heap, (r[0], order, p))
                order += 1

        while heap:
            # 1. Find the lowest non-empty bucket. Drain everything at that
            #    rank in one pass. We collect candidates, sort by position
            #    (HF leftmost rule), then non-overlap filter.
            cur_rank = heap[0][0]
            candidates: List[int] = []
            while heap and heap[0][0] == cur_rank:
                _, _, p = heapq.heappop(heap)
                r = pair_rank(p)
                if r is None:
                    continue  # stale
                if r[0] != cur_rank:
                    # The pair's rank changed since it was enqueued — push
                    # it back into the correct bucket. (Can happen when
                    # someone merged its left neighbor and tokens[p]
                    # changed.)
                    heapq.heappush(heap, (r[0], order, p))
                    order += 1
                    continue
                candidates.append(p)

            if not candidates:
                continue

            # Sort by position to mimic HF leftmost.
            candidates.sort()
            # Non-overlap filter: keep p only if its left neighbor wasn't
            # just selected. Walking sorted by position means we just check
            # that prev[p] isn't already in our selected set, which is
            # equivalent to "the position s most recently selected is not
            # the left operand of p" — i.e., next[s] != p.
            selected: List[int] = []
            last_kept = NIL
            for p in candidates:
                if last_kept != NIL and next_[last_kept] == p:
                    continue  # overlaps with previous selection — skip
                selected.append(p)
                last_kept = p

            # The candidates we skipped because of overlap must be
            # re-considered after the merges fire (they may still be valid
            # at this rank if their pair survives). The simplest correct
            # rule is: push them back into the bucket at the current rank;
            # they will be re-popped and re-validated. Most will become
            # stale because the left neighbor's merge clobbered the pair,
            # but the bookkeeping is trivial.
            for p in candidates:
                if p not in selected:
                    heapq.heappush(heap, (cur_rank, order, p))
                    order += 1

            # 2. Apply selected merges in two passes (mirrors CUDA):
            #    (a) Capture pre-merge DLL state for each selected p.
            #    (b) Update tokens[] and DLL pointers.
            pre: Dict[int, Tuple[int, int, int]] = {}  # p -> (old_next, old_old_next, new_id)
            for p in selected:
                rb = pair_rank(p)
                assert rb is not None and rb[0] == cur_rank, "validated above"
                old_next = next_[p]
                old_old_next = next_[old_next]
                new_id = rb[1]
                pre[p] = (old_next, old_old_next, new_id)

            for p in selected:
                old_next, old_old_next, new_id = pre[p]
                tokens[p] = new_id
                next_[p] = old_old_next
                if old_old_next != NIL:
                    prev_[old_old_next] = p
                # Kill the right-operand position.
                next_[old_next] = DEAD
                prev_[old_next] = DEAD

            # 3. Insert new bucket entries for the merge results' new
            #    neighbors. Left side: prev[p] now has a new right pair.
            #    Right side: p now has a new right pair.
            for p in selected:
                left = prev_[p]
                if left != NIL and left != DEAD:
                    r = pair_rank(left)
                    if r is not None:
                        heapq.heappush(heap, (r[0], order, left))
                        order += 1
                r = pair_rank(p)
                if r is not None:
                    heapq.heappush(heap, (r[0], order, p))
                    order += 1

        # 4. Walk the DLL from the first live position to produce the
        #    output sequence. The leftmost live position is the smallest p
        #    with prev[p] != DEAD (we never moved the head); since we only
        #    kill right-operand positions, prev[0] is always NIL or 0 is
        #    alive.
        out: List[int] = []
        # Find head: first p with prev[p] == NIL and next[p] != DEAD.
        head = 0
        while head < n and (next_[head] == DEAD and prev_[head] == DEAD):
            head += 1
        if head >= n:
            return []
        p = head
        while p != NIL:
            out.append(tokens[p])
            p = next_[p]
        return out


# --------------------------------------------------------------------------
# Self-test / smoke test
# --------------------------------------------------------------------------
if __name__ == "__main__":
    # Tiny synthetic merges table to sanity-check the algorithm.
    enc = BPEEncoder()
    enc.byte_to_id = list(range(256))
    A = ord("A")
    enc.merges = {
        (A, A):                            (0, 256),   # AA -> 256
        (256, 256):                        (1, 257),   # AAAA -> 257
        (257, A):                          (2, 258),   # AAAAA -> 258
    }
    enc.num_merges = len(enc.merges)

    # AAAAA → after rank-0 merges: AA, AA, A → after rank-1: AAAA, A → after rank-2: AAAAA
    assert enc.encode("AAAAA") == [258], enc.encode("AAAAA")
    # AAAA: rank-0 → AA, AA → rank-1 → AAAA → [257]
    assert enc.encode("AAAA") == [257], enc.encode("AAAA")
    # AAA: rank-0 leftmost → AA, A → no rank-1 applies (we have (256,256) only) → [256, A]
    assert enc.encode("AAA") == [256, A], enc.encode("AAA")
    # AA: rank-0 → [256]
    assert enc.encode("AA") == [256]
    # A: no merge → [A]
    assert enc.encode("A") == [A]
    # empty
    assert enc.encode("") == []

    print("Phase 3 reference: synthetic merges PASS")
