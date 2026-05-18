// dnatok_bpe.cu — DNA-specialised GPU BPE kernel.
//
// What this is and what it isn't:
//
//   * Implements HF Algorithm-1 BPE (find globally lowest-rank applicable
//     pair, merge LEFTMOST occurrence, repeat) — empirically validated
//     bit-identical to HF native for DNABERT-2, GENA-LM, METAGENE-1
//     across 256 random sequences, 4-bp..8-kbp lengths, with N runs and
//     mixed case.
//
//   * Working buffers live in **global memory**, not shared memory.
//     GPUTOK's stock kernel caps at chunk_tokens=2048 because its working
//     buffers must fit in ~48 KB of static shared memory. We don't chunk
//     at all — the entire input sequence is processed in one kernel call
//     and chunk-boundary merge errors cannot occur.
//
//   * Genomic-input specialisations actually shipped here:
//        - Direct uint8 byte input (no encode_text_to_tokens CPU step;
//          no per-character std::string allocation on the host).
//        - Returns torch CUDA tensors (ids[B,T_max] int32, lengths[B]
//          int32) directly. No Python list-of-list round-trip.
//        - Rank-batched merge loop: instead of one merge per outer
//          iteration (GPUTOK baseline), we apply every non-overlapping
//          leftmost merge of the globally-lowest rank in one pass.
//          Empirically cuts outer-iteration count 3-5× on random DNA;
//          on inputs > ~2 kbp HF's heap-based scheme still wins because
//          the tail of the schedule degenerates to one merge per iter.
//          The Python wrapper routes long inputs to HF for that reason.
//
//   * NOT specialised: the initial byte→id LUT is the IDENTITY map
//     (byte b → token id b). The kernel's internal vocab seeds 256 byte
//     symbols at ids 0..255 in build_vocab_from_merges(); a remap LUT in
//     the Python wrapper translates to the HF tokenizer's id space.
//     Working buffers are int32, not uint16. Both could be tightened in
//     a future revision.
//
//   * Algorithm reference: BlockBPE paper §3 Algorithm 1
//       https://arxiv.org/pdf/2507.11941
//
// Build:
//
//   torch.utils.cpp_extension.load(
//       name="dnatok_bpe",
//       sources=[".../dnatok_bpe.cu"],
//       extra_include_paths=[
//           "/path/to/cuCollections/include",
//           "/path/to/cccl/cub",
//           "/path/to/cccl/thrust",
//           "/path/to/cccl/libcudacxx/include",
//       ],
//       extra_cuda_cflags=["-O3", "-std=c++17", "--expt-extended-lambda"],
//   )

#include <torch/extension.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <thrust/device_vector.h>
#include <cuco/static_map.cuh>
#include <cub/block/block_merge_sort.cuh>

namespace {

// --------------------------------------------------------------------------
// Error handling
// --------------------------------------------------------------------------

// Lightweight CUDA error-check macro. Throws std::runtime_error with the
// CUDA error string + call site on failure. Used for host-side cudaMalloc /
// cudaMemcpy / cudaFree calls; kernel-launch failures are checked separately
// via cudaGetLastError().
#define DNATOK_CHECK_CUDA(call) do {                                       \
    cudaError_t err__ = (call);                                            \
    if (err__ != cudaSuccess) {                                            \
      throw std::runtime_error(std::string("CUDA error: ") +               \
                                cudaGetErrorString(err__) +                \
                                " (at " #call ")");                        \
    }                                                                      \
  } while (0)

// --------------------------------------------------------------------------
// Build-side support
// --------------------------------------------------------------------------

// Pair packing: little-endian (a << 32 | b). Token IDs are int32 (we don't
// constrain to int16 in the key because cuCollections needs a 64-bit key for
// the open-addressing hashmap; the IDs in the *output* tensor are int32 too,
// which matches HF's id space).
using PairKey = std::uint64_t;
using PairVal = std::uint64_t;  // packed (rank, new_token)

struct PairInfo {
  std::uint32_t rank;
  std::int32_t  new_token;
};

__host__ __device__ inline PairKey pack_pair(std::int32_t a, std::int32_t b) {
  return (static_cast<PairKey>(static_cast<std::uint32_t>(a)) << 32) |
         static_cast<PairKey>(static_cast<std::uint32_t>(b));
}

__host__ __device__ inline PairVal pack_val(PairInfo info) {
  return (static_cast<PairVal>(static_cast<std::uint32_t>(info.new_token)) << 32) |
         static_cast<PairVal>(info.rank);
}

__host__ __device__ inline PairInfo unpack_val(PairVal v) {
  PairInfo info;
  info.rank      = static_cast<std::uint32_t>(v & 0xFFFFFFFFu);
  info.new_token = static_cast<std::int32_t>((v >> 32) & 0xFFFFFFFFu);
  return info;
}

// Sentinel pair-key (= no entry). Chosen to never collide with a real merge:
// a real merge has at least one operand id < 2^31, so the high bit of both
// halves cannot simultaneously be set — UINT64_MAX is safe.
static constexpr PairKey EMPTY_KEY = std::numeric_limits<PairKey>::max();
static constexpr PairVal EMPTY_VAL = std::numeric_limits<PairVal>::max();

using PairHasher = cuco::default_hash_function<PairKey>;
using PairEq     = thrust::equal_to<PairKey>;
using PairProbe  = cuco::linear_probing<1, PairHasher>;

using PairMap = cuco::static_map<PairKey, PairVal,
                                 cuco::extent<std::size_t>,
                                 cuda::thread_scope_device,
                                 PairEq,
                                 PairProbe>;
using PairMapRef = decltype(std::declval<PairMap>().ref(cuco::find));

// --------------------------------------------------------------------------
// BPE merge kernel — Algorithm 1, working buffers in global memory
// --------------------------------------------------------------------------

#ifndef DNATOK_BLOCK_SIZE
#define DNATOK_BLOCK_SIZE 256
#endif

// The block-wide min reduction (lines below) uses a halving-stride pattern
// that requires DNATOK_BLOCK_SIZE to be a power of two. CUB's BlockScan is
// also typically tuned for power-of-two block sizes.
static_assert((DNATOK_BLOCK_SIZE & (DNATOK_BLOCK_SIZE - 1)) == 0,
              "DNATOK_BLOCK_SIZE must be a power of two");

using BlockScan = cub::BlockScan<int, DNATOK_BLOCK_SIZE>;

// Rank-batched Algorithm-1 BPE:
//
//   Stock HF (Rust) processes merges one at a time in (rank, position)
//   order: pop the lowest-rank applicable pair, apply it, push new
//   candidates, repeat. With a heap it runs in O(T log T) per sequence.
//
//   We exploit the fact that when N positions share the global minimum
//   rank AND they don't overlap (their pairs are not adjacent), the
//   order HF processes them in doesn't change the result. So we batch
//   all such positions into a single iteration and merge them in
//   parallel. For overlapping same-rank positions (e.g. consecutive
//   (A,A) pairs), we keep only the leftmost of every run — the rest
//   will fire in later iterations exactly as HF would do them next.
//
//   Empirically: number of outer iterations drops 3-5× vs the naive
//   "one merge per iter" baseline. Per-iter cost is O(T) (rank scan +
//   sequential selection scan + compaction). Total time:
//
//      ≤ ~2 kbp inputs:  we beat HF native by ~2× on B≥128 batches
//                        because the per-iter overhead is amortised
//                        across many parallel merges.
//      > ~2 kbp:         tail iterations only batch ~1 merge each, so
//                        cost regresses toward O(T²). HF's heap stays
//                        O(T log T). The Python wrapper routes long
//                        inputs to HF until a future heap-on-GPU pass.
//
// Per-block global workspace layout (the caller allocates):
//   d_tokens_a     [B, T_max] int32  — primary token buffer (in/out)
//   d_tokens_b     [B, T_max] int32  — scratch for compaction
//   d_ranks        [B, T_max] uint32 — pair ranks per position
//   d_new_tokens   [B, T_max] int32  — merge result per position
//   d_selected     [B, T_max] uint8  — 1 if this position will merge
//
__global__ void bpe_algorithm1_kernel(
    PairMapRef                       map_ref,
    const std::uint8_t* __restrict__ d_bytes,         // [total_bytes]
    const std::int32_t* __restrict__ d_byte_offsets,  // [B+1]
    const std::int32_t* __restrict__ d_byte_to_id,    // [256]
    std::int32_t*                    d_tokens_a,      // workspace [B * T_max] (in + out)
    std::int32_t*                    d_tokens_b,      // workspace [B * T_max] (compaction scratch)
    std::uint32_t*                   d_ranks,         // workspace [B * T_max] (per-pair ranks)
    std::int32_t*                    d_new_tokens,    // workspace [B * T_max] (merge results)
    std::uint8_t*                    d_selected,      // workspace [B * T_max] (selection flags)
    std::int32_t* __restrict__       d_out_lengths,   // [B]
    std::int32_t                     T_max,           // max sequence length budgeted
    std::int32_t                     max_iters)
{
  int seq_id = blockIdx.x;
  int tid    = threadIdx.x;
  int nthr   = blockDim.x;

  std::int32_t byte_start = d_byte_offsets[seq_id];
  std::int32_t byte_end   = d_byte_offsets[seq_id + 1];
  std::int32_t len        = byte_end - byte_start;
  if (len <= 0) {
    if (tid == 0) d_out_lengths[seq_id] = 0;
    return;
  }
  if (len > T_max) {
    // Caller is responsible for guaranteeing len <= T_max. We refuse to
    // produce wrong output rather than truncate.
    if (tid == 0) d_out_lengths[seq_id] = -1;
    return;
  }

  // Per-block working buffer slices (no shared-memory chunk limit).
  std::int32_t*  g_tokens_in   = d_tokens_a    + static_cast<std::size_t>(seq_id) * T_max;
  std::int32_t*  g_tokens_out  = d_tokens_b    + static_cast<std::size_t>(seq_id) * T_max;
  std::uint32_t* g_ranks       = d_ranks       + static_cast<std::size_t>(seq_id) * T_max;
  std::int32_t*  g_new_tokens  = d_new_tokens  + static_cast<std::size_t>(seq_id) * T_max;
  std::uint8_t*  g_selected    = d_selected    + static_cast<std::size_t>(seq_id) * T_max;

  // 1. Initial byte → token id mapping (one byte = one initial token).
  for (int i = tid; i < len; i += nthr) {
    g_tokens_in[i] = d_byte_to_id[d_bytes[byte_start + i]];
  }

  __shared__ int sh_len;
  __shared__ std::uint32_t sh_min_rank;
  __shared__ int sh_total_deleted;
  __shared__ std::uint32_t sh_red_rank[DNATOK_BLOCK_SIZE];
  __shared__ typename BlockScan::TempStorage scan_storage;

  if (tid == 0) sh_len = len;
  __syncthreads();

  const std::uint32_t INF_RANK = 0xFFFFFFFFu;

  // 2. Rank-batched Algorithm 1 merge loop.
  int iter = 0;
  while (iter < max_iters) {
    int cur_len = sh_len;
    if (cur_len < 2) break;

    // 2a. Compute pair ranks + merge results for every adjacent pair.
    std::uint32_t local_min = INF_RANK;
    for (int i = tid; i < cur_len - 1; i += nthr) {
      PairKey key = pack_pair(g_tokens_in[i], g_tokens_in[i + 1]);
      auto    it  = map_ref.find(key);
      if (it == map_ref.end()) {
        g_ranks[i]      = INF_RANK;
        g_new_tokens[i] = -1;
      } else {
        PairInfo info   = unpack_val((*it).second);
        g_ranks[i]      = info.rank;
        g_new_tokens[i] = info.new_token;
        if (info.rank < local_min) local_min = info.rank;
      }
    }
    // Last position has no right neighbor → no pair.
    if (tid == 0 && cur_len > 0) {
      g_ranks[cur_len - 1]      = INF_RANK;
      g_new_tokens[cur_len - 1] = -1;
    }

    // 2b. Block-wide min reduction over local_min.
    sh_red_rank[tid] = local_min;
    __syncthreads();
    for (int stride = nthr / 2; stride > 0; stride >>= 1) {
      if (tid < stride) {
        std::uint32_t a = sh_red_rank[tid];
        std::uint32_t b = sh_red_rank[tid + stride];
        if (b < a) sh_red_rank[tid] = b;
      }
      __syncthreads();
    }
    if (tid == 0) sh_min_rank = sh_red_rank[0];
    __syncthreads();
    std::uint32_t rmin = sh_min_rank;
    if (rmin == INF_RANK) break;  // no merges applicable → converged

    // 2c. Selection of non-overlapping leftmost positions with rank == rmin.
    //
    // We pick positions that have rank == rmin AND are at an even offset
    // from the start of their maximal run-of-rmin. That gives leftmost
    // non-overlapping selections in every run, which is exactly what HF
    // would do (HF processes same-rank merges in left-to-right order;
    // for same-rank adjacent positions only the leftmost survives, the
    // others are recomputed next iteration).
    //
    // The "offset within run" is a parallel-scan operation with a custom
    // associative op. For simplicity (and because the sequential pass
    // is O(T) per iter, same as the rest of the loop), we do it in a
    // single thread. The block has nthr-1 idle threads during this step;
    // they pick back up at 2d below. On 8 kbp inputs this sequential
    // section is ~10 µs/iter — acceptable next to the parallel work.
    if (tid == 0) {
      int run_off = 0;
      bool prev_marked = false;
      for (int i = 0; i < cur_len - 1; ++i) {
        bool m = (g_ranks[i] == rmin);
        if (m) {
          if (!prev_marked) run_off = 0;
          else              run_off += 1;
          g_selected[i] = (run_off % 2 == 0) ? 1 : 0;
        } else {
          g_selected[i] = 0;
        }
        prev_marked = m;
      }
      g_selected[cur_len - 1] = 0;
    }
    __syncthreads();

    // 2d. Build compacted output:
    //   For each position i:
    //     - if i > 0 && selected[i-1]: this position is the right operand
    //       of a merge → drop it (delete_flag = 1).
    //     - else if selected[i]: emit the merge result (new_tokens[i]).
    //     - else: emit g_tokens_in[i].
    //
    //   delete_flag → exclusive prefix sum → write_index per position.
    //
    // The compaction is two-regime, same as before:
    //   small seq → CUB BlockScan
    //   long seq  → tiled prefix sum across multiple block strides.
    //
    // For long sequences we serialise the prefix-sum across tiles via
    // sh_total_deleted (running total maintained by thread 0).
    if (tid == 0) sh_total_deleted = 0;
    __syncthreads();

    if (cur_len <= nthr) {
      int flag = 0;
      int val  = 0;
      bool keep = true;
      if (tid < cur_len) {
        val = g_tokens_in[tid];
        if (tid > 0 && g_selected[tid - 1]) {
          flag = 1;
          keep = false;
        } else if (g_selected[tid]) {
          val = g_new_tokens[tid];
        }
      }
      int prefix = 0, block_sum = 0;
      BlockScan(scan_storage).ExclusiveSum(flag, prefix, block_sum);
      __syncthreads();

      if (tid < cur_len && keep) {
        int dst = tid - prefix;
        g_tokens_out[dst] = val;
      }
      __syncthreads();
      if (tid == 0) sh_len = cur_len - block_sum;
    } else {
      // Multi-tile compaction. Each tile is a BlockScan over nthr positions;
      // tile prefix sums are added to a running total maintained in sh_total_deleted.
      for (int base = 0; base < cur_len; base += nthr) {
        int i = base + tid;
        int flag = 0;
        int val  = 0;
        bool keep = true;
        bool in_range = (i < cur_len);
        if (in_range) {
          val = g_tokens_in[i];
          if (i > 0 && g_selected[i - 1]) {
            flag = 1;
            keep = false;
          } else if (g_selected[i]) {
            val = g_new_tokens[i];
          }
        }
        int prefix = 0, tile_sum = 0;
        BlockScan(scan_storage).ExclusiveSum(flag, prefix, tile_sum);
        __syncthreads();
        int dst_base = sh_total_deleted;
        if (in_range && keep) {
          int dst = i - dst_base - prefix;
          g_tokens_out[dst] = val;
        }
        __syncthreads();
        if (tid == 0) sh_total_deleted += tile_sum;
        __syncthreads();
      }
      if (tid == 0) sh_len = cur_len - sh_total_deleted;
    }
    __syncthreads();

    // Swap in / out buffers for next iteration.
    std::int32_t* tmp = g_tokens_in;
    g_tokens_in  = g_tokens_out;
    g_tokens_out = tmp;
    __syncthreads();

    ++iter;
  }

  // 3. Persist final tokens. After the swap loop, the "logical" output
  //    lives in g_tokens_in. Copy it back to d_tokens_a (canonical output
  //    slot) so the Python side always reads from the same tensor.
  int final_len = sh_len;
  if (g_tokens_in != d_tokens_a + static_cast<std::size_t>(seq_id) * T_max) {
    // Output is in the b-buffer; copy back into the a-buffer.
    std::int32_t* canonical = d_tokens_a + static_cast<std::size_t>(seq_id) * T_max;
    for (int i = tid; i < final_len; i += nthr) {
      canonical[i] = g_tokens_in[i];
    }
  }
  if (tid == 0) d_out_lengths[seq_id] = final_len;
}

// --------------------------------------------------------------------------
// Phase 3 — rank-bucket scheduling kernel  ***BROKEN, DO NOT USE***
// --------------------------------------------------------------------------
//
// STATUS (2026-05-19): the rank-bucket scheduling design below has a
// fundamental correctness flaw that cannot be fixed without changing the
// data structure. The kernel compiles and runs but does NOT produce
// HF-equivalent output. See PHASE3_DESIGN.md "Status" section for the
// counterexample (CGTT under (T,T)=0, (G,TT)=1, (G,T)=5 fails).
//
// The flaw: this kernel assumes "current_rank only advances upward".
// That holds for fresh inserts but NOT for stale entries — a position
// whose pair changes (because a neighbor merged) may now belong in a
// lower-rank bucket than the one it's currently sitting in. To handle
// that we'd need either a min-heap on GPU or a separate pool of bucket
// entries so positions can be in multiple buckets simultaneously.
//
// The Python reference in phase3_reference.py implements the correct
// algorithm using a min-heap, and passes the bit-identical gate against
// the (validated) Phase 2 kernel on 262/262 sequences up to 16 kbp.
// The reference is the canonical Phase 3 algorithm; this CUDA kernel
// is a starting point for a future port using the entry-pool design.
//
// Why a second kernel (kept for context):
//
//   The Phase 2 kernel above does O(T) work per outer iteration regardless
//   of how few merges fire that iteration. For long inputs the tail of the
//   schedule has only 1-2 merges per iter, so total work degrades to O(T²).
//   HF's heap stays O(T log T) and beats us above ~2 kbp.
//
// What's new:
//
//   * Doubly-linked list of LIVE positions: nxt/prv arrays. A merge kills
//     the right-operand position by writing nxt[p] = prv[p] = DEAD.
//
//   * Per-rank bucket linked lists: bucket_head[B, num_merges] points to
//     the first position whose current pair has that rank; bucket_next[p]
//     chains them within a bucket. Inserts are atomic LIFO; the
//     bucket_state[p] flag prevents double-insertion of the same position.
//
//   * current_rank pointer that walks upward monotonically through ranks.
//     BPE training invariant: a rank-R merge result only forms pairs of
//     rank >= R+1, so once bucket R is drained it stays empty.
//
// Stale-on-pop is the rule for handling out-of-date bucket entries: when
// we pop position p from bucket R, we re-validate (tokens[p], tokens[nxt[p]])
// against the merge map. If p is dead, or its pair has rank > R, we skip
// or re-route accordingly.
//
// Within a bucket, HF's "leftmost first" tie-break is enforced by sorting
// bucket entries by position before applying the non-overlap filter. This
// is the only mandatory O(k log k) cost per drain (k = bucket size).
//
// Design doc: see PHASE3_DESIGN.md in this directory.

static constexpr std::int32_t V2_NIL  = -1;
static constexpr std::int32_t V2_DEAD = -2;

// Per-block global workspace layout (caller allocates):
//   d_tokens     [B, T_max]      int32  — current tokens, length T (initial = byte count)
//   d_nxt        [B, T_max]      int32  — DLL next-pointer (V2_DEAD if killed)
//   d_prv        [B, T_max]      int32  — DLL prev-pointer
//   d_bucket_h   [B, num_merges] int32  — bucket head per rank
//   d_bucket_n   [B, T_max]      int32  — bucket next-pointer per position
//   d_bucket_st  [B, T_max]      int32  — bucket state (V2_NIL = not in any bucket; else rank R)
//   d_scratch    [B, T_max]      int32  — drain scratch (collect + sort buffer)
__global__ void bpe_v2_kernel(
    PairMapRef                       map_ref,
    const std::uint8_t* __restrict__ d_bytes,         // [total_bytes]
    const std::int32_t* __restrict__ d_byte_offsets,  // [B+1]
    const std::int32_t* __restrict__ d_byte_to_id,    // [256]
    std::int32_t*                    d_tokens,        // [B*T_max] in+out
    std::int32_t*                    d_nxt,           // [B*T_max]
    std::int32_t*                    d_prv,           // [B*T_max]
    std::int32_t*                    d_bucket_h,      // [B*num_merges]
    std::int32_t*                    d_bucket_n,      // [B*T_max]
    std::int32_t*                    d_bucket_st,     // [B*T_max]
    std::int32_t*                    d_scratch,       // [B*T_max]
    std::int32_t* __restrict__       d_out_lengths,   // [B]
    std::int32_t                     T_max,
    std::int32_t                     num_merges)
{
  int seq_id = blockIdx.x;
  int tid    = threadIdx.x;
  int nthr   = blockDim.x;

  std::int32_t byte_start = d_byte_offsets[seq_id];
  std::int32_t byte_end   = d_byte_offsets[seq_id + 1];
  std::int32_t n          = byte_end - byte_start;
  if (n <= 0) {
    if (tid == 0) d_out_lengths[seq_id] = 0;
    return;
  }
  if (n > T_max) {
    if (tid == 0) d_out_lengths[seq_id] = -1;
    return;
  }

  std::int32_t* tokens     = d_tokens     + static_cast<std::size_t>(seq_id) * T_max;
  std::int32_t* nxt        = d_nxt        + static_cast<std::size_t>(seq_id) * T_max;
  std::int32_t* prv        = d_prv        + static_cast<std::size_t>(seq_id) * T_max;
  std::int32_t* bucket_h   = d_bucket_h   + static_cast<std::size_t>(seq_id) * num_merges;
  std::int32_t* bucket_n   = d_bucket_n   + static_cast<std::size_t>(seq_id) * T_max;
  std::int32_t* bucket_st  = d_bucket_st  + static_cast<std::size_t>(seq_id) * T_max;
  std::int32_t* scratch    = d_scratch    + static_cast<std::size_t>(seq_id) * T_max;

  // 1. Initialize tokens, DLL, bucket state.
  for (int i = tid; i < n; i += nthr) {
    tokens[i]    = d_byte_to_id[d_bytes[byte_start + i]];
    nxt[i]       = (i < n - 1) ? (i + 1) : V2_NIL;
    prv[i]       = (i > 0)     ? (i - 1) : V2_NIL;
    bucket_st[i] = V2_NIL;
    bucket_n[i]  = V2_NIL;
  }
  for (int r = tid; r < num_merges; r += nthr) bucket_h[r] = V2_NIL;
  __syncthreads();

  // 2. Initial bucket fill: parallel over positions [0, n-1).
  //    Each thread looks up its pair, atomicCAS-pushes onto the head of
  //    the corresponding rank bucket. bucket_st[p] = r marks p enqueued.
  for (int i = tid; i < n - 1; i += nthr) {
    PairKey k = pack_pair(tokens[i], tokens[i + 1]);
    auto    it = map_ref.find(k);
    if (it != map_ref.end()) {
      int r = static_cast<int>(unpack_val((*it).second).rank);
      // Claim slot (always succeeds since bucket_st[i] is V2_NIL here).
      bucket_st[i] = r;
      // Atomic LIFO push.
      int old_head;
      do {
        old_head    = bucket_h[r];
        bucket_n[i] = old_head;
      } while (atomicCAS(&bucket_h[r], old_head, i) != old_head);
    }
  }
  __syncthreads();

  __shared__ int sh_cur_rank;
  __shared__ int sh_n_candidates;
  __shared__ int sh_n_selected;
  __shared__ int sh_live_count;

  if (tid == 0) {
    sh_live_count = n;
    // Advance to first non-empty bucket.
    int r = 0;
    while (r < num_merges && bucket_h[r] == V2_NIL) ++r;
    sh_cur_rank = r;
  }
  __syncthreads();

  // 3. Main outer loop. Each iteration drains one rank bucket.
  while (sh_cur_rank < num_merges) {
    int R = sh_cur_rank;

    // 3a. Collect valid bucket entries into scratch (single-thread).
    //
    //   * Stale-DEAD: skip and clear bucket_st.
    //   * Stale-rank: re-route into the correct bucket; clear bucket_st (it's
    //     already going to be set by the re-route below).
    //   * Valid (rank == R): add to scratch.
    if (tid == 0) {
      int cnt = 0;
      int p   = bucket_h[R];
      bucket_h[R] = V2_NIL;
      while (p != V2_NIL) {
        int np = bucket_n[p];   // capture before bucket_n[p] is reused
        bool dead = (nxt[p] == V2_DEAD);
        bool valid = false;
        int new_r = -1;
        if (!dead) {
          int nb = nxt[p];
          if (nb != V2_NIL && nb != V2_DEAD) {
            PairKey k = pack_pair(tokens[p], tokens[nb]);
            auto it = map_ref.find(k);
            if (it != map_ref.end()) {
              new_r = static_cast<int>(unpack_val((*it).second).rank);
              if (new_r == R) valid = true;
            }
          }
        }
        if (valid) {
          scratch[cnt++] = p;
          // The entry is removed from the bucket linked list. Clear its
          // bucket_st so step 3e can re-insert it (its new pair after the
          // merge will have a higher rank, which means we want to
          // atomicCAS NIL -> new_r).
          bucket_st[p] = V2_NIL;
        } else if (dead || new_r < 0) {
          bucket_st[p] = V2_NIL;
        } else {
          // new_r > R (since R was the lowest non-empty; if it were less
          // the schedule invariant would be violated). Re-route to new_r.
          bucket_st[p] = new_r;
          bucket_n[p]  = bucket_h[new_r];
          bucket_h[new_r] = p;
        }
        p = np;
      }
      sh_n_candidates = cnt;
    }
    __syncthreads();

    int n_cand = sh_n_candidates;
    if (n_cand == 0) {
      if (tid == 0) {
        int r = R + 1;
        while (r < num_merges && bucket_h[r] == V2_NIL) ++r;
        sh_cur_rank = r;
      }
      __syncthreads();
      continue;
    }

    // 3b. Sort scratch[0..n_cand) by position (HF leftmost rule).
    //     Single-thread insertion sort — n_cand is small after the first
    //     few iterations. For the initial drain (n_cand ~= n) we pay
    //     O(n²) on one thread, which dominates init cost. For now this is
    //     acceptable; we can swap in CUB BlockRadixSort if init becomes
    //     the bottleneck.
    if (tid == 0) {
      for (int i = 1; i < n_cand; ++i) {
        int v = scratch[i];
        int j = i - 1;
        while (j >= 0 && scratch[j] > v) {
          scratch[j + 1] = scratch[j];
          --j;
        }
        scratch[j + 1] = v;
      }
    }
    __syncthreads();

    // 3c. Non-overlap filter (single-thread): walk left-to-right, keep p
    //     only if its left neighbor wasn't just selected. Overlapping
    //     skipped entries are re-inserted into bucket R (they may still
    //     be valid next time R is drained — although by the schedule
    //     invariant we never drain R again, so these become stale and
    //     are detected as "dead" in their next rank's drain).
    //
    //     Wait — actually R is drained ONLY ONCE. Skipped-because-of-overlap
    //     entries have their LEFT operand merged this iteration, so their
    //     pair (tokens[p], tokens[nxt[p]]) changes (nxt[p] was the right
    //     operand of the left-neighbor merge, now killed; nxt[p] now
    //     points further right, and tokens[p] is unchanged). The skipped
    //     entries' new pair may have a different rank — we re-insert into
    //     bucket R and let stale-on-pop sort it out next time bucket R is
    //     visited. But R isn't visited again unless we DELAY advancing
    //     cur_rank. To handle this, after each drain we re-check whether
    //     bucket R was repopulated; if so, drain again.
    if (tid == 0) {
      int n_sel = 0;
      int last_kept = V2_NIL;
      for (int i = 0; i < n_cand; ++i) {
        int p = scratch[i];
        if (last_kept != V2_NIL && nxt[last_kept] == p) {
          // Overlaps with prior selection — re-insert into bucket R as
          // tentative. Stale-on-pop will validate next time.
          bucket_st[p] = R;
          bucket_n[p]  = bucket_h[R];
          bucket_h[R]  = p;
        } else {
          scratch[n_sel++] = p;
          last_kept = p;
        }
      }
      sh_n_selected = n_sel;
    }
    __syncthreads();

    int n_sel = sh_n_selected;
    if (n_sel == 0) {
      if (tid == 0) {
        // Bucket R may have been re-populated by overlap re-inserts but
        // those are now stale (their left neighbor's pair just changed
        // ... wait, no merges fired this iter because n_sel = 0). If
        // n_sel = 0 and we re-inserted some entries into R, we'd loop
        // forever. So in this branch we must advance even if R is non-empty
        // — but it's only non-empty due to overlaps with positions we
        // never merged, which can't happen if n_sel == 0 (no merges =>
        // no last_kept => no overlap skips). Therefore bucket R is
        // empty here and we can safely advance.
        int r = R + 1;
        while (r < num_merges && bucket_h[r] == V2_NIL) ++r;
        sh_cur_rank = r;
      }
      __syncthreads();
      continue;
    }

    // 3d. Apply merges in parallel. Each selected p:
    //   old_next = nxt[p];                     // operand to be killed
    //   old_old_next = nxt[old_next];          // new right neighbor of p
    //   tokens[p] = new_tok;
    //   nxt[p] = old_old_next; if (old_old_next != NIL) prv[old_old_next] = p
    //   nxt[old_next] = prv[old_next] = DEAD
    //   bucket_st[old_next] = NIL              // no longer in any bucket
    //
    // Non-overlap guarantees the writes from different threads target
    // disjoint positions:
    //   - tokens[p]: each p is unique in selected.
    //   - nxt[p], prv[p]: p is unique.
    //   - nxt[old_next], prv[old_next]: old_next is unique (non-overlap).
    //   - prv[old_old_next] = p: old_old_next might be the next selected
    //     position p'. Then prv[p'] gets written by us, and tokens[p'],
    //     nxt[p'], prv[p'] are also written by p's owner thread. But p's
    //     prv write happens BEFORE the __syncthreads, and p' is processed
    //     concurrently. Conflict on prv[p'].
    //   We resolve this in a two-pass scheme: first apply tokens + nxt
    //   updates + DEAD marks; sync; then update prv pointers.
    for (int i = tid; i < n_sel; i += nthr) {
      int p        = scratch[i];
      int old_next = nxt[p];
      PairKey k    = pack_pair(tokens[p], tokens[old_next]);
      auto it      = map_ref.find(k);
      int new_tok  = unpack_val((*it).second).new_token;
      int ool      = nxt[old_next];

      tokens[p] = new_tok;
      nxt[p]    = ool;
      nxt[old_next] = V2_DEAD;
      bucket_st[old_next] = V2_NIL;
    }
    __syncthreads();

    // Pass 2: update prv pointers. For each selected p, prv[new_next] = p
    // where new_next = nxt[p] (post-pass-1 value).
    for (int i = tid; i < n_sel; i += nthr) {
      int p   = scratch[i];
      int nn  = nxt[p];  // post-pass-1 right neighbor
      // Mark old right operand's prv as DEAD too (we kept prv[old_next]
      // alive in pass 1 because pass 2 might need to read it to find
      // prv[p]'s identity — but it doesn't, since each thread has its own
      // p). Mark dead now.
      // Find old_next via... we don't have it cheap here, but pass 1
      // already set nxt[old_next] = DEAD; we just need prv[old_next] = DEAD.
      // Trick: walk one step right to find it. But nxt[p] now skips
      // old_next, so we can't. Easier: just leave prv[old_next] alone
      // — it'll only be read if someone uses old_next as their right
      // neighbor, which they won't because nxt[old_next] = DEAD signals
      // it's killed; readers check nxt to detect DEAD.
      if (nn != V2_NIL) prv[nn] = p;
    }
    __syncthreads();

    if (tid == 0) sh_live_count -= n_sel;
    __syncthreads();

    // 3e. Insert new bucket entries for merged positions' new pairs.
    //     Each selected p produces up to two new candidate pairs:
    //       left:  (tokens[prv[p]], tokens[p])  — inserted at prv[p]
    //       right: (tokens[p],      tokens[nxt[p]]) — inserted at p
    //     Non-overlap guarantees no two selected p's share prv or p, so
    //     the candidate insert-points are disjoint. But the same position
    //     might be a left-insert target of one merge AND a right-insert
    //     target of another (when two non-overlapping merges meet head-
    //     to-tail). In that case both inserts target the SAME position;
    //     bucket_st atomicCAS ensures at most one succeeds.
    for (int i = tid; i < n_sel; i += nthr) {
      int p = scratch[i];

      // Left pair: rooted at prv[p].
      int left = prv[p];
      if (left != V2_NIL && left != V2_DEAD) {
        int lnb = nxt[left];
        if (lnb != V2_NIL && lnb != V2_DEAD) {
          PairKey k = pack_pair(tokens[left], tokens[lnb]);
          auto it = map_ref.find(k);
          if (it != map_ref.end()) {
            int r = static_cast<int>(unpack_val((*it).second).rank);
            int prev_st = atomicCAS(&bucket_st[left], V2_NIL, r);
            if (prev_st == V2_NIL) {
              int old_head;
              do {
                old_head      = bucket_h[r];
                bucket_n[left] = old_head;
              } while (atomicCAS(&bucket_h[r], old_head, left) != old_head);
            }
            // else: left already enqueued somewhere; stale-on-pop handles it.
          }
        }
      }

      // Right pair: rooted at p.
      int right = nxt[p];
      if (right != V2_NIL && right != V2_DEAD) {
        PairKey k = pack_pair(tokens[p], tokens[right]);
        auto it = map_ref.find(k);
        if (it != map_ref.end()) {
          int r = static_cast<int>(unpack_val((*it).second).rank);
          int prev_st = atomicCAS(&bucket_st[p], V2_NIL, r);
          if (prev_st == V2_NIL) {
            int old_head;
            do {
              old_head   = bucket_h[r];
              bucket_n[p] = old_head;
            } while (atomicCAS(&bucket_h[r], old_head, p) != old_head);
          }
        }
      }
    }
    __syncthreads();

    // 3f. Advance cur_rank. Bucket R may still have entries from overlap
    //     re-inserts in 3c — but those are now stale (their left
    //     neighbor was just merged, so their pair changed). We need to
    //     either re-drain R (handle the stale entries) OR detect that R
    //     should advance.
    //
    //     Simple rule: if bucket R is non-empty, stay; otherwise advance.
    //     A re-drain will call stale-on-pop on those entries and re-route
    //     them to higher ranks.
    if (tid == 0) {
      if (bucket_h[R] != V2_NIL) {
        // stay at R for another pass — drain the re-inserts
      } else {
        int r = R + 1;
        while (r < num_merges && bucket_h[r] == V2_NIL) ++r;
        sh_cur_rank = r;
      }
    }
    __syncthreads();
  }

  // 4. Walk the DLL from head to produce the output sequence.
  if (tid == 0) {
    // Find head: smallest p with nxt[p] != DEAD and prv[p] == NIL.
    int head = 0;
    while (head < n && nxt[head] == V2_DEAD) ++head;
    if (head >= n) {
      d_out_lengths[seq_id] = 0;
      return;
    }
    // (Defensive) head must have prv == NIL since head positions stay anchored
    // at the leftmost live position throughout — we never kill position 0 if
    // it was the left operand of a merge (we kill the RIGHT operand).
    int p = head;
    int cnt = 0;
    while (p != V2_NIL) {
      // Write into d_tokens row (overwriting the in-place buffer is safe
      // because cnt <= p always — we're shifting left).
      d_tokens[static_cast<std::size_t>(seq_id) * T_max + cnt] = tokens[p];
      ++cnt;
      p = nxt[p];
    }
    d_out_lengths[seq_id] = cnt;
  }
}

// --------------------------------------------------------------------------
// Phase 3 v3 — entry-pool bucket scheduling (CORRECT)
// --------------------------------------------------------------------------
//
// v3 fixes v2's correctness bug by allowing a position to be in multiple
// bucket linked lists simultaneously. Each bucket entry is a separate
// node in a global entry pool; positions are no longer their own bucket
// linked-list nodes. New pairs created by merges always insert a fresh
// entry into the correct rank's bucket, regardless of whether the
// position has stale entries elsewhere. Stale entries are discarded by
// rank-revalidation at drain time.
//
// Per-iter bucket-min scan is parallel (O(num_merges / nthr) work per
// thread, then a block-wide min reduction).
//
// Capacity: each position can be inserted at most O(merges-touching-its-
// neighborhood) times, so total entries ≤ ~4*T. We allocate 4*T_max and
// fail fast on overflow.
//
// Correctness is anchored on phase3_reference.py, which is bit-identical
// against the Phase 2 kernel (which is bit-identical against HF) on
// 262/262 sequences up to 16 kbp on two real merge tables.

// Entry pool sizing factor. Each position generates at most ~3-4 bucket
// entries across the entire merge schedule (initial + left-insert +
// right-insert + a small slack). 8× is generous.
#ifndef DNATOK_V3_ENTRY_FACTOR
#define DNATOK_V3_ENTRY_FACTOR 8
#endif

// Per-iter bucket sort uses CUB BlockMergeSort on the candidate array.
// The sort handles up to (DNATOK_BLOCK_SIZE * DNATOK_V3_ITEMS_PER_THREAD)
// items in a single parallel pass; larger buckets fall back to a
// single-thread insertion sort. For T_max ≤ 8 kbp the initial drain at
// rank 0 has at most ~2 kbp candidates (one per ~4 bp on random DNA), so
// 4096 covers the publication-grade benchmark range.
#ifndef DNATOK_V3_ITEMS_PER_THREAD
#define DNATOK_V3_ITEMS_PER_THREAD 16
#endif

__global__ void bpe_v3_kernel(
    PairMapRef                       map_ref,
    const std::uint8_t* __restrict__ d_bytes,
    const std::int32_t* __restrict__ d_byte_offsets,
    const std::int32_t* __restrict__ d_byte_to_id,
    std::int32_t*                    d_tokens,       // [B*T_max]
    std::int32_t*                    d_nxt,          // [B*T_max]
    std::int32_t*                    d_prv,          // [B*T_max]
    std::int32_t*                    d_bucket_h,     // [B*num_merges]
    std::int32_t*                    d_entry_pos,    // [B*entry_pool_size]
    std::int32_t*                    d_entry_next,   // [B*entry_pool_size]
    std::int32_t*                    d_entry_count,  // [B] atomic counter
    std::int32_t*                    d_scratch,      // [B*entry_pool_size]
    std::int32_t*                    d_overflow,     // [B] flag, set if pool overflowed
    std::int32_t*                    d_bucket_bits,  // [B*bits_words] non-empty summary
    std::int32_t* __restrict__       d_out_lengths,  // [B]
    std::int32_t                     T_max,
    std::int32_t                     entry_pool_size,
    std::int32_t                     num_merges,
    std::int32_t                     bits_words)     // = ceil(num_merges/32)
{
  int seq_id = blockIdx.x;
  int tid    = threadIdx.x;
  int nthr   = blockDim.x;

  std::int32_t byte_start = d_byte_offsets[seq_id];
  std::int32_t byte_end   = d_byte_offsets[seq_id + 1];
  std::int32_t n          = byte_end - byte_start;
  if (n <= 0) {
    if (tid == 0) d_out_lengths[seq_id] = 0;
    return;
  }
  if (n > T_max) {
    if (tid == 0) d_out_lengths[seq_id] = -1;
    return;
  }

  std::int32_t* tk = d_tokens     + static_cast<std::size_t>(seq_id) * T_max;
  std::int32_t* nx = d_nxt        + static_cast<std::size_t>(seq_id) * T_max;
  std::int32_t* pv = d_prv        + static_cast<std::size_t>(seq_id) * T_max;
  std::int32_t* bh = d_bucket_h   + static_cast<std::size_t>(seq_id) * num_merges;
  std::int32_t* ep = d_entry_pos  + static_cast<std::size_t>(seq_id) * entry_pool_size;
  std::int32_t* en = d_entry_next + static_cast<std::size_t>(seq_id) * entry_pool_size;
  // Scratch is sized [B, entry_pool_size] to accommodate worst-case
  // bucket walks (single bucket can hold ~3T entries on pathological
  // inputs). Stride = entry_pool_size, NOT T_max.
  std::int32_t* sc   = d_scratch    + static_cast<std::size_t>(seq_id) * entry_pool_size;
  std::int32_t* ec   = d_entry_count + seq_id;
  std::int32_t* ovf  = d_overflow + seq_id;
  // bit-array summary of bh: bits[i] bit j set ⟺ bh[i*32+j] non-empty.
  // 32 ranks per uint32; we cast d_bucket_bits to uint32* for atomicOr.
  unsigned int* bits = reinterpret_cast<unsigned int*>(d_bucket_bits)
                       + static_cast<std::size_t>(seq_id) * bits_words;

  // Step 1: initialize tokens, DLL, bucket heads, counters.
  for (int i = tid; i < n; i += nthr) {
    tk[i] = d_byte_to_id[d_bytes[byte_start + i]];
    nx[i] = (i < n - 1) ? (i + 1) : V2_NIL;
    pv[i] = (i > 0)     ? (i - 1) : V2_NIL;
  }
  for (int r = tid; r < num_merges; r += nthr) bh[r] = V2_NIL;
  for (int w = tid; w < bits_words;  w += nthr) bits[w] = 0u;
  if (tid == 0) { *ec = 0; *ovf = 0; }
  __syncthreads();

  // Step 2: initial bucket fill.
  for (int i = tid; i < n - 1; i += nthr) {
    PairKey k = pack_pair(tk[i], tk[i + 1]);
    auto it = map_ref.find(k);
    if (it != map_ref.end()) {
      int r = static_cast<int>(unpack_val((*it).second).rank);
      int slot = atomicAdd(ec, 1);
      if (slot < entry_pool_size) {
        ep[slot] = i;
        int old_head;
        do {
          old_head = bh[r];
          en[slot] = old_head;
        } while (atomicCAS(&bh[r], old_head, slot) != old_head);
        atomicOr(&bits[r >> 5], 1u << (r & 31));
      } else {
        atomicExch(ovf, 1);
      }
    }
  }
  __syncthreads();

  if (*ovf) {
    if (tid == 0) d_out_lengths[seq_id] = -2;  // signal pool overflow
    return;
  }

  __shared__ int sh_cur_rank;
  __shared__ int sh_n_candidates;
  __shared__ int sh_n_selected;
  __shared__ int sh_red[DNATOK_BLOCK_SIZE];

  // sh_min_insert tracks the lowest rank that step 3e inserts into in
  // the *current* iteration. It bounds the cur_rank scan in step 3f:
  //   * The new lowest non-empty bucket is in [R+1, sh_min_insert].
  //   * We scan only that range (bounded above by sh_min_insert) and
  //     combine with sh_min_insert itself if it's < num_merges.
  // This avoids the full O(num_merges) scan that dominates v3 on
  // tokenisers with large vocabularies (GENA-LM: 32k merges).
  __shared__ int sh_min_insert;

  // CUB BlockMergeSort temp storage. Union with sh_red because the sort
  // and the rank-min reduction are never live simultaneously (sort runs
  // strictly between drain and the next reduction). NOTE: kept SEPARATE
  // here for clarity; the compiler typically packs shared memory
  // efficiently anyway.
  using BlockMergeSortT = cub::BlockMergeSort<int, DNATOK_BLOCK_SIZE,
                                              DNATOK_V3_ITEMS_PER_THREAD>;
  __shared__ typename BlockMergeSortT::TempStorage sort_storage;

  // Initial advance to lowest non-empty bucket — scan the bits array
  // instead of bh directly. Each thread looks at its share of words,
  // contributes the lowest rank it finds.
  {
    int local_min = num_merges;
    for (int w = tid; w < bits_words; w += nthr) {
      unsigned int bw = bits[w];
      if (bw != 0u) {
        int candidate = w * 32 + __ffs(bw) - 1;
        if (candidate < local_min) local_min = candidate;
      }
    }
    sh_red[tid] = local_min;
    __syncthreads();
    for (int stride = nthr / 2; stride > 0; stride >>= 1) {
      if (tid < stride) {
        int a = sh_red[tid], b = sh_red[tid + stride];
        if (b < a) sh_red[tid] = b;
      }
      __syncthreads();
    }
    if (tid == 0) sh_cur_rank = (sh_red[0] < num_merges) ? sh_red[0] : num_merges;
    __syncthreads();
  }

  while (sh_cur_rank < num_merges) {
    int R = sh_cur_rank;
    if (tid == 0) sh_min_insert = num_merges;
    __syncthreads();

    // Step 3a: drain bucket R, collect positions into scratch, validate
    // in parallel. The linked-list walk is unavoidably sequential, but
    // the map.find() per entry is parallelisable — that's the dominant
    // cost in the original single-threaded drain.
    //
    //   Phase A (single-thread): walk the linked-list, dump raw positions
    //   into sc[0..n_raw). Cheap pointer-follows only.
    //
    //   Phase B (parallel): validate each sc[i]. Invalid → mark sc[i] = -1.
    //   Step 3b's sort will move all -1 sentinels to the front; step 3c
    //   skips them.
    if (tid == 0) {
      int slot = bh[R];
      bh[R] = V2_NIL;
      // Clear bit R — bucket is now empty (may be repopulated by step 3e).
      bits[R >> 5] &= ~(1u << (R & 31));
      int cnt = 0;
      // A single bucket can hold up to ~3T entries in pathological cases
      // (initial + 2 inserts/merge across all merges). The scratch
      // buffer is sized to the entry pool capacity (entry_pool_size),
      // matching the worst-case bucket size. The conditional bound is
      // defensive; in measured workloads cnt stays well under T_max.
      while (slot != V2_NIL && cnt < entry_pool_size) {
        sc[cnt++] = ep[slot];
        slot = en[slot];
      }
      if (slot != V2_NIL) {
        // Would have to grow the scratch buffer to handle this case.
        // Set overflow flag to signal the host; kernel exits cleanly.
        atomicExch(ovf, 1);
      }
      sh_n_candidates = cnt;
    }
    __syncthreads();

    if (*ovf) {
      if (tid == 0) d_out_lengths[seq_id] = -2;
      return;
    }

    int n_raw = sh_n_candidates;
    for (int i = tid; i < n_raw; i += nthr) {
      int p = sc[i];
      bool valid = false;
      if (nx[p] != V2_DEAD) {
        int nb = nx[p];
        if (nb != V2_NIL && nb != V2_DEAD) {
          PairKey k = pack_pair(tk[p], tk[nb]);
          auto it = map_ref.find(k);
          if (it != map_ref.end()) {
            int r = static_cast<int>(unpack_val((*it).second).rank);
            if (r == R) valid = true;
          }
        }
      }
      if (!valid) sc[i] = -1;
    }
    __syncthreads();

    int n_cand = n_raw;
    if (n_cand == 0) {
      // No merges fired → no inserts → no entries < R+1 are possible.
      // Scan the bits summary from word(R+1) upward.
      int start_word = (R + 1) >> 5;
      int start_bit  = (R + 1) & 31;
      int local_min  = num_merges;
      // Boundary word: only consider bits >= start_bit.
      if (tid == 0 && start_word < bits_words) {
        unsigned int bw = bits[start_word];
        if (start_bit > 0) bw &= ~((1u << start_bit) - 1u);
        if (bw != 0u) {
          int candidate = start_word * 32 + __ffs(bw) - 1;
          if (candidate < local_min) local_min = candidate;
        }
      }
      // Subsequent words: every thread takes its share.
      for (int w = start_word + 1 + tid; w < bits_words; w += nthr) {
        unsigned int bw = bits[w];
        if (bw != 0u) {
          int candidate = w * 32 + __ffs(bw) - 1;
          if (candidate < local_min) local_min = candidate;
        }
      }
      sh_red[tid] = local_min;
      __syncthreads();
      for (int stride = nthr / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
          int a = sh_red[tid], b = sh_red[tid + stride];
          if (b < a) sh_red[tid] = b;
        }
        __syncthreads();
      }
      if (tid == 0) sh_cur_rank = (sh_red[0] < num_merges) ? sh_red[0] : num_merges;
      __syncthreads();
      continue;
    }

    // Step 3b: sort sc[0..n_cand). After Phase B above, sc has positions
    // mixed with -1 sentinels. Sort puts -1 sentinels at the front and
    // valid positions in ascending order at the back.
    //
    // CUB BlockMergeSort handles up to ITEMS_PER_THREAD * BLOCK_SIZE
    // items in one collective call (default 16 * 256 = 4096). Larger
    // buckets fall back to single-thread insertion sort.
    constexpr int IPT          = DNATOK_V3_ITEMS_PER_THREAD;
    constexpr int SORT_CAP     = IPT * DNATOK_BLOCK_SIZE;
    // Threshold for BlockMergeSort vs single-thread insertion sort. The
    // parallel sort has fixed setup overhead (shared memory init, CUB
    // merge passes) that exceeds the insertion-sort cost for small
    // buckets. Measured crossover on GB10 is around n=64 — below that,
    // insertion sort wins. Use a slightly higher threshold for safety
    // since the parallel path also pays for cross-block sync.
    constexpr int SORT_THRESH  = 96;
    if (n_cand >= SORT_THRESH && n_cand <= SORT_CAP) {
      int my_items[IPT];
      #pragma unroll
      for (int j = 0; j < IPT; ++j) {
        int idx = tid + j * DNATOK_BLOCK_SIZE;
        my_items[j] = (idx < n_cand) ? sc[idx] : 0x7FFFFFFF;
      }
      __syncthreads();
      BlockMergeSortT(sort_storage).Sort(
          my_items,
          [] __device__ (int a, int b) { return a < b; });
      __syncthreads();
      // BlockMergeSort outputs in BLOCKED layout — thread t holds the
      // items at sorted positions [t*IPT, (t+1)*IPT). Write them out.
      #pragma unroll
      for (int j = 0; j < IPT; ++j) {
        int dst = tid * IPT + j;
        if (dst < n_cand) sc[dst] = my_items[j];
      }
      __syncthreads();
    } else if (n_cand >= 2) {
      // Single-thread insertion sort — used for small buckets (< 96
      // candidates, the common case in the tail) AND for very large
      // buckets that exceed SORT_CAP (rare, only when T_max > SORT_CAP).
      if (tid == 0) {
        for (int i = 1; i < n_cand; ++i) {
          int v = sc[i]; int j = i - 1;
          while (j >= 0 && sc[j] > v) { sc[j + 1] = sc[j]; --j; }
          sc[j + 1] = v;
        }
      }
      __syncthreads();
    }
    // (n_cand <= 1: nothing to sort.)

    // Step 3c: single-pass walk over the sorted sc[0..n_cand) that:
    //   - skips -1 sentinels (invalid entries from Phase B).
    //   - dedups consecutive identical positions (entry pool can have
    //     duplicates when adjacent merges insert the same (position,
    //     rank) pair).
    //   - applies the non-overlap filter (skip p if last_kept's right
    //     neighbour in the DLL is p — that would conflict on the shared
    //     operand position).
    // Single-threaded because the filter decision at p depends on the
    // last-kept selection.
    if (tid == 0) {
      int n_sel = 0;
      int last_kept = V2_NIL;
      int prev_p    = V2_NIL;  // for dedup
      for (int i = 0; i < n_cand; ++i) {
        int p = sc[i];
        if (p < 0) continue;            // skip -1 sentinel
        if (p == prev_p) continue;      // skip dup
        prev_p = p;
        if (last_kept != V2_NIL && nx[last_kept] == p) {
          // Overlap with prior selection — skip; p will be killed by it.
          continue;
        }
        sc[n_sel++] = p;
        last_kept = p;
      }
      sh_n_selected = n_sel;
    }
    __syncthreads();
    int n_sel = sh_n_selected;

    // Step 3d: apply merges in two passes.
    //   Pass 1: tokens, nxt, mark right-operand DEAD.
    //   Pass 2: prv pointers for new right neighbors.
    for (int i = tid; i < n_sel; i += nthr) {
      int p = sc[i];
      int old_next = nx[p];
      PairKey k = pack_pair(tk[p], tk[old_next]);
      auto it = map_ref.find(k);
      int new_tok = unpack_val((*it).second).new_token;
      int ool = nx[old_next];

      tk[p] = new_tok;
      nx[p] = ool;
      nx[old_next] = V2_DEAD;
    }
    __syncthreads();

    for (int i = tid; i < n_sel; i += nthr) {
      int p = sc[i];
      int nn = nx[p];
      if (nn != V2_NIL) pv[nn] = p;
    }
    __syncthreads();

    // Step 3e: insert new bucket entries (parallel).
    //   For each selected p: left pair (pv[p]) and right pair (p).
    //   Each insert allocates a fresh entry slot — no per-position
    //   uniqueness constraint, so always-correct even for stale entries.
    //   Each insert also atomicMin's its rank into sh_min_insert so step
    //   3f can bound its scan range.
    for (int i = tid; i < n_sel; i += nthr) {
      int p = sc[i];

      // Left pair: rooted at pv[p].
      int left = pv[p];
      if (left >= 0) {
        int lnb = nx[left];
        if (lnb != V2_NIL && lnb != V2_DEAD) {
          PairKey k = pack_pair(tk[left], tk[lnb]);
          auto it = map_ref.find(k);
          if (it != map_ref.end()) {
            int r = static_cast<int>(unpack_val((*it).second).rank);
            int slot = atomicAdd(ec, 1);
            if (slot < entry_pool_size) {
              ep[slot] = left;
              int old_head;
              do {
                old_head = bh[r];
                en[slot] = old_head;
              } while (atomicCAS(&bh[r], old_head, slot) != old_head);
              atomicOr(&bits[r >> 5], 1u << (r & 31));
              atomicMin(&sh_min_insert, r);
            } else {
              atomicExch(ovf, 1);
            }
          }
        }
      }

      // Right pair: rooted at p.
      int right = nx[p];
      if (right >= 0 && right != V2_DEAD) {
        PairKey k = pack_pair(tk[p], tk[right]);
        auto it = map_ref.find(k);
        if (it != map_ref.end()) {
          int r = static_cast<int>(unpack_val((*it).second).rank);
          int slot = atomicAdd(ec, 1);
          if (slot < entry_pool_size) {
            ep[slot] = p;
            int old_head;
            do {
              old_head = bh[r];
              en[slot] = old_head;
            } while (atomicCAS(&bh[r], old_head, slot) != old_head);
            atomicOr(&bits[r >> 5], 1u << (r & 31));
            atomicMin(&sh_min_insert, r);
          } else {
            atomicExch(ovf, 1);
          }
        }
      }
    }
    __syncthreads();

    if (*ovf) {
      if (tid == 0) d_out_lengths[seq_id] = -2;
      return;
    }

    // Step 3f: advance cur_rank via the bits summary.
    //
    // sh_min_insert is the lowest rank inserted this iter (num_merges if
    // no inserts). It tightens the upper bound on the scan range:
    //   - If sh_min_insert < num_merges, the actual min is in
    //     [R+1, sh_min_insert] — we scan the bits words covering
    //     [R+1, sh_min_insert) and combine with sh_min_insert itself.
    //   - Otherwise the scan covers [R+1, num_merges) over the bits.
    //
    // Word-level scanning means we look at num_merges/32 words instead of
    // num_merges entries — a 32× reduction in memory traffic, decisive
    // for tokenisers with large vocabularies (GENA-LM: 32k merges →
    // 1024 words).
    int K = sh_min_insert;
    int start_word = (R + 1) >> 5;
    int start_bit  = (R + 1) & 31;
    int end_word   = (K < num_merges) ? (K + 31) >> 5 : bits_words;
    if (end_word > bits_words) end_word = bits_words;
    int local_min = (K < num_merges) ? K : num_merges;

    // Boundary word handled by tid 0 (mask off bits < start_bit).
    if (tid == 0 && start_word < end_word) {
      unsigned int bw = bits[start_word];
      if (start_bit > 0) bw &= ~((1u << start_bit) - 1u);
      if (bw != 0u) {
        int candidate = start_word * 32 + __ffs(bw) - 1;
        if (candidate < local_min) local_min = candidate;
      }
    }
    // Subsequent words spread across all threads.
    for (int w = start_word + 1 + tid; w < end_word; w += nthr) {
      unsigned int bw = bits[w];
      if (bw != 0u) {
        int candidate = w * 32 + __ffs(bw) - 1;
        if (candidate < local_min) local_min = candidate;
      }
    }
    sh_red[tid] = local_min;
    __syncthreads();
    for (int stride = nthr / 2; stride > 0; stride >>= 1) {
      if (tid < stride) {
        int a = sh_red[tid], b = sh_red[tid + stride];
        if (b < a) sh_red[tid] = b;
      }
      __syncthreads();
    }
    if (tid == 0) sh_cur_rank = (sh_red[0] < num_merges) ? sh_red[0] : num_merges;
    __syncthreads();
  }

  // Step 4: walk DLL to produce output sequence.
  if (tid == 0) {
    int head = 0;
    while (head < n && nx[head] == V2_DEAD) ++head;
    if (head >= n) { d_out_lengths[seq_id] = 0; return; }
    int p = head;
    int cnt = 0;
    while (p != V2_NIL) {
      d_tokens[static_cast<std::size_t>(seq_id) * T_max + cnt] = tk[p];
      ++cnt;
      p = nx[p];
    }
    d_out_lengths[seq_id] = cnt;
  }
}

// --------------------------------------------------------------------------
// Host-side helpers — build vocab + merge table from a merges.txt file
// --------------------------------------------------------------------------

struct VocabBuild {
  std::unordered_map<std::string, std::int32_t> token_to_id;
  std::vector<std::string>                      id_to_token;
  std::vector<std::pair<PairKey, PairVal>>      pairs;

  std::int32_t add_token(const std::string& s) {
    auto it = token_to_id.find(s);
    if (it != token_to_id.end()) return it->second;
    std::int32_t id = static_cast<std::int32_t>(id_to_token.size());
    id_to_token.push_back(s);
    token_to_id.emplace(s, id);
    return id;
  }
};

VocabBuild build_vocab_from_merges(const std::string& merges_path) {
  VocabBuild v;
  // Seed with 256 byte symbols (printable ASCII == identity; we never need
  // GPT-2's safe-unicode mapping because DNA is pure ASCII).
  for (int b = 0; b < 256; ++b) {
    v.add_token(std::string(1, static_cast<char>(b)));
  }

  std::ifstream in(merges_path);
  if (!in) throw std::runtime_error("Failed to open merges file: " + merges_path);

  std::string line;
  // First line is "#version: ..." per GPT-2 merges.txt convention.
  if (!std::getline(in, line)) throw std::runtime_error("Empty merges file");

  std::uint32_t rank = 0;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::istringstream iss(line);
    std::string a, b;
    if (!(iss >> a >> b)) continue;
    std::int32_t id_a = v.add_token(a);
    std::int32_t id_b = v.add_token(b);
    std::string merged = a + b;
    std::int32_t new_id = v.add_token(merged);
    PairInfo info{rank, new_id};
    v.pairs.emplace_back(pack_pair(id_a, id_b), pack_val(info));
    ++rank;
  }
  return v;
}

// --------------------------------------------------------------------------
// Public class
// --------------------------------------------------------------------------

class DNATokBPE {
public:
  DNATokBPE(const std::string& merges_path,
            std::vector<std::int32_t> byte_to_id_lut,
            std::int32_t              max_iters)
    : max_iters_{max_iters}
  {
    if (byte_to_id_lut.size() != 256) {
      throw std::invalid_argument("byte_to_id_lut must have length 256");
    }
    if (max_iters_ <= 0) max_iters_ = 1024;

    // Build vocab + merge pairs on host.
    vocab_ = build_vocab_from_merges(merges_path);

    // Caller-supplied byte→initial-id LUT. The Python wrapper currently
    // passes the IDENTITY map (byte b → id b) because our internal vocab
    // seeds byte symbols at exactly those slots in build_vocab_from_merges
    // above. We accept a LUT parameter so a future caller can specialise
    // it (e.g. fold non-DNA bytes to UNK before the merge loop sees them).
    DNATOK_CHECK_CUDA(cudaMalloc(&d_byte_to_id_, 256 * sizeof(std::int32_t)));
    DNATOK_CHECK_CUDA(cudaMemcpy(d_byte_to_id_, byte_to_id_lut.data(),
                                  256 * sizeof(std::int32_t),
                                  cudaMemcpyHostToDevice));

    // Build device merge map (cuCollections static_map).
    std::size_t cap = std::max<std::size_t>(static_cast<std::size_t>(vocab_.pairs.size() * 2 + 32),
                                              std::size_t{64});
    map_ = std::make_unique<PairMap>(cap, cuco::empty_key{EMPTY_KEY},
                                      cuco::empty_value{EMPTY_VAL});
    if (!vocab_.pairs.empty()) {
      thrust::device_vector<cuco::pair<PairKey, PairVal>> d_pairs(vocab_.pairs.size());
      std::vector<cuco::pair<PairKey, PairVal>> h_pairs;
      h_pairs.reserve(vocab_.pairs.size());
      for (auto& p : vocab_.pairs) h_pairs.emplace_back(p.first, p.second);
      DNATOK_CHECK_CUDA(cudaMemcpy(thrust::raw_pointer_cast(d_pairs.data()),
                                    h_pairs.data(),
                                    h_pairs.size() * sizeof(cuco::pair<PairKey, PairVal>),
                                    cudaMemcpyHostToDevice));
      map_->insert(d_pairs.begin(), d_pairs.end());
    }
  }

  ~DNATokBPE() {
    // Destructor must not throw — log & swallow any CUDA error.
    if (d_byte_to_id_) {
      cudaError_t err = cudaFree(d_byte_to_id_);
      if (err != cudaSuccess) {
        std::cerr << "[dnatok_bpe] cudaFree(d_byte_to_id_) failed: "
                  << cudaGetErrorString(err) << std::endl;
      }
      d_byte_to_id_ = nullptr;
    }
  }

  std::int32_t vocab_size() const { return static_cast<std::int32_t>(vocab_.id_to_token.size()); }

  const std::string& id_to_token(std::int32_t id) const {
    if (id < 0 || id >= static_cast<std::int32_t>(vocab_.id_to_token.size()))
      throw std::out_of_range("id out of range");
    return vocab_.id_to_token[id];
  }

  // tokenize_batch:
  //   Inputs (host): list of UTF-8 strings (treated as raw byte sequences).
  //   Output: (ids_tensor[B, T_max], lengths_tensor[B]) on the requested device.
  std::pair<torch::Tensor, torch::Tensor>
  tokenize_batch(const std::vector<std::string>& texts) {
    const int B = static_cast<int>(texts.size());
    if (B == 0) {
      return {torch::empty({0, 0}, torch::dtype(torch::kInt32).device(torch::kCUDA)),
              torch::empty({0},   torch::dtype(torch::kInt32).device(torch::kCUDA))};
    }

    // Compute byte offsets and validate T_max.
    int T_max = 0;
    std::vector<std::int32_t> byte_offsets(B + 1, 0);
    for (int i = 0; i < B; ++i) {
      byte_offsets[i + 1] = byte_offsets[i] + static_cast<std::int32_t>(texts[i].size());
      if (static_cast<int>(texts[i].size()) > T_max) T_max = static_cast<int>(texts[i].size());
    }
    if (T_max == 0) T_max = 1;  // at least 1 column so the output tensor has well-defined shape

    // Stage input bytes in one contiguous buffer (pinned for fast H2D).
    std::int32_t total_bytes = byte_offsets[B];
    auto bytes_cpu  = torch::empty({total_bytes}, torch::dtype(torch::kUInt8).pinned_memory(true));
    std::uint8_t* bytes_ptr = bytes_cpu.data_ptr<std::uint8_t>();
    for (int i = 0; i < B; ++i) {
      std::memcpy(bytes_ptr + byte_offsets[i], texts[i].data(), texts[i].size());
    }
    // Use torch::tensor (allocates + copies in one shot) rather than
    // from_blob().clone() so the dependency on the lifetime of
    // byte_offsets is obvious.
    auto offsets_cpu = torch::tensor(byte_offsets, torch::kInt32);

    auto bytes_dev   = bytes_cpu.to(torch::kCUDA, /*non_blocking=*/true);
    auto offsets_dev = offsets_cpu.to(torch::kCUDA, /*non_blocking=*/true);

    // Workspace + outputs (int32 to match HF id space).
    //
    // We cache the workspace tensors as members; the first call
    // (re)allocates, subsequent calls with (B, T_max) ≤ cached capacity
    // hit the cache. The kernel always strides by the *cached* T (call it
    // T_kernel) so its address arithmetic matches the storage layout, and
    // the caller-visible output is narrowed to [B, T_max] at the end.
    //
    // The output tensor's "tail" (positions beyond each sequence's
    // length) must be zero before the kernel runs — the downstream remap
    // LUT indexes it, and any out-of-vocab value would trip a CUDA assert.
    // Zero is the NUL byte symbol, always a valid index.
    ensure_workspace(B, T_max);
    int T_kernel = static_cast<int>(ws_ids_out_.size(1));
    // The kernel only writes to positions [0, len_i) of each row, so zero
    // the visible [0, T_max) range we're about to return. (Positions
    // T_max..T_kernel are not part of the returned narrow view.)
    ws_ids_out_.narrow(0, 0, B).narrow(1, 0, T_max).zero_();
    auto ids_out      = ws_ids_out_.narrow(0, 0, B);
    auto workspace_b  = ws_workspace_b_.narrow(0, 0, B);
    auto ranks_buf    = ws_ranks_buf_.narrow(0, 0, B);
    auto new_tok_buf  = ws_new_tok_buf_.narrow(0, 0, B);
    auto selected_buf = ws_selected_buf_.narrow(0, 0, B);
    auto lengths_out  = ws_lengths_out_.narrow(0, 0, B);

    auto map_ref = map_->ref(cuco::find);
    dim3 grid(B);
    dim3 block(DNATOK_BLOCK_SIZE);
    // Each merge reduces length by exactly 1, so the maximum number of
    // outer iterations to reach convergence is T_max - 1. We give the
    // kernel T_max as the cap (slightly conservative) and respect a
    // caller-supplied higher cap if set.
    int eff_max_iters = std::max(max_iters_, T_max);
    bpe_algorithm1_kernel<<<grid, block>>>(
        map_ref,
        bytes_dev.data_ptr<std::uint8_t>(),
        offsets_dev.data_ptr<std::int32_t>(),
        d_byte_to_id_,
        ids_out.data_ptr<std::int32_t>(),
        workspace_b.data_ptr<std::int32_t>(),
        reinterpret_cast<std::uint32_t*>(ranks_buf.data_ptr<std::int32_t>()),
        new_tok_buf.data_ptr<std::int32_t>(),
        selected_buf.data_ptr<std::uint8_t>(),
        lengths_out.data_ptr<std::int32_t>(),
        T_kernel,        // row stride = cached capacity, matches storage
        eff_max_iters);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      throw std::runtime_error(std::string("dnatok_bpe kernel launch failed: ") +
                                cudaGetErrorString(err));
    }
    // Narrow the output to the user-visible shape [B, T_max]. The slice is
    // a view into the cached storage; the caller may clone if they need
    // independent ownership.
    return {ids_out.narrow(1, 0, T_max), lengths_out};
  }

  // tokenize_batch_v3: Phase 3 entry-pool bucket scheduling — CORRECT.
  //   The v2 path above is left in source as a buggy reference; this is
  //   the one that should be used. Identical I/O contract to
  //   tokenize_batch().
  std::pair<torch::Tensor, torch::Tensor>
  tokenize_batch_v3(const std::vector<std::string>& texts) {
    const int B = static_cast<int>(texts.size());
    if (B == 0) {
      return {torch::empty({0, 0}, torch::dtype(torch::kInt32).device(torch::kCUDA)),
              torch::empty({0},   torch::dtype(torch::kInt32).device(torch::kCUDA))};
    }
    int T_max = 0;
    std::vector<std::int32_t> byte_offsets(B + 1, 0);
    for (int i = 0; i < B; ++i) {
      byte_offsets[i + 1] = byte_offsets[i] + static_cast<std::int32_t>(texts[i].size());
      if (static_cast<int>(texts[i].size()) > T_max) T_max = static_cast<int>(texts[i].size());
    }
    if (T_max == 0) T_max = 1;
    std::int32_t total_bytes = byte_offsets[B];
    auto bytes_cpu  = torch::empty({total_bytes}, torch::dtype(torch::kUInt8).pinned_memory(true));
    std::uint8_t* bytes_ptr = bytes_cpu.data_ptr<std::uint8_t>();
    for (int i = 0; i < B; ++i) {
      std::memcpy(bytes_ptr + byte_offsets[i], texts[i].data(), texts[i].size());
    }
    auto offsets_cpu = torch::tensor(byte_offsets, torch::kInt32);
    auto bytes_dev   = bytes_cpu.to(torch::kCUDA, /*non_blocking=*/true);
    auto offsets_dev = offsets_cpu.to(torch::kCUDA, /*non_blocking=*/true);

    int num_merges = static_cast<int>(vocab_.pairs.size());
    int entry_pool_size = std::max(64, T_max * DNATOK_V3_ENTRY_FACTOR);
    int bits_words = (num_merges + 31) >> 5;
    ensure_workspace_v3(B, T_max, num_merges, entry_pool_size, bits_words);
    int T_kernel = static_cast<int>(ws_v3_tokens_.size(1));
    int E_kernel = static_cast<int>(ws_v3_entry_pos_.size(1));
    int W_kernel = static_cast<int>(ws_v3_bucket_bits_.size(1));
    ws_v3_tokens_.narrow(0, 0, B).narrow(1, 0, T_max).zero_();

    auto tokens_buf    = ws_v3_tokens_.narrow(0, 0, B);
    auto nxt_buf       = ws_v3_nxt_.narrow(0, 0, B);
    auto prv_buf       = ws_v3_prv_.narrow(0, 0, B);
    auto bucket_h_buf  = ws_v3_bucket_h_.narrow(0, 0, B);
    auto entry_pos_buf = ws_v3_entry_pos_.narrow(0, 0, B);
    auto entry_next_buf = ws_v3_entry_next_.narrow(0, 0, B);
    auto entry_cnt_buf  = ws_v3_entry_count_.narrow(0, 0, B);
    auto scratch_buf    = ws_v3_scratch_.narrow(0, 0, B);
    auto overflow_buf   = ws_v3_overflow_.narrow(0, 0, B);
    auto bits_buf       = ws_v3_bucket_bits_.narrow(0, 0, B);
    auto lengths_out    = ws_v3_lengths_out_.narrow(0, 0, B);

    auto map_ref = map_->ref(cuco::find);
    dim3 grid(B);
    dim3 block(DNATOK_BLOCK_SIZE);

    bpe_v3_kernel<<<grid, block>>>(
        map_ref,
        bytes_dev.data_ptr<std::uint8_t>(),
        offsets_dev.data_ptr<std::int32_t>(),
        d_byte_to_id_,
        tokens_buf.data_ptr<std::int32_t>(),
        nxt_buf.data_ptr<std::int32_t>(),
        prv_buf.data_ptr<std::int32_t>(),
        bucket_h_buf.data_ptr<std::int32_t>(),
        entry_pos_buf.data_ptr<std::int32_t>(),
        entry_next_buf.data_ptr<std::int32_t>(),
        entry_cnt_buf.data_ptr<std::int32_t>(),
        scratch_buf.data_ptr<std::int32_t>(),
        overflow_buf.data_ptr<std::int32_t>(),
        bits_buf.data_ptr<std::int32_t>(),
        lengths_out.data_ptr<std::int32_t>(),
        T_kernel,
        E_kernel,
        num_merges,
        W_kernel);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      throw std::runtime_error(std::string("dnatok_bpe v3 kernel launch failed: ") +
                                cudaGetErrorString(err));
    }
    return {tokens_buf.narrow(1, 0, T_max), lengths_out};
  }

  // tokenize_batch_v2: Phase 3 kernel — rank-bucket scheduling.
  //   Identical API to tokenize_batch(); intended for long inputs where
  //   Phase 2's O(T²) tail iteration becomes the bottleneck.
  //   See PHASE3_DESIGN.md and bpe_v2_kernel above for details.
  std::pair<torch::Tensor, torch::Tensor>
  tokenize_batch_v2(const std::vector<std::string>& texts) {
    const int B = static_cast<int>(texts.size());
    if (B == 0) {
      return {torch::empty({0, 0}, torch::dtype(torch::kInt32).device(torch::kCUDA)),
              torch::empty({0},   torch::dtype(torch::kInt32).device(torch::kCUDA))};
    }

    int T_max = 0;
    std::vector<std::int32_t> byte_offsets(B + 1, 0);
    for (int i = 0; i < B; ++i) {
      byte_offsets[i + 1] = byte_offsets[i] + static_cast<std::int32_t>(texts[i].size());
      if (static_cast<int>(texts[i].size()) > T_max) T_max = static_cast<int>(texts[i].size());
    }
    if (T_max == 0) T_max = 1;

    std::int32_t total_bytes = byte_offsets[B];
    auto bytes_cpu  = torch::empty({total_bytes}, torch::dtype(torch::kUInt8).pinned_memory(true));
    std::uint8_t* bytes_ptr = bytes_cpu.data_ptr<std::uint8_t>();
    for (int i = 0; i < B; ++i) {
      std::memcpy(bytes_ptr + byte_offsets[i], texts[i].data(), texts[i].size());
    }
    auto offsets_cpu = torch::tensor(byte_offsets, torch::kInt32);
    auto bytes_dev   = bytes_cpu.to(torch::kCUDA, /*non_blocking=*/true);
    auto offsets_dev = offsets_cpu.to(torch::kCUDA, /*non_blocking=*/true);

    int num_merges = static_cast<int>(vocab_.pairs.size());
    ensure_workspace_v2(B, T_max, num_merges);
    int T_kernel = static_cast<int>(ws_v2_tokens_.size(1));
    // Zero only the visible [0, T_max) tail per row of the output — the
    // kernel writes positions [0, len_i) but the downstream remap LUT
    // indexes the whole [0, T_max). Same convention as Phase 2.
    ws_v2_tokens_.narrow(0, 0, B).narrow(1, 0, T_max).zero_();

    auto tokens_buf    = ws_v2_tokens_.narrow(0, 0, B);
    auto nxt_buf       = ws_v2_nxt_.narrow(0, 0, B);
    auto prv_buf       = ws_v2_prv_.narrow(0, 0, B);
    auto bucket_h_buf  = ws_v2_bucket_h_.narrow(0, 0, B);
    auto bucket_n_buf  = ws_v2_bucket_n_.narrow(0, 0, B);
    auto bucket_st_buf = ws_v2_bucket_st_.narrow(0, 0, B);
    auto scratch_buf   = ws_v2_scratch_.narrow(0, 0, B);
    auto lengths_out   = ws_v2_lengths_out_.narrow(0, 0, B);

    auto map_ref = map_->ref(cuco::find);
    dim3 grid(B);
    dim3 block(DNATOK_BLOCK_SIZE);

    bpe_v2_kernel<<<grid, block>>>(
        map_ref,
        bytes_dev.data_ptr<std::uint8_t>(),
        offsets_dev.data_ptr<std::int32_t>(),
        d_byte_to_id_,
        tokens_buf.data_ptr<std::int32_t>(),
        nxt_buf.data_ptr<std::int32_t>(),
        prv_buf.data_ptr<std::int32_t>(),
        bucket_h_buf.data_ptr<std::int32_t>(),
        bucket_n_buf.data_ptr<std::int32_t>(),
        bucket_st_buf.data_ptr<std::int32_t>(),
        scratch_buf.data_ptr<std::int32_t>(),
        lengths_out.data_ptr<std::int32_t>(),
        T_kernel,
        num_merges);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      throw std::runtime_error(std::string("dnatok_bpe v2 kernel launch failed: ") +
                                cudaGetErrorString(err));
    }
    return {tokens_buf.narrow(1, 0, T_max), lengths_out};
  }

private:
  // Grow workspace tensors so they cover at least (B, T_max). The narrow()
  // views in tokenize_batch slice to the actual requested shape; the
  // underlying storage outlives a single call so we avoid per-call alloc.
  void ensure_workspace(int B, int T_max) {
    auto need_2d_int = [&](torch::Tensor& t) {
      if (!t.defined() || t.size(0) < B || t.size(1) < T_max) {
        int Bh = std::max(B, t.defined() ? static_cast<int>(t.size(0)) : 0);
        int Th = std::max(T_max, t.defined() ? static_cast<int>(t.size(1)) : 0);
        t = torch::empty({Bh, Th}, torch::dtype(torch::kInt32).device(torch::kCUDA));
      }
    };
    auto need_2d_u8 = [&](torch::Tensor& t) {
      if (!t.defined() || t.size(0) < B || t.size(1) < T_max) {
        int Bh = std::max(B, t.defined() ? static_cast<int>(t.size(0)) : 0);
        int Th = std::max(T_max, t.defined() ? static_cast<int>(t.size(1)) : 0);
        t = torch::empty({Bh, Th}, torch::dtype(torch::kUInt8).device(torch::kCUDA));
      }
    };
    auto need_1d_int = [&](torch::Tensor& t) {
      if (!t.defined() || t.size(0) < B) {
        int Bh = std::max(B, t.defined() ? static_cast<int>(t.size(0)) : 0);
        t = torch::empty({Bh}, torch::dtype(torch::kInt32).device(torch::kCUDA));
      }
    };
    need_2d_int(ws_ids_out_);
    need_2d_int(ws_workspace_b_);
    need_2d_int(ws_ranks_buf_);
    need_2d_int(ws_new_tok_buf_);
    need_2d_u8 (ws_selected_buf_);
    need_1d_int(ws_lengths_out_);
  }

  // Workspace for Phase 3 (v2). Allocates the position-indexed buffers
  // (tokens / nxt / prv / bucket_n / bucket_st / scratch) at [Bcap, Tcap],
  // and the rank-indexed bucket_h at [Bcap, num_merges]. The latter
  // doesn't grow as a function of T_max because it's per-rank, not
  // per-position.
  void ensure_workspace_v2(int B, int T_max, int num_merges) {
    auto need_2d_int_T = [&](torch::Tensor& t) {
      if (!t.defined() || t.size(0) < B || t.size(1) < T_max) {
        int Bh = std::max(B, t.defined() ? static_cast<int>(t.size(0)) : 0);
        int Th = std::max(T_max, t.defined() ? static_cast<int>(t.size(1)) : 0);
        t = torch::empty({Bh, Th}, torch::dtype(torch::kInt32).device(torch::kCUDA));
      }
    };
    auto need_2d_int_R = [&](torch::Tensor& t) {
      if (!t.defined() || t.size(0) < B || t.size(1) < num_merges) {
        int Bh = std::max(B, t.defined() ? static_cast<int>(t.size(0)) : 0);
        int Rh = std::max(num_merges, t.defined() ? static_cast<int>(t.size(1)) : 0);
        t = torch::empty({Bh, Rh}, torch::dtype(torch::kInt32).device(torch::kCUDA));
      }
    };
    auto need_1d_int = [&](torch::Tensor& t) {
      if (!t.defined() || t.size(0) < B) {
        int Bh = std::max(B, t.defined() ? static_cast<int>(t.size(0)) : 0);
        t = torch::empty({Bh}, torch::dtype(torch::kInt32).device(torch::kCUDA));
      }
    };
    need_2d_int_T(ws_v2_tokens_);
    need_2d_int_T(ws_v2_nxt_);
    need_2d_int_T(ws_v2_prv_);
    need_2d_int_R(ws_v2_bucket_h_);
    need_2d_int_T(ws_v2_bucket_n_);
    need_2d_int_T(ws_v2_bucket_st_);
    need_2d_int_T(ws_v2_scratch_);
    need_1d_int (ws_v2_lengths_out_);
  }

  // Workspace for Phase 3 v3 (entry-pool bucket scheduling). The entry
  // pool size E grows independently of T; for safety we size at
  // E = max(64, T_max * DNATOK_V3_ENTRY_FACTOR). W = bits_words.
  void ensure_workspace_v3(int B, int T_max, int num_merges, int E, int W) {
    auto need_2d_int_T = [&](torch::Tensor& t) {
      if (!t.defined() || t.size(0) < B || t.size(1) < T_max) {
        int Bh = std::max(B, t.defined() ? static_cast<int>(t.size(0)) : 0);
        int Th = std::max(T_max, t.defined() ? static_cast<int>(t.size(1)) : 0);
        t = torch::empty({Bh, Th}, torch::dtype(torch::kInt32).device(torch::kCUDA));
      }
    };
    auto need_2d_int_R = [&](torch::Tensor& t) {
      if (!t.defined() || t.size(0) < B || t.size(1) < num_merges) {
        int Bh = std::max(B, t.defined() ? static_cast<int>(t.size(0)) : 0);
        int Rh = std::max(num_merges, t.defined() ? static_cast<int>(t.size(1)) : 0);
        t = torch::empty({Bh, Rh}, torch::dtype(torch::kInt32).device(torch::kCUDA));
      }
    };
    auto need_2d_int_E = [&](torch::Tensor& t) {
      if (!t.defined() || t.size(0) < B || t.size(1) < E) {
        int Bh = std::max(B, t.defined() ? static_cast<int>(t.size(0)) : 0);
        int Eh = std::max(E, t.defined() ? static_cast<int>(t.size(1)) : 0);
        t = torch::empty({Bh, Eh}, torch::dtype(torch::kInt32).device(torch::kCUDA));
      }
    };
    auto need_1d_int = [&](torch::Tensor& t) {
      if (!t.defined() || t.size(0) < B) {
        int Bh = std::max(B, t.defined() ? static_cast<int>(t.size(0)) : 0);
        t = torch::empty({Bh}, torch::dtype(torch::kInt32).device(torch::kCUDA));
      }
    };
    auto need_2d_int_W = [&](torch::Tensor& t) {
      if (!t.defined() || t.size(0) < B || t.size(1) < W) {
        int Bh = std::max(B, t.defined() ? static_cast<int>(t.size(0)) : 0);
        int Wh = std::max(W, t.defined() ? static_cast<int>(t.size(1)) : 0);
        t = torch::empty({Bh, Wh}, torch::dtype(torch::kInt32).device(torch::kCUDA));
      }
    };
    need_2d_int_T(ws_v3_tokens_);
    need_2d_int_T(ws_v3_nxt_);
    need_2d_int_T(ws_v3_prv_);
    need_2d_int_R(ws_v3_bucket_h_);
    need_2d_int_E(ws_v3_entry_pos_);
    need_2d_int_E(ws_v3_entry_next_);
    need_1d_int (ws_v3_entry_count_);
    // Scratch sized to E (entry pool capacity) so the worst-case
    // bucket-walk in step 3a can always fit. In measured workloads the
    // largest bucket is ~T/4 (≪ E), but pathological inputs could push
    // a single bucket to ~3T entries; sizing scratch to E covers them.
    need_2d_int_E(ws_v3_scratch_);
    need_1d_int (ws_v3_overflow_);
    need_2d_int_W(ws_v3_bucket_bits_);
    need_1d_int (ws_v3_lengths_out_);
  }

  std::int32_t                max_iters_;
  VocabBuild                  vocab_;
  std::int32_t*               d_byte_to_id_ = nullptr;
  std::unique_ptr<PairMap>    map_;

  // Cached workspace tensors. Empty-by-default; grown lazily by
  // ensure_workspace(). Re-used across calls when the shape fits.
  torch::Tensor ws_ids_out_;       // int32 [Bcap, Tcap]
  torch::Tensor ws_workspace_b_;   // int32 [Bcap, Tcap]
  torch::Tensor ws_ranks_buf_;     // int32 [Bcap, Tcap] (cast to uint32)
  torch::Tensor ws_new_tok_buf_;   // int32 [Bcap, Tcap]
  torch::Tensor ws_selected_buf_;  // uint8 [Bcap, Tcap]
  torch::Tensor ws_lengths_out_;   // int32 [Bcap]

  // Phase 3 v2 workspace tensors (BROKEN — kept only because the kernel
  // is in source for reference).
  torch::Tensor ws_v2_tokens_;
  torch::Tensor ws_v2_nxt_;
  torch::Tensor ws_v2_prv_;
  torch::Tensor ws_v2_bucket_h_;
  torch::Tensor ws_v2_bucket_n_;
  torch::Tensor ws_v2_bucket_st_;
  torch::Tensor ws_v2_scratch_;
  torch::Tensor ws_v2_lengths_out_;

  // Phase 3 v3 workspace tensors (entry-pool bucket scheduling — CORRECT).
  torch::Tensor ws_v3_tokens_;       // int32 [Bcap, Tcap]
  torch::Tensor ws_v3_nxt_;          // int32 [Bcap, Tcap]
  torch::Tensor ws_v3_prv_;          // int32 [Bcap, Tcap]
  torch::Tensor ws_v3_bucket_h_;     // int32 [Bcap, num_merges]
  torch::Tensor ws_v3_entry_pos_;    // int32 [Bcap, Ecap]
  torch::Tensor ws_v3_entry_next_;   // int32 [Bcap, Ecap]
  torch::Tensor ws_v3_entry_count_;  // int32 [Bcap]
  torch::Tensor ws_v3_scratch_;      // int32 [Bcap, Tcap]
  torch::Tensor ws_v3_overflow_;     // int32 [Bcap]
  torch::Tensor ws_v3_bucket_bits_;  // int32 [Bcap, ceil(num_merges/32)]
  torch::Tensor ws_v3_lengths_out_;  // int32 [Bcap]
};

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "DNA-specialised GPU BPE — HF Algorithm-1, global-memory working buffer";
  pybind11::class_<DNATokBPE>(m, "DNATokBPE")
    .def(pybind11::init<const std::string&, std::vector<std::int32_t>, std::int32_t>(),
         pybind11::arg("merges_path"),
         pybind11::arg("byte_to_id_lut"),
         pybind11::arg("max_iters") = 1024)
    .def("vocab_size", &DNATokBPE::vocab_size)
    .def("id_to_token", &DNATokBPE::id_to_token)
    .def("tokenize_batch_v2", &DNATokBPE::tokenize_batch_v2,
         pybind11::arg("texts"),
         "Phase 3 v2 (BROKEN — DO NOT USE). Kept for reference. "
         "See dnatok_bpe.cu bpe_v2_kernel docs.")
    .def("tokenize_batch_v3", &DNATokBPE::tokenize_batch_v3,
         pybind11::arg("texts"),
         "Phase 3 v3: entry-pool bucket scheduling. Identical I/O to "
         "tokenize_batch(); intended to win on long inputs.")
    .def("tokenize_batch", &DNATokBPE::tokenize_batch,
         pybind11::arg("texts"),
         "Encode a batch of byte strings. Returns (ids[B,T_max] int32 cuda, lengths[B] int32 cuda).");
}
