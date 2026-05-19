// dnatok_bpe.cu — DNA-specialised GPU BPE kernel.
//
// Implements HF Algorithm-1 BPE (pop the globally lowest-rank applicable
// pair, merge its leftmost occurrence, repeat) with one thread block per
// input sequence. Bit-identical to HuggingFace native on every measured
// genomic BPE tokeniser (DNABERT-2, GENA-LM, METAGENE-1) across input
// lengths from 1 bp up to 32 kbp; correctness gate lives in
// tests/test_gputok_bpe_backend.py.
//
// Key design choices:
//   * Entry-pool bucket scheduling — per-rank linked lists of bucket
//     entries over a single shared pool, allowing a position to appear
//     in multiple buckets concurrently while its pair rank evolves.
//     Stale entries are discarded by re-validation at drain time.
//   * Bit-array summary of non-empty buckets (1 bit per rank, scanned
//     with __ffs) — O(num_merges / 32) min-rank lookup per iteration.
//   * Doubly-linked list of live positions in a global-memory working
//     buffer — no shared-memory chunk cap, so the entire sequence is
//     processed in a single kernel call regardless of length.
//   * CUB BlockMergeSort for in-bucket position sorting (parallel above
//     ~96 candidates; single-thread insertion sort below).
//   * Direct uint8 byte input + int32 CUDA tensor output — no Python
//     list-of-list round-trip and no host-side string allocation.
//
// The initial byte → token-id LUT is the identity map (byte b → id b);
// the kernel's internal vocab seeds 256 byte symbols at ids 0..255 in
// build_vocab_from_merges(), and a remap LUT in the Python wrapper
// translates to the HF tokenizer's id space.
//
// Algorithm reference: BlockBPE paper §3 Algorithm 1
//   https://arxiv.org/pdf/2507.11941
//
// Build:
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
// Block configuration
// --------------------------------------------------------------------------

#ifndef DNATOK_BLOCK_SIZE
#define DNATOK_BLOCK_SIZE 256
#endif

// The cur-rank min-reduction uses a halving-stride pattern that requires
// DNATOK_BLOCK_SIZE to be a power of two. CUB BlockMergeSort is also
// typically tuned for power-of-two block sizes.
static_assert((DNATOK_BLOCK_SIZE & (DNATOK_BLOCK_SIZE - 1)) == 0,
              "DNATOK_BLOCK_SIZE must be a power of two");

// DLL pointer sentinels used throughout the kernel:
//   * NIL_POS  marks "no neighbour" (head/tail of the DLL).
//   * DEAD_POS marks "this position has been merged away" — when the
//     merge at p kills its right operand q, we set nxt[q] = DEAD_POS so
//     subsequent stale bucket-entries for q can be filtered cheaply.
static constexpr std::int32_t NIL_POS  = -1;
static constexpr std::int32_t DEAD_POS = -2;


// --------------------------------------------------------------------------
// BPE kernel — entry-pool bucket scheduling, HF Algorithm-1 semantics
// --------------------------------------------------------------------------
//
// One block per input sequence. Each block maintains:
//   * a doubly-linked list of live positions over the working buffer;
//   * a shared entry pool, where each entry represents "position p has
//     a candidate pair at rank r" — a position can appear in multiple
//     rank buckets concurrently;
//   * per-rank linked-list heads (bh[]) and a bit-array summary (bits[])
//     of which buckets are non-empty, for O(num_merges / 32) min-rank
//     lookup;
//   * CUB BlockMergeSort scratch for sorting candidates by position
//     before the non-overlap filter.
//
// Each outer iteration drains the lowest non-empty rank bucket, applies
// every non-overlapping leftmost merge of that rank in parallel,
// inserts the resulting new pairs into their rank buckets, and advances
// to the next-lowest rank. Stale entries (whose pair changed since
// insertion) are discarded by re-validation at drain time.
//
// Bit-identical to HuggingFace native on every measured BPE genomic
// tokenizer (DNABERT-2 / GENA-LM / METAGENE-1) across inputs up to
// 32 kbp; correctness gate in tests/test_gputok_bpe_backend.py.

// Entry pool sizing factor. Each position generates at most ~3-4 bucket
// entries across the merge schedule (initial + left-insert + right-
// insert + slack). 8× covers worst-case adversarial inputs.
#ifndef DNATOK_BPE_ENTRY_FACTOR
#define DNATOK_BPE_ENTRY_FACTOR 8
#endif

// Per-iter bucket sort uses CUB BlockMergeSort on the candidate array.
// Handles up to (DNATOK_BLOCK_SIZE * DNATOK_BPE_ITEMS_PER_THREAD) items
// in a single parallel pass; larger buckets fall back to single-thread
// insertion sort. For T_max ≤ 8 kbp the initial drain at rank 0 has at
// most ~2 kbp candidates (one per ~4 bp on random DNA), so the default
// 4096 covers the publication-grade benchmark range.
#ifndef DNATOK_BPE_ITEMS_PER_THREAD
#define DNATOK_BPE_ITEMS_PER_THREAD 16
#endif

// Block-wide min reduction with warp-shuffle fast-path.
//
// Input  : per-thread value (local_min).
// Output : the block-wide minimum, broadcast to all threads.
// Uses sh_red as scratch; assumes blockDim.x is a power of two ≥ 32.
//
// vs the straight halving-stride pattern, this saves __syncthreads on the
// last 5 reduction levels (intra-warp) which adds up over the hundreds of
// outer-loop iterations a long sequence produces. The shuffle-based final
// reduction is also a single 5-instruction sequence instead of an explicit
// loop.
__device__ inline int block_min_reduce(int local_min, int* sh_red) {
  const int tid  = threadIdx.x;
  const int nthr = blockDim.x;
  sh_red[tid] = local_min;
  __syncthreads();
  // Cross-warp halving down to one value per warp (32 values).
  for (int stride = nthr / 2; stride >= 32; stride >>= 1) {
    if (tid < stride) {
      int a = sh_red[tid], b = sh_red[tid + stride];
      if (b < a) sh_red[tid] = b;
    }
    __syncthreads();
  }
  // Intra-warp shuffle reduction over the first 32 lanes.
  int v = (tid < 32) ? sh_red[tid] : 0x7FFFFFFF;
  v = min(v, __shfl_xor_sync(0xFFFFFFFF, v, 16));
  v = min(v, __shfl_xor_sync(0xFFFFFFFF, v, 8));
  v = min(v, __shfl_xor_sync(0xFFFFFFFF, v, 4));
  v = min(v, __shfl_xor_sync(0xFFFFFFFF, v, 2));
  v = min(v, __shfl_xor_sync(0xFFFFFFFF, v, 1));
  // Lane 0 of warp 0 has the min; publish via shared memory and
  // broadcast.
  if (tid == 0) sh_red[0] = v;
  __syncthreads();
  return sh_red[0];
}

__global__ void bpe_kernel(
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
    nx[i] = (i < n - 1) ? (i + 1) : NIL_POS;
    pv[i] = (i > 0)     ? (i - 1) : NIL_POS;
  }
  for (int r = tid; r < num_merges; r += nthr) bh[r] = NIL_POS;
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
  // This avoids the full O(num_merges) scan that dominates on tokenisers
  // with large vocabularies (GENA-LM: 32k merges).
  __shared__ int sh_min_insert;

  // CUB BlockMergeSort temp storage. The sort and the rank-min reduction
  // are never live simultaneously (sort runs strictly between drain and
  // the next reduction), so the compiler is free to overlap their
  // shared-memory regions.
  using BlockMergeSortT = cub::BlockMergeSort<int, DNATOK_BLOCK_SIZE,
                                              DNATOK_BPE_ITEMS_PER_THREAD>;
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
    int result = block_min_reduce(local_min, sh_red);
    if (tid == 0) sh_cur_rank = (result < num_merges) ? result : num_merges;
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
      bh[R] = NIL_POS;
      // Clear bit R — bucket is now empty (may be repopulated by step 3e).
      bits[R >> 5] &= ~(1u << (R & 31));
      int cnt = 0;
      // A single bucket can hold up to ~3T entries in pathological cases
      // (initial + 2 inserts/merge across all merges). The scratch
      // buffer is sized to the entry pool capacity (entry_pool_size),
      // matching the worst-case bucket size. The conditional bound is
      // defensive; in measured workloads cnt stays well under T_max.
      while (slot != NIL_POS && cnt < entry_pool_size) {
        sc[cnt++] = ep[slot];
        slot = en[slot];
      }
      if (slot != NIL_POS) {
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
      if (nx[p] != DEAD_POS) {
        int nb = nx[p];
        if (nb != NIL_POS && nb != DEAD_POS) {
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
      int result = block_min_reduce(local_min, sh_red);
      if (tid == 0) sh_cur_rank = (result < num_merges) ? result : num_merges;
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
    constexpr int IPT          = DNATOK_BPE_ITEMS_PER_THREAD;
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
      int last_kept = NIL_POS;
      int prev_p    = NIL_POS;  // for dedup
      for (int i = 0; i < n_cand; ++i) {
        int p = sc[i];
        if (p < 0) continue;            // skip -1 sentinel
        if (p == prev_p) continue;      // skip dup
        prev_p = p;
        if (last_kept != NIL_POS && nx[last_kept] == p) {
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
      nx[old_next] = DEAD_POS;
    }
    __syncthreads();

    for (int i = tid; i < n_sel; i += nthr) {
      int p = sc[i];
      int nn = nx[p];
      if (nn != NIL_POS) pv[nn] = p;
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
        if (lnb != NIL_POS && lnb != DEAD_POS) {
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
      if (right >= 0 && right != DEAD_POS) {
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
    int result = block_min_reduce(local_min, sh_red);
    if (tid == 0) sh_cur_rank = (result < num_merges) ? result : num_merges;
    __syncthreads();
  }

  // Step 4: walk DLL to produce output sequence.
  if (tid == 0) {
    int head = 0;
    while (head < n && nx[head] == DEAD_POS) ++head;
    if (head >= n) { d_out_lengths[seq_id] = 0; return; }
    int p = head;
    int cnt = 0;
    while (p != NIL_POS) {
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


  // tokenize_batch: entry-pool bucket scheduling.
  //   The v2 path above is left in source as a buggy reference; this is
  //   the one that should be used. Identical I/O contract to
  //   tokenize_batch().
  std::pair<torch::Tensor, torch::Tensor>
  tokenize_batch(const std::vector<std::string>& texts) {
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
    int entry_pool_size = std::max(64, T_max * DNATOK_BPE_ENTRY_FACTOR);
    int bits_words = (num_merges + 31) >> 5;
    ensure_workspace(B, T_max, num_merges, entry_pool_size, bits_words);
    int T_kernel = static_cast<int>(ws_tokens_.size(1));
    int E_kernel = static_cast<int>(ws_entry_pos_.size(1));
    int W_kernel = static_cast<int>(ws_bucket_bits_.size(1));
    ws_tokens_.narrow(0, 0, B).narrow(1, 0, T_max).zero_();

    auto tokens_buf    = ws_tokens_.narrow(0, 0, B);
    auto nxt_buf       = ws_nxt_.narrow(0, 0, B);
    auto prv_buf       = ws_prv_.narrow(0, 0, B);
    auto bucket_h_buf  = ws_bucket_h_.narrow(0, 0, B);
    auto entry_pos_buf = ws_entry_pos_.narrow(0, 0, B);
    auto entry_next_buf = ws_entry_next_.narrow(0, 0, B);
    auto entry_cnt_buf  = ws_entry_count_.narrow(0, 0, B);
    auto scratch_buf    = ws_scratch_.narrow(0, 0, B);
    auto overflow_buf   = ws_overflow_.narrow(0, 0, B);
    auto bits_buf       = ws_bucket_bits_.narrow(0, 0, B);
    auto lengths_out    = ws_lengths_out_.narrow(0, 0, B);

    auto map_ref = map_->ref(cuco::find);
    dim3 grid(B);
    dim3 block(DNATOK_BLOCK_SIZE);

    bpe_kernel<<<grid, block>>>(
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
      throw std::runtime_error(std::string("dnatok_bpe kernel launch failed: ") +
                                cudaGetErrorString(err));
    }
    return {tokens_buf.narrow(1, 0, T_max), lengths_out};
  }


private:


  // Workspace for the bucket-scheduling kernel. The entry
  // pool size E grows independently of T; for safety we size at
  // E = max(64, T_max * DNATOK_BPE_ENTRY_FACTOR). W = bits_words.
  void ensure_workspace(int B, int T_max, int num_merges, int E, int W) {
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
    need_2d_int_T(ws_tokens_);
    need_2d_int_T(ws_nxt_);
    need_2d_int_T(ws_prv_);
    need_2d_int_R(ws_bucket_h_);
    need_2d_int_E(ws_entry_pos_);
    need_2d_int_E(ws_entry_next_);
    need_1d_int (ws_entry_count_);
    // Scratch sized to E (entry pool capacity) so the worst-case
    // bucket-walk in step 3a can always fit. In measured workloads the
    // largest bucket is ~T/4 (≪ E), but pathological inputs could push
    // a single bucket to ~3T entries; sizing scratch to E covers them.
    need_2d_int_E(ws_scratch_);
    need_1d_int (ws_overflow_);
    need_2d_int_W(ws_bucket_bits_);
    need_1d_int (ws_lengths_out_);
  }

  std::int32_t                max_iters_;
  VocabBuild                  vocab_;
  std::int32_t*               d_byte_to_id_ = nullptr;
  std::unique_ptr<PairMap>    map_;



  // Workspace tensors. Empty-by-default; grown lazily by
  // ensure_workspace(). Re-used across calls when the shape fits.
  torch::Tensor ws_tokens_;       // int32 [Bcap, Tcap]
  torch::Tensor ws_nxt_;          // int32 [Bcap, Tcap]
  torch::Tensor ws_prv_;          // int32 [Bcap, Tcap]
  torch::Tensor ws_bucket_h_;     // int32 [Bcap, num_merges]
  torch::Tensor ws_entry_pos_;    // int32 [Bcap, Ecap]
  torch::Tensor ws_entry_next_;   // int32 [Bcap, Ecap]
  torch::Tensor ws_entry_count_;  // int32 [Bcap]
  torch::Tensor ws_scratch_;      // int32 [Bcap, Tcap]
  torch::Tensor ws_overflow_;     // int32 [Bcap]
  torch::Tensor ws_bucket_bits_;  // int32 [Bcap, ceil(num_merges/32)]
  torch::Tensor ws_lengths_out_;  // int32 [Bcap]
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
    .def("tokenize_batch", &DNATokBPE::tokenize_batch,
         pybind11::arg("texts"),
         "Encode a batch of byte strings. Returns "
         "(ids[B,T_max] int32 cuda, lengths[B] int32 cuda).");
}
