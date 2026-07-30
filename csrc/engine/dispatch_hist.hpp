#pragma once

#include "cudagraph_dispatcher.hpp"

namespace infinilm::engine::dispatch_hist {

/// Record dispatcher mode (FULL / PIECEWISE / NONE + NONE reason). Rank-0 only.
void record_mode(CudaGraphRuntimeMode mode, const char *none_reason);

/// One step-level count for a PIECEWISE run_prefill entry after pad_up / mid_chunk
/// are known. Priority: pad_up > mid_chunk > missing_graph > exact_cg.
void record_piecewise_eager(bool pad_up, bool mid_chunk, bool missing_graph);

/// Legacy noisy dump gated by INFINI_RANK_WORKER_PROFILE / hang_trace.
void log_profile_hist(const char *tag);

/// Periodic / teardown one-line dump gated by INFINI_CG_EAGER_HIST=1.
/// Call after each mode record (period) and on process teardown / RankWorker::close.
void maybe_dump_eager(const char *tag);
void dump_eager_final(const char *tag = "teardown");

/// Ensure atexit final dump is registered (idempotent).
void ensure_atexit_dump();

} // namespace infinilm::engine::dispatch_hist
