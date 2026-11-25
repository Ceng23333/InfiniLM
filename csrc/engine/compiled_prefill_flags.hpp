#pragma once

namespace infinilm::engine {
namespace {
} // namespace

/// Bench-frozen production path. Do not reintroduce a deployment policy knob.
inline const char *cudagraph_policy() {
    // Empty is the dispatcher's legacy-policy representation.
    return "";
}

/// Latest bench path: retain the C++ monolithic prefill graph.
inline bool skip_cpp_prefill_graph() {
    return false;
}

/// Latest bench path: capture the monolithic full-forward decode graph.
inline bool skip_monolithic_decode_graph() {
    return false;
}

/// Latest bench path: use the monolithic FULL decode graph.
inline bool native_piecewise_decode_enabled() {
    return false;
}

/// Latest bench path: do not enable native C++ piecewise prefill.
inline bool native_piecewise_prefill_enabled() {
    return false;
}

/// When true, allow paged decode CUDAGraph capture/replay under tensor parallel (tp>1).
inline bool decode_cg_tp_enabled() {
    return false;
}

} // namespace infinilm::engine
