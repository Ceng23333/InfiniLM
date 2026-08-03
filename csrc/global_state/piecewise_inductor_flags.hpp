#pragma once

#include <cstdlib>
#include <string>

namespace infinilm::global_state {

/// When true, piecewise pre/post segments use AOTInductor kernels (M4 path).
inline bool piecewise_inductor_segment_enabled() {
    const char *v = std::getenv("INFINI_PIECEWISE_INDUCTOR_SEGMENT");
    return v != nullptr && v[0] != '\0' && std::string(v) != "0";
}

/// Production default ON when inductor segment is enabled; set to 0 to bisect.
inline bool scoped_inductor_pre_attn_enabled() {
    const char *v = std::getenv("INFINI_PIECEWISE_SCOPED_INDUCTOR");
    if (v == nullptr || v[0] == '\0') {
        return piecewise_inductor_segment_enabled();
    }
    return v[0] != '0' && std::string(v) != "false";
}

inline bool repro_skip_midchunk_eager() {
    const char *v = std::getenv("INFINI_PIECEWISE_REPRO_SKIP_MIDCHUNK_EAGER");
    return v != nullptr && v[0] == '1' && v[1] == '\0';
}

/// Part A / Gate D bisect: allow exact-width mid-chunk CG replay (final graphs).
/// Keeps allow_inductor_pre_attn=false for mid unless SKIP_MIDCHUNK_EAGER also set.
/// Do not enable in product defaults.
inline bool repro_allow_midchunk_cg() {
    const char *v = std::getenv("INFINI_PIECEWISE_REPRO_ALLOW_MIDCHUNK_CG");
    return v != nullptr && v[0] == '1' && v[1] == '\0';
}

/// MIXED mid dual-capture bisect: force prefer ``compiled_mixed_mid_`` when present
/// even if row layout is not the capture shape (n_req=2 decode 1 + mid~2047).
/// Microbench only — do not enable in product defaults.
inline bool repro_mixed_mid_cg() {
    const char *v = std::getenv("INFINI_PIECEWISE_REPRO_MIXED_MID_CG");
    return v != nullptr && v[0] == '1' && v[1] == '\0';
}

/// Product opt-in for MIXED mid CG replay (default OFF). Capture still runs when
/// native piecewise is on; prefer requires this or REPRO_MIXED_MID_CG until
/// LongBench quality gates pass (see midchunk Phase MIXED mid gate notes).
inline bool enable_mixed_mid_cg() {
    const char *v = std::getenv("INFINI_PIECEWISE_ENABLE_MIXED_MID_CG");
    return v != nullptr && v[0] == '1' && v[1] == '\0';
}

inline bool repro_skip_final_inductor() {
    const char *v = std::getenv("INFINI_PIECEWISE_REPRO_SKIP_FINAL_INDUCTOR");
    return v != nullptr && v[0] == '1' && v[1] == '\0';
}

/// Tail bucket eligible for inductor pre_attn inside hcGraph (B4 only).
inline bool bucket_is_inductor_eligible(size_t bucket) {
    constexpr size_t kInductorTailBucket = 4;
    return bucket == kInductorTailBucket;
}

} // namespace infinilm::global_state
