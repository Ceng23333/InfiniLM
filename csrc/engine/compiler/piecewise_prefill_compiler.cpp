#include "piecewise_prefill_compiler.hpp"

#include "../../global_state/ar_profile.hpp"
#include "../../global_state/global_state.hpp"
#include "../../global_state/hang_trace.hpp"
#include "../compiled_prefill_flags.hpp"
#include "../../global_state/piecewise_inductor_flags.hpp"
#include "../../utils.hpp"
#include "../dispatch_hist.hpp"
#include "piecewise_bucket_policy.hpp"
#include "attn_metadata_utils.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <numeric>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <spdlog/spdlog.h>
#include <infinirt.h>
#include "infinicore/context/context.hpp"
#include "infinicore/graph/graph.hpp"
#include "infinicore/ops/inductor_segment.hpp"

namespace infinilm::engine {

namespace {

bool graph_capture_audit_enabled_() {
    const char *v = std::getenv("INFINI_GRAPH_CAPTURE_AUDIT");
    return v != nullptr && v[0] != '\0' && std::string(v) != "0";
}

/// When MoE is device-capturable, post_attn must be a single hcGraph segment
/// (true one-piece: o_proj+AR+LN+MoE). Fail fast under capture audit.
void assert_post_one_device_seg_(
    const char *where,
    size_t layer,
    const std::shared_ptr<infinicore::graph::Graph> &g) {
    if (!graph_capture_audit_enabled_()
        || !infinicore::context::moeTritonCaptureAllowed()
        || !g) {
        return;
    }
    const size_t n = g->device_segment_count();
    if (n == 1) {
        return;
    }
    spdlog::error(
        "[capture_audit] {} layer={} post device_segments={} expected=1 "
        "(MoE capturable — true one-piece post requires a single device seg)",
        where,
        layer,
        n);
    throw std::runtime_error(
        std::string("[capture_audit] post device_segments!=1 when MoE capturable (")
        + where + ")");
}

bool rank_worker_profile_enabled() {
    static int cached = -1;
    if (cached < 0) {
        const char *raw = std::getenv("INFINI_RANK_WORKER_PROFILE");
        cached = (raw != nullptr && raw[0] == '1' && raw[1] == '\0') ? 1 : 0;
    }
    return cached == 1;
}

/// Trim redundant post-allreduce barrier (next layer's pre-graph barrier subsumes it).
/// Default on; opt out for bisect: INFINI_PIECEWISE_KEEP_BARRIERS=1
bool piecewise_trim_barriers() {
    static int cached = -1;
    if (cached < 0) {
        const char *raw = std::getenv("INFINI_PIECEWISE_KEEP_BARRIERS");
        cached = (raw != nullptr && raw[0] == '1' && raw[1] == '\0') ? 0 : 1;
    }
    return cached == 1;
}

bool repro_skip_midchunk_eager() {
    return infinilm::global_state::repro_skip_midchunk_eager();
}

bool repro_allow_midchunk_cg() {
    return infinilm::global_state::repro_allow_midchunk_cg();
}

bool repro_mixed_mid_cg() {
    return infinilm::global_state::repro_mixed_mid_cg();
}

bool enable_mixed_mid_cg() {
    return infinilm::global_state::enable_mixed_mid_cg();
}

bool scoped_inductor_pre_attn() {
    return infinilm::global_state::scoped_inductor_pre_attn_enabled();
}

bool repro_skip_final_inductor() {
    return infinilm::global_state::repro_skip_final_inductor();
}

bool bucket_inductor_capture_enabled(size_t bucket) {
    return infinilm::global_state::piecewise_inductor_segment_enabled()
           && infinilm::global_state::scoped_inductor_pre_attn_enabled()
           && infinilm::global_state::bucket_is_inductor_eligible(bucket);
}

bool layer_capture_inductor_pre_attn(size_t layer, size_t bucket) {
    return bucket_inductor_capture_enabled(bucket)
           && infinicore::op::inductor_segment_impl::has_package(
               infinicore::op::PiecewiseInductorSegmentId::PreAttn, layer, bucket);
}

void verify_inductor_packages_(const std::shared_ptr<InfinilmModel> &model,
                               const std::vector<size_t> &capture_buckets) {
    if (!infinilm::global_state::piecewise_inductor_segment_enabled()) {
        return;
    }
    if (!infinilm::global_state::scoped_inductor_pre_attn_enabled()) {
        return;
    }
    const size_t num_layers = model->native_piecewise_num_layers();
    const int tp_rank = infinilm::global_state::get_tensor_model_parallel_rank();
    for (size_t bucket : capture_buckets) {
        if (!infinilm::global_state::bucket_is_inductor_eligible(bucket)) {
            continue;
        }
        for (size_t layer = 0; layer < num_layers; ++layer) {
            if (!infinicore::op::inductor_segment_impl::has_package(
                    infinicore::op::PiecewiseInductorSegmentId::PreAttn, layer, bucket)) {
                throw std::runtime_error(
                    "piecewise inductor: missing AOT pre_attn package layer="
                    + std::to_string(layer) + " bucket=" + std::to_string(bucket)
                    + " tp_rank=" + std::to_string(tp_rank)
                    + " (compile with aot_compile_piecewise_segments.py --buckets "
                    + std::to_string(bucket) + " --layers all)");
            }
        }
    }
}

double monotonic_ms() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(clock::now().time_since_epoch()).count();
}

size_t device_free_bytes_() {
    const auto device = infinicore::context::getDevice();
    size_t free_b = 0;
    size_t total_b = 0;
    const auto st = infinirtGetMemInfo(static_cast<infiniDevice_t>(device.getType()),
                                       static_cast<int>(device.getIndex()),
                                       &free_b,
                                       &total_b);
    if (st != INFINI_STATUS_SUCCESS) {
        return 0;
    }
    return free_b;
}

size_t tensor_nbytes_(const infinicore::Tensor &t) {
    return t ? t->nbytes() : 0;
}

size_t graph_arena_nbytes_(const std::shared_ptr<infinicore::graph::Graph> &g) {
    return g ? g->capture_arena_bytes() : 0;
}

size_t bucket_arena_nbytes_(const PiecewisePrefillCompiler::BucketGraphs &g) {
    size_t n = graph_arena_nbytes_(g.lm_head);
    for (const auto &seg : g.pre_attn) {
        n += graph_arena_nbytes_(seg);
    }
    for (const auto &seg : g.post_attn) {
        n += graph_arena_nbytes_(seg);
    }
    return n;
}

double bytes_to_gib_(size_t n) {
    return static_cast<double>(n) / (1024.0 * 1024.0 * 1024.0);
}

size_t compute_prefill_len(const InfinilmModel::Input &input) {
    if (input.input_offsets.has_value()) {
        const auto &offsets = input.input_offsets.value();
        const size_t n = offsets->size(0);
        if (n >= 2) {
            auto cpu_offsets = offsets->to(infinicore::Device::cpu());
            const auto *data = reinterpret_cast<const int32_t *>(cpu_offsets->data());
            return static_cast<size_t>(data[n - 1] - data[0]);
        }
    }
    if (input.input_ids.has_value()) {
        return input.input_ids.value()->size(1);
    }
    return 0;
}

void set_attn_metadata(const InfinilmModel::Input &input) {
    attn_metadata_utils::set_attn_metadata(input);
}

void set_attn_metadata_for_varlen_batch(const InfinilmModel::Input &compiled,
                                        const InfinilmModel::Input &runtime) {
    attn_metadata_utils::set_attn_metadata_for_varlen_batch(compiled, runtime);
}

void zero_tensor_tail_seq_(infinicore::Tensor &tensor, size_t valid_len, size_t bucket) {
    if (valid_len >= bucket) {
        return;
    }
    auto tail = tensor->narrow({{1, valid_len, bucket - valid_len}});
    set_zeros(tail);
}

void clear_stale_bucket_tails_(global_state::PiecewisePrefillState &piecewise,
                               infinicore::Tensor &logits_holder,
                               size_t valid_seq_len,
                               size_t bucket) {
    if (valid_seq_len >= bucket) {
        return;
    }
    zero_tensor_tail_seq_(piecewise.hidden_states, valid_seq_len, bucket);
    zero_tensor_tail_seq_(piecewise.residual, valid_seq_len, bucket);
    for (auto &staging : piecewise.layer_staging) {
        zero_tensor_tail_seq_(staging.q_rope, valid_seq_len, bucket);
        zero_tensor_tail_seq_(staging.k_rope, valid_seq_len, bucket);
        zero_tensor_tail_seq_(staging.v_rope, valid_seq_len, bucket);
        zero_tensor_tail_seq_(staging.attn_output, valid_seq_len, bucket);
    }
    if (piecewise.ar_staging) {
        zero_tensor_tail_seq_(piecewise.ar_staging, valid_seq_len, bucket);
    }
    if (logits_holder) {
        zero_tensor_tail_seq_(logits_holder, valid_seq_len, bucket);
    }
}

} // namespace

PiecewisePrefillCompiler::PiecewisePrefillCompiler(const std::shared_ptr<InfinilmModel> &model,
                                                   RankBarrier *barrier)
    : model_(model), barrier_(barrier) {
    enabled_ = native_piecewise_prefill_enabled() && model_->supports_native_piecewise_prefill();
    if (!enabled_) {
        return;
    }
    max_seq_len_ = compile_max_seq_from_env();
    prefill_chunk_size_ = prefill_chunk_size_from_env();
    const size_t chunk_cap = prefill_chunk_size_;
    const bool vllm_ladder = vllm_capture_ladder_enabled();
    if (const char *raw = std::getenv("INFINI_NATIVE_CG_CAPTURE_BUCKETS")) {
        capture_buckets_.clear();
        std::string spec(raw);
        size_t start = 0;
        while (start < spec.size()) {
            const size_t comma = spec.find(',', start);
            const std::string token = spec.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
            if (!token.empty()) {
                capture_buckets_.push_back(static_cast<size_t>(std::stoul(token)));
            }
            if (comma == std::string::npos) {
                break;
            }
            start = comma + 1;
        }
        // Runtime pad ladder: keep full vLLM ladder when enabled; capture list may be a subset
        // (e.g. Qwen3-32B captures B8192 only to avoid multi-bucket CG OOM under TP=4).
        if (vllm_ladder) {
            bs_to_padded_ = build_bs_to_padded_bucket(
                piecewise_compile_buckets_vllm(max_seq_len_, chunk_cap));
        } else {
            bs_to_padded_ = build_bs_to_padded_bucket(capture_buckets_);
        }
    } else if (vllm_ladder) {
        capture_buckets_ = piecewise_capture_buckets_vllm(max_seq_len_, chunk_cap);
        bs_to_padded_ = build_bs_to_padded_bucket(capture_buckets_);
    } else {
        auto pad_ladder = piecewise_compile_buckets(max_seq_len_);
        bs_to_padded_ = build_bs_to_padded_bucket(pad_ladder);
        capture_buckets_ = piecewise_capture_buckets(max_seq_len_);
    }
    std::sort(capture_buckets_.begin(), capture_buckets_.end(), std::greater<size_t>());
}

void PiecewisePrefillCompiler::allocate_shared_banks_(size_t max_bucket,
                                                      size_t num_layers,
                                                      size_t n_req) {
    shared_banks_ = SharedPrefillBanks{};
    shared_banks_.max_bucket = max_bucket;
    shared_banks_.num_layers = num_layers;
    const auto device = infinicore::context::getDevice();
    const auto &model_config = infinilm::global_state::get_infinilm_config().model_config;
    const auto dtype = model_config->get_dtype();
    const size_t hidden = model_config->get<size_t>("hidden_size");
    const size_t vocab_size = model_config->get<size_t>("vocab_size");
    const size_t tp_size = std::max<size_t>(
        1, static_cast<size_t>(infinilm::global_state::get_tensor_model_parallel_world_size()));
    const size_t num_heads = model_config->get<size_t>("num_attention_heads") / static_cast<size_t>(tp_size);
    const size_t total_kv = model_config->get<size_t>("num_key_value_heads");
    const size_t num_kv_heads = total_kv < tp_size ? 1 : total_kv / tp_size;
    const size_t head_dim = model_config->get_head_dim();

    shared_banks_.layer_staging.resize(num_layers);
    for (size_t i = 0; i < num_layers; ++i) {
        auto &st = shared_banks_.layer_staging[i];
        st.q_rope = infinicore::Tensor::empty({1, max_bucket, num_heads, head_dim}, dtype, device);
        st.k_rope = infinicore::Tensor::empty({1, max_bucket, num_kv_heads, head_dim}, dtype, device);
        st.v_rope = infinicore::Tensor::empty({1, max_bucket, num_kv_heads, head_dim}, dtype, device);
        st.attn_output = infinicore::Tensor::empty({1, max_bucket, num_heads * head_dim}, dtype, device);
    }
    shared_banks_.hidden_states = infinicore::Tensor::empty({1, max_bucket, hidden}, dtype, device);
    shared_banks_.residual = infinicore::Tensor::empty({1, max_bucket, hidden}, dtype, device);
    shared_banks_.ar_staging = infinicore::Tensor::empty({1, max_bucket, hidden}, dtype, device);

    shared_banks_.input_ids =
        infinicore::Tensor::empty({1, max_bucket}, infinicore::DataType::I64, device);
    shared_banks_.position_ids =
        infinicore::Tensor::empty({max_bucket}, infinicore::DataType::I64, device);
    shared_banks_.slot_mapping =
        infinicore::Tensor::empty({max_bucket}, infinicore::DataType::I64, device);
    shared_banks_.past_sequence_lengths =
        infinicore::Tensor::empty({n_req}, infinicore::DataType::I32, device);
    shared_banks_.total_sequence_lengths =
        infinicore::Tensor::empty({n_req}, infinicore::DataType::I32, device);
    shared_banks_.input_offsets =
        infinicore::Tensor::empty({n_req + 1}, infinicore::DataType::I32, device);
    shared_banks_.cu_seqlens =
        infinicore::Tensor::empty({n_req + 1}, infinicore::DataType::I32, device);
    shared_banks_.logits_holder =
        infinicore::Tensor::empty({1, max_bucket, vocab_size}, dtype, device);

    size_t staging = tensor_nbytes_(shared_banks_.hidden_states)
                     + tensor_nbytes_(shared_banks_.residual)
                     + tensor_nbytes_(shared_banks_.ar_staging);
    for (const auto &st : shared_banks_.layer_staging) {
        staging += tensor_nbytes_(st.q_rope) + tensor_nbytes_(st.k_rope)
                   + tensor_nbytes_(st.v_rope) + tensor_nbytes_(st.attn_output);
    }
    const size_t io = tensor_nbytes_(shared_banks_.input_ids)
                      + tensor_nbytes_(shared_banks_.position_ids)
                      + tensor_nbytes_(shared_banks_.slot_mapping)
                      + tensor_nbytes_(shared_banks_.past_sequence_lengths)
                      + tensor_nbytes_(shared_banks_.total_sequence_lengths)
                      + tensor_nbytes_(shared_banks_.input_offsets)
                      + tensor_nbytes_(shared_banks_.cu_seqlens)
                      + tensor_nbytes_(shared_banks_.logits_holder);
    shared_banks_.staging_physical_bytes = staging;
    shared_banks_.io_physical_bytes = io;
    spdlog::info(
        "native piecewise CG: shared max-bucket banks max_bucket={} layers={} "
        "staging_GiB={:.3f} io_GiB={:.3f}",
        max_bucket,
        num_layers,
        bytes_to_gib_(staging),
        bytes_to_gib_(io));
}

void PiecewisePrefillCompiler::bind_bucket_staging_(size_t bucket, size_t num_layers) {
    if (!shared_banks_.hidden_states || bucket > shared_banks_.max_bucket) {
        throw std::runtime_error(
            "bind_bucket_staging_: shared banks missing or bucket exceeds max");
    }
    if (num_layers > shared_banks_.num_layers) {
        throw std::runtime_error("bind_bucket_staging_: num_layers exceeds shared bank");
    }
    auto &piecewise = infinilm::global_state::get_forward_context().piecewise;
    piecewise.bucket_seq_len = bucket;
    piecewise.layer_staging.clear();
    piecewise.layer_staging.resize(num_layers);
    for (size_t i = 0; i < num_layers; ++i) {
        const auto &src = shared_banks_.layer_staging[i];
        auto &st = piecewise.layer_staging[i];
        st.q_rope = src.q_rope->narrow({{1, 0, bucket}});
        st.k_rope = src.k_rope->narrow({{1, 0, bucket}});
        st.v_rope = src.v_rope->narrow({{1, 0, bucket}});
        st.attn_output = src.attn_output->narrow({{1, 0, bucket}});
    }
    piecewise.hidden_states = shared_banks_.hidden_states->narrow({{1, 0, bucket}});
    piecewise.residual = shared_banks_.residual->narrow({{1, 0, bucket}});
    piecewise.ar_staging = shared_banks_.ar_staging->narrow({{1, 0, bucket}});
}

InfinilmModel::Input PiecewisePrefillCompiler::make_bucket_input_(size_t bucket,
                                                                  size_t nblocks,
                                                                  size_t n_req,
                                                                  bool mid_chunk_capture) const {
    if (!shared_banks_.input_ids || bucket > shared_banks_.max_bucket) {
        throw std::runtime_error("make_bucket_input_: shared I/O banks not allocated");
    }
    InfinilmModel::Input input;
    input.input_ids = shared_banks_.input_ids->narrow({{1, 0, bucket}});
    input.position_ids = shared_banks_.position_ids->narrow({{0, 0, bucket}});
    input.past_sequence_lengths = shared_banks_.past_sequence_lengths;
    input.total_sequence_lengths = shared_banks_.total_sequence_lengths;
    input.input_offsets = shared_banks_.input_offsets;
    input.cu_seqlens = shared_banks_.cu_seqlens;
    input.slot_mapping = shared_banks_.slot_mapping->narrow({{0, 0, bucket}});
    set_zeros(input.input_ids.value());
    set_zeros(input.past_sequence_lengths.value());
    set_zeros(input.total_sequence_lengths.value());

    const size_t chunk_size =
        prefill_chunk_size_ > 0 ? prefill_chunk_size_ : prefill_chunk_size_from_env();
    // Mid dual-capture: representative continuing chunk after one full prior chunk.
    const size_t past =
        mid_chunk_capture ? (bucket == chunk_size ? bucket : chunk_size) : 0;
    const int64_t pos_start = mid_chunk_capture
        ? static_cast<int64_t>(past)
        : ((bucket < chunk_size) ? static_cast<int64_t>(chunk_size) : int64_t{0});
    std::vector<int64_t> position_ids_vec(bucket);
    std::iota(position_ids_vec.begin(), position_ids_vec.end(), pos_start);
    infinicore::context::memcpyH2D(
        input.position_ids.value()->data(), position_ids_vec.data(), bucket * sizeof(int64_t), false);

    const int32_t per_req = static_cast<int32_t>(bucket / std::max<size_t>(1, n_req));
    std::vector<int32_t> past_lengths_vec(n_req, static_cast<int32_t>(past));
    std::vector<int32_t> total_lengths_vec(
        n_req, static_cast<int32_t>(past) + per_req);
    infinicore::context::memcpyH2D(
        input.past_sequence_lengths.value()->data(),
        past_lengths_vec.data(),
        n_req * sizeof(int32_t),
        false);
    infinicore::context::memcpyH2D(
        input.total_sequence_lengths.value()->data(),
        total_lengths_vec.data(),
        n_req * sizeof(int32_t),
        false);

    std::vector<int32_t> input_offsets_vec(n_req + 1, 0);
    for (size_t i = 0; i <= n_req; ++i) {
        input_offsets_vec[i] = static_cast<int32_t>(std::min<size_t>(bucket, i * per_req));
    }
    input_offsets_vec[n_req] = static_cast<int32_t>(bucket);
    infinicore::context::memcpyH2D(
        input.input_offsets.value()->data(),
        input_offsets_vec.data(),
        (n_req + 1) * sizeof(int32_t),
        false);

    std::vector<int32_t> cu_seqlens_vec(n_req + 1, 0);
    for (size_t i = 0; i <= n_req; ++i) {
        cu_seqlens_vec[i] = static_cast<int32_t>(std::min<size_t>(bucket, i * per_req));
    }
    cu_seqlens_vec[n_req] = static_cast<int32_t>(bucket);
    infinicore::context::memcpyH2D(
        input.cu_seqlens.value()->data(),
        cu_seqlens_vec.data(),
        (n_req + 1) * sizeof(int32_t),
        false);

    const size_t block_per_req = nblocks;
    input.block_tables = block_tables_holder_->as_strided({n_req, block_per_req}, {(ptrdiff_t)block_per_req, 1});
    const auto *paged_config = dynamic_cast<const cache::PagedKVCacheConfig *>(model_->get_cache_config());
    const size_t block_size = paged_config != nullptr ? paged_config->block_size() : 256;
    const size_t span_tokens = past + bucket;
    const size_t blocks_needed = (span_tokens + block_size - 1) / block_size;
    for (size_t row = 0; row < n_req; ++row) {
        std::vector<int32_t> block_row(block_per_req, -1);
        const size_t row_offset = row * blocks_needed;
        for (size_t b = 0; b < blocks_needed && b < block_per_req; ++b) {
            block_row[b] = static_cast<int32_t>(row_offset + b);
        }
        auto row_tensor = input.block_tables.value()->narrow({{0, row, 1}});
        infinicore::context::memcpyH2D(
            row_tensor->data(), block_row.data(), block_per_req * sizeof(int32_t), false);
    }

    std::vector<int64_t> slot_mapping_vec(bucket);
    std::iota(slot_mapping_vec.begin(),
              slot_mapping_vec.end(),
              static_cast<int64_t>(past));
    infinicore::context::memcpyH2D(
        input.slot_mapping.value()->data(), slot_mapping_vec.data(), bucket * sizeof(int64_t), false);
    // Mid capture is a continuing chunk (not final); leave flag empty → treated as final for
    // lm_head dry-run only when we explicitly run lm_head below. Runtime mid uses is_final=false.
    return input;
}

InfinilmModel::Input PiecewisePrefillCompiler::make_mixed_mid_bucket_input_(size_t bucket,
                                                                           size_t nblocks) const {
    if (!shared_banks_.input_ids || bucket > shared_banks_.max_bucket) {
        throw std::runtime_error("make_mixed_mid_bucket_input_: shared I/O banks not allocated");
    }
    if (bucket < 2) {
        throw std::runtime_error("make_mixed_mid_bucket_input_: bucket must be >= 2");
    }
    const size_t n_req = mixed_mid_capture_req_;
    // Match v1 scheduler RUNNING order: decode rows often precede mid-prefills.
    // Dominant LongBench MIXED@2048: 1 decode token + continuing mid fills bucket-1.
    const size_t decode_q = 1;
    const size_t mid_q = bucket - 1;
    const size_t decode_past = bucket;
    const size_t mid_past = bucket;
    const size_t decode_total = decode_past + decode_q;
    const size_t mid_total = mid_past + mid_q;

    InfinilmModel::Input input;
    input.input_ids = shared_banks_.input_ids->narrow({{1, 0, bucket}});
    input.position_ids = shared_banks_.position_ids->narrow({{0, 0, bucket}});
    // Narrow metadata to capture n_req so FA max_seqlens do not see zero-pad tails.
    input.past_sequence_lengths = shared_banks_.past_sequence_lengths->narrow({{0, 0, n_req}});
    input.total_sequence_lengths = shared_banks_.total_sequence_lengths->narrow({{0, 0, n_req}});
    input.input_offsets = shared_banks_.input_offsets->narrow({{0, 0, n_req + 1}});
    input.cu_seqlens = shared_banks_.cu_seqlens->narrow({{0, 0, n_req + 1}});
    input.slot_mapping = shared_banks_.slot_mapping->narrow({{0, 0, bucket}});
    set_zeros(input.input_ids.value());
    set_zeros(input.past_sequence_lengths.value());
    set_zeros(input.total_sequence_lengths.value());

    std::vector<int64_t> position_ids_vec(bucket);
    position_ids_vec[0] = static_cast<int64_t>(decode_past);
    for (size_t i = 0; i < mid_q; ++i) {
        position_ids_vec[decode_q + i] = static_cast<int64_t>(mid_past + i);
    }
    infinicore::context::memcpyH2D(
        input.position_ids.value()->data(), position_ids_vec.data(), bucket * sizeof(int64_t), false);

    const std::vector<int32_t> past_lengths_vec{
        static_cast<int32_t>(decode_past), static_cast<int32_t>(mid_past)};
    const std::vector<int32_t> total_lengths_vec{
        static_cast<int32_t>(decode_total), static_cast<int32_t>(mid_total)};
    infinicore::context::memcpyH2D(
        input.past_sequence_lengths.value()->data(),
        past_lengths_vec.data(),
        n_req * sizeof(int32_t),
        false);
    infinicore::context::memcpyH2D(
        input.total_sequence_lengths.value()->data(),
        total_lengths_vec.data(),
        n_req * sizeof(int32_t),
        false);

    const std::vector<int32_t> input_offsets_vec{
        0, static_cast<int32_t>(decode_q), static_cast<int32_t>(bucket)};
    // cu_seqlens = cumulative KV lengths (matches processor MIXED packing).
    const std::vector<int32_t> cu_seqlens_vec{
        0, static_cast<int32_t>(decode_total), static_cast<int32_t>(decode_total + mid_total)};
    infinicore::context::memcpyH2D(
        input.input_offsets.value()->data(),
        input_offsets_vec.data(),
        (n_req + 1) * sizeof(int32_t),
        false);
    infinicore::context::memcpyH2D(
        input.cu_seqlens.value()->data(),
        cu_seqlens_vec.data(),
        (n_req + 1) * sizeof(int32_t),
        false);

    const size_t block_per_req = nblocks;
    input.block_tables = block_tables_holder_->as_strided({n_req, block_per_req}, {(ptrdiff_t)block_per_req, 1});
    const auto *paged_config = dynamic_cast<const cache::PagedKVCacheConfig *>(model_->get_cache_config());
    const size_t block_size = paged_config != nullptr ? paged_config->block_size() : 256;
    const size_t spans[2] = {decode_total, mid_total};
    size_t row_block_cursor = 0;
    for (size_t row = 0; row < n_req; ++row) {
        std::vector<int32_t> block_row(block_per_req, -1);
        const size_t blocks_needed = (spans[row] + block_size - 1) / block_size;
        for (size_t b = 0; b < blocks_needed && b < block_per_req; ++b) {
            block_row[b] = static_cast<int32_t>(row_block_cursor + b);
        }
        row_block_cursor += blocks_needed;
        auto row_tensor = input.block_tables.value()->narrow({{0, row, 1}});
        infinicore::context::memcpyH2D(
            row_tensor->data(), block_row.data(), block_per_req * sizeof(int32_t), false);
    }

    std::vector<int64_t> slot_mapping_vec(bucket);
    slot_mapping_vec[0] = static_cast<int64_t>(decode_past);
    for (size_t i = 0; i < mid_q; ++i) {
        slot_mapping_vec[decode_q + i] = static_cast<int64_t>(mid_past + i);
    }
    infinicore::context::memcpyH2D(
        input.slot_mapping.value()->data(), slot_mapping_vec.data(), bucket * sizeof(int64_t), false);

    // Row0 decode (lm_head + sample); row1 continuing mid (no sample).
    input.is_final_prefill_chunk = {true, false};
    return input;
}

void PiecewisePrefillCompiler::capture_bucket_(size_t bucket, bool mid_chunk_capture) {
    const auto rank_device = infinilm::global_state::get_tensor_model_parallel_rank_info().device;
    infinicore::context::setDevice(rank_device);
    // Prefill: FA host-break + AOT MoE under full_and_piecewise.
    infinicore::context::InferencePhaseGuard phase_guard(
        infinicore::context::InferencePhase::Prefill);

    auto &piecewise_flag = infinilm::global_state::get_forward_context().piecewise;
    struct CaptureGuard {
        infinilm::global_state::PiecewisePrefillState &pw;
        explicit CaptureGuard(infinilm::global_state::PiecewisePrefillState &p) : pw(p) { pw.compile_capture_active = true; }
        ~CaptureGuard() { pw.compile_capture_active = false; }
    } capture_guard(piecewise_flag);

    const size_t nblocks = dynamic_cast<const cache::PagedKVCacheConfig *>(model_->get_cache_config())->num_blocks();
    const size_t num_layers = model_->native_piecewise_num_layers();
    bind_bucket_staging_(bucket, num_layers);
    auto bucket_input = make_bucket_input_(bucket, nblocks, max_capture_req_, mid_chunk_capture);
    set_attn_metadata(bucket_input);

    BucketGraphs graphs;
    graphs.input = std::move(bucket_input);

    auto &piecewise = infinilm::global_state::get_forward_context().piecewise;
    piecewise.valid_seq_len = bucket;
    // Mid dual-capture: never bake inductor-in-CG (A2 remains opt-in via SKIP_MIDCHUNK_EAGER).
    piecewise.allow_inductor_pre_attn =
        mid_chunk_capture ? false : bucket_inductor_capture_enabled(bucket);
    piecewise.phase = global_state::PiecewiseCapturePhase::None;

    auto &hidden = piecewise.hidden_states;
    auto &residual = piecewise.residual;

    size_t capture_layers = num_layers;
    if (const char *raw = std::getenv("INFINI_NATIVE_CG_MAX_LAYERS")) {
        capture_layers = std::min(num_layers, static_cast<size_t>(std::stoul(raw)));
    }

    graphs.logits_holder = shared_banks_.logits_holder->narrow({{1, 0, bucket}});

    // Eager warmup dry-run before capture (NONE-equivalent).
    // Dry-run / capture use valid_seq_len==bucket (exact). Runtime mid-chunk
    // with seq_len==graph_bucket shares this staging; pad-up does not.
    model_->native_piecewise_embed(graphs.input, hidden);
    for (size_t layer = 0; layer < capture_layers; ++layer) {
        model_->native_piecewise_pre_attn_layer(layer, graphs.input, hidden, residual);
        if (infinilm::global_state::piecewise_inductor_segment_enabled()) {
            infinicore::context::syncDevice();
            barrier_->wait("piecewise_dry_run_pre_attn");
        }
        model_->native_piecewise_eager_attn_layer(layer, graphs.input);
        if (infinilm::global_state::piecewise_inductor_segment_enabled()) {
            infinicore::context::syncDevice();
            barrier_->wait("piecewise_dry_run_eager_attn");
        }
        model_->native_piecewise_post_attn_cg_layer(layer, graphs.input, hidden, residual);
        if (infinilm::global_state::piecewise_inductor_segment_enabled()) {
            infinicore::context::syncDevice();
            barrier_->wait("piecewise_dry_run_post_attn");
        }
        if (infinilm::global_state::piecewise_inductor_segment_enabled()) {
            infinicore::context::syncDevice();
        }
    }
    // Mid chunks do not sample; skip lm_head dry-run/capture for mid key.
    if (!mid_chunk_capture) {
        model_->native_piecewise_lm_head(graphs.input, hidden, residual, graphs.logits_holder);
    }
    graphs.pre_attn.resize(capture_layers);
    graphs.post_attn.resize(capture_layers);

    set_zeros(piecewise.residual);
    model_->native_piecewise_embed(graphs.input, hidden);

    for (size_t layer = 0; layer < capture_layers; ++layer) {
        piecewise.active_layer = layer;
        piecewise.phase = global_state::PiecewiseCapturePhase::PreAttn;

        barrier_->wait("piecewise_capture_pre_attn");
        const bool capture_inductor =
            !mid_chunk_capture && layer_capture_inductor_pre_attn(layer, bucket);
        piecewise.allow_inductor_pre_attn = capture_inductor;
        infinicore::context::startGraphRecording();
        model_->native_piecewise_pre_attn_layer(layer, graphs.input, hidden, residual);
        graphs.pre_attn[layer] = infinicore::context::stopGraphRecording();
        infinicore::context::syncStream();

        piecewise.phase = global_state::PiecewiseCapturePhase::EagerAttn;
        model_->native_piecewise_eager_attn_layer(layer, graphs.input);

        piecewise.phase = global_state::PiecewiseCapturePhase::PostAttn;
        barrier_->wait("piecewise_capture_post_attn");
        infinicore::context::startGraphRecording();
        model_->native_piecewise_post_attn_cg_layer(layer, graphs.input, hidden, residual);
        graphs.post_attn[layer] = infinicore::context::stopGraphRecording();
        assert_post_one_device_seg_("piecewise_prefill_post", layer, graphs.post_attn[layer]);
        barrier_->wait("piecewise_capture_post_attn_sync");
    }

    if (!mid_chunk_capture) {
        piecewise.phase = global_state::PiecewiseCapturePhase::LmHead;
        barrier_->wait("piecewise_capture_lm_head");
        infinicore::context::startGraphRecording();
        model_->native_piecewise_lm_head(graphs.input, hidden, residual, graphs.logits_holder);
        graphs.lm_head = infinicore::context::stopGraphRecording();
        barrier_->wait("piecewise_capture_lm_head_sync");
    }

    piecewise.phase = global_state::PiecewiseCapturePhase::None;
    graphs.hidden_states = piecewise.hidden_states;
    graphs.residual = piecewise.residual;
    graphs.ar_staging = piecewise.ar_staging;
    graphs.layer_staging = piecewise.layer_staging;
    if (mid_chunk_capture) {
        compiled_mid_[bucket] = std::move(graphs);
    } else {
        compiled_[bucket] = std::move(graphs);
    }
    const size_t captured_segments =
        capture_layers * 2 + (mid_chunk_capture ? 0 : 1);
    spdlog::info(
        "native piecewise CG: captured bucket={} mid={} layers={} segments={}",
        bucket,
        mid_chunk_capture,
        capture_layers,
        captured_segments);
}

void PiecewisePrefillCompiler::capture_mixed_mid_bucket_(size_t bucket) {
    const auto rank_device = infinilm::global_state::get_tensor_model_parallel_rank_info().device;
    infinicore::context::setDevice(rank_device);
    infinicore::context::InferencePhaseGuard phase_guard(
        infinicore::context::InferencePhase::Prefill);

    auto &piecewise_flag = infinilm::global_state::get_forward_context().piecewise;
    struct CaptureGuard {
        infinilm::global_state::PiecewisePrefillState &pw;
        explicit CaptureGuard(infinilm::global_state::PiecewisePrefillState &p) : pw(p) {
            pw.compile_capture_active = true;
        }
        ~CaptureGuard() { pw.compile_capture_active = false; }
    } capture_guard(piecewise_flag);

    const size_t nblocks =
        dynamic_cast<const cache::PagedKVCacheConfig *>(model_->get_cache_config())->num_blocks();
    const size_t num_layers = model_->native_piecewise_num_layers();
    bind_bucket_staging_(bucket, num_layers);
    auto bucket_input = make_mixed_mid_bucket_input_(bucket, nblocks);
    set_attn_metadata(bucket_input);

    BucketGraphs graphs;
    graphs.input = std::move(bucket_input);

    auto &piecewise = infinilm::global_state::get_forward_context().piecewise;
    piecewise.valid_seq_len = bucket;
    // MIXED mid dual-capture: never bake inductor-in-CG (same as homo mid).
    piecewise.allow_inductor_pre_attn = false;
    piecewise.phase = global_state::PiecewiseCapturePhase::None;

    auto &hidden = piecewise.hidden_states;
    auto &residual = piecewise.residual;

    size_t capture_layers = num_layers;
    if (const char *raw = std::getenv("INFINI_NATIVE_CG_MAX_LAYERS")) {
        capture_layers = std::min(num_layers, static_cast<size_t>(std::stoul(raw)));
    }

    graphs.logits_holder = shared_banks_.logits_holder->narrow({{1, 0, bucket}});

    // Eager warmup dry-run (includes lm_head — decode row needs logits).
    model_->native_piecewise_embed(graphs.input, hidden);
    for (size_t layer = 0; layer < capture_layers; ++layer) {
        model_->native_piecewise_pre_attn_layer(layer, graphs.input, hidden, residual);
        if (infinilm::global_state::piecewise_inductor_segment_enabled()) {
            infinicore::context::syncDevice();
            barrier_->wait("piecewise_dry_run_pre_attn");
        }
        model_->native_piecewise_eager_attn_layer(layer, graphs.input);
        if (infinilm::global_state::piecewise_inductor_segment_enabled()) {
            infinicore::context::syncDevice();
            barrier_->wait("piecewise_dry_run_eager_attn");
        }
        model_->native_piecewise_post_attn_cg_layer(layer, graphs.input, hidden, residual);
        if (infinilm::global_state::piecewise_inductor_segment_enabled()) {
            infinicore::context::syncDevice();
            barrier_->wait("piecewise_dry_run_post_attn");
        }
        if (infinilm::global_state::piecewise_inductor_segment_enabled()) {
            infinicore::context::syncDevice();
        }
    }
    model_->native_piecewise_lm_head(graphs.input, hidden, residual, graphs.logits_holder);
    graphs.pre_attn.resize(capture_layers);
    graphs.post_attn.resize(capture_layers);

    set_zeros(piecewise.residual);
    model_->native_piecewise_embed(graphs.input, hidden);

    for (size_t layer = 0; layer < capture_layers; ++layer) {
        piecewise.active_layer = layer;
        piecewise.phase = global_state::PiecewiseCapturePhase::PreAttn;

        barrier_->wait("piecewise_capture_pre_attn");
        piecewise.allow_inductor_pre_attn = false;
        infinicore::context::startGraphRecording();
        model_->native_piecewise_pre_attn_layer(layer, graphs.input, hidden, residual);
        graphs.pre_attn[layer] = infinicore::context::stopGraphRecording();
        infinicore::context::syncStream();

        piecewise.phase = global_state::PiecewiseCapturePhase::EagerAttn;
        model_->native_piecewise_eager_attn_layer(layer, graphs.input);

        piecewise.phase = global_state::PiecewiseCapturePhase::PostAttn;
        barrier_->wait("piecewise_capture_post_attn");
        infinicore::context::startGraphRecording();
        model_->native_piecewise_post_attn_cg_layer(layer, graphs.input, hidden, residual);
        graphs.post_attn[layer] = infinicore::context::stopGraphRecording();
        assert_post_one_device_seg_("piecewise_prefill_post", layer, graphs.post_attn[layer]);
        barrier_->wait("piecewise_capture_post_attn_sync");
    }

    piecewise.phase = global_state::PiecewiseCapturePhase::LmHead;
    barrier_->wait("piecewise_capture_lm_head");
    infinicore::context::startGraphRecording();
    model_->native_piecewise_lm_head(graphs.input, hidden, residual, graphs.logits_holder);
    graphs.lm_head = infinicore::context::stopGraphRecording();
    barrier_->wait("piecewise_capture_lm_head_sync");

    piecewise.phase = global_state::PiecewiseCapturePhase::None;
    graphs.hidden_states = piecewise.hidden_states;
    graphs.residual = piecewise.residual;
    graphs.ar_staging = piecewise.ar_staging;
    graphs.layer_staging = piecewise.layer_staging;
    compiled_mixed_mid_[bucket] = std::move(graphs);
    const size_t captured_segments = capture_layers * 2 + 1;
    spdlog::info(
        "native piecewise CG: captured bucket={} mixed_mid=1 n_req={} layers={} segments={}",
        bucket,
        mixed_mid_capture_req_,
        capture_layers,
        captured_segments);
}

void PiecewisePrefillCompiler::warmup_inductor_segments_(size_t nblocks, size_t n_req) {
    if (!infinilm::global_state::piecewise_inductor_segment_enabled()) {
        return;
    }
    const size_t num_layers = model_->native_piecewise_num_layers();
    if (num_layers == 0 || capture_buckets_.empty()) {
        return;
    }
    size_t warmed_buckets = 0;
    for (size_t bucket : capture_buckets_) {
        if (!infinilm::global_state::bucket_is_inductor_eligible(bucket)) {
            continue;
        }
        bind_bucket_staging_(bucket, num_layers);
        auto bucket_input = make_bucket_input_(bucket, nblocks, n_req);
        const auto &positions = bucket_input.position_ids.value();
        auto positions_padded = infinicore::Tensor::zeros(
            {1, bucket}, infinicore::DataType::I64, infinicore::context::getDevice());
        auto &piecewise = infinilm::global_state::get_forward_context().piecewise;
        for (size_t layer = 0; layer < num_layers; ++layer) {
            if (!infinicore::op::inductor_segment_impl::has_package(
                    infinicore::op::PiecewiseInductorSegmentId::PreAttn, layer, bucket)) {
                continue;
            }
            infinicore::op::inductor_warmup_pre_attn_bucket(
                positions,
                positions_padded,
                piecewise.hidden_states,
                piecewise.residual,
                layer,
                bucket,
                bucket);
            infinicore::context::syncDevice();
        }
        ++warmed_buckets;
    }
    if (warmed_buckets > 0) {
        auto &piecewise = infinilm::global_state::get_forward_context().piecewise;
        piecewise.layer_staging.clear();
        piecewise.hidden_states = infinicore::Tensor();
        piecewise.residual = infinicore::Tensor();
        piecewise.ar_staging = infinicore::Tensor();
        infinicore::context::setDevice(
            infinilm::global_state::get_tensor_model_parallel_rank_info().device);
        infinicore::context::syncDevice();
        spdlog::info(
            "piecewise inductor: eager AOT warmup buckets={} (per-rank, before CG capture)",
            warmed_buckets);
    }
}

void PiecewisePrefillCompiler::compile() {
    if (!native_piecewise_prefill_enabled() || !model_->supports_native_piecewise_prefill()) {
        enabled_ = false;
        return;
    }
    if (model_->get_cache_config() == nullptr
        || dynamic_cast<const cache::PagedKVCacheConfig *>(model_->get_cache_config()) == nullptr) {
        spdlog::debug("PiecewisePrefillCompiler: defer capture until paged cache is configured");
        return;
    }
    enabled_ = true;

    const auto *paged_config =
        dynamic_cast<const cache::PagedKVCacheConfig *>(model_->get_cache_config());
    const size_t nblocks = paged_config->num_blocks();
    // Capture width ≥ MAX_BATCH_SIZE so mixed/multi-req PIECEWISE can replay with
    // runtime_n_req ≤ compiled_n_req (vLLM mixed_mode=PIECEWISE). Opt-in override:
    // INFINI_MAX_PREFILL_BATCH. Default 8 (Band C). Set to 1 only for Gate-D bisect.
    max_capture_req_ = 8;
    if (const char *raw = std::getenv("INFINI_MAX_PREFILL_BATCH")) {
        max_capture_req_ = std::max<size_t>(1, std::stoul(raw));
    } else if (const char *raw = std::getenv("MAX_BATCH_SIZE")) {
        max_capture_req_ = std::max<size_t>(1, std::stoul(raw));
    }
    size_t max_bucket = capture_buckets_.empty() ? 0 : capture_buckets_.front();
    block_tables_holder_ = infinicore::Tensor::empty(
        {max_capture_req_ * nblocks}, infinicore::DataType::I32, infinicore::context::getDevice());
    set_zeros(block_tables_holder_);
    const size_t num_layers = model_->native_piecewise_num_layers();
    const size_t free_before = device_free_bytes_();
    // M2: allocate shared banks before inductor warmup so warmup also uses prefix views.
    allocate_shared_banks_(max_bucket, num_layers, max_capture_req_);
    spdlog::info(
        "native piecewise CG: capture warmup n_req={} (metadata only, hidden [1,bucket] shared max={})",
        max_capture_req_,
        max_bucket);

    const int tp_rank = infinilm::global_state::get_tensor_model_parallel_rank();
    const int tp_size = infinilm::global_state::get_tensor_model_parallel_world_size();
    if (tp_size > 1) {
        barrier_->wait("piecewise_inductor_aot_warmup", tp_rank);
    }
    verify_inductor_packages_(model_, capture_buckets_);
    warmup_inductor_segments_(nblocks, max_capture_req_);
    if (tp_size > 1) {
        barrier_->wait("piecewise_inductor_aot_warmup_done", tp_rank);
    }

    compiled_.clear();
    compiled_mid_.clear();
    compiled_mixed_mid_.clear();
    mixed_mid_capture_req_ = std::min<size_t>(2, max_capture_req_);
    for (size_t bucket : capture_buckets_) {
        capture_bucket_(bucket, /*mid_chunk_capture=*/false);
        infinicore::context::syncDevice();
        // Dual capture: exact chunk-sized bucket also gets homo mid + MIXED mid keys.
        if (bucket == prefill_chunk_size_ && prefill_chunk_size_ > 0) {
            capture_bucket_(bucket, /*mid_chunk_capture=*/true);
            infinicore::context::syncDevice();
            if (mixed_mid_capture_req_ >= 2 && bucket >= 2) {
                capture_mixed_mid_bucket_(bucket);
                infinicore::context::syncDevice();
            }
        }
    }
    infinicore::context::syncDevice();
    const size_t free_after = device_free_bytes_();
    // Physical staging/I-O = shared max bank (O(max)); arenas = MoE scratch via
    // shared Python workspace after M3 (still summed per Graph CaptureArena).
    const size_t staging_bytes = shared_banks_.staging_physical_bytes;
    const size_t io_bytes =
        shared_banks_.io_physical_bytes + tensor_nbytes_(block_tables_holder_);
    size_t arena_bytes = 0;
    for (const auto &kv : compiled_) {
        arena_bytes += bucket_arena_nbytes_(kv.second);
    }
    for (const auto &kv : compiled_mid_) {
        arena_bytes += bucket_arena_nbytes_(kv.second);
    }
    for (const auto &kv : compiled_mixed_mid_) {
        arena_bytes += bucket_arena_nbytes_(kv.second);
    }
    const size_t staging_io_bytes = staging_bytes + io_bytes;
    const size_t free_delta =
        free_before >= free_after ? free_before - free_after : 0;
    std::ostringstream oss;
    for (size_t i = 0; i < capture_buckets_.size(); ++i) {
        if (i > 0) {
            oss << ',';
        }
        oss << capture_buckets_[i];
    }
    spdlog::info(
        "native piecewise CG: capture_buckets=[{}] max_seq={} chunk_size={} mid_keys={} "
        "mixed_mid_keys={}",
        oss.str(),
        max_seq_len_,
        prefill_chunk_size_,
        compiled_mid_.size(),
        compiled_mixed_mid_.size());
    // M1: greppable one-line CG VRAM split (before /health). Staging is physical O(max) after M2.
    spdlog::info(
        "cg_mem_budget tag=piecewise_prefill buckets={} mid_buckets={} mixed_mid_buckets={} "
        "staging_bytes={} io_bytes={} "
        "staging_io_bytes={} arena_bytes={} free_delta_bytes={} "
        "staging_GiB={:.3f} io_GiB={:.3f} staging_io_GiB={:.3f} arena_GiB={:.3f} "
        "free_delta_GiB={:.3f}",
        compiled_.size(),
        compiled_mid_.size(),
        compiled_mixed_mid_.size(),
        staging_bytes,
        io_bytes,
        staging_io_bytes,
        arena_bytes,
        free_delta,
        bytes_to_gib_(staging_bytes),
        bytes_to_gib_(io_bytes),
        bytes_to_gib_(staging_io_bytes),
        bytes_to_gib_(arena_bytes),
        bytes_to_gib_(free_delta));
}

size_t PiecewisePrefillCompiler::padded_bucket_for(size_t seq_len) const {
    return padded_bucket_for_seq_len(seq_len, bs_to_padded_, max_seq_len_);
}

void PiecewisePrefillCompiler::copy_runtime_into_bucket_(BucketGraphs &bucket_graphs,
                                                         const InfinilmModel::Input &runtime,
                                                         size_t valid_seq_len) const {
    const size_t bucket = bucket_graphs.input.input_ids.value()->size(1);
    auto &graph_input = bucket_graphs.input;

    graph_input.input_ids.value()
        ->narrow({{1, 0, valid_seq_len}})
        ->copy_from(runtime.input_ids.value());
    graph_input.position_ids.value()
        ->narrow({{0, 0, valid_seq_len}})
        ->copy_from(runtime.position_ids.value());


    const size_t runtime_n_req = runtime.block_tables.value()->size(0);
    const size_t compiled_n_req = graph_input.block_tables.value()->size(0);
    if (runtime_n_req > compiled_n_req) {
        throw std::runtime_error("block_tables batch exceeds compiled capture warmup width");
    }

    if (graph_input.past_sequence_lengths.has_value() && runtime.past_sequence_lengths.has_value()) {
        graph_input.past_sequence_lengths.value()
            ->narrow({{0, 0, runtime_n_req}})
            ->copy_from(runtime.past_sequence_lengths.value());
        if (runtime_n_req < compiled_n_req) {
            auto past_tail = graph_input.past_sequence_lengths.value()->narrow(
                {{0, runtime_n_req, compiled_n_req - runtime_n_req}});
            set_zeros(past_tail);
        }
    }
    graph_input.total_sequence_lengths.value()
        ->narrow({{0, 0, runtime_n_req}})
        ->copy_from(runtime.total_sequence_lengths.value());
    if (runtime_n_req < compiled_n_req) {
        auto total_tail = graph_input.total_sequence_lengths.value()->narrow(
            {{0, runtime_n_req, compiled_n_req - runtime_n_req}});
        set_zeros(total_tail);
    }
    graph_input.input_offsets.value()
        ->narrow({{0, 0, runtime_n_req + 1}})
        ->copy_from(runtime.input_offsets.value());
    graph_input.cu_seqlens.value()
        ->narrow({{0, 0, runtime_n_req + 1}})
        ->copy_from(runtime.cu_seqlens.value());
    // Pad unused capture-width rows with the last cumulative value (not zeros).
    // Zero-fill would make cu_seqlens/input_offsets non-monotonic
    // (…, total_tokens, 0, …) → negative FA varlen spans if anything reads
    // compiled_n_req width.
    if (runtime_n_req < compiled_n_req) {
        const size_t off_tail = compiled_n_req - runtime_n_req;
        auto last_off = graph_input.input_offsets.value()
                            ->narrow({{0, runtime_n_req, 1}})
                            ->to(infinicore::Device::cpu());
        const int32_t last_val =
            *reinterpret_cast<const int32_t *>(last_off->data());
        std::vector<int32_t> pad(off_tail, last_val);
        auto offsets_tail = graph_input.input_offsets.value()->narrow(
            {{0, runtime_n_req + 1, off_tail}});
        infinicore::context::memcpyH2D(
            offsets_tail->data(), pad.data(), pad.size() * sizeof(int32_t), false);
        auto cu_tail = graph_input.cu_seqlens.value()->narrow(
            {{0, runtime_n_req + 1, off_tail}});
        infinicore::context::memcpyH2D(
            cu_tail->data(), pad.data(), pad.size() * sizeof(int32_t), false);
    }

    const size_t block_per_req = runtime.block_tables.value()->size(1);
    const size_t compiled_block_per_req = graph_input.block_tables.value()->size(1);
    if (block_per_req > compiled_block_per_req) {
        throw std::runtime_error("block_tables width exceeds compiled bucket");
    }

    auto &graph_block_tables = graph_input.block_tables.value();
    set_minus_one(graph_block_tables);
    graph_block_tables
        ->narrow({{0, 0, runtime_n_req}, {1, 0, block_per_req}})
        ->copy_from(runtime.block_tables.value());

    if (runtime_n_req < compiled_n_req) {
        auto stale_rows = graph_block_tables->narrow(
            {{0, runtime_n_req, compiled_n_req - runtime_n_req}, {1, 0, compiled_block_per_req}});
        set_minus_one(stale_rows);
    }

    graph_input.slot_mapping.value()
        ->narrow({{0, 0, valid_seq_len}})
        ->copy_from(runtime.slot_mapping.value());
    if (valid_seq_len < bucket) {
        auto slot_tail = graph_input.slot_mapping.value()->narrow({{0, valid_seq_len, bucket - valid_seq_len}});
        set_minus_one(slot_tail);
        auto ids_tail = graph_input.input_ids.value()->narrow({{1, valid_seq_len, bucket - valid_seq_len}});
        set_zeros(ids_tail);
        auto pos_tail = graph_input.position_ids.value()->narrow({{0, valid_seq_len, bucket - valid_seq_len}});
        set_zeros(pos_tail);
    }
}

std::optional<infinicore::Tensor> PiecewisePrefillCompiler::run_prefill(const InfinilmModel::Input &input) {
    if (!enabled_) {
        return std::nullopt;
    }
    // Phase-scoped MoE: Prefill must not take Triton-in-graph path.
    // Under full_and_piecewise MoE is host-break in Prefill (Decode-only adaptive).
    infinicore::context::InferencePhaseGuard phase_guard(
        infinicore::context::InferencePhase::Prefill);
    last_prefill_executed_ = false;
    const bool profile = rank_worker_profile_enabled();
    const double t_total0 = profile ? monotonic_ms() : 0.0;
    const size_t seq_len = compute_prefill_len(input);
    const size_t padded = padded_bucket_for(seq_len);
    const size_t graph_bucket = graph_replay_bucket_for_padded(padded);
    const size_t runtime_n_req = input.block_tables.has_value() ? input.block_tables.value()->size(0) : 1;
    // Decode rows set is_final=True for sampling; that must not hide mid-prefill
    // rows (MIXED). mid_chunk ← any False flag; need_lm_head ← any True / empty.
    const bool mid_chunk =
        InfinilmModel::has_nonfinal_prefill_chunk(input.is_final_prefill_chunk);
    const bool need_lm_head =
        InfinilmModel::any_final_prefill_chunk(input.is_final_prefill_chunk);
    if (profile) {
        spdlog::info(
            "rank_worker_profile: piecewise run_prefill begin seq_len={} padded={} graph_bucket={} "
            "n_req={} mid_chunk={} need_lm_head={}",
            seq_len,
            padded,
            graph_bucket,
            runtime_n_req,
            mid_chunk,
            need_lm_head);
    }
    if (compiled_.find(graph_bucket) == compiled_.end()) {
        ++prefill_misses_;
        return std::nullopt;
    }
    if (runtime_n_req > max_capture_req_) {
        ++prefill_misses_;
        spdlog::debug(
            "piecewise run_prefill: runtime_n_req={} > max_capture_req={} → eager",
            runtime_n_req,
            max_capture_req_);
        return std::nullopt;
    }
    const bool inductor_mode = infinilm::global_state::piecewise_inductor_segment_enabled();
    // Pad-up (seq_len < graph_bucket): CG PlannedMeta bakes valid_len==bucket at
    // capture; replaying that meta on a short chat SIGSEGVs. Use eager inductor
    // with runtime piecewise.valid_seq_len instead (same AOT package).
    const bool pad_up = seq_len != graph_bucket;

    const bool have_mid_graphs = [&]() {
        auto it = compiled_mid_.find(graph_bucket);
        if (it == compiled_mid_.end()) {
            return false;
        }
        return !it->second.pre_attn.empty() && it->second.pre_attn[0]
            && !it->second.post_attn.empty() && it->second.post_attn[0];
    }();
    const bool have_mixed_mid_graphs = [&]() {
        auto it = compiled_mixed_mid_.find(graph_bucket);
        if (it == compiled_mixed_mid_.end()) {
            return false;
        }
        return !it->second.pre_attn.empty() && it->second.pre_attn[0]
            && !it->second.post_attn.empty() && it->second.post_attn[0]
            && static_cast<bool>(it->second.lm_head);
    }();
    // Runtime past of request 0 (representative). Mid dual-capture baked past>0;
    // past==0 mid (first chunk of a long prompt) still SIGSEGVs under CG (Gate D /
    // homo_mid 20260731) — keep that case eager. Continuing mids (past>0) use mid graphs.
    int32_t runtime_past0 = 0;
    if (input.past_sequence_lengths.has_value()
        && input.past_sequence_lengths.value()->size(0) > 0) {
        auto cpu = input.past_sequence_lengths.value()->to(infinicore::Device::cpu());
        runtime_past0 = reinterpret_cast<const int32_t *>(cpu->data())[0];
    }
    const bool past_matches_mid_capture = runtime_past0 > 0;
    // MIXED layout must match dual-capture: n_req=2 decode(1) + mid(bucket-1).
    // v1 RUNNING pack typically emits decode before continuing mid.
    bool mixed_layout_matches_capture = false;
    int32_t mixed_q0 = -1;
    int32_t mixed_q1 = -1;
    int32_t runtime_mid_past = runtime_past0;
    if (runtime_n_req == mixed_mid_capture_req_ && input.input_offsets.has_value()
        && input.input_offsets.value()->size(0) >= runtime_n_req + 1) {
        auto off_cpu = input.input_offsets.value()->to(infinicore::Device::cpu());
        const auto *offs = reinterpret_cast<const int32_t *>(off_cpu->data());
        mixed_q0 = offs[1] - offs[0];
        mixed_q1 = offs[2] - offs[1];
        const int32_t expect_mid_q = static_cast<int32_t>(graph_bucket) - 1;
        mixed_layout_matches_capture = (mixed_q0 == 1 && mixed_q1 == expect_mid_q);
        if (mixed_layout_matches_capture && input.past_sequence_lengths.has_value()
            && input.past_sequence_lengths.value()->size(0) >= 2) {
            auto past_cpu = input.past_sequence_lengths.value()->to(infinicore::Device::cpu());
            // Row1 is continuing mid under decode-first capture.
            runtime_mid_past = reinterpret_cast<const int32_t *>(past_cpu->data())[1];
        }
    }
    // Product: MIXED mid CG prefer is opt-in (ENABLE_MIXED_MID_CG / REPRO_MIXED_MID_CG).
    // Default off: ENABLE=1 collapses LongBench EM (215547 lb_em≈0.017 vs Phase2 0.305).
    const bool prefer_mixed_mid_graphs =
        mid_chunk && need_lm_head && !pad_up && have_mixed_mid_graphs
        && runtime_mid_past > 0
        && (enable_mixed_mid_cg() || repro_mixed_mid_cg())
        && (mixed_layout_matches_capture || repro_mixed_mid_cg());
    // Product: exact homogeneous mid + past>0 → mid dual-capture graphs.
    // MIXED without mixed-mid hit stays eager (Phase2 gate 200231 if homo mid reused).
    // Part A bisect: ALLOW_MIDCHUNK_CG forces final graphs on mid (Gate D / 143915).
    const bool prefer_mid_graphs =
        !prefer_mixed_mid_graphs
        && mid_chunk && !need_lm_head && !pad_up && have_mid_graphs
        && past_matches_mid_capture && !repro_allow_midchunk_cg();
    auto &bucket_graphs = prefer_mixed_mid_graphs
        ? compiled_mixed_mid_.at(graph_bucket)
        : (prefer_mid_graphs ? compiled_mid_.at(graph_bucket) : compiled_.at(graph_bucket));

    // Mid CG ok when dual mid / mixed-mid graphs selected, or repro lifts the gate onto final graphs.
    const bool mid_cg_ok =
        prefer_mid_graphs
        || prefer_mixed_mid_graphs
        || (mid_chunk && !pad_up
            && (repro_allow_midchunk_cg() || repro_skip_midchunk_eager()));
    const bool force_mid_eager = mid_chunk && !mid_cg_ok;
    // Step-level eager hist: pad_up > mid_chunk > missing > exact_cg.
    {
        const bool missing_graph =
            bucket_graphs.pre_attn.empty() || !bucket_graphs.pre_attn[0]
            || bucket_graphs.post_attn.empty() || !bucket_graphs.post_attn[0];
        const bool mid_chunk_eager = force_mid_eager;
        dispatch_hist::record_piecewise_eager(pad_up, mid_chunk_eager, missing_graph);
    }
    const double t_copy0 = profile ? monotonic_ms() : 0.0;
    copy_runtime_into_bucket_(bucket_graphs, input, seq_len);
    // Keep sample/lm_head row flags aligned with the live MIXED pack (capture baked
    // decode-first {true,false}; runtime must match for eager lm_head gather).
    if (!input.is_final_prefill_chunk.empty()) {
        bucket_graphs.input.is_final_prefill_chunk = input.is_final_prefill_chunk;
    }
    set_attn_metadata_for_varlen_batch(bucket_graphs.input, input);
    if (profile) {
        spdlog::info(
            "rank_worker_profile: piecewise copy_runtime+metadata_ms={:.3f}",
            monotonic_ms() - t_copy0);
    }

    auto &piecewise = infinilm::global_state::get_forward_context().piecewise;
    piecewise.valid_seq_len = seq_len;
    piecewise.bucket_seq_len = padded;
    piecewise.hidden_states = bucket_graphs.hidden_states;
    piecewise.residual = bucket_graphs.residual;
    piecewise.ar_staging = bucket_graphs.ar_staging;
    piecewise.layer_staging = bucket_graphs.layer_staging;
    if (!mid_chunk) {
        // Exact-width and pad-up final chunks both use inductor when enabled.
        // Pad-up still sets piecewise.valid_seq_len=L; AOT pads positions/hidden to bucket.
        piecewise.allow_inductor_pre_attn = inductor_mode && !repro_skip_final_inductor();
    } else if (repro_skip_midchunk_eager()) {
        // A2: stress inductor-in-CG mid (out of product default).
        piecewise.allow_inductor_pre_attn = inductor_mode;
    } else {
        // Mid dual-capture / product mid: keep inductor out of mid path.
        piecewise.allow_inductor_pre_attn = false;
    }

    // Fresh residual each replay (matches capture warmup); hidden prefix comes from embed.
    set_zeros(piecewise.residual);
    // Pad-up: clear stale CG tails on staging/hidden/logits before embed so pad
    // positions do not pollute layernorm / MoE (RC-4).
    if (seq_len < graph_bucket) {
        clear_stale_bucket_tails_(piecewise, bucket_graphs.logits_holder, seq_len, graph_bucket);
    }

    model_->native_piecewise_embed(bucket_graphs.input, piecewise.hidden_states);


    const size_t num_layers = bucket_graphs.pre_attn.size();
    // Mid with dual mid graphs: CG replay; inductor-in-CG layers stay eager unless A2.
    // Final-chunk inductor pre-attn needs the same eager post replay (B4 tail).
    // Pad-up: always eager post/lm_head (CG PlannedMeta bakes full-bucket shapes).
    // Homo mid graphs omit lm_head — MIXED without mixed-mid hit uses eager lm_head.
    // MIXED mid prefer: eager pre_attn (CG QKV/RoPE on decode+mid packs garbles when
    // runtime past ≫ capture — ENABLE=1 LIMIT=8 EM 0.375→0); CG post_attn OK to keep;
    // lm_head stays eager for sample-row safety.
    const bool use_eager_post =
        pad_up
        || force_mid_eager
        || (inductor_mode && piecewise.allow_inductor_pre_attn);
    const bool use_eager_lm_head =
        pad_up
        || force_mid_eager
        || prefer_mid_graphs
        || prefer_mixed_mid_graphs
        || !bucket_graphs.lm_head
        || (inductor_mode && piecewise.allow_inductor_pre_attn);
    {
        // Telemetry for Part A/C: first past=0 mid and first past>0 mid at exact bucket.
        static bool logged_mid_past0 = false;
        static bool logged_mid_past_gt0 = false;
        static bool logged_mixed_mid = false;
        if (mid_chunk && !pad_up && graph_bucket >= 2048) {
            const bool do_log = (runtime_past0 <= 0 && !logged_mid_past0)
                || (runtime_past0 > 0 && !logged_mid_past_gt0)
                || (prefer_mixed_mid_graphs && !logged_mixed_mid)
                || (need_lm_head && runtime_past0 > 0 && !logged_mixed_mid);
            if (do_log) {
                if (runtime_past0 <= 0) {
                    logged_mid_past0 = true;
                } else if (!need_lm_head) {
                    logged_mid_past_gt0 = true;
                }
                if (need_lm_head && runtime_past0 > 0) {
                    logged_mixed_mid = true;
                }
                spdlog::info(
                    "mid_chunk_cg_probe: mid_chunk={} past_len={} mid_past={} seq_len={} "
                    "graph_bucket={} force_mid_eager={} prefer_mid_graphs={} "
                    "prefer_mixed_mid_graphs={} mid_cg_ok={} past_matches_mid_capture={} "
                    "mixed_layout_matches_capture={} mixed_q0={} mixed_q1={} n_req={} "
                    "allow_inductor_pre_attn={} use_eager_post={}",
                    mid_chunk,
                    runtime_past0,
                    runtime_mid_past,
                    seq_len,
                    graph_bucket,
                    force_mid_eager,
                    prefer_mid_graphs,
                    prefer_mixed_mid_graphs,
                    mid_cg_ok,
                    past_matches_mid_capture,
                    mixed_layout_matches_capture,
                    mixed_q0,
                    mixed_q1,
                    runtime_n_req,
                    piecewise.allow_inductor_pre_attn,
                    use_eager_post);
            }
        }
    }
    double layers_ms = 0.0;
    const double t_layers0 = profile ? monotonic_ms() : 0.0;
    for (size_t layer = 0; layer < num_layers; ++layer) {
        const double t_layer0 = profile ? monotonic_ms() : 0.0;
        barrier_->wait("piecewise_replay_pre_attn");
        const bool inductor_layer_mid =
            mid_chunk && layer_capture_inductor_pre_attn(layer, graph_bucket)
            && !repro_skip_midchunk_eager();
        // MIXED mid prefer: keep mixed dual-capture buffers + FA meta, but do not
        // replay CG pre_attn when runtime past ≫ capture past — CG QKV/RoPE on
        // decode+mid packs garbles LongBench (ENABLE=1 LIMIT=8 EM 0.375→0).
        // Pre stays eager until a past-robust mixed pre capture lands; post/lm_head
        // already eager on this path. Hist still counts prefer as exact_cg.
        const bool use_eager_pre_attn =
            pad_up
            || force_mid_eager
            || prefer_mixed_mid_graphs
            || inductor_layer_mid
            || !bucket_graphs.pre_attn[layer]
            || (inductor_mode && piecewise.allow_inductor_pre_attn
                && !prefer_mid_graphs && !prefer_mixed_mid_graphs && mid_chunk);
        if (use_eager_pre_attn) {
            model_->native_piecewise_pre_attn_layer(
                layer, bucket_graphs.input, piecewise.hidden_states, piecewise.residual);
        } else if (bucket_graphs.pre_attn[layer]) {
            bucket_graphs.pre_attn[layer]->run();
            ++segment_replays_;
        } else {
            model_->native_piecewise_pre_attn_layer(
                layer, bucket_graphs.input, piecewise.hidden_states, piecewise.residual);
        }
        const double t_pre_attn = profile ? monotonic_ms() : 0.0;
        piecewise.phase = global_state::PiecewiseCapturePhase::EagerAttn;
        model_->native_piecewise_eager_attn_layer(layer, bucket_graphs.input);
        const double t_eager_attn = profile ? monotonic_ms() : 0.0;
        barrier_->wait("piecewise_replay_post_attn");
        if (use_eager_post || inductor_layer_mid || !bucket_graphs.post_attn[layer]) {
            model_->native_piecewise_post_attn_cg_layer(
                layer, bucket_graphs.input, piecewise.hidden_states, piecewise.residual);
            barrier_->wait("piecewise_replay_post_attn_sync");
        } else {
            bucket_graphs.post_attn[layer]->run();
            ++segment_replays_;
            barrier_->wait("piecewise_replay_post_attn_sync");
        }
        if (profile) {
            spdlog::info(
                "rank_worker_profile: piecewise layer={} pre_attn_ms={:.3f} eager_attn_ms={:.3f} "
                "post_attn_ms={:.3f} layer_total_ms={:.3f}",
                layer,
                t_pre_attn - t_layer0,
                t_eager_attn - t_pre_attn,
                monotonic_ms() - t_eager_attn,
                monotonic_ms() - t_layer0);
        }
    }
    if (profile) {
        layers_ms = monotonic_ms() - t_layers0;
    }
    if (need_lm_head) {
        const double t_lm0 = profile ? monotonic_ms() : 0.0;
        barrier_->wait("piecewise_replay_lm_head");
        if (use_eager_lm_head) {
            model_->native_piecewise_lm_head(
                bucket_graphs.input, piecewise.hidden_states, piecewise.residual, bucket_graphs.logits_holder);
        } else {
            bucket_graphs.lm_head->run();
            ++segment_replays_;
        }
        barrier_->wait("piecewise_replay_lm_head_sync");
        if (profile) {
            spdlog::info(
                "rank_worker_profile: piecewise lm_head_ms={:.3f}",
                monotonic_ms() - t_lm0);
        }
    }

    piecewise.phase = global_state::PiecewiseCapturePhase::None;
    ++prefill_hits_;
    last_prefill_executed_ = true;
    if (profile) {
        spdlog::info(
            "rank_worker_profile: piecewise run_prefill end layers_total_ms={:.3f} total_ms={:.3f} "
            "mid_chunk={} need_lm_head={} pad_up={}",
            layers_ms,
            monotonic_ms() - t_total0,
            mid_chunk,
            need_lm_head,
            pad_up);
    }
    if (global_state::ar_profile::enabled()
        && global_state::get_tensor_model_parallel_rank() == 0) {
        global_state::ar_profile::log_barrier_chunk_summary(
            "piecewise_replay", seq_len, runtime_n_req);
    }
    if (!need_lm_head) {
        return std::nullopt;
    }
    return bucket_graphs.logits_holder->narrow({{1, 0, seq_len}});
}

} // namespace infinilm::engine
