#include "static_batching_compiler.hpp"
#include "../../cache/cache.hpp"
#include "../../global_state/global_state.hpp"
#include "../../utils.hpp"

#include <spdlog/spdlog.h>

namespace infinilm::engine {
StaticBatchingCompiler::StaticBatchingCompiler(const std::shared_ptr<InfinilmModel> &model, RankBarrier *barrier)
    : GraphCompiler(model, barrier) {
}

void StaticBatchingCompiler::compile() {
    if (model_->get_cache_config() != nullptr && dynamic_cast<const cache::StaticKVCacheConfig *>(model_->get_cache_config())) {
        // StaticAttentionImpl (static_attn.cpp) reads past/total sequence lengths on the
        // host to narrow KV cache views and attention windows:
        //   cache_pos = past_sequence_lengths.cpu()[0]
        //   total_seq_len = total_sequence_lengths.cpu()[0]
        // CUDA graph capture runs that host control flow once. This compiler seeds
        // past=0 and total=1 for capture, so replay after a long prefill still
        // writes new KV at offset 0 and attends only length 1. On MiniCPM5/QY that
        // shows up as decode loops on low-id special tokens (e.g. <|fim_prefix|>)
        // and LongBench CoT EM=0 with --enable-graph.
        //
        // Device-side kv_caching_ + length-aware attention are required before a
        // static-KV decode graph is safe. Until then, skip capture so RUN falls
        // back to eager (correct) when --enable-graph is set.
        spdlog::warn(
            "[StaticBatchingCompiler] skipping CUDA graph capture for Static KV: "
            "static_attn host-side past/total sequence lengths are not capture-safe");
        return;
    }
}

StaticBatchingCompiler::Compiled StaticBatchingCompiler::get_compiled(
    const InfinilmModel::Input &input) {
    if (model_->get_cache_config() != nullptr && dynamic_cast<const cache::StaticKVCacheConfig *>(model_->get_cache_config())) {
        size_t batch_size = input.input_ids.value()->size(0);
        size_t seqlen = input.input_ids.value()->size(1);
        auto result = compiled_map_.find(std::make_tuple(batch_size, seqlen));
        if (result == compiled_map_.end()) {
            return std::make_tuple(nullptr, nullptr);
        } else {
            auto &graph_input = result->second.input;
            graph_input.input_ids.value()->copy_from(input.input_ids.value());
            graph_input.position_ids.value()->copy_from(input.position_ids.value());
            graph_input.past_sequence_lengths.value()->copy_from(input.past_sequence_lengths.value());
            graph_input.total_sequence_lengths.value()->copy_from(input.total_sequence_lengths.value());

            auto graph = std::get<0>(result->second.compiled);
            auto shared_output = std::shared_ptr<InfinilmModel::Output>(new InfinilmModel::Output{std::get<1>(result->second.compiled)->logits->resume_from_blob_()});
            return std::make_tuple(graph, shared_output);
        }
    } else {
        return std::make_tuple(nullptr, nullptr);
    }
}
} // namespace infinilm::engine
