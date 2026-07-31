#pragma once

#include "../../global_state/forward_context.hpp"
#include "graph_compiler.hpp"

#include <unordered_map>

namespace infinilm::engine {
class PagedCompiler : public GraphCompiler {
public:
    struct GraphStats {
        size_t prefill_graph_hits{0};
        size_t prefill_graph_misses{0};
        size_t decode_graph_hits{0};
        size_t decode_graph_misses{0};
        size_t piecewise_segment_replays{0};
        size_t piecewise_prefill_hits{0};
        size_t piecewise_prefill_misses{0};
        size_t piecewise_decode_hits{0};
        size_t piecewise_decode_misses{0};
        size_t piecewise_decode_device_segments{0};
    };

    PagedCompiler(const std::shared_ptr<InfinilmModel> &model, RankBarrier *barrier);

    void compile() override;

    Compiled get_compiled(const InfinilmModel::Input &input) override;

    void record_graph_hit(bool is_prefill);
    void record_graph_miss(bool is_prefill);
    GraphStats graph_stats() const;

    struct CompiledResult {
        InfinilmModel::Input input;
        Compiled compiled;
        std::shared_ptr<std::vector<global_state::DeferredAllreduce>> post_graph_allreduces;
    };

    /// M4: max-batch decode I/O bank; smaller batches use prefix views.
    struct SharedDecodeBanks {
        size_t max_batch{0};
        size_t io_physical_bytes{0};
        infinicore::Tensor input_ids;
        infinicore::Tensor position_ids;
        infinicore::Tensor total_sequence_lengths;
        infinicore::Tensor input_offsets;
        infinicore::Tensor cu_seqlens;
        infinicore::Tensor slot_mapping;
    };

private:
    CompiledResult capture_forward_graph_(InfinilmModel::Input input);
    void allocate_shared_decode_banks_(size_t max_batch);
    InfinilmModel::Input make_decode_input_(size_t batch, size_t nblocks) const;

    std::vector<size_t> decode_batch_sizes_;
    std::vector<size_t> prefill_seq_buckets_{4096};

    infinicore::Tensor block_tables_holder_;
    SharedDecodeBanks shared_decode_;

    std::unordered_map<
        size_t, // num_requests
        CompiledResult>
        compiled_map_decode_;

    std::unordered_map<
        size_t, // prefill sequence bucket length
        CompiledResult>
        compiled_map_prefill_;

    size_t prefill_graph_hits_{0};
    size_t prefill_graph_misses_{0};
    size_t decode_graph_hits_{0};
    size_t decode_graph_misses_{0};
};
} // namespace infinilm::engine
