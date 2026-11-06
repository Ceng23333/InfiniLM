#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "../tensor.hpp"
#include "common/op.hpp"

namespace infinicore::op {

// Device-graph safety is ctor-gated (moeTritonCaptureAllowed + QY decode gate).
// QY Decode defaults to host-break unless INFINI_MOE_QY_ALLOW_GRAPH=1; Prefill
// stays capture-safe. Non-QY follows moeTritonCaptureAllowed only.
class MoeFusedDense : public graph::DispatchableGraphOperator {
public:
    using schema = void (*)(Tensor,
                            const Tensor &,
                            const Tensor &,
                            const Tensor &,
                            const Tensor &,
                            const Tensor &,
                            const Tensor &,
                            const Tensor &,
                            const Tensor &);
    using plan_schema = void *(*)(Tensor,
                                  const Tensor &,
                                  const Tensor &,
                                  const Tensor &,
                                  const Tensor &,
                                  const Tensor &,
                                  const Tensor &,
                                  const Tensor &,
                                  const Tensor &);

    static common::OpDispatcher<plan_schema> &plan_dispatcher();
    static common::OpDispatcher<run_schema> &run_dispatcher();
    static common::OpDispatcher<cleanup_schema> &cleanup_dispatcher();

    MoeFusedDense(Tensor output,
                  const Tensor &hidden_states,
                  const Tensor &w13,
                  const Tensor &w2,
                  const Tensor &topk_weights,
                  const Tensor &topk_ids,
                  const Tensor &sorted_token_ids,
                  const Tensor &expert_ids,
                  const Tensor &num_tokens_post_padded);

    static void execute(Tensor output,
                        const Tensor &hidden_states,
                        const Tensor &w13,
                        const Tensor &w2,
                        const Tensor &topk_weights,
                        const Tensor &topk_ids,
                        const Tensor &sorted_token_ids,
                        const Tensor &expert_ids,
                        const Tensor &num_tokens_post_padded);

    bool is_device_graph_capture_safe() const override {
        return device_graph_capture_safe_;
    }

private:
    bool device_graph_capture_safe_;
};

Tensor moe_fused_dense(
    const Tensor &hidden_states,
    const Tensor &w13,
    const Tensor &w2,
    const Tensor &topk_weights,
    const Tensor &topk_ids,
    const Tensor &sorted_token_ids,
    const Tensor &expert_ids,
    const Tensor &num_tokens_post_padded);

void moe_fused_dense_(
    Tensor output,
    const Tensor &hidden_states,
    const Tensor &w13,
    const Tensor &w2,
    const Tensor &topk_weights,
    const Tensor &topk_ids,
    const Tensor &sorted_token_ids,
    const Tensor &expert_ids,
    const Tensor &num_tokens_post_padded);

} // namespace infinicore::op
