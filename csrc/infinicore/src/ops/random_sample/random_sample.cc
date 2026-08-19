#include "infinicore/ops/random_sample.hpp"

#include "../../utils.hpp"
#include "custom_types.h"

#include <cmath>
#include <cstdint>

#ifdef ENABLE_INFINIOPS_API
#include "../infiniops_impl.hpp"

#include "base/argmax.h"
#endif

namespace infinicore::op {
namespace {

bool tryGreedyHost(
    Tensor indices, Tensor logits,
    float /*random_value*/, float /*top_p*/, int top_k, float /*temperature*/) {
    if (top_k != 1
        || logits->ndim() != 1
        || logits->numel() == 0
        || !logits->is_contiguous()
        || indices->numel() != 1
        || !indices->is_contiguous()) {
        return false;
    }

    auto cpu_logits = logits->contiguous()->to(Device{Device::Type::kCpu});
    const size_t n = cpu_logits->numel();
    size_t best = 0;
    float best_v = -INFINITY;
    auto consider = [&](size_t i, float v) {
        if (v > best_v) {
            best_v = v;
            best = i;
        }
    };

    if (cpu_logits->dtype() == DataType::kFloat32) {
        const auto *p = reinterpret_cast<const float *>(cpu_logits->data());
        for (size_t i = 0; i < n; ++i) {
            consider(i, p[i]);
        }
    } else if (cpu_logits->dtype() == DataType::kFloat16) {
        const auto *p = reinterpret_cast<const fp16_t *>(cpu_logits->data());
        for (size_t i = 0; i < n; ++i) {
            consider(i, _f16_to_f32(p[i]));
        }
    } else if (cpu_logits->dtype() == DataType::kBFloat16) {
        const auto *p = reinterpret_cast<const bf16_t *>(cpu_logits->data());
        for (size_t i = 0; i < n; ++i) {
            consider(i, _bf16_to_f32(p[i]));
        }
    } else {
        return false;
    }

    if (indices->dtype() == DataType::kInt64) {
        int64_t value = static_cast<int64_t>(best);
        auto cpu_idx = Tensor::from_blob(
            &value, {}, DataType::kInt64, Device{Device::Type::kCpu});
        indices->copy_from(cpu_idx);
        return true;
    }
    if (indices->dtype() == DataType::kInt32) {
        int32_t value = static_cast<int32_t>(best);
        auto cpu_idx = Tensor::from_blob(
            &value, {}, DataType::kInt32, Device{Device::Type::kCpu});
        indices->copy_from(cpu_idx);
        return true;
    }
    return false;
}

#ifdef ENABLE_INFINIOPS_API
bool tryGreedyWithInfiniOps(
    Tensor indices, Tensor logits,
    float random_value, float top_p, int top_k, float temperature) {
    const auto dtype = logits->dtype();
    if (logits->device().type() != Device::Type::kNvidia
        || (random_value != 0.0f
            && top_p != 0.0f
            && top_k != 1
            && temperature != 0.0f)
        || logits->ndim() != 1
        || logits->numel() == 0
        || !logits->is_contiguous()
        || (dtype != DataType::kFloat16 && dtype != DataType::kBFloat16 && dtype != DataType::kFloat32)
        || indices->numel() != 1
        || indices->dtype() != DataType::kInt64
        || !indices->is_contiguous()) {
        return false;
    }

    infini::ops::Handle handle;
    handle.set_stream(context::getStream());
    infini::ops::Config config;
    config.set_implementation_index(8);
    const std::optional<int64_t> no_dim;
    infini::ops::Argmax::Call(
        handle,
        config,
        infiniops::TensorMeta(logits).tensor(logits),
        no_dim,
        false,
        infiniops::TensorMeta(indices).tensor(indices));
    return true;
}
#endif

} // namespace

common::OpDispatcher<RandomSample::schema> &RandomSample::dispatcher() {
    static common::OpDispatcher<RandomSample::schema> dispatcher_;
    return dispatcher_;
};

void RandomSample::execute(
    Tensor indices, Tensor logits,
    float random_val, float topp, int topk, float temperature) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(indices, logits);
    infinicore::context::setDevice(logits->device());
#ifdef ENABLE_INFINIOPS_API
    if (tryGreedyWithInfiniOps(
            indices, logits, random_val, topp, topk, temperature)) {
        return;
    }
#endif
    if (tryGreedyHost(indices, logits, random_val, topp, topk, temperature)) {
        return;
    }

    dispatcher().lookup(logits->device().type())(
        indices, logits, random_val, topp, topk, temperature);
}

Tensor random_sample(
    Tensor logits,
    float random_val,
    float topp,
    int topk,
    float temperature) {
    auto indices = Tensor::empty({}, DataType::kInt32, logits->device());
    random_sample_(indices, logits, random_val, topp, topk, temperature);
    return indices;
}

void random_sample_(
    Tensor indices,
    Tensor logits,
    float random_val,
    float topp,
    int topk,
    float temperature) {
    RandomSample::execute(indices, logits, random_val, topp, topk, temperature);
}

} // namespace infinicore::op
