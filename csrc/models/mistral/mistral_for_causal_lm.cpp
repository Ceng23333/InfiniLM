#include "mistral_for_causal_lm.hpp"
#include "../models_registry.hpp"

namespace infinilm::models::mistral {

std::shared_ptr<infinilm::config::ModelConfig> create_mistral_model_config(std::shared_ptr<infinilm::config::ModelConfig> model_config) {
    const std::string &model_type = model_config->get<std::string>("model_type");
    if ("mistral" != model_type) {
        throw std::runtime_error(
            "infinilm::models::mistral::create_mistral_model_config: model_type is not mistral");
    }

    nlohmann::json &config_json = model_config->get_config_json();

    if (!config_json.contains("head_dim")) {
        size_t head_dim = model_config->get<size_t>("hidden_size")
            / model_config->get<size_t>("num_attention_heads");
        config_json["head_dim"] = head_dim;
    }

    if (!config_json.contains("attention_bias")) {
        config_json["attention_bias"] = false;
    }

    return model_config;
}

} // namespace infinilm::models::mistral

namespace {

INFINILM_REGISTER_CAUSAL_LM_MODEL(
    mistral,
    infinilm::models::mistral::MistralForCausalLM,
    infinilm::models::mistral::create_mistral_model_config);

} // namespace
