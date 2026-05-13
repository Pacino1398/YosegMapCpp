#pragma once

#include "yoseg/infer/infer.hpp"
#include "rknn_runtime_adapter.hpp"

#include <string>
#include <vector>

namespace yoseg::infer {

class RknnEngine final : public IInferEngine {
public:
    RknnEngine() = default;
    ~RknnEngine() override;

    bool init(const std::string& model_path) override;
    bool run(const InferInput& input, InferOutput& output) override;
    const char* name() const override;

private:
    bool load_model(const std::string& model_path);
    bool init_runtime();
    void release_runtime();

private:
    bool initialized_ = false;
    std::string model_path_;
    std::vector<std::uint8_t> model_blob_;
    RknnRuntimeAdapter runtime_;
};

} // namespace yoseg::infer
