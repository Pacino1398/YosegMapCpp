#pragma once

#include "yoseg/infer/infer.hpp"

#include <cstdint>
#include <vector>

namespace yoseg::infer {

class RknnRuntimeAdapter {
public:
    bool init(const std::vector<std::uint8_t>& model_blob);
    bool run(const InferInput& input, InferOutput& output);
    void release();
    bool available() const;

private:
#if defined(YOSEG_WITH_RKNN_RUNTIME) && YOSEG_WITH_RKNN_RUNTIME
    std::uint64_t ctx_ = 0;
#endif
    std::uint32_t input_count_ = 0;
    std::uint32_t output_count_ = 0;
    bool ready_ = false;
};

} // namespace yoseg::infer
