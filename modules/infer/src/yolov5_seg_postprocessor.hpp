#pragma once

#include "yoseg/infer/infer.hpp"

namespace yoseg::infer {

class YoloV5SegPostprocessor final : public IPostprocessor {
public:
    explicit YoloV5SegPostprocessor(PostprocessConfig cfg);
    bool run(const InferInput& input, const InferOutput& infer_output, PostprocessOutput& post_output) override;

private:
    PostprocessConfig cfg_;
};

} // namespace yoseg::infer
