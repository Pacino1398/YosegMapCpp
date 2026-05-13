#pragma once

#include "yoseg/core/perf_counters.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace yoseg::infer {

struct Tensor {
    std::vector<float> data;
    std::vector<int> shape;
};

struct InferInput {
    int width = 0;
    int height = 0;
    int channels = 3;
    std::vector<std::uint8_t> data;
};

struct PreprocessConfig {
    int target_width = 640;
    int target_height = 640;
    bool bgr_to_rgb = false;
};

struct InferOutput {
    std::vector<Tensor> tensors;
};

struct PostprocessOutput {
    int obstacle_count = 0;
    std::vector<std::uint8_t> occupancy;
    int map_width = 0;
    int map_height = 0;
};

struct PostprocessConfig {
    float obstacle_threshold = 0.5f;
    int grid_width = 640;
    int grid_height = 640;
};

class IInferEngine {
public:
    virtual ~IInferEngine() = default;
    virtual bool init(const std::string& model_path) = 0;
    virtual bool run(const InferInput& input, InferOutput& output) = 0;
    virtual const char* name() const = 0;
};

class IPostprocessor {
public:
    virtual ~IPostprocessor() = default;
    virtual bool run(const InferInput& input, const InferOutput& infer_output, PostprocessOutput& post_output) = 0;
};

std::unique_ptr<IInferEngine> create_engine(const std::string& backend);
void set_perf_counters(yoseg::core::PerfCounters* perf);
void set_preprocess_config(const PreprocessConfig& cfg);
bool preprocess(const InferInput& raw_input, InferInput& net_input);
bool preprocess_move(InferInput&& raw_input, InferInput& net_input);
void set_postprocess_config(const PostprocessConfig& cfg);
void set_postprocessor(std::unique_ptr<IPostprocessor> postprocessor);
bool postprocess(const InferInput& input, const InferOutput& infer_output, PostprocessOutput& post_output);

} // namespace yoseg::infer
