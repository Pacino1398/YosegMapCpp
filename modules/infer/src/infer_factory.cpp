#include "yoseg/infer/infer.hpp"
#include "rknn_engine.hpp"
#include "yolov5_seg_postprocessor.hpp"

#include <memory>
#include <utility>

namespace yoseg::infer {

namespace {
PreprocessConfig g_pre_cfg{};
PostprocessConfig g_post_cfg{};
yoseg::core::PerfCounters* g_perf = nullptr;

class CpuStubEngine final : public IInferEngine {
public:
    bool init(const std::string& model_path) override {
        (void)model_path;
        return true;
    }

    bool run(const InferInput& input, InferOutput& output) override {
        if (output.tensors.size() != 1) {
            output.tensors.resize(1);
        }
        Tensor& t = output.tensors[0];
        t.shape = {1, 1};
        if (t.data.size() != 1) {
            t.data.resize(1);
        }
        t.data[0] = static_cast<float>(input.width * input.height);
        return true;
    }

    const char* name() const override {
        return "cpu_stub";
    }
};

std::unique_ptr<IPostprocessor> g_postprocessor = std::make_unique<YoloV5SegPostprocessor>(g_post_cfg);

} // namespace

std::unique_ptr<IInferEngine> create_engine(const std::string& backend) {
    if (backend == "rknn") {
        return std::make_unique<RknnEngine>();
    }
    return std::make_unique<CpuStubEngine>();
}

void set_perf_counters(yoseg::core::PerfCounters* perf) {
    g_perf = perf;
}

void set_preprocess_config(const PreprocessConfig& cfg) {
    g_pre_cfg = cfg;
}

bool preprocess(const InferInput& raw_input, InferInput& net_input) {
    if (raw_input.width <= 0 || raw_input.height <= 0 || raw_input.channels <= 0) {
        return false;
    }
    // TODO: replace with RK3588-optimized path (RGA / zero-copy) during device integration.
    if (g_perf != nullptr) {
        g_perf->add_alloc();
        g_perf->add_copy_pre(static_cast<std::uint64_t>(raw_input.data.size()));
    }
    net_input = raw_input;
    if (g_pre_cfg.target_width > 0) {
        net_input.width = g_pre_cfg.target_width;
    }
    if (g_pre_cfg.target_height > 0) {
        net_input.height = g_pre_cfg.target_height;
    }
    return true;
}

bool preprocess_move(InferInput&& raw_input, InferInput& net_input) {
    if (raw_input.width <= 0 || raw_input.height <= 0 || raw_input.channels <= 0) {
        return false;
    }
    net_input = std::move(raw_input);
    if (g_pre_cfg.target_width > 0) {
        net_input.width = g_pre_cfg.target_width;
    }
    if (g_pre_cfg.target_height > 0) {
        net_input.height = g_pre_cfg.target_height;
    }
    return true;
}

void set_postprocess_config(const PostprocessConfig& cfg) {
    g_post_cfg = cfg;
    g_postprocessor = std::make_unique<YoloV5SegPostprocessor>(g_post_cfg);
}

void set_postprocessor(std::unique_ptr<IPostprocessor> postprocessor) {
    if (postprocessor) {
        g_postprocessor = std::move(postprocessor);
    }
}

bool postprocess(const InferInput& input, const InferOutput& infer_output, PostprocessOutput& post_output) {
    const bool ok = g_postprocessor->run(input, infer_output, post_output);
    if (ok && g_perf != nullptr) {
        g_perf->add_copy_post(static_cast<std::uint64_t>(post_output.occupancy.size()));
    }
    return ok;
}

} // namespace yoseg::infer
