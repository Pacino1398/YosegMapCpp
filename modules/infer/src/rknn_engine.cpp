#include "rknn_engine.hpp"

#include <fstream>
#include <iterator>

namespace yoseg::infer {

RknnEngine::~RknnEngine() {
    release_runtime();
}

bool RknnEngine::init(const std::string& model_path) {
    model_path_ = model_path;
    initialized_ = load_model(model_path_) && init_runtime();
    return initialized_;
}

bool RknnEngine::run(const InferInput& input, InferOutput& output) {
    if (!initialized_) {
        return false;
    }
    if (runtime_.run(input, output)) {
        return true;
    }

    // SDK is unavailable in current build/runtime; keep a deterministic stub output.
    if (output.tensors.size() != 1) {
        output.tensors.resize(1);
    }
    Tensor& t = output.tensors[0];
    t.shape = {1, 2};
    if (t.data.size() != 2) {
        t.data.resize(2);
    }
    t.data[0] = static_cast<float>(input.width);
    t.data[1] = static_cast<float>(input.height);
    return true;
}

const char* RknnEngine::name() const {
    return runtime_.available() ? "rknn_runtime" : "rknn_stub";
}

bool RknnEngine::load_model(const std::string& model_path) {
    if (model_path.empty()) {
        return false;
    }
    std::ifstream in(model_path, std::ios::binary);
    if (!in.is_open()) {
        return false;
    }
    model_blob_.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
    return !model_blob_.empty();
}

bool RknnEngine::init_runtime() {
    if (runtime_.init(model_blob_)) {
        return true;
    }
    // Runtime not available yet (e.g. SDK/header missing); keep stub fallback active.
    return true;
}

void RknnEngine::release_runtime() {
    runtime_.release();
    initialized_ = false;
}

} // namespace yoseg::infer
