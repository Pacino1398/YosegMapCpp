#include "rknn_runtime_adapter.hpp"

#include <cstring>
#include <utility>

#if defined(YOSEG_WITH_RKNN_RUNTIME) && YOSEG_WITH_RKNN_RUNTIME
#if __has_include(<rknn_api.h>)
#include <rknn_api.h>
#define YOSEG_RKNN_HEADER_OK 1
#else
#define YOSEG_RKNN_HEADER_OK 0
#endif
#else
#define YOSEG_RKNN_HEADER_OK 0
#endif

namespace yoseg::infer {

bool RknnRuntimeAdapter::init(const std::vector<std::uint8_t>& model_blob) {
#if YOSEG_RKNN_HEADER_OK
    if (model_blob.empty()) {
        return false;
    }
    rknn_context ctx = 0;
    const int ret = rknn_init(&ctx, model_blob.data(), model_blob.size(), 0, nullptr);
    if (ret != RKNN_SUCC) {
        ready_ = false;
        return false;
    }
    rknn_input_output_num io_num{};
    const int qret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (qret != RKNN_SUCC) {
        (void)rknn_destroy(ctx);
        ready_ = false;
        return false;
    }
    ctx_ = static_cast<std::uint64_t>(ctx);
    input_count_ = io_num.n_input;
    output_count_ = io_num.n_output;
    ready_ = true;
    return true;
#else
    (void)model_blob;
    ready_ = false;
    return false;
#endif
}

bool RknnRuntimeAdapter::run(const InferInput& input, InferOutput& output) {
#if YOSEG_RKNN_HEADER_OK
    if (!ready_ || ctx_ == 0 || input_count_ == 0 || output_count_ == 0) {
        return false;
    }
    rknn_context ctx = static_cast<rknn_context>(ctx_);
    rknn_input rinput{};
    rinput.index = 0;
    rinput.type = RKNN_TENSOR_UINT8;
    rinput.size = static_cast<std::uint32_t>(input.data.size());
    rinput.fmt = RKNN_TENSOR_NHWC;
    rinput.pass_through = 0;
    rinput.buf = const_cast<std::uint8_t*>(input.data.data());

    int ret = rknn_inputs_set(ctx, 1, &rinput);
    if (ret != RKNN_SUCC) {
        return false;
    }
    ret = rknn_run(ctx, nullptr);
    if (ret != RKNN_SUCC) {
        return false;
    }

    std::vector<rknn_output> outputs(output_count_);
    for (std::uint32_t i = 0; i < output_count_; ++i) {
        outputs[i].want_float = 1;
        outputs[i].is_prealloc = 0;
    }
    ret = rknn_outputs_get(ctx, output_count_, outputs.data(), nullptr);
    if (ret != RKNN_SUCC) {
        return false;
    }

    // Template path: capture output metadata now; full decode is implemented in postprocess module.
    if (output.tensors.size() != output_count_) {
        output.tensors.resize(output_count_);
    }
    for (std::uint32_t i = 0; i < output_count_; ++i) {
        Tensor& t = output.tensors[i];
        t.shape = {1, static_cast<int>(outputs[i].size / sizeof(float))};
        if (outputs[i].buf != nullptr && outputs[i].size >= sizeof(float)) {
            const auto count = static_cast<std::size_t>(outputs[i].size / sizeof(float));
            if (t.data.size() != count) {
                t.data.resize(count);
            }
            std::memcpy(t.data.data(), outputs[i].buf, count * sizeof(float));
        } else {
            t.data.clear();
        }
    }
    (void)rknn_outputs_release(ctx, output_count_, outputs.data());
    return true;
#else
    (void)input;
    (void)output;
    return false;
#endif
}

void RknnRuntimeAdapter::release() {
#if YOSEG_RKNN_HEADER_OK
    if (ctx_ != 0) {
        const auto ctx = static_cast<rknn_context>(ctx_);
        (void)rknn_destroy(ctx);
        ctx_ = 0;
    }
#endif
    input_count_ = 0;
    output_count_ = 0;
    ready_ = false;
}

bool RknnRuntimeAdapter::available() const {
#if YOSEG_RKNN_HEADER_OK
    return ready_ && ctx_ != 0;
#else
    return false;
#endif
}

} // namespace yoseg::infer
