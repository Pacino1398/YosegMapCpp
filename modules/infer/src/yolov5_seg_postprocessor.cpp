#include "yolov5_seg_postprocessor.hpp"

#include <algorithm>
#include <cstddef>

namespace yoseg::infer {

YoloV5SegPostprocessor::YoloV5SegPostprocessor(PostprocessConfig cfg) : cfg_(cfg) {}

bool YoloV5SegPostprocessor::run(
    const InferInput& input,
    const InferOutput& infer_output,
    PostprocessOutput& post_output) {
    if (input.width <= 0 || input.height <= 0) {
        return false;
    }
    const int map_w = cfg_.grid_width > 0 ? cfg_.grid_width : input.width;
    const int map_h = cfg_.grid_height > 0 ? cfg_.grid_height : input.height;
    post_output.map_width = map_w;
    post_output.map_height = map_h;
    const std::size_t total = static_cast<std::size_t>(map_w) * static_cast<std::size_t>(map_h);
    if (post_output.occupancy.size() != total) {
        post_output.occupancy.assign(total, 0);
    } else {
        std::fill(post_output.occupancy.begin(), post_output.occupancy.end(), static_cast<std::uint8_t>(0));
    }

    // Decode skeleton for YOLOv5-seg:
    // 1) infer_output.tensors[0] -> candidate detections + mask coefficients
    // 2) infer_output.tensors[1] -> proto mask feature map
    // 3) NMS + mask projection -> binary obstacle occupancy
    if (!infer_output.tensors.empty() && !infer_output.tensors[0].data.empty()) {
        const auto& logits = infer_output.tensors[0].data;
        for (std::size_t i = 0; i < total; ++i) {
            const float v = logits[i % logits.size()];
            const float norm = std::max(0.0f, std::min(1.0f, v / 640.0f));
            post_output.occupancy[i] = (norm > cfg_.obstacle_threshold) ? static_cast<std::uint8_t>(100)
                                                                         : static_cast<std::uint8_t>(0);
        }
    }
    post_output.obstacle_count = 0;
    for (std::uint8_t x : post_output.occupancy) {
        if (x > 0) {
            ++post_output.obstacle_count;
        }
    }
    // Approximate postprocess write volume to occupancy map.
    // Exact tracking can be refined when full YOLOv5-seg decode lands.
    return true;
}

} // namespace yoseg::infer
