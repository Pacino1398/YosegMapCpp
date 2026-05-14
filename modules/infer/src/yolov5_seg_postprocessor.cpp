#include "yolov5_seg_postprocessor.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#if defined(YOSEG_INFER_HAS_OPENCV) && YOSEG_INFER_HAS_OPENCV
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#endif

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
        post_output.occupancy.resize(total);
    }
    std::fill(post_output.occupancy.begin(), post_output.occupancy.end(), static_cast<std::uint8_t>(0));
    post_output.obstacle_count = 0;

    if (infer_output.tensors.size() < 2) {
        return true;
    }
    const Tensor& det = infer_output.tensors[0];
    const Tensor& proto = infer_output.tensors[1];
    if (det.data.empty() || proto.data.empty()) {
        return true;
    }

    constexpr int kMaskDim = 32;
    constexpr int kProtoH = 160;
    constexpr int kProtoW = 160;
    constexpr int kProtoPixels = kProtoH * kProtoW;  // 25600
    constexpr int kDetStride = 16;                   // x,y,w,h,conf,...,mask32 (fallback-padded)
    constexpr int kMaxAnchors = 25200;
    constexpr float kNmsIou = 0.45f;

    if (proto.data.size() < static_cast<std::size_t>(kMaskDim * kProtoPixels)) {
        return true;
    }

    const std::size_t det_count = det.data.size() / static_cast<std::size_t>(kDetStride);
    if (det_count == 0) {
        return true;
    }
    const int anchors = static_cast<int>(std::min<std::size_t>(det_count, static_cast<std::size_t>(kMaxAnchors)));
    const float conf_th = std::max(0.0f, std::min(1.0f, cfg_.obstacle_threshold));

    struct Candidate {
        float x1;
        float y1;
        float x2;
        float y2;
        float conf;
        int anchor_idx;
    };
    std::vector<Candidate> candidates;
    candidates.reserve(static_cast<std::size_t>(anchors));

    const float in_w = static_cast<float>(std::max(1, input.width));
    const float in_h = static_cast<float>(std::max(1, input.height));
    const float proto_sx = static_cast<float>(kProtoW) / in_w;
    const float proto_sy = static_cast<float>(kProtoH) / in_h;

    const float* det_ptr = det.data.data();
    for (int i = 0; i < anchors; ++i) {
        const float* row = det_ptr + static_cast<std::size_t>(i) * static_cast<std::size_t>(kDetStride);
        const float conf = row[4];
        if (conf < conf_th) {
            continue;
        }
        const float cx = row[0];
        const float cy = row[1];
        const float bw = std::max(0.0f, row[2]);
        const float bh = std::max(0.0f, row[3]);
        float x1 = (cx - bw * 0.5f) * proto_sx;
        float y1 = (cy - bh * 0.5f) * proto_sy;
        float x2 = (cx + bw * 0.5f) * proto_sx;
        float y2 = (cy + bh * 0.5f) * proto_sy;
        x1 = std::max(0.0f, std::min(static_cast<float>(kProtoW - 1), x1));
        y1 = std::max(0.0f, std::min(static_cast<float>(kProtoH - 1), y1));
        x2 = std::max(0.0f, std::min(static_cast<float>(kProtoW - 1), x2));
        y2 = std::max(0.0f, std::min(static_cast<float>(kProtoH - 1), y2));
        if (x2 <= x1 || y2 <= y1) {
            continue;
        }
        candidates.push_back(Candidate{x1, y1, x2, y2, conf, i});
    }

    if (candidates.empty()) {
        return true;
    }
    std::sort(candidates.begin(), candidates.end(), [](const Candidate& a, const Candidate& b) {
        return a.conf > b.conf;
    });

    auto iou = [](const Candidate& a, const Candidate& b) {
        const float xx1 = std::max(a.x1, b.x1);
        const float yy1 = std::max(a.y1, b.y1);
        const float xx2 = std::min(a.x2, b.x2);
        const float yy2 = std::min(a.y2, b.y2);
        const float w = std::max(0.0f, xx2 - xx1);
        const float h = std::max(0.0f, yy2 - yy1);
        const float inter = w * h;
        const float area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
        const float area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
        const float uni = area_a + area_b - inter;
        return uni > 1e-6f ? inter / uni : 0.0f;
    };

    std::vector<Candidate> selected;
    selected.reserve(candidates.size());
    for (const Candidate& c : candidates) {
        bool keep = true;
        for (const Candidate& s : selected) {
            if (iou(c, s) > kNmsIou) {
                keep = false;
                break;
            }
        }
        if (keep) {
            selected.push_back(c);
        }
    }
    if (selected.empty()) {
        return true;
    }

#if defined(YOSEG_INFER_HAS_OPENCV) && YOSEG_INFER_HAS_OPENCV
    cv::Mat coeffs(static_cast<int>(selected.size()), kMaskDim, CV_32F);
    for (int r = 0; r < static_cast<int>(selected.size()); ++r) {
        float* dst = coeffs.ptr<float>(r);
        const float* src = det_ptr + static_cast<std::size_t>(selected[static_cast<std::size_t>(r)].anchor_idx) *
                                         static_cast<std::size_t>(kDetStride);
        for (int c = 0; c < kMaskDim; ++c) {
            const int det_col = 5 + c;
            dst[c] = det_col < kDetStride ? src[det_col] : 0.0f;
        }
    }

    // Proto tensor: [1,32,160,160] treated as [32,25600].
    cv::Mat proto_mat(kMaskDim, kProtoPixels, CV_32F, const_cast<float*>(proto.data.data()));
    cv::Mat mask_logits;
    cv::gemm(coeffs, proto_mat, 1.0, cv::Mat(), 0.0, mask_logits);

    if (!mask_logits.empty()) {
        cv::Mat one(kProtoH, kProtoW, CV_8U);
        cv::Mat resized_mask(map_h, map_w, CV_8U);
        for (int r = 0; r < mask_logits.rows; ++r) {
            cv::Mat row = mask_logits.row(r).reshape(1, kProtoH);
            cv::Mat neg_row;
            cv::multiply(row, cv::Scalar(-1.0f), neg_row);
            cv::exp(neg_row, neg_row);
            neg_row += 1.0f;
            cv::divide(1.0f, neg_row, row);  // row now sigmoid
            cv::threshold(row, one, conf_th, 255.0, cv::THRESH_BINARY);

            // Box crop in proto grid to reduce false fill outside detection.
            const Candidate& b = selected[static_cast<std::size_t>(r)];
            const int x1 = std::max(0, std::min(kProtoW - 1, static_cast<int>(std::floor(b.x1))));
            const int y1 = std::max(0, std::min(kProtoH - 1, static_cast<int>(std::floor(b.y1))));
            const int x2 = std::max(0, std::min(kProtoW - 1, static_cast<int>(std::ceil(b.x2))));
            const int y2 = std::max(0, std::min(kProtoH - 1, static_cast<int>(std::ceil(b.y2))));
            if (x2 > x1 && y2 > y1) {
                cv::Mat keep = cv::Mat::zeros(kProtoH, kProtoW, CV_8U);
                one(cv::Rect(x1, y1, x2 - x1, y2 - y1)).copyTo(keep(cv::Rect(x1, y1, x2 - x1, y2 - y1)));
                cv::resize(keep, resized_mask, cv::Size(map_w, map_h), 0.0, 0.0, cv::INTER_NEAREST);
            } else {
                cv::resize(one, resized_mask, cv::Size(map_w, map_h), 0.0, 0.0, cv::INTER_NEAREST);
            }

            const std::uint8_t* src = resized_mask.ptr<std::uint8_t>(0);
            std::uint8_t* dst = post_output.occupancy.data();
            for (std::size_t i = 0; i < total; ++i) {
                if (src[i] != 0) {
                    dst[i] = 100;
                }
            }
        }
    } else {
        // Fallback: have boxes but mask fusion failed, fill boxes directly.
        for (const Candidate& b : selected) {
            const int x1 = std::max(0, std::min(map_w - 1, static_cast<int>(std::floor(b.x1 * map_w / kProtoW))));
            const int y1 = std::max(0, std::min(map_h - 1, static_cast<int>(std::floor(b.y1 * map_h / kProtoH))));
            const int x2 = std::max(0, std::min(map_w, static_cast<int>(std::ceil(b.x2 * map_w / kProtoW))));
            const int y2 = std::max(0, std::min(map_h, static_cast<int>(std::ceil(b.y2 * map_h / kProtoH))));
            for (int y = y1; y < y2; ++y) {
                std::uint8_t* row = post_output.occupancy.data() + static_cast<std::size_t>(y) * static_cast<std::size_t>(map_w);
                std::fill(row + x1, row + x2, static_cast<std::uint8_t>(100));
            }
        }
    }
#else
    // Fallback without OpenCV: box-only fill, still respects grid-index mapping.
    for (const Candidate& b : selected) {
        const int x1 = std::max(0, std::min(map_w - 1, static_cast<int>(std::floor(b.x1 * map_w / kProtoW))));
        const int y1 = std::max(0, std::min(map_h - 1, static_cast<int>(std::floor(b.y1 * map_h / kProtoH))));
        const int x2 = std::max(0, std::min(map_w, static_cast<int>(std::ceil(b.x2 * map_w / kProtoW))));
        const int y2 = std::max(0, std::min(map_h, static_cast<int>(std::ceil(b.y2 * map_h / kProtoH))));
        for (int y = y1; y < y2; ++y) {
            std::uint8_t* row = post_output.occupancy.data() + static_cast<std::size_t>(y) * static_cast<std::size_t>(map_w);
            std::fill(row + x1, row + x2, static_cast<std::uint8_t>(100));
        }
    }
#endif

    int count = 0;
    for (std::uint8_t v : post_output.occupancy) {
        if (v != 0) {
            ++count;
        }
    }
    post_output.obstacle_count = count;
    return true;
}

} // namespace yoseg::infer
