#include "yoseg/capture/capture.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#if defined(YOSEG_HAS_OPENCV) && YOSEG_HAS_OPENCV
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#endif

namespace yoseg::capture {

struct CaptureSource::Impl {
    bool opened = false;
    bool synthetic = true;
    std::size_t synthetic_tick = 0;
#if defined(YOSEG_HAS_OPENCV) && YOSEG_HAS_OPENCV
    cv::VideoCapture cap;
#endif
};

CaptureSource::CaptureSource() : impl_(new Impl()) {}

CaptureSource::~CaptureSource() {
    close();
    delete impl_;
    impl_ = nullptr;
}

bool CaptureSource::open(const std::string& source) {
#if defined(YOSEG_HAS_OPENCV) && YOSEG_HAS_OPENCV
    bool opened = false;
    if (!source.empty()) {
        bool numeric = true;
        for (char c : source) {
            if (c < '0' || c > '9') {
                numeric = false;
                break;
            }
        }
        if (numeric) {
            opened = impl_->cap.open(std::atoi(source.c_str()));
        } else {
            opened = impl_->cap.open(source);
        }
    }
    if (opened) {
        impl_->opened = true;
        impl_->synthetic = false;
        impl_->synthetic_tick = 0;
        return true;
    }
#endif
    impl_->opened = true;
    impl_->synthetic = true;
    impl_->synthetic_tick = 0;
    return true;  // fallback to synthetic source to keep pipeline alive
}

bool CaptureSource::read(Frame& frame) {
    if (!impl_->opened) {
        return false;
    }
    if (!impl_->synthetic) {
#if defined(YOSEG_HAS_OPENCV) && YOSEG_HAS_OPENCV
        cv::Mat bgr;
        if (!impl_->cap.read(bgr) || bgr.empty()) {
            return false;
        }
        cv::Mat resized;
        cv::resize(bgr, resized, cv::Size(640, 640), 0.0, 0.0, cv::INTER_LINEAR);

        frame.width = resized.cols;
        frame.height = resized.rows;
        frame.channels = resized.channels();
        const std::size_t size = static_cast<std::size_t>(frame.width) * static_cast<std::size_t>(frame.height) *
                                 static_cast<std::size_t>(frame.channels);
        if (frame.data.size() != size) {
            frame.data.resize(size);
        }
        std::memcpy(frame.data.data(), resized.data, size);
        return true;
#endif
    }

    frame.width = 640;
    frame.height = 640;
    frame.channels = 3;
    const std::size_t size = static_cast<std::size_t>(frame.width) * static_cast<std::size_t>(frame.height) *
                             static_cast<std::size_t>(frame.channels);
    if (frame.data.size() != size) {
        frame.data.resize(size);
    }

    const std::uint8_t base = static_cast<std::uint8_t>(impl_->synthetic_tick % 255);
    std::fill(frame.data.begin(), frame.data.end(), base);
    ++impl_->synthetic_tick;
    return true;
}

bool CaptureSource::warmup(int frames, int expected_width, int expected_height, int expected_channels) {
    if (!impl_->opened) {
        return false;
    }
    Frame tmp;
    tmp.width = expected_width > 0 ? expected_width : 640;
    tmp.height = expected_height > 0 ? expected_height : 640;
    tmp.channels = expected_channels > 0 ? expected_channels : 3;
    const std::size_t size = static_cast<std::size_t>(tmp.width) * static_cast<std::size_t>(tmp.height) *
                             static_cast<std::size_t>(tmp.channels);
    tmp.data.resize(size);
    for (int i = 0; i < frames; ++i) {
        if (!read(tmp)) {
            return false;
        }
    }
    return true;
}

void CaptureSource::close() {
#if defined(YOSEG_HAS_OPENCV) && YOSEG_HAS_OPENCV
    if (impl_->cap.isOpened()) {
        impl_->cap.release();
    }
#endif
    impl_->opened = false;
    impl_->synthetic = true;
}

bool CaptureSource::is_open() const {
    return impl_->opened;
}

} // namespace yoseg::capture
