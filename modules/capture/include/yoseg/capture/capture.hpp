#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace yoseg::capture {

struct Frame {
    int width = 0;
    int height = 0;
    int channels = 3;
    std::vector<std::uint8_t> data;
};

class CaptureSource {
public:
    CaptureSource();
    ~CaptureSource();

    bool open(const std::string& source);
    bool read(Frame& frame);
    bool warmup(int frames, int expected_width, int expected_height, int expected_channels);
    void close();
    bool is_open() const;

private:
    struct Impl;
    Impl* impl_;
};

} // namespace yoseg::capture
