#pragma once

#include <vector>
#include <cstint>
#include <cuda_runtime.h>

namespace cuEDI {

    std::vector<uint8_t> upscale(const std::vector<uint8_t>& input_image,
                                int input_width,
                                int input_height,
                                int channels,
                                float scale_factor);
}