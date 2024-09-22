#pragma once
#include <vector>

namespace Bicubic {
    std::vector<unsigned char> upscale(const std::vector<unsigned char>& input,
                                       int input_width, int input_height, int channels,
                                       int output_width, int output_height);
}