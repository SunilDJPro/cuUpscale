#pragma once
#include <vector>

enum class UpscaleMethod {
    Bicubic,
    Lanczos,
    EDI
};

class Upscaler {
public:
    static std::vector<unsigned char> upscale(const std::vector<unsigned char>& input,
                                              int input_width, int input_height, int channels,
                                              int output_width, int output_height,
                                              UpscaleMethod method);
};