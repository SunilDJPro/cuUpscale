#include "upscaler.h"
#include "bicubic.h"
#include "lanczos.h"
#include "edi.h"
#include <stdexcept>

std::vector<unsigned char> Upscaler::upscale(const std::vector<unsigned char>& input,
                                             int input_width, int input_height, int channels,
                                             int output_width, int output_height,
                                             UpscaleMethod method) {
    switch (method) {
        case UpscaleMethod::Bicubic:
            return Bicubic::upscale(input, input_width, input_height, channels, output_width, output_height);
        case UpscaleMethod::Lanczos:
            return Lanczos::upscale(input, input_width, input_height, channels, output_width, output_height);
        case UpscaleMethod::EDI: {
            EDIUpscaler edi;
            float scale_factor = static_cast<float>(output_width) / input_width;
            return edi.upscale(input, input_width, input_height, channels, scale_factor);
        }
        default:
            throw std::runtime_error("Unknown upscale method");
    }
}