#include "edi.h"
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <omp.h>
//Edge Detected Interpolation

EDIUpscaler::EDIUpscaler() {}

EDIUpscaler::~EDIUpscaler() {}

std::vector<uint8_t> EDIUpscaler::upscale(const std::vector<uint8_t>& input_image, 
                                          int input_width, 
                                          int input_height, 
                                          int channels, 
                                          float scale_factor) {
    std::vector<float> preprocessed_image = preprocess(input_image);
    std::vector<float> upscaled_image = applyEDI(preprocessed_image, input_width, input_height, channels, scale_factor);
    return postprocess(upscaled_image);
}

std::vector<float> EDIUpscaler::preprocess(const std::vector<uint8_t>& input_image) {
    std::vector<float> preprocessed(input_image.size());
    #pragma omp parallel for
    for (size_t i = 0; i < input_image.size(); ++i) {
        preprocessed[i] = input_image[i] / 255.0f;
    }
    return preprocessed;
}

std::vector<uint8_t> EDIUpscaler::postprocess(const std::vector<float>& output_image) {
    std::vector<uint8_t> postprocessed(output_image.size());
    #pragma omp parallel for
    for (size_t i = 0; i < output_image.size(); ++i) {
        postprocessed[i] = static_cast<uint8_t>(std::min(std::max(output_image[i] * 255.0f, 0.0f), 255.0f));
    }
    return postprocessed;
}

std::vector<float> EDIUpscaler::applyEDI(const std::vector<float>& input, int width, int height, int channels, float scale_factor) {
    int output_width = static_cast<int>(width * scale_factor);
    int output_height = static_cast<int>(height * scale_factor);
    std::vector<float> output(output_width * output_height * channels);

    #pragma omp parallel for collapse(2)
    for (int y = 0; y < output_height; ++y) {
        for (int x = 0; x < output_width; ++x) {
            float src_x = x / scale_factor;
            float src_y = y / scale_factor;

            for (int c = 0; c < channels; ++c) {
                float value = interpolatePixel(input, width, height, channels, src_x, src_y, c);
                output[(y * output_width + x) * channels + c] = value;
            }
        }
    }

    return output;
}

float EDIUpscaler::interpolatePixel(const std::vector<float>& input, int width, int height, int channels, float x, float y, int channel) {
    int x0 = static_cast<int>(std::floor(x));
    int y0 = static_cast<int>(std::floor(y));
    int x1 = std::min(x0 + 1, width - 1);
    int y1 = std::min(y0 + 1, height - 1);

    float fx = x - x0;
    float fy = y - y0;

    float a = input[(y0 * width + x0) * channels + channel];
    float b = input[(y0 * width + x1) * channels + channel];
    float c = input[(y1 * width + x0) * channels + channel];
    float d = input[(y1 * width + x1) * channels + channel];

    float gx = calculateGradient(a, b, c, d);
    float gy = calculateGradient(a, c, b, d);

    if (std::abs(gx) > std::abs(gy)) {
        //Interpolate along y-axis
        float i1 = a + fy * (c - a);
        float i2 = b + fy * (d - b);
        return i1 + fx * (i2 - i1);
    } else {
        //Interpolate along x-axis
        float i1 = a + fx * (b - a);
        float i2 = c + fx * (d - c);
        return i1 + fy * (i2 - i1);
    }
}

float EDIUpscaler::calculateGradient(float a, float b, float c, float d) {
    return std::abs(a - b) + std::abs(c - d);
}