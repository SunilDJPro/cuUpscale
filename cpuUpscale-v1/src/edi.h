#pragma once

#include <vector>
#include <cstdint>

class EDIUpscaler {
public:
    EDIUpscaler();
    ~EDIUpscaler();

    std::vector<uint8_t> upscale(const std::vector<uint8_t>& input_image, 
                                 int input_width, 
                                 int input_height, 
                                 int channels, 
                                 float scale_factor);

private:
    std::vector<float> preprocess(const std::vector<uint8_t>& input_image);
    std::vector<uint8_t> postprocess(const std::vector<float>& output_image);
    std::vector<float> applyEDI(const std::vector<float>& input, int width, int height, int channels, float scale_factor);
    float interpolatePixel(const std::vector<float>& input, int width, int height, int channels, float x, float y, int channel);
    float calculateGradient(float a, float b, float c, float d);
};