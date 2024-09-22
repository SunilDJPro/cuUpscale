#include "bicubic.h"
#include <cmath>
#include <algorithm>
#include <omp.h>

namespace {
    double cubic(double x) {
        x = std::abs(x);
        if (x <= 1.0) return (1.5 * x - 2.5) * x * x + 1.0;
        else if (x < 2.0) return ((-0.5 * x + 2.5) * x - 4.0) * x + 2.0;
        return 0.0;
    }
}

namespace Bicubic {
    std::vector<unsigned char> upscale(const std::vector<unsigned char>& input,
                                       int input_width, int input_height, int channels,
                                       int output_width, int output_height) {
        std::vector<unsigned char> output(output_width * output_height * channels);
        double x_ratio = static_cast<double>(input_width - 1) / (output_width - 1);
        double y_ratio = static_cast<double>(input_height - 1) / (output_height - 1);

        #pragma omp parallel for collapse(2)
        for (int y = 0; y < output_height; ++y) {
            for (int x = 0; x < output_width; ++x) {
                double x_l = x * x_ratio;
                double y_l = y * y_ratio;
                int x_i = static_cast<int>(x_l);
                int y_i = static_cast<int>(y_l);

                for (int c = 0; c < channels; ++c) {
                    double result = 0.0;
                    double normalizer = 0.0;

                    for (int m = -1; m <= 2; ++m) {
                        for (int n = -1; n <= 2; ++n) {
                            int cur_x = std::clamp(x_i + m, 0, input_width - 1);
                            int cur_y = std::clamp(y_i + n, 0, input_height - 1);
                            double weight = cubic(x_l - cur_x) * cubic(y_l - cur_y);

                            result += weight * input[(cur_y * input_width + cur_x) * channels + c];
                            normalizer += weight;
                        }
                    }

                    output[(y * output_width + x) * channels + c] = 
                        static_cast<unsigned char>(std::clamp(result / normalizer, 0.0, 255.0));
                }
            }
        }

        return output;
    }
}