#include "lanczos.h"
#include <cmath>
#include <algorithm>
#include <omp.h>
//lanczos v1 cpu (OpenMP)
namespace {
    double sinc(double x) {
        if (x == 0) return 1.0;
        return std::sin(M_PI * x) / (M_PI * x);
    }

    double lanczos(double x, int a) {
        if (x == 0) return 1.0;
        if (x > -a && x < a) return sinc(x) * sinc(x / a);
        return 0.0;
    }
}

namespace Lanczos {
    std::vector<unsigned char> upscale(const std::vector<unsigned char>& input,
                                       int input_width, int input_height, int channels,
                                       int output_width, int output_height,
                                       int a) {
        std::vector<unsigned char> output(output_width * output_height * channels);
        double x_ratio = static_cast<double>(input_width) / output_width;
        double y_ratio = static_cast<double>(input_height) / output_height;

        #pragma omp parallel for collapse(2)
        for (int y = 0; y < output_height; ++y) {
            for (int x = 0; x < output_width; ++x) {
                double x_l = (x + 0.5) * x_ratio - 0.5;
                double y_l = (y + 0.5) * y_ratio - 0.5;
                int x_i = static_cast<int>(x_l);
                int y_i = static_cast<int>(y_l);

                for (int c = 0; c < channels; ++c) {
                    double result = 0.0;
                    double normalizer = 0.0;

                    for (int m = -a + 1; m <= a; ++m) {
                        for (int n = -a + 1; n <= a; ++n) {
                            int cur_x = std::clamp(x_i + m, 0, input_width - 1);
                            int cur_y = std::clamp(y_i + n, 0, input_height - 1);
                            double weight = lanczos(x_l - cur_x, a) * lanczos(y_l - cur_y, a);

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