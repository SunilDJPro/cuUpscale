#include "lanczos.cuh"
#include "cuda_utils.cuh"
#include <cmath>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>

namespace {
    __device__ double sinc(double x) {
        if (x == 0) return 1.0;
        return sin(M_PI * x) / (M_PI * x);
    }

    __device__ double lanczos(double x, int a) {
        if (x == 0) return 1.0;
        if (x > -a && x < a) return sinc(x) * sinc(x / a);
        return 0.0;
    }

    __global__ void lanczosKernel(const unsigned char* input, unsigned char* output,
                                  int input_width, int input_height, int channels,
                                  int output_width, int output_height, int a,
                                  double x_ratio, double y_ratio) {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x >= output_width || y >= output_height) return;

        double x_l = (x + 0.5) * x_ratio - 0.5;
        double y_l = (y + 0.5) * y_ratio - 0.5;
        int x_i = static_cast<int>(x_l);
        int y_i = static_cast<int>(y_l);

        for (int c = 0; c < channels; ++c) {
            double result = 0.0;
            double normalizer = 0.0;

            for (int m = -a + 1; m <= a; ++m) {
                for (int n = -a + 1; n <= a; ++n) {
                    int cur_x = max(0, min(x_i + m, input_width - 1));
                    int cur_y = max(0, min(y_i + n, input_height - 1));
                    double weight = lanczos(x_l - cur_x, a) * lanczos(y_l - cur_y, a);
                    result += weight * input[(cur_y * input_width + cur_x) * channels + c];
                    normalizer += weight;
                }
            }

            output[(y * output_width + x) * channels + c] =
                static_cast<unsigned char>(max(0.0, min(result / normalizer, 255.0)));
        }
    }
}


namespace cuLanczos {
    std::vector<unsigned char> upscale(const std::vector<unsigned char>& input,
                                       int input_width, int input_height, int channels,
                                       int output_width, int output_height,
                                       int a) {
        GPUInfo gpuInfo = getGPUInfo();
        if (gpuInfo.deviceCount == 0) {
            fprintf(stderr, "No CUDA-capable GPU found\n");
            exit(EXIT_FAILURE);
        }

        cudaDeviceProp SMprop;
        //Set device (Single GPU Only)
        int device = 0;
        CUDA_CHECK(cudaSetDevice(device));
        cudaGetDeviceProperties(&SMprop, device);
        int numSMs = SMprop.multiProcessorCount;

        //Allocate device memory (Unified->MallocManaged)
        unsigned char *d_input, *d_output;
        size_t input_size = input_width * input_height * channels * sizeof(unsigned char);
        size_t output_size = output_width * output_height * channels * sizeof(unsigned char);

        CUDA_CHECK(cudaMallocManaged(&d_input, input_size));
        CUDA_CHECK(cudaMallocManaged(&d_output, output_size));

        memcpy(d_input, input.data(), input_size);
        //Copy input data to device mem
        CUDA_CHECK(cudaMemPrefetchAsync(d_input, input_size, device, nullptr));

        //Calculate grid and block dimensions
        int threadsPerBlock = 16;
        dim3 blockDim(threadsPerBlock, threadsPerBlock);
        dim3 gridDim((output_width + blockDim.x - 1) / blockDim.x,
                     (output_height + blockDim.y - 1) / blockDim.y);

        int blockPerSM = 4; //Temp before profile
        gridDim.x *= numSMs * blockPerSM; //grid allocation based on no of SMs available
       
        double x_ratio = static_cast<double>(input_width) / output_width;
        double y_ratio = static_cast<double>(input_height) / output_height;

         // Kernel Launch
        lanczosKernel<<<gridDim, blockDim>>>(d_input, d_output,
                                             input_width, input_height, channels,
                                             output_width, output_height, a,
                                             x_ratio, y_ratio);

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemPrefetchAsync(d_output, output_size, cudaCpuDeviceId, nullptr));

        //cpy result to host system
        std::vector<unsigned char> output(output_size);
        memcpy(output.data(), d_output, output_size);

        //Free device mem
        CUDA_CHECK(cudaFree(d_input));
        CUDA_CHECK(cudaFree(d_output));

        return output;
    }
}