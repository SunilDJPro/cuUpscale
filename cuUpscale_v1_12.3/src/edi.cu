#include "edi.cuh"
#include "cuda_runtime.h"
#include "cuda_utils.cuh"
#include <cmath>
#include <device_launch_parameters.h>
#include <algorithm>

__device__ float calculateGradient(float a, float b, float c, float d) {
    return fabsf(a - b) + fabsf(c - d);
}

__device__ float interpolatePixel(const float* input, int width, int height, int channels,
                                    float x, float y, int channel) {
                                
    int x0 = static_cast<int>(floor(x));
    int y0 = static_cast<int>(floor(y));


    int x1 = min(x0 + 1, width - 1);
    int y1 = min(y0 + 1, height - 1);

    float fx = x - x0;
    float fy = y - y0;

    float a = input[(y0 * width + x0) * channels + channel];
    float b = input[(y0 * width + x1) * channels + channel];
    float c = input[(y1 * width + x0) * channels + channel];
    float d = input[(y1 * width + x1) * channels + channel];

    float gx = calculateGradient(a, b, c, d);
    float gy = calculateGradient(a, c, b, d);

    if (fabsf(gx) > fabsf(gy)) {
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

__global__ void EDIKernel(const float* input, float* output, int input_width, int input_height,
                          int output_width, int output_height, int channels, float scale_factor) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < output_width && y < output_height) {
        float src_x = x / scale_factor;
        float src_y = y / scale_factor;

        for (int c = 0; c < channels; ++c) {
            float value = interpolatePixel(input, input_width, input_height, channels, src_x, src_y, c);
            output[(y * output_width + x) * channels + c] = value;
        }
    }
}


namespace cuEDI {

std::vector<uint8_t> upscale(const std::vector<uint8_t>& input_image,
                             int input_width,
                             int input_height,
                             int channels,
                             float scale_factor) {

    GPUInfo gpuInfo = getGPUInfo();
        if (gpuInfo.deviceCount == 0) {
            fprintf(stderr, "No CUDA-capable GPU found\n");
            exit(EXIT_FAILURE);
        }


    int output_width = static_cast<int>(input_width * scale_factor);
    int output_height = static_cast<int>(input_height * scale_factor);

    //Set device to use (Single GPU Only)
    cudaDeviceProp SMprop;
    int device = 0;
    CUDA_CHECK(cudaSetDevice(device));
    cudaGetDeviceProperties(&SMprop, device);
    int numSMs = SMprop.multiProcessorCount;

    //Allocate device memory
    float *d_input, *d_output;
    size_t input_size = input_width * input_height * channels * sizeof(float);
    size_t output_size = output_width * output_height * channels * sizeof(float);

    CUDA_CHECK(cudaMallocManaged(&d_input, input_size));
    CUDA_CHECK(cudaMallocManaged(&d_output, output_size));

    //Preprocess and copy input data to unified memory
    #pragma omp parallel for
    for (size_t i = 0; i < input_image.size(); ++i) {
        d_input[i] = input_image[i] / 255.0f;
    }

    //Prefetch input data to GPU
    CUDA_CHECK(cudaMemPrefetchAsync(d_input, input_size, device, nullptr));

    int threadsPerBlock = 16;

    //Calculate grid and block dimensions
    dim3 blockDim(threadsPerBlock, threadsPerBlock);
    dim3 gridDim((output_width + blockDim.x - 1) / blockDim.x,
                 (output_height + blockDim.y - 1) / blockDim.y);

    int blockPerSM = 4; //Temp before profile
    gridDim.x *= numSMs * blockPerSM; //Grid allocation based on no of SMs available

    //Launch kernel
    EDIKernel<<<gridDim, blockDim>>>(d_input, d_output,
                                     input_width, input_height,
                                     output_width, output_height,
                                     channels, scale_factor);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    //Prefetch output
    CUDA_CHECK(cudaMemPrefetchAsync(d_output, output_size, cudaCpuDeviceId, nullptr));

    std::vector<uint8_t> output(output_width * output_height * channels);
    #pragma omp parallel for
    for (size_t i = 0; i < output.size(); ++i) {
        output[i] = static_cast<uint8_t>(std::min(std::max(d_output[i] * 255.0f, 0.0f), 255.0f));
    }

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    return output;
}

}