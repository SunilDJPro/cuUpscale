#include "cuda_utils.cuh"

GPUInfo getGPUInfo() {
    GPUInfo info;

    CUDA_CHECK(cudaGetDeviceCount(&info.deviceCount));

    if (info.deviceCount > 0) {

        CUDA_CHECK(cudaGetDeviceProperties(&info.deviceProp, 0));

        printf("NVIDIA GPU information: \n\n");
        printf("  Device name: %s\n", info.deviceProp.name);
        printf("  Compute capability: %d.%d\n", info.deviceProp.major, info.deviceProp.minor);
        printf("  Number of SMs %d\n", info.deviceProp.multiProcessorCount);
        printf("  Global Memory: %.2f GB\n\n\n", static_cast<float>(info.deviceProp.totalGlobalMem) / (1024 * 1024 * 1024));
    }

    return info;
}