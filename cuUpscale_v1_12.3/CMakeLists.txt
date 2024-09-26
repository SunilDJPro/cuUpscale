cmake_minimum_required(VERSION 3.18)
project(CUDAUpscale CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 14)

find_package(JPEG REQUIRED)
find_package(CUDA 12.4 REQUIRED)

include_directories(${JPEG_INCLUDE_DIR})
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
add_executable(upscale
    src/main.cu
    src/jpeg_cpu.cpp
    src/lanczos.cu
    src/cuda_utils.cu
)

target_link_libraries(upscale ${JPEG_LIBRARIES} ${CUDA_LIBRARIES})
set_target_properties(upscale PROPERTIES CUDA_SEPARABLE_COMPILATION ON)