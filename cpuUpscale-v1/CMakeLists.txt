cmake_minimum_required(VERSION 3.10)
project(cpuUpscale CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(JPEG REQUIRED)
find_package(OpenMP REQUIRED)

set(SOURCES
    src/main.cpp
    src/jpeg_cpu.cpp
    src/upscaler.cpp
    src/bicubic.cpp
    src/lanczos.cpp
    src/edi.cpp 
)

add_executable(cpuUpscale ${SOURCES})

target_include_directories(cpuUpscale PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(cpuUpscale PRIVATE ${JPEG_LIBRARIES} OpenMP::OpenMP_CXX)
target_include_directories(cpuUpscale PRIVATE ${JPEG_INCLUDE_DIRS})

if(OpenMP_CXX_FOUND)
    target_link_libraries(cpuUpscale PRIVATE OpenMP::OpenMP_CXX)
endif()

target_compile_options(cpuUpscale PRIVATE -O3)