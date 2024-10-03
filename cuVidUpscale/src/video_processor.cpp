#include "video_processor.hpp"
#include <iostream>
#include "nvdec_wrapper.hpp"
#include "nvenc_wrapper.hpp"
#include "lanczos_upscaler.cuh"
#include "audio_processsor.hpp"
#include "cuda_utils.cuh"
#include <stdexcept>
#include <iomanip>
#include <chrono>
#include <format>
#include <string_view>
#include <algorithm>

VideoProcessor::VideoProcessor(const std::string& input_file, const std::string& output_file, float scale_factor)
    : m_input_file(input_file), m_output_file(output_file), m_scale_factor(scale_factor),
      m_input_width(0), m_input_height(0), m_output_width(0), m_output_height(0), 
      m_frame_rate(0), m_total_frames(0)
{
    initialize();
}

VideoProcessor::~VideoProcessor() {
    finalize();
}