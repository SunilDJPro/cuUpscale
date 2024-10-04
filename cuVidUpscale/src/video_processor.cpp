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

void VideoProcessor::initialize() {
    std::cout << "Initialize video processor...\n";

    CUDA_CHECK(cudaStreamCreate(&m_cuda_stream));

    m_decoder = std::make_unique<NvDecWrapper>(m_input_file); //Init Decoder Engine
    m_input_width = m_decoder->getWidth();
    m_input_height = m_decoder->getHeigth();
    m_frame_rate = m_decoder->getFrameRate();
    m_total_frames = m_decoder->getTotalFrames();

    m_output_width = static_cast<int>(m_input_width * m_scale_factor); //Output dimensions
    m_output_height = static_cast<int>(m_input_height * m_scale_factor);

    m_encoder = std::make_unique<NvEncWrapper>(m_output_file, m_output_width, m_output_height, m_frame_rate);

    m_upscaler = std::make_unique<LanczosUpscaler>(m_input_width, m_input_height, m_output_weight, m_output_height, m_cuda_stream);

    m_audio_prcessor = std::make_unique<AudioProcessor>(m_input_size, m_output);

    m_num_threads = omp_get_max_threads(); //Getting thread count available (CPU)
    m_batch_size = std::min(m_num_threads * 2, 16);

    size_t input_frame_size = m_input_width * m_input_height * 3;
    size_t output_frame_size = m_output_width * m_output_height * 3;

    m_input_frame_buffers.resize(m_batch_size, std::vector<uint8_t>(input_frame_size)); //Allocation of frame buffers
    m_output_frame_buffers.resize(m_batch_size, std::vector<uint8_t>(output_frame_size));

    printInputInfo();
}

void VideoProcessor::printInputInfo() {

    std::cout << std::format("Input file: {}\n", m_input_file);
    std::cout << std::format("Input resoution: {}x{}\n", m_input_width, m_input_height);
    std::cout << std::format("Frame rate: {:.2f} FPS\n", m_frame_rate);
    std::cout << std::format("Total frames: {}\n ", m_total_frames);
    std::cout << std::format("Scale factor: {:.2f}\n", m_scale_factor);
    std::cout << std::format("Output resolution: {}x{}\n", m_output_width, m_output_height);
    std::cout << std::format("Output file: {}\n", m_output_file);
    std::cout << std::format("OpenMP Number of threads: {}\n", m_num_threads);
}

void VideoProcessor::process() {

    std::cout << "Starting video processing...\n";

    auto start_time = std::chrono::high_resolution_clock::now();

    std::cout << "Extracting audio... \n";
    m_audio_processor->extractAudio();

    std::cout << "Processing Video frames...\n";
    processFrames();

    std::cout << "Processing Video frames...\n";
    m_audio_prcessor->muxAudio();

    audo end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);

    std::cout << std::format("Video processing completed in {} seconds. \n", duration.count());
    printOutputSummary();
}

void VideoProcessor::processFrames() {

    int processed_frames = 0;
    int report_interval = m_total_frames / 100; //Progress
    report_interval = std::max(1, report_interval);

    auto start_time = std::chrono::high_resolution_clock::now();

    while (processed_frames < m_total_frames) {
        int remaining_frames = m_total_frames - processed_frames;
        int batch_frames = std::min(m_batch_size, remaining_frames);

        for (int i = 0; i < batch_frames; ++i) {
            if (!m_decoder->decodeNextFrame(m_input_frame_buffers[i])) {
                throw std::runtime_error("Failed to decode frame");
            }
        }

        processFrameBatch(m_input_frame_buffers, m_output_frame_buffers, processed_frames, processed_frames + batch_frames);

        for (int i = 0; i < batch_frames; ++i) {
            if (!m_encoder->encodeFrame(m_output_frame_buffers[i])) {
                throw std::runtime_error("Failed to encode frame");
            }
        }

        processed_frames += batch_frames;

        if (processed_frames % report_interval == 0 || processed_frames == m_total_frames) {
            auto current_time = std::chrono::high_resolution_clock:now();
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(current_time - start_time);
            double progress = static_cast<double>(processed_frames) / m_total_frames;
            double fps = static_cast<double>(processed_frames) / elapsed.count();

            std::cout << std::format("\rProgress: {:.1f}% ({}/{} frames) | FPS: {:.2f} | Elapsed: {}s",
                                        progress * 100, processed_frames, m_total_frames, fps, elapsed.count());

            std::cout.flush();
        }
    }

    std::cout << "\n";

    m_encoder->flush();

}


void VideoProcessor::processFrameBatch(std::vector<std::vector<uint8_t>>& input_batch,
                                        std::vector<std::vector<uint8_t>>& output_batch,
                                        int start_frame, int end_frame) {

        #pragma omp parallel for num_threads(m_num_threads)
        for (int i = 0; i < end_frame - start_frame; ++i) {
            m_upscaler->upscale(input_batch[i].data(), output_batch[i].data());
        }

}

void VideoProcessor::printOutputSummary() {
    std::cout << "\nOutput Summary: \n";
    std::cout << std::format("Output file: {}\n", m_output_file);
    std::cout << std::format("Output resolution: {}x{}\n", m_output_width, m_output_height);
    std::cout << std::format("Frame rate: {:.2f} FPS\n", m_frame_rate);
    std::cout << std::format("Total Frames: {}\n", m_total_frames);
}

void VideoProcessor::finalize() {
    
    std::cout << "Process completed!\n";

    if (m_cuda_stream) {
        cudaStreamDestroy(m_cuda_stream);
        m_cuda_stream = nullptr;
    }

    m_decoder.reset();
    m_encoder.reset();
    m_upscaler.reset();
    m_audio_processor.reset();

    std::cout << "Video prcessor cleaned and reset.";
}

