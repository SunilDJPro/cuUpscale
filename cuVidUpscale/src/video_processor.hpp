#pragma once
#include <string>
#include <memory>
#include <vector>
#include <cuda_runtime.h>
#include <omp.h>



class NvDecWrapper;
class NvEncWrapper;
class LanczosUpscaler;
class AudioProcessor;


class VideoProcessor {
    public: 
        VideoProcessor(const std::string& input_file, const std::string& output_file, float scale_factor);
        ~VideoProcessor(); //Destructor

        void process();

    private:
        void initialize();
        void processFrames();
        void processFrameBatch(std::vector<std::vector<uint8_t>>& input_batch, 
                                std::vector<std::vector<uint8_t>>& output_batch,
                                int start_frame, int end_frame);

        void finalize();
        void printInputDigest();
        void printOutputSummary();

        std::string m_input_file;
        std::string m_output_file;
        float m_scale_factor;

        std::unique_ptr<NvDecWrapper> m_decoder;
        std::unique_ptr<NvEncWrapper> m_encoder;
        std::unique_ptr<LanczosUpscaler> m_upscaler;
        std::unique_ptr<AudioProcessor> m_audio_prcessor;

        cudaStream_t m_cuda_stream; //CuStream define

        std::vector<std::vector<uint8_t>> m_input_frame_buffers;
        std::vector<std::vector<uint8_t>> m_output_frame_buffers;

        int m_input_width; //Video Properties
        int m_input_height;
        int m_output_width;
        float m_frame_rate;
        int m_total_frames;

        int m_num_threads; //OpenMP
        int m_batch_size;

};

