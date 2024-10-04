#pragma once

#include <string>
#include <memory>
#include <vector>
#include <cuda.h>
#include <nvcuvid.h>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
}

class NvDecWrapper {
public:
    NvDecWrapper(const std::string& input_file);
    ~NvDecWrapper();

    bool decodeNextFrame(std::vector<uint8_t>& frame_buffer);
    int getWidth() const { return m_width; }
    int getHeight() const { return m_height; }
    float getFrameRate() const { return m_frame_rate; }
    int getTotalFrames() const { return m_total_frames; }

private:
    bool initialize();
    void cleanup();

    static int CUDAAPI handleVideoSequence(void* user_data, CUVIDEOFORMAT* format);
    static int CUDAAPI handlePictureDecode(void* user_data, CUVIDPICPARAMS* pic_params);
    static int CUDAAPI handlePictureDisplay(void* user_data, CUVIDPARSERDISPINFO* disp_info);

    std::string m_input_file;
    CUcontext m_cuda_context;
    CUvideoctxlock m_ctx_lock;
    CUvideoparser m_video_parser;
    CUvideodecoder m_video_decoder;

    AVFormatContext* m_format_context;
    AVCodecContext* m_codec_context;
    AVPacket* m_packet;
    AVFrame* m_frame;
    SwsContext* m_sws_context;

    int m_width;
    int m_height;
    float m_frame_rate;
    int m_total_frames;

    std::vector<uint8_t> m_decoded_frame_buffer;
};