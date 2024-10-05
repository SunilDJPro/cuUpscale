#include "nvdec_wrapper.hpp"
#include "cuda_utils.cuh"
#include <stdexcept>
#include <iostream>

NvDecWrapper::NvDecWrapper(const std::string& input_file)
    : m_input_file(input_file), m_cuda_context(nullptr), m_ctx_lock(nullptr),
      m_video_parser(nullptr), m_video_decoder(nullptr), m_format_context(nullptr),
      m_codec_context(nullptr), m_packet(nullptr), m_frame(nullptr), m_sws_context(nullptr),
      m_width(0), m_height(0), m_frame_rate(0), m_total_frames(0) {
    if (!initialize()) {
        throw std::runtime_error("Failed to initialize NvDecWrapper");
    }
}

NvDecWrapper::~NvDecWrapper() {
    cleanup();
}

bool NvDecWrapper::initialize() {
    //Init CUDA
    CUdevice cuda_device;
    CUDA_CHECK(cuInit(0));
    CUDA_CHECK(cuDeviceGet(&cuda_device, 0));
    CUDA_CHECK(cuCtxCreate(&m_cuda_context, 0, cuda_device));
    CUDA_CHECK(cuvidCtxLockCreate(&m_ctx_lock, m_cuda_context));

    //Init FFmpeg
    avformat_network_init();
    m_format_context = avformat_alloc_context();
    if (avformat_open_input(&m_format_context, m_input_file.c_str(), nullptr, nullptr) != 0) {
        std::cerr << "Could not open input file: " << m_input_file << std::endl;
        return false;
    }

    if (avformat_find_stream_info(m_format_context, nullptr) < 0) {
        std::cerr << "Could not find stream information" << std::endl;
        return false;
    }

    int video_stream_index = av_find_best_stream(m_format_context, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
    if (video_stream_index < 0) {
        std::cerr << "Could not find video stream in the input file" << std::endl;
        return false;
    }

    AVStream* video_stream = m_format_context->streams[video_stream_index];
    const AVCodec* codec = avcodec_find_decoder(video_stream->codecpar->codec_id);
    m_codec_context = avcodec_alloc_context3(codec);
    avcodec_parameters_to_context(m_codec_context, video_stream->codecpar);

    if (avcodec_open2(m_codec_context, codec, nullptr) < 0) {
        std::cerr << "Could not open codec" << std::endl;
        return false;
    }

    m_width = m_codec_context->width;
    m_height = m_codec_context->height;
    m_frame_rate = static_cast<float>(video_stream->avg_frame_rate.num) / video_stream->avg_frame_rate.den;
    m_total_frames = static_cast<int>(video_stream->nb_frames);

    //Init NVDEC
    CUVIDPARSERPARAMS parser_params = {};
    parser_params.CodecType = ffmpeg_to_cuvid_codec(m_codec_context->codec_id);
    parser_params.ulMaxNumDecodeSurfaces = 1;
    parser_params.ulClockRate = 1000000;
    parser_params.ulErrorThreshold = 100;
    parser_params.ulMaxDisplayDelay = 0;
    parser_params.uVidPicStruct = 1;
    parser_params.pUserData = this;
    parser_params.pfnSequenceCallback = handleVideoSequence;
    parser_params.pfnDecodePicture = handlePictureDecode;
    parser_params.pfnDisplayPicture = handlePictureDisplay;

    if (cuvidCreateVideoParser(&m_video_parser, &parser_params) != CUDA_SUCCESS) {
        std::cerr << "Could not create CUVID parser" << std::endl;
        return false;
    }

    m_packet = av_packet_alloc();
    m_frame = av_frame_alloc();

    m_decoded_frame_buffer.resize(m_width * m_height * 3);

    return true;
}

bool NvDecWrapper::decodeNextFrame(std::vector<uint8_t>& frame_buffer) {
    if (av_read_frame(m_format_context, m_packet) >= 0) {
        CUVIDSOURCEDATAPACKET cuda_packet = {};
        cuda_packet.payload = m_packet->data;
        cuda_packet.payload_size = m_packet->size;
        cuda_packet.flags = CUVID_PKT_TIMESTAMP;
        cuda_packet.timestamp = m_packet->pts;

        if (cuvidParseVideoData(m_video_parser, &cuda_packet) != CUDA_SUCCESS) {
            std::cerr << "Error parsing video data" << std::endl;
            return false;
        }

        //Copy decoded frame from GPU to CPU -> temp
        if (!m_decoded_frame_buffer.empty()) {
            frame_buffer = m_decoded_frame_buffer;
            m_decoded_frame_buffer.clear();
            return true;
        }
    }

    return false;
}

void NvDecWrapper::cleanup() {
    if (m_video_decoder) {
        cuvidDestroyDecoder(m_video_decoder);
    }
    if (m_video_parser) {
        cuvidDestroyVideoParser(m_video_parser);
    }
    if (m_ctx_lock) {
        cuvidCtxLockDestroy(m_ctx_lock);
    }
    if (m_cuda_context) {
        cuCtxDestroy(m_cuda_context);
    }

    if (m_sws_context) {
        sws_freeContext(m_sws_context);
    }
    if (m_frame) {
        av_frame_free(&m_frame);
    }
    if (m_packet) {
        av_packet_free(&m_packet);
    }
    if (m_codec_context) {
        avcodec_free_context(&m_codec_context);
    }
    if (m_format_context) {
        avformat_close_input(&m_format_context);
    }
}

int CUDAAPI NvDecWrapper::handleVideoSequence(void* user_data, CUVIDEOFORMAT* format) {
    NvDecWrapper* decoder = static_cast<NvDecWrapper*>(user_data);

    CUVIDDECODECREATEINFO decode_create_info = {};
    decode_create_info.CodecType = format->codec;
    decode_create_info.ChromaFormat = format->chroma_format;
    decode_create_info.OutputFormat = cudaVideoSurfaceFormat_NV12;
    decode_create_info.bitDepthMinus8 = format->bit_depth_luma_minus8;
    decode_create_info.DeinterlaceMode = cudaVideoDeinterlaceMode_Adaptive;
    decode_create_info.ulNumOutputSurfaces = 1;
    decode_create_info.ulCreationFlags = cudaVideoCreate_PreferCUVID;
    decode_create_info.ulNumDecodeSurfaces = 1;
    decode_create_info.vidLock = decoder->m_ctx_lock;
    decode_create_info.ulWidth = format->coded_width;
    decode_create_info.ulHeight = format->coded_height;
    decode_create_info.ulMaxWidth = format->coded_width;
    decode_create_info.ulMaxHeight = format->coded_height;

    if (cuvidCreateDecoder(&decoder->m_video_decoder, &decode_create_info) != CUDA_SUCCESS) {
        std::cerr << "Could not create CUVID decoder" << std::endl;
        return 0;
    }

    return 1;
}

int CUDAAPI NvDecWrapper::handlePictureDecode(void* user_data, CUVIDPICPARAMS* pic_params) {
    NvDecWrapper* decoder = static_cast<NvDecWrapper*>(user_data);
    
    if (cuvidDecodePicture(decoder->m_video_decoder, pic_params) != CUDA_SUCCESS) {
        std::cerr << "Error decoding picture" << std::endl;
        return 0;
    }

    return 1;
}

int CUDAAPI NvDecWrapper::handlePictureDisplay(void* user_data, CUVIDPARSERDISPINFO* disp_info) {
    NvDecWrapper* decoder = static_cast<NvDecWrapper*>(user_data);

    CUVIDPROCPARAMS proc_params = {};
    proc_params.progressive_frame = disp_info->progressive_frame;
    proc_params.second_field = 0;
    proc_params.top_field_first = disp_info->top_field_first;
    proc_params.unpaired_field = disp_info->repeat_first_field;

    CUdeviceptr mapped_frame;
    unsigned int pitch;
    if (cuvidMapVideoFrame(decoder->m_video_decoder, disp_info->picture_index, &mapped_frame, &pitch, &proc_params) == CUDA_SUCCESS) {
        CUDA_MEMCPY2D memcpy_params = {};
        memcpy_params.srcMemoryType = CU_MEMORYTYPE_DEVICE;
        memcpy_params.srcDevice = mapped_frame;
        memcpy_params.srcPitch = pitch;
        memcpy_params.dstMemoryType = CU_MEMORYTYPE_HOST;
        memcpy_params.dstHost = decoder->m_decoded_frame_buffer.data();
        memcpy_params.dstPitch = decoder->m_width * 3;
        memcpy_params.WidthInBytes = decoder->m_width * 3;
        memcpy_params.Height = decoder->m_height;

        cuMemcpy2D(&memcpy_params);
        cuvidUnmapVideoFrame(decoder->m_video_decoder, mapped_frame);
    }

    return 1;
}

cudaVideoCodec NvDecWrapper::ffmpeg_to_cuvid_codec(AVCodecID codec_id) {
    switch (codec_id) {
        case AV_CODEC_ID_H264: return cudaVideoCodec_H264;
        case AV_CODEC_ID_HEVC: return cudaVideoCodec_HEVC;
        default: throw std::runtime_error("Unsupported codec");
    }
}