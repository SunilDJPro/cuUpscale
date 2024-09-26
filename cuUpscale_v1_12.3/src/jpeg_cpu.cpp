#include "jpeg_cpu.h"
#include <jpeglib.h>
#include <iostream>

void JPEGProcessor::read_jpeg_file(const std::string& filename, std::vector<unsigned char>& image_data, int& width, int& height, int& channels) {
    struct jpeg_decompress_struct cinfo;
    struct jpeg_error_mgr jerr;

    FILE* infile = fopen(filename.c_str(), "rb");
    if (!infile) {
        std::cerr << "Error opening input file: " << filename << std::endl;
        return;
    }

    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_decompress(&cinfo);
    jpeg_stdio_src(&cinfo, infile);

    jpeg_read_header(&cinfo, TRUE);
    jpeg_start_decompress(&cinfo);

    width = cinfo.output_width;
    height = cinfo.output_height;
    channels = cinfo.output_components;

    int row_stride = width * channels;
    image_data.resize(height * row_stride);

    while (cinfo.output_scanline < cinfo.output_height) {
        unsigned char* row_pointer = &image_data[cinfo.output_scanline * row_stride];
        jpeg_read_scanlines(&cinfo, &row_pointer, 1);
    }

    jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);
    fclose(infile);
}

void JPEGProcessor::write_jpeg_file(const std::string& filename, const std::vector<unsigned char>& image_data, int width, int height, int channels, int quality) {
    struct jpeg_compress_struct cinfo;
    struct jpeg_error_mgr jerr;

    FILE* outfile = fopen(filename.c_str(), "wb");
    if (!outfile) {
        std::cerr << "Error opening output file: " << filename << std::endl;
        return;
    }

    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_compress(&cinfo);
    jpeg_stdio_dest(&cinfo, outfile);

    cinfo.image_width = width;
    cinfo.image_height = height;
    cinfo.input_components = channels;
    cinfo.in_color_space = (channels == 3) ? JCS_RGB : JCS_GRAYSCALE;

    jpeg_set_defaults(&cinfo);
    jpeg_set_quality(&cinfo, quality, TRUE);

    jpeg_start_compress(&cinfo, TRUE);

    int row_stride = width * channels;
    while (cinfo.next_scanline < cinfo.image_height) {
        const unsigned char* row_pointer = &image_data[cinfo.next_scanline * row_stride];
        jpeg_write_scanlines(&cinfo, const_cast<JSAMPARRAY>(&row_pointer), 1);
    }

    jpeg_finish_compress(&cinfo);
    jpeg_destroy_compress(&cinfo);
    fclose(outfile);
}