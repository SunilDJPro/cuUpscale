#pragma once
#include <vector>
#include <string>

class JPEGProcessor {
public:
    static void read_jpeg_file(const std::string& filename, std::vector<unsigned char>& image_data, int& width, int& height, int& channels);
    static void write_jpeg_file(const std::string& filename, const std::vector<unsigned char>& image_data, int width, int height, int channels, int quality);
};