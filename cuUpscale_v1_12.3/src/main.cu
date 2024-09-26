#include "jpeg_cpu.h"
#include "lanczos.cuh"
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <input_file> <output_file> <scale_factor>" << std::endl;
        return 1;
    }

    std::string input_file = argv[1];
    std::string output_file = argv[2];
    float scale_factor = std::stof(argv[3]);

    std::vector<unsigned char> image_data;
    int width, height, channels;

    JPEGProcessor::read_jpeg_file(input_file, image_data, width, height, channels);

    int new_width = static_cast<int>(width * scale_factor);
    int new_height = static_cast<int>(height * scale_factor);

    std::vector<unsigned char> upscaled_image = cuLanczos::upscale(image_data, width, height, channels, new_width, new_height);

    JPEGProcessor::write_jpeg_file(output_file, upscaled_image, new_width, new_height, channels, 90);

    std::cout << "Image upscaled and saved successfully." << std::endl;

    return 0;
}