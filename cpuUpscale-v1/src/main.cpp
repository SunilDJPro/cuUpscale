#include "jpeg_cpu.h"
#include "upscaler.h"
#include <iostream>

int main(int argc, char** argv) {
    if (argc != 5) {
        std::cerr << "Usage: " << argv[0] << " <input.jpg> <output.jpg> <scale_factor> <method>" << std::endl;
        return 1;
    }

    std::vector<unsigned char> image_data;
    int width, height, channels;

    // Read JPEG
    JPEGProcessor::read_jpeg_file(argv[1], image_data, width, height, channels);
    std::cout << "Image read: " << width << "x" << height << " with " << channels << " channels" << std::endl;

    // Parse scale factor and method
    float scale_factor = std::stof(argv[3]);
    UpscaleMethod method = (std::string(argv[4]) == "bicubic") ? UpscaleMethod::Bicubic : UpscaleMethod::Lanczos;

    // Upscale
    int new_width = static_cast<int>(width * scale_factor);
    int new_height = static_cast<int>(height * scale_factor);
    std::vector<unsigned char> upscaled_image = Upscaler::upscale(image_data, width, height, channels, new_width, new_height, method);

    // Write JPEG
    JPEGProcessor::write_jpeg_file(argv[2], upscaled_image, new_width, new_height, channels, 90);
    std::cout << "Upscaled image written to " << argv[2] << std::endl;

    return 0;
}