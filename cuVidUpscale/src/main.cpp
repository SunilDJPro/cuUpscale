#include <iostream>
#include <string>
#include <vector>
#include <stdexcept>
#include "video_processor.hpp"

int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <input_file> <output_file> <scale_factor>" << std::endl;
        return 1;
    }

    std::string input_file = argv[1];
    std::string output_file = argv[2];
    float scale_factor = std::stdof(argv[3]);

    try {
        VideoProcessor processor(input_file, output_file, scale_factor);
        processor.prcess();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "Video processed successfully." << std::endl;
    return 0;
}