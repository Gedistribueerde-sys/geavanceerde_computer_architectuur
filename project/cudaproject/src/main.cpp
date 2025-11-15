#include "las_reader.h"


int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <input.las>\n";
        return 1;
    }
    
    // 1. Read the las file
    std::string inputFile = argv[1];
    PointCloudVecs pcVecs = readLASFileNormalized(inputFile);
    
    size_t totalPoints = pcVecs.x.size();   
    std::cout << "Total points read: " << totalPoints << "\n";
    return 0;
}