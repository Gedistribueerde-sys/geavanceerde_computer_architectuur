#include "las_reader.h"


int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <input.las>\n";
        return 1;
    }
    
    std::string inputFile = argv[1];
    std::vector<Point> points = readLASFileNormalized(inputFile);

    for (int i = 0; i < 50 && i < points.size(); i++) {
        const Point& p = points[i];
        std::cout << "Point " << i << ": (" << p.x << ", " << p.y << ", " << p.z  << ")"
                  << ", Color: (" << static_cast<int>(p.r) << ", " 
                  << static_cast<int>(p.g) << ", " 
                  << static_cast<int>(p.b) << ")\n";
    }
    
    std::cout << "Total points read: " << points.size() << "\n";
    return 0;
}