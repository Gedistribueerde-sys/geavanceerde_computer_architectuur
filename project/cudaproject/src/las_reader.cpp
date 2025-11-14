#include "las_reader.h"

std::vector<Point> readLASFileNormalized(const std::string& filename) {
    using namespace pdal;

    std::cout << "Reading LAS file: " << filename << "\n";
    
    Options options;
    options.add("filename", filename);
    
    StageFactory factory;
    Stage* reader(factory.createStage("readers.las"));
    reader->setOptions(options);
    
    PointTable table;
    reader->prepare(table);
    PointViewSet viewSet = reader->execute(table);
    
    std::vector<Point> points;
    
    std::cout << "Number of point views: " << viewSet.size() << "\n";
    
    // First pass: read points and find minX, minY, minZ
    float minX = std::numeric_limits<float>::max();
    float minY = std::numeric_limits<float>::max();
    float minZ = std::numeric_limits<float>::max();
    for (auto& view: viewSet) {
        for (PointId i = 0; i < view->size(); i++) {
            float x = static_cast<float>(view->getFieldAs<double>(Dimension::Id::X, i));
            float y = static_cast<float>(view->getFieldAs<double>(Dimension::Id::Y, i));
            float z = static_cast<float>(view->getFieldAs<double>(Dimension::Id::Z, i));

            minX = std::min(minX, x);
            minY = std::min(minY, y);
            minZ = std::min(minZ, z);
        }
    }

    
    // Second pass: read points and normalize
    for (auto& view : viewSet) {
        std::cout << "Processing view with " << view->size() << " points\n";
        
        for (PointId i = 0; i < view->size(); ++i) {
            Point p = {};
            
            try {
                p.x = static_cast<float>(view->getFieldAs<double>(Dimension::Id::X, i)) - minX;
                p.y = static_cast<float>(view->getFieldAs<double>(Dimension::Id::Y, i)) - minY;
                p.z = static_cast<float>(view->getFieldAs<double>(Dimension::Id::Z, i)) - minZ;
            
                
                // RGB values are 16-bit in LAS
                if (table.layout()->hasDim(Dimension::Id::Red)) {
                    p.r = static_cast<uint8_t>(view->getFieldAs<uint16_t>(Dimension::Id::Red, i) >> 8);
                    p.g = static_cast<uint8_t>(view->getFieldAs<uint16_t>(Dimension::Id::Green, i) >> 8);
                    p.b = static_cast<uint8_t>(view->getFieldAs<uint16_t>(Dimension::Id::Blue, i) >> 8);
                }
                
                points.push_back(p);
            } catch (const std::exception& e) {
                std::cerr << "Error reading point " << i << ": " << e.what() << "\n";
                continue;
            }
        }
    }
    
    std::cout << "Successfully loaded " << points.size() << " points\n";
    
    return points;
}