#include "BaseData.h"
#include <sstream>
#include <stdexcept>


BaseData::BaseData(const std::string& input) {
    std::istringstream stream(input);
    image.resize(28, std::vector<int>(28));


    for (int i = 0; i < 28; ++i) {
        for (int j = 0; j < 28; ++j) {
            if (!(stream >> image[i][j])) {
                throw std::invalid_argument("Khong du 28*28");
            }
        }
    }


    if (!(stream >> label)) {
        throw std::invalid_argument("Label khong hop le.");
    }


    int extra;
    if (stream >> extra) {
        throw std::invalid_argument("Qua nhieu gia tri.");
    }
}


const std::vector<std::vector<int>>& BaseData::getImage() const {
    return image;
}

vector<float> BaseData::get_float_image()
{
    vector<float> output;
    for (int r = 0; r < image.size(); r++)
    {
        for (int c = 0; c < image[0].size(); c++)
        {
            output.push_back(static_cast<float>(image[r][c]));
        }
    }
    return output;
}


int BaseData::getLabel() const {
    return label;
}

float BaseData::get_float_label() {
    return static_cast<float>(label);
}
