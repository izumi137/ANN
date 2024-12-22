#pragma once
#include "module.h"
class Softmax : public Module {
    

public:
    vector<vector<float>> forward(const vector<vector<float>>& input);

    vector<vector<float>> backward(const vector<vector<float>>& grad);
};

