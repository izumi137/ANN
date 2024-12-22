#pragma once
#include "module.h"

class ReLU : public Module {
    vector<vector<float>> input;

public:
    vector<vector<float>> forward(const vector<vector<float>>& input);
    vector<vector<float>> backward(const vector<vector<float>>& grad);
};


