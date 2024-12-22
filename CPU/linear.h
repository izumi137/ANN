#pragma once

#include "module.h"
class Linear : public Module
{
    vector<vector<float>> weights, grad_w;
    vector<float> biases, grad_b;
    vector<vector<float>> input, grad_input;
    //int lr;
public:
    Linear(int inFeatures, int outFeatures);

    vector<vector<float>> forward(const vector<vector<float>>& _input);

    vector<vector<float>> backward(const vector<vector<float>>& grad);

    void update_weights(float lr);

};

