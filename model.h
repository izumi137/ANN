#pragma once

#include "linear.h"
#include "relu.h"
#include "crossentropy.h"
#include "softmax.h"

class Model {
private:
    vector<Module*> layers;
    int n_in, n_out, h1;
    float lr;
    int mode; // 0 : test_cpu, 1 : train_cpu
public:
    Model(int _n_in, int _h1, int _n_out, float _lr);
    vector<vector<float>> forward(const vector<vector<float>>& input);
    void backward(const vector<vector<float>>& grad_output);
    void train(const int _mode = 1);
    void test(const int _mode = 0);
};
