#pragma once
#include <iostream>
#include <vector>


using namespace std;

class Module {
public:
    virtual vector<vector<float>> forward(const vector<vector<float>>& input) = 0;
    virtual vector<vector<float>> backward(const vector<vector<float>>& grad) = 0;
    virtual void update_weights(float lr) {}
};


