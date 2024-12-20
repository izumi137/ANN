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
public:
    Model(int _n_in, int _h1, int _n_out, float _lr) {
        n_in = _n_in;
        n_out = _n_out;
        h1 = _h1;
        lr = _lr;
        layers.push_back(new Linear(_n_in, _h1));
        layers.push_back(new ReLU());
        layers.push_back(new Linear(_h1, _h1));
        layers.push_back(new ReLU());
        layers.push_back(new Linear(_h1, _n_out));
        layers.push_back(new Softmax());
    }

    vector<vector<float>> forward(const vector<vector<float>>& input) {
        vector<vector<float>> output = input;
        int n_layer = layers.size();
        for (int i = 0; i < n_layer; ++i)
        {
            output = layers[i]->forward(output);
        }
            

        return output;
    }

    void backward(const vector<vector<float>>& grad_output) {
        vector<vector<float>> grad = grad_output;
        int n_layer = layers.size();

        for (int i = n_layer-1; i > 0; --i)
        {
            grad = layers[i]->backward(grad);
            layers[i]->update_weights(lr);
        }
    }

    //void update_weights(float lr) {
    //    for (auto& layer : layers)
    //        
    //}
};
