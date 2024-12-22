#include "linear.h"
#include "utils.h"

Linear::Linear(int inFeatures, int outFeatures)
{
    biases.resize(outFeatures, 0.0f);
    weights = kaiming_init(weights, inFeatures, outFeatures);
}


vector<vector<float>> Linear::forward(const vector<vector<float>>& _input) 
{
    this->input = _input;
    vector<vector<float>> output(_input.size(), vector<float>(weights[0].size(), 0.0f));

    output = matMul(input, weights);
    addBias(output, biases);
    return output;
}

void Linear::update_weights(float lr) {
    //for (auto& row : weights)
    //    for (auto& w : row)
    //        w -= lr*w;
    //for (auto& b : biases)
    //    b -= lr*b;

    for (int b = 0; b < weights.size(); ++b)
        for (int i = 0; i < weights[0].size(); ++i)
            weights[b][i] -= lr * grad_w[b][i];

    for (int i = 0; i < biases.size(); ++i)
        biases[i] -= lr * grad_b[i];
}

vector<vector<float>> Linear::backward(const vector<vector<float>>& grad) {
    size_t batch_size = grad.size();
    size_t input_size = input[0].size();
    size_t output_size = grad[0].size();

    // Gradient for input, weight, bias
    grad_input = vector<vector<float>>(batch_size, vector<float>(input_size, 0.0f));
    grad_w = vector<vector<float>>(weights.size(), vector<float>(weights[0].size(), 0.0f));
    grad_b = vector<float> (biases.size(), 0.0f);

    grad_w = matMul(input, grad, 2);
    grad_input = matMul(grad, weights, 1);
    for (int i = 0; i < output_size; ++i)
    {
        float sum = 0;
        for (int b = 0; b < batch_size; ++b)
        {
            sum += grad[b][i];
        }
        grad_b[i] = sum;
    }


    return grad_input;
}

