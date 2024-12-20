#include "relu.h"

vector<vector<float>> ReLU::forward(const vector<vector<float>>& _input)  {
    this->input = _input;
    int batch_size = input.size();
    int sz = input[0].size();
    vector<vector<float>> output = _input;
    for (int b = 0; b < batch_size; b++)
        for (int i = 0; i < sz; i++)
            output[b][i] = max(0.0f, output[b][i]);
    return output;
}

vector<vector<float>> ReLU::backward(const vector<vector<float>>& grad) {
    vector<vector<float>> gradInput = grad;
    for (size_t i = 0; i < input.size(); ++i)
        for (size_t j = 0; j < input[0].size(); ++j)
            gradInput[i][j] *= (input[i][j] > 0 ? 1.0f : 0.0f);
    return gradInput;
}