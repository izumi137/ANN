#include "softmax.h"
#include <algorithm>
vector<vector<float>> Softmax::forward(const vector<vector<float>>& input)  {
    vector<vector<float>> output = input;
    for (int i = 0; i < input.size(); ++i) {
        
        float maxVal = input[i][0];
        for (int j = 0; j < output[i].size(); ++j)
            maxVal = fmaxf(maxVal, output[i][j]);

        float sumExp = 0.0f;
        for (int j = 0; j < input[i].size(); ++j) {
            output[i][j] = exp(output[i][j] - maxVal);
            sumExp += output[i][j];
        }

        const float epsilon = 1e-7f;
        for (int j = 0; j < input[i].size(); ++j) {
            output[i][j] = fmaxf((output[i][j] / sumExp), epsilon);
        }
    }

    return output;
}

vector<vector<float>> Softmax::backward(const vector<vector<float>>& grad)  {
    return grad; // Assumes softmax is followed by cross-entropy loss
}