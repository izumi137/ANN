#include "crossentropy.h"

float CE::crossEntropyLoss(const vector<vector<float>>& predictions, const vector<vector<int>>& labels) {
    float loss = 0.0f;
    
    for (size_t i = 0; i < predictions.size(); ++i) {
        int label = labels[i][0];
        loss -= log(predictions[i][label] + 1e-9f);
    }
    float sz = static_cast<float>(predictions.size());
    loss /= sz;
    return loss;
}