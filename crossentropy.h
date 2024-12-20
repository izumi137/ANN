#pragma once
#include <vector>
using namespace std;
class CE
{
public:
    float crossEntropyLoss(const vector<vector<float>>& predictions, const vector<vector<int>>& labels);
};

