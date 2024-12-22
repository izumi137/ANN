#include "utils.h"
vector<vector<float>> kaiming_init(vector<vector<float>>& w, int n_in, int n_out) {
    float std = sqrt(2 / (float)n_in);

    random_device rd;
    mt19937 gen(rd());
    normal_distribution<float> dist(0.0f, std);

    for (int i = 0; i < n_in; i++)
    {
        vector<float> row;
        for (int j = 0; j < n_out; j++)
            row.push_back(dist(gen));
        w.push_back(row);
    }
    return w;
}

vector<vector<int>> get_arg_max(vector<vector<float>> inp)
{
    vector<vector<int>> output(inp.size(), vector<int>(1, 0));
    for (size_t b = 0; b < inp.size(); b++)
    {
        int max_idx = 0;
        for (size_t i = 0; i < inp[0].size(); i++)
        {
            if (inp[b][i] > inp[b][max_idx])
                max_idx = static_cast<int>(i);
        }
        output[b][0] = max_idx;
    }
    return output;
}

template <typename T>
void print_vec(const vector<vector<T>>& inp)
{
    for (const auto& row : inp)
    {
        for (const auto& element : row)
        {
            cout << element << " ";
        }
        cout << endl;
    }
}

vector<vector<float>> batchnorm(vector<vector<float>> inp)
{
    size_t batch_size = inp.size();
    size_t feature_size = inp[0].size();
    float minRange = 0, maxRange = 1;
    
    vector<vector<float>> output = inp;
    for (int b = 0; b < batch_size; b ++)
        for (int i=0; i < feature_size; i ++)
            output[b][i] /= 255.0f;
    // Find the minimum and maximum values for each feature
    //vector<float> minVals(feature_size, numeric_limits<float>::max());
    //vector<float> maxVals(feature_size, numeric_limits<float>::lowest());

    //for (const auto& row : inp) {
    //    for (size_t j = 0; j < feature_size; ++j) {
    //        minVals[j] = min(minVals[j], row[j]);
    //        maxVals[j] = max(maxVals[j], row[j]);
    //    }
    //}

    //// Normalize the input
    //vector<vector<float>> output(batch_size, vector<float>(feature_size, 0.0f));
    //for (size_t i = 0; i < batch_size; ++i) {
    //    for (size_t j = 0; j < feature_size; ++j) {
    //        // Handle division by zero if max and min are the same
    //        if (maxVals[j] != minVals[j]) {
    //            output[i][j] = ((inp[i][j] - minVals[j]) / (maxVals[j] - minVals[j])) * (maxRange - minRange) + minRange;
    //        }
    //        else {
    //            output[i][j] = static_cast<float>(minRange); // If all values are the same, set to minRange
    //        }
    //    }
    //}

    return output;
}

float accuracy(vector<vector<int>> v1, vector<vector<int>> v2)
{
    //cout << "len: " << v1.size() << " " << v2.size();
    float sum = 0;
    for (int i = 0; i < v1.size(); i++)
        if (v1[i][0] == v2[i][0])
            sum += 1.0f / v1.size();
    return sum;
}

float maxf(const float& a, const float& b)
{
    if (a > b)
        return a;
    return b;
}

vector<vector<float>> matMul(const vector<vector<float>>& a, const vector<vector<float>>& b, int operation) {
    if (a.empty() || b.empty()) {
        throw invalid_argument("Input matrices cannot be empty");
    }

    size_t rows_a = a.size();
    size_t cols_a = a[0].size();
    size_t rows_b = b.size();
    size_t cols_b = b[0].size();

    vector<vector<float>> result;

    switch (operation) {
    case 0: // a @ b
        if (cols_a != rows_b) {
            throw invalid_argument("Invalid dimensions for a @ b");
        }
        result = vector<vector<float>>(rows_a, vector<float>(cols_b, 0.0f));
        for (size_t i = 0; i < rows_a; ++i) {
            for (size_t j = 0; j < cols_b; ++j) {
                for (size_t k = 0; k < cols_a; ++k) {
                    result[i][j] += a[i][k] * b[k][j];
                }
            }
        }
        break;

    case 1: // a @ b^T
        if (cols_a != cols_b) {
            throw invalid_argument("Invalid dimensions for a @ b^T");
        }
        result = vector<vector<float>>(rows_a, vector<float>(rows_b, 0.0f));
        for (size_t i = 0; i < rows_a; ++i) {
            for (size_t j = 0; j < rows_b; ++j) {
                for (size_t k = 0; k < cols_a; ++k) {
                    result[i][j] += a[i][k] * b[j][k];
                }
            }
        }
        break;

    case 2: // a^T @ b
        if (rows_a != rows_b) {
            throw invalid_argument("Invalid dimensions for a^T @ b");
        }
        result = vector<vector<float>>(cols_a, vector<float>(cols_b, 0.0f));
        for (size_t i = 0; i < cols_a; ++i) {
            for (size_t j = 0; j < cols_b; ++j) {
                for (size_t k = 0; k < rows_a; ++k) {
                    result[i][j] += a[k][i] * b[k][j];
                }
            }
        }
        break;

    default:
        throw invalid_argument("Invalid operation type");
    }

    return result;
}

void addBias(vector<vector<float>>& a, const vector<float>& bias)
{
    int batch_size = a.size();
    int size = a[0].size();
    for (int b = 0; b < batch_size; b++)
        for (int i = 0; i < size; i++)
            a[b][i] += bias[i];
}
//int random_int(int min, int max) {
//    random_device rd;
//    mt19937 gen(rd());
//    uniform_int_distribution<int> dist(min, max);
//    return dist(gen);
//}


bool getBatchData(
    const vector<vector<float>>& x,
    const vector<vector<int>>& y,
    vector<vector<float>>& xbatch,
    vector<vector<int>>& ybatch,
    const int& batch,
    const int& batch_size)
{
    int start = batch * batch_size;
    int end = start + batch_size;
    if (end > x.size())
        return 0;

    for (int i = start, i_b = 0; i < end; ++i, ++i_b)
    {
        for (int j = 0; j < xbatch[0].size(); ++j)
            xbatch[i_b][j] = x[i][j];
        for (int j = 0; j < ybatch[0].size(); ++j)
            ybatch[i_b][j] = y[i][j];
    }
    return 1;
}


