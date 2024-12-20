#pragma once
#include <vector>
#include <iostream>
#include <cmath>
#include <random>

using namespace std;

vector<vector<float>> kaiming_init(vector<vector<float>>& w, int n_in, int n_out);
//int random_int(int min, int max);
vector<vector<int>> get_arg_max(vector<vector<float>> inp);
vector<vector<float>> batchnorm(vector<vector<float>> inp);
template <typename T> void print_vec(const vector<vector<T>>& inp);
float accuracy(vector<vector<int>> v1, vector<vector<int>> v2);
float maxf(const float& a, const float& b);
vector<vector<float>> matMul(const vector<vector<float>>& a, const vector<vector<float>>& b, int operation = 0);
void addBias(vector<vector<float>>& a, const vector<float> &bias);
