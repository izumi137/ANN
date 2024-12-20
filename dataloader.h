#pragma once

#include "BaseData.h"
#include <fstream>


using namespace std;

class Data
{
private:
	vector<BaseData> _data;
	//int bs;

public:
	Data();
	void load_data(const string& filepath, int maxsize = -1);
	void print_sample(int idx = 0);
	vector<vector<float>> get_batch_data(const int& bs, const int& idx, bool& flag);
	int size() { return _data.size(); };
	vector<vector<int>> get_batch_label(const int& bs, const int& idx, bool& flag);
	vector<vector<float>> get_all_data();
	vector<vector<int>> get_all_label();
};



