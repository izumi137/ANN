#pragma once
#include <vector>
#include <string>
#include <iostream>

using namespace std;


class BaseData
{
private:
	vector<vector<int>> image;
	int label;
public:
	BaseData(const std::string& input);

	const std::vector<std::vector<int>>& getImage() const;

	int getLabel() const;
	float get_float_label();
	vector<float> get_float_image();
};