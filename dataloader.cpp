#include "Dataloader.h"

int min(const int& a, const int& b)
{
    if (a > b)
        return b;
    return a;
}

void Data::load_data(const string& filepath, int maxsize)
{
    ifstream file(filepath, ios::binary);
    if (!file.is_open()) {
        throw runtime_error("Cannot open file: " + filepath);
    }

    string line;

    int count = 0;

    while (getline(file, line)) {
        if (count == maxsize)
            break;
        if (line.empty()) continue;
        BaseData data(line);
        _data.push_back(data);
        count++;
    }
    file.close();
    cout << "Loaded " << count << " data from " << filepath;

}

void Data::print_sample(int idx)
{
    cout << endl << "Print sample train[" << idx << "]" << endl;
    vector<vector<int>> image = _data[idx].getImage();
    int label = _data[idx].getLabel();

    for (int i = 0; i < image.size(); i++)
    {
        for (int j = 0; j < image[0].size(); j++)
        {
            cout << image[i][j] << ' ';
        }
        cout << endl;
    }

    cout << endl << "Label: " << label << endl;
}

vector<vector<float>> Data::get_batch_data(const int& bs, const int& idx, bool& flag)
{
    vector<vector<float>> batch_data;
    int start_i = idx * bs;
    int end_i = min(start_i + bs, _data.size());

    flag = true;
    if (end_i - start_i < bs)
        flag = false;
    for (int i = start_i; i < end_i; i++)
    {
        batch_data.push_back(_data[i].get_float_image());
    }

    return batch_data;
}

vector<vector<int>> Data::get_batch_label(const int& bs, const int& idx, bool& flag)
{
    vector<vector<int>> batch_label;
    int start_i = idx * bs;
    int end_i = min(start_i + bs, _data.size());

    flag = true;
    if (end_i - start_i < bs)
        flag = false;

    for (int i = start_i; i < end_i; i++)
    {
        vector<int> tmp;
        tmp.push_back(_data[i].getLabel());
        batch_label.push_back(tmp);
    }

    return batch_label;
}

vector<vector<float>> Data::get_all_data()
{
    vector<vector<float>> all_data;
    for (int i = 0; i < _data.size(); i++)
    {
        all_data.push_back(_data[i].get_float_image());
    }

    return all_data;
}

vector<vector<int>> Data::get_all_label()
{
    vector<vector<int>> all_label;
    for (int i = 0; i < _data.size(); i++)
    {
        vector<int> tmp;
        tmp.push_back(_data[i].getLabel());
        all_label.push_back(tmp);
    }

    return all_label;
}