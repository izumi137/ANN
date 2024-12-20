
//#include <iostream>
#include "Dataloader.h"
#include <chrono>

#include "model.h"
#include "utils.h"
using namespace std;
const int SIZE = 28;

int main()
{
    int n_in = SIZE * SIZE;
    int n_hidden = 128;
    int n_out = 10;
    int n_epochs = 10;
    int bs = 32;
    float lr = 0.01f;
    chrono::steady_clock::time_point begin, end;
    string train_file = "D:\\AI_HCMUS\\Nam4\\Parallel Programing\\pj\\ANN\\train.txt";
    string test_file = "D:\\AI_HCMUS\\Nam4\\Parallel Programing\\pj\\ANN\\test.txt";
    // Load data
    std::cout << "Loading data...\n";
    Data trainData, testData;
    begin = std::chrono::steady_clock::now();
    trainData.load_data(train_file, -1);
    testData.load_data(test_file, -1);
    end = std::chrono::steady_clock::now();
    std::cout << "Data reading time: " << (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) / 1000000.0f << std::endl;
    //trainData.print_sample();

    //print_sample(train_data[0]);


    // Model
    Model model(n_in, n_hidden, n_out, lr);

    // Lossfunction
    CE ce;

    // Training

    vector<vector<float>> batchdata;
    vector<vector<int>> batchlabel;

    vector<vector<float>> alltraindt = batchnorm(trainData.get_all_data());
    vector<vector<int>> alltrainlb = trainData.get_all_label();
    float loss = 0;
    cout << "Training..." << endl;
    begin = std::chrono::steady_clock::now();
    for (int epoch = 0; epoch < n_epochs; epoch++) {
        chrono::steady_clock::time_point begin_e, end_e;
        begin_e = std::chrono::steady_clock::now();
        vector<vector<int>> predicts, output;
        for (int b = 0; b < trainData.size(); b += bs)
        {
            bool flag = false;
            batchdata = trainData.get_batch_data(bs, b, flag);
            batchlabel = trainData.get_batch_label(bs, b, flag);
            if (flag == false)
                continue;

            vector<vector<float>> inputs = batchnorm(batchdata);

            vector<vector<float>> predictions = model.forward(inputs);
            loss = ce.crossEntropyLoss(predictions, batchlabel);

            vector<vector<float>> grad_output = predictions;

            for (size_t i = 0; i < batchlabel.size(); ++i)
            {
                int label = batchlabel[i][0];
                grad_output[i][label] -= 1.0;
            }

            model.backward(grad_output);
            //model.update_weights(lr);

        }

        vector<vector<float>>predictions = model.forward(alltraindt);
        predicts = get_arg_max(predictions);
        float acc = accuracy(predicts, alltrainlb);
        //cout << endl << "Predict: ";
        //for (const auto& row : predicts)
        //{
        //   cout << row[0] << " ";
        //}
        cout << "Epoch " << epoch + 1 << ", Loss: " << loss  << ", Acc: " << acc << endl;
        end_e = std::chrono::steady_clock::now();
        std::cout << "Epoch runtime: " << (std::chrono::duration_cast<std::chrono::microseconds>(end_e - begin_e).count()) / 1000000.0f << std::endl;
    }
    end = std::chrono::steady_clock::now();
    std::cout << "Total training time: " << (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) / 1000000.0f << std::endl;


    //--------------------Test-----------------------------
    cout << "Tesing..." << endl;
    begin = std::chrono::steady_clock::now();
    vector<vector<float>> xtest = batchnorm(testData.get_all_data());
    vector<vector<int>> ytest = testData.get_all_label();

    vector<vector<float>>predictions = model.forward(xtest);
    vector<vector<int>> predicts = get_arg_max(predictions);
    float acc = accuracy(predicts, ytest);
    cout << endl << "Predict: ";
    //for (const auto& row : predicts)
    //{
    //   cout << row[0] << " ";
    //}
    cout << "Loss: " << loss << ", Acc: " << acc << endl;
    end = std::chrono::steady_clock::now();
    std::cout << "Total testing time: " << (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) / 1000000.0f << std::endl;
    /*Linear_CPU* linear1 = new Linear_CPU(BATCHSIZE, n_in, n_hidden);
    ReLU_CPU* relu1 = new ReLU_CPU(n_hidden);
    Linear_CPU* linear2 = new Linear_CPU(BATCHSIZE, n_hidden, n_hidden);
    ReLU_CPU* relu2 = new ReLU_CPU(n_hidden);
    Linear_CPU* linear3 = new Linear_CPU(BATCHSIZE, n_hidden, n_out);
    SoftMax_CPU* softmax1 = new SoftMax_CPU(n_out);

    std::vector<Module*> layers = { linear1, relu1, linear2, relu2, linear3, softmax1 };
    Sequential_CPU seq(layers);

    begin = std::chrono::steady_clock::now();
    train_cpu(seq, trainData, BATCHSIZE, n_in, n_epochs);
    end = std::chrono::steady_clock::now();
    std::cout << "Training time: " << (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) / 1000000.0f << std::endl;*/

    return 0;
}

