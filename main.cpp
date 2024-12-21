
//#include <iostream>
#include "Dataloader.h"
#include <chrono>

#include "model.h"
#include "utils.h"
using namespace std;
const int SIZE = 28;
struct CFG
{
    int epochs, batch_size, n_in, n_out;
    float lr;
    int h1;
};

float train(
    const vector<vector<float>>& xtrain,
    const vector<vector<int>>& ytrain,
    const vector<vector<float>>& xtest,
    const vector<vector<int>>& ytest,
    const CFG& cfg,
    Model& model,
    CE& ce
);

void test(
    const vector<vector<float>>& xtest,
    const vector<vector<int>>& ytest,
    Model& model,
    float& acc
);


int main()
{
    CFG cfg;
    cfg.batch_size = 32;
    cfg.epochs = 10;
    cfg.h1 = 128;
    cfg.lr = 0.01f;
    cfg.n_in = SIZE * SIZE;
    cfg.n_out = 10;

    int n_in = SIZE * SIZE;
    int n_hidden = 128;
    int n_out = 10;
    int n_epochs = 10;
    int bs = 32;
    float lr = 0.01f;
    chrono::steady_clock::time_point begin, end;
    string train_file = "D:\\AI_HCMUS\\Nam4\\Parallel Programing\\pj\\ANN\\train.txt";
    string test_file = "D:\\AI_HCMUS\\Nam4\\Parallel Programing\\pj\\ANN\\test.txt";
    string valid_file = "D:\\AI_HCMUS\\Nam4\\Parallel Programing\\pj\\ANN\\valid.txt";
    // Load data
    std::cout << "Loading data...\n";

    Data trainData, testData, validData;
    begin = std::chrono::steady_clock::now();
    trainData.load_data(train_file, -1);
    testData.load_data(test_file, -1);
    validData.load_data(valid_file, -1);
    end = std::chrono::steady_clock::now();
    std::cout << "Data reading time: " << (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) / 1000000.0f << std::endl;
    //trainData.print_sample();

    // Model
    Model model(n_in, n_hidden, n_out, lr);

    // Lossfunction
    CE ce;

    // -------------------Load data---------------------------
    vector<vector<float>> x_train = batchnorm(trainData.get_all_data());
    vector<vector<int>> y_train = trainData.get_all_label();

    vector<vector<float>> x_valid = batchnorm(validData.get_all_data());
    vector<vector<int>> y_valid = validData.get_all_label();

    vector<vector<float>> x_test = batchnorm(testData.get_all_data());
    vector<vector<int>> y_test = testData.get_all_label();

    //--------------------Train-----------------------------
    cout << "Training..." << endl;
    float loss = train(x_train, y_train, x_valid, y_valid, cfg, model, ce);
    end = std::chrono::steady_clock::now();
    std::cout << "Total training time: " << (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) / 1000000.0f << std::endl;


    //--------------------Test-----------------------------
    cout << "Tesing..." << endl;

    begin = std::chrono::steady_clock::now();
    float acc = 0.0f;
    test(x_test, y_test, model, acc);
    end = std::chrono::steady_clock::now();
    float runtime = (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) / 1000000.0f;
    std::cout << "Total testing time: " << runtime << std::endl;

    // --------------------Write result-----------------------------
    std::ofstream outfile;
    string filename = "result.txt";
    outfile.open(filename, std::ios::app);

    if (!outfile.is_open()) {
        std::cerr << "Couldn't open file " << filename << std::endl;
        return 1;
    }

    outfile << "CPU: " << "AVG Loss: " << loss << ", " << "Test accuracy: " << acc << endl;
    outfile.close();
    return 0;
}


void test(
    const vector<vector<float>>& xtest,
    const vector<vector<int>>& ytest,
    Model& model,
    float& acc
)
{
    vector<vector<float>> x = batchnorm(xtest);
    vector<vector<float>>predictions = model.forward(x);
    auto predicts = get_arg_max(predictions);
    acc = accuracy(predicts, ytest);
}

float train(
    const vector<vector<float>>& xtrain,
    const vector<vector<int>>& ytrain,
    const vector<vector<float>>& xtest,
    const vector<vector<int>>& ytest,
    const CFG& cfg,
    Model& model,
    CE& ce
)
{
    chrono::steady_clock::time_point begin, end;
    begin = std::chrono::steady_clock::now();
    float loss = 0;
    for (int epoch = 0; epoch < cfg.epochs; epoch++) {
        
        chrono::steady_clock::time_point begin_e, end_e;
        vector<vector<float>> xbatch(cfg.batch_size, vector<float>(cfg.n_in, 0.0f));
        vector<vector<int>> ybatch(cfg.batch_size, vector<int>(1, 0));

        begin_e = std::chrono::steady_clock::now();
        vector<vector<int>> predicts, output;
        for (int b = 0; b < xtrain.size(); ++b)
        {
            bool flag = true;
            flag = getBatchData(xtrain, ytrain, xbatch, ybatch, b, cfg.batch_size);
            if (flag == 0) // drop last batch if not enough
                break;

            // Device pixel value by 255
            vector<vector<float>> inputs = batchnorm(xbatch);

            // Forward
            vector<vector<float>> predictions = model.forward(inputs);

            // Calculate loss
            loss = ce.crossEntropyLoss(predictions, ybatch);

            // Backward and update
            vector<vector<float>> grad_output = predictions;

            for (size_t i = 0; i < ybatch.size(); ++i)
            {
                int label = ybatch[i][0];
                grad_output[i][label] -= 1.0; // Minus the correct layer probability by 1
            }

            model.backward(grad_output);
            //model.update_weights(lr);
        }
        float acc;
        test(xtest, ytest, model, acc);
        cout << "Epoch " << epoch + 1 << ", Loss: " << loss << ", Acc: " << acc << endl;
        end_e = std::chrono::steady_clock::now();
        std::cout << "Epoch runtime: " << (std::chrono::duration_cast<std::chrono::microseconds>(end_e - begin_e).count()) / 1000000.0f << std::endl;
    }

    // Calculate runtime
    float runtime = (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) / 1000000.0f;
    return runtime;
}