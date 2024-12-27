#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <cuda_runtime.h>

typedef struct {
    float *W1, *W2, *W3;
    float *d_W1, *d_W2, *d_W3;

    float *b1, *b2, *b3;
    float *d_b1, *d_b2, *d_b3;

    float *Z1, *Z2, *Z3, *A1, *A2;
    float *d_Z1, *d_Z2, *d_Z3, *d_A1, *d_A2;
    
    float *Y_pred;

    float *X_train, *Y_train;
    float *X_valid, *Y_valid;
    float *X_test,  *Y_test;
} ANN;

struct GpuTimer
{
    cudaEvent_t start;
    cudaEvent_t stop;

    GpuTimer()
    {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }

    ~GpuTimer()
    {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    void Start()
    {
        cudaEventRecord(start, 0);
        cudaEventSynchronize(start);
    }

    void Stop()
    {
        cudaEventRecord(stop, 0);
    }

    float Elapsed()
    {
        float elapsed;
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed, start, stop);
        return elapsed;
    }
};

// C(mxk) = A(mxn) @ B(nxk)
void matMulAB(float *C, float *A, float *B, int m, int n, int k)
{
    for (int row = 0; row < m; ++row)
    {
        for (int col = 0; col < k; ++col)
        {
            float sum = 0.0f;
            for (int i = 0; i < n; ++i) 
                sum += A[row * n + i] * B[i * k + col];

            C[row * k + col] = sum;
        }
    }
}

// C(mxk) = A(nxm)^T @ B(nxk)
void matMulATB(float *C, float *A, float *B, int m, int n, int k)
{
    for (int row = 0; row < m; ++row)
    {
        for (int col = 0; col < k; ++col)
        {
            float sum = 0.0f;
            for (int i = 0; i < n; ++i) 
                sum += A[i * m + row] * B[i * k + col];

            C[row * k + col] = sum;
        }
    }
}

// C(mxk) = A(mxn) @ B(kxn)^T / d
void matMulABT(float *C, float *A, float *B, int m, int n, int k)
{
    for (int row = 0; row < m; ++row)
    {
        for (int col = 0; col < k; ++col)
        {
            float sum = 0.0f;
            for (int i = 0; i < n; ++i) 
                sum += A[row * n + i] * B[col * n + i];

            C[row * k + col] = sum;
        }
    }
}

// Z(mxk) + b(1xk)
void addBias(float *Z, float *b, int m, int k) 
{
    for (int row = 0; row < m; ++row)
        for (int col = 0; col < k; ++col)
            Z[row * k + col] += b[col];
}

// C(mxk) = A(mxk) - B(mxk)
void matSub(float *C, float *A, float *B, int m, int k)
{
    for (int row = 0; row < m; ++row)
    {
        for (int col = 0; col < k; ++col)
        {
            int idx = row * k + col;
            C[idx] = A[idx] - B[idx];
        }
    }
}

// relu(A(mxk)) = max(Z(mxk), 0)
void relu(float *A, float *Z, int m, int k) 
{
    for (int row = 0; row < m; ++row)
    {
        for (int col = 0; col < k; ++col)
        {
            int idx = row * k + col;
            A[idx] = fmaxf(0.0f, Z[idx]);
        }
    }
}

// drelu(d_Z(mxk)) = d_A*(Z > 0) 
void drelu(float *d_Z, float *d_A, float *Z, int m, int k) 
{
    for (int row = 0; row < m; ++row)
    {
        for (int col = 0; col < k; ++col)
        {
            int idx = row * k + col;
            d_Z[idx] = d_A[idx] * ((Z[idx] > 0.0f) ? 1.0f : 0.0f);
        }
    }
}

// A(BATCHSIZExLENGTH) = softmax(Z(BATCH_SIZExLENGTH))
void softmax(float *A, float *Z, int BATCH_SIZE, int length) 
{
    for (int b = 0; b < BATCH_SIZE; ++b) 
    {
        float mx = Z[b * length];
        for (int i = 1; i < length; ++i) 
            mx = fmaxf(mx, Z[b * length + i]);

        float sum = 0.0f;
        for (int i = 0; i < length; ++i) 
        {
            A[b * length + i] = expf(Z[b * length + i] - mx);
            sum += A[b * length + i];
        }

        const float epsilon = 1e-7f;
        for (int i = 0; i < length; ++i) 
            A[b * length + i] = A[b * length + i] / fmaxf(sum, epsilon); //avoid divide by zero
    }
}

// d_b = sumBatch(d_Z) = sum_batch(BATCHSIZExLENGTH) = (1xLENGTH)
void sumBatch(float *d_b, float *d_Z, int BATCH_SIZE, int length)
{
    for (int l = 0; l < length; ++l)
    {
        float sum = 0.0f;
        for (int i = 0; i < BATCH_SIZE; ++i)
            sum += d_Z[i * length + l];

        d_b[l] = sum;
    }
}

// W(mxk) -= LR * d_W(mxk)
void updateWeight2D(float *W, float *d_W, int m, int k, float LEARNING_RATE)
{
    for (int row = 0; row < m; ++row)
    {
        for (int col = 0; col < k; ++col)
        {
            int idx = row * k + col;
            W[idx] -= LEARNING_RATE * d_W[idx];
        }
    }
}

// b(1xk) -= LR * d_b(1xk)
void updateWeight1D(float *b, float *d_b, int k, float LEARNING_RATE)
{
    for (int idx = 0; idx < k; ++idx)
        b[idx] -= LEARNING_RATE * d_b[idx];
}

// Initialize weight matrix W (size mxk)
// Credit: Claude 3.5 Sonnet
void initWeight(float *W, int m, int k)
{
    float scale = sqrtf(2.0f / k); // He initialization scaling factor
    for (int i = 0; i < m * k; i++) 
        W[i] = ((float)rand() / RAND_MAX) * 2 * scale - scale; // Uniform distribution [-scale, scale]
}

void initBias(float *b, int k)
{
    for (int i = 0; i < k; i++) 
        b[i] = 0.0f; 
}


void initANN(ANN *nn, int BATCH_SIZE)
{
    nn->W1 = (float*)malloc(128 * 784 * sizeof(float));
    nn->W2 = (float*)malloc(128 * 128 * sizeof(float));
    nn->W3 = (float*)malloc(10  * 128 * sizeof(float));

    nn->d_W1 = (float*)malloc(128 * 784 * sizeof(float));
    nn->d_W2 = (float*)malloc(128 * 128 * sizeof(float));
    nn->d_W3 = (float*)malloc(10  * 128 * sizeof(float));

    nn->b1 = (float*)malloc(128 * sizeof(float));
    nn->b2 = (float*)malloc(128 * sizeof(float));
    nn->b3 = (float*)malloc(10  * sizeof(float));

    nn->d_b1 = (float*)malloc(128 * sizeof(float));
    nn->d_b2 = (float*)malloc(128 * sizeof(float));
    nn->d_b3 = (float*)malloc(10  * sizeof(float));

    nn->Z2 = (float*)malloc(50000 * 128 * sizeof(float));
    nn->Z3 = (float*)malloc(50000 * 10  * sizeof(float));
    nn->A1 = (float*)malloc(50000 * 128 * sizeof(float));
    nn->A2 = (float*)malloc(50000 * 128 * sizeof(float));
    nn->Z1 = (float*)malloc(50000 * 128 * sizeof(float));

    nn->d_Z1 = (float*)malloc(BATCH_SIZE * 128 * sizeof(float));
    nn->d_Z2 = (float*)malloc(BATCH_SIZE * 128 * sizeof(float));
    nn->d_Z3 = (float*)malloc(BATCH_SIZE * 10  * sizeof(float));
    nn->d_A1 = (float*)malloc(BATCH_SIZE * 128 * sizeof(float));
    nn->d_A2 = (float*)malloc(BATCH_SIZE * 128 * sizeof(float));

    nn->X_train = (float*)malloc(50000 * 784 * sizeof(float));
    nn->X_valid = (float*)malloc(10000 * 784 * sizeof(float));
    nn->X_test  = (float*)malloc(10000 * 784 * sizeof(float));
    nn->Y_train = (float*)malloc(50000 * 10  * sizeof(float));
    nn->Y_valid = (float*)malloc(10000 * 10  * sizeof(float));
    nn->Y_test  = (float*)malloc(10000 * 10  * sizeof(float));

    nn->Y_pred  = (float*)malloc(50000 * 10  * sizeof(float));

    initWeight(nn->W1, 128, 784);
    initWeight(nn->W2, 128, 128);
    initWeight(nn->W3, 10 , 128);

    initBias(nn->b1, 128);
    initBias(nn->b2, 128);
    initBias(nn->b3, 10);
}


void forward(ANN *nn, float *X, int BATCH_SIZE)
{   
    // Z1 = X @ W1^T + b1 = (32x784) @ (128x784)^T = (32x128) 
    matMulABT(nn->Z1, X, nn->W1, BATCH_SIZE, 784, 128);
    addBias(nn->Z1, nn->b1, BATCH_SIZE, 128);
    // A1 = relu(Z1) = (32x128)
    relu(nn->A1, nn->Z1, BATCH_SIZE, 128);

    // Z2 = A1 @ W2^T + b2 = (32x128) @ (128x128)^T = (32x128)
    matMulABT(nn->Z2, nn->A1, nn->W2, BATCH_SIZE, 128, 128);
    addBias(nn->Z2, nn->b2, BATCH_SIZE, 128);
    // A2 = relu(Z2) = (32x128)
    relu(nn->A2, nn->Z2, BATCH_SIZE, 128);

    // Z3 = A2 @ W3^T + b3 = (32x128) @ (10x128)^T = (32x10)
    matMulABT(nn->Z3, nn->A2, nn->W3, BATCH_SIZE, 128, 10);
    addBias(nn->Z3, nn->b3, BATCH_SIZE, 10);
    // Y_pred = softmax(Z3) = (32x10)
    softmax(nn->Y_pred, nn->Z3, BATCH_SIZE, 10);
}


void backward(ANN *nn, float *X, float *Y_true, int BATCH_SIZE, dim3 bs2 = dim3(32, 32), dim3 bs1 = dim3(32))
{
    // Layer: Output
    // d_Z3 = Y_pred - Y_true = (32x10)
    matSub(nn->d_Z3, nn->Y_pred, Y_true, BATCH_SIZE, 10);
    // d_W3 = d_Z3^T @ A2 = (32x10)^T @ (32x128) = (10x128)
    matMulATB(nn->d_W3, nn->d_Z3, nn->A2, 10, BATCH_SIZE, 128);
    // d_b3 = sum_batch(d_Z3) = sum_batch(32x10) = (1, 10)
    sumBatch(nn->d_b3, nn->d_Z3, BATCH_SIZE, 10);

    // Layer: Hidden 2
    // d_A2 = d_Z3 @ W3 = (32x10) x (10x128) = (32x128)
    matMulAB(nn->d_A2, nn->d_Z3, nn->W3, BATCH_SIZE, 10, 128);
    // d_Z2 = d_relu(d_A2, Z2) = d_relu(32x128) = (32x128)
    drelu(nn->d_Z2, nn->d_A2, nn->Z2, BATCH_SIZE, 128);
    // d_W2 = d_Z2^T @ A_1= (32x128)^T @ (32x128) = (128x128)
    matMulATB(nn->d_W2, nn->d_Z2, nn->A1, 128, BATCH_SIZE, 128);
    // d_b2 = sum_batch(d_Z2) = sum_batch(32x128) = (1, 128)  
    sumBatch(nn->d_b2, nn->d_Z2, BATCH_SIZE, 128);

    // Layer: Hidden 1
    // d_A1 = d_Z2 @ W2 = (32x128) x (128x128) = (32x128)
    matMulAB(nn->d_A1, nn->d_Z2, nn->W2, BATCH_SIZE, 128, 128);
    // d_Z1 = d_relu(d_A1, Z1) = d_relu(32x128) = (32x128)
    drelu(nn->d_Z1, nn->d_A1, nn->Z1, BATCH_SIZE, 128);
    // d_W1 = d_Z1^T @ X = (32x128)^T @ (32x784) = (128x784)
    matMulATB(nn->d_W1, nn->d_Z1, X, 128, BATCH_SIZE, 784);
    // d_b1 = sum_batch(d_Z1) = sum_batch(32x128) = (1, 128)  
    sumBatch(nn->d_b1, nn->d_Z1, BATCH_SIZE, 128);

    // UPDATE WEIGHTS
    float LEARNING_RATE = 0.001f;
    updateWeight2D(nn->W1, nn->d_W1, 128, 784, LEARNING_RATE);
    updateWeight1D(nn->b1, nn->d_b1, 128, LEARNING_RATE);

    updateWeight2D(nn->W2, nn->d_W2, 128, 128, LEARNING_RATE);
    updateWeight1D(nn->b2, nn->d_b2, 128, LEARNING_RATE);

    updateWeight2D(nn->W3, nn->d_W3, 10, 128, LEARNING_RATE);
    updateWeight1D(nn->b3, nn->d_b3, 10, LEARNING_RATE);
}

void write_log(const char* filename, float* data, int size, bool end = false)
{
    FILE* file = fopen(filename, "a");
    for (int i = 0; i < size; i++)
        fprintf(file, "%f ", data[i]);
    if (end == true)
        fprintf(file, "\n");
    fclose(file);
}

void eval(ANN *nn, int mode, float* X, float *Y_true, bool save_log = false)
{
    int size = 10000;
    if (mode == 0)
        size = 50000;

    forward(nn, X, size);
    float *Y_pred = nn->Y_pred;
    
    float loss = 0, acc = 0;
    for (int i = 0; i < size; ++i)
    {
        float mx = 0.0f;
        int mx_idx = 0;
        int label_idx = 0;
        for (int j = 0; j < 10; ++j)
        {
            int idx = 10 * i + j;
            if (Y_pred[idx] > mx)
            {
                mx = Y_pred[idx];
                mx_idx = j;
            }
            if (Y_true[idx] == 1)
            {
                label_idx = j;
                loss -= logf(fmaxf(Y_pred[idx], 1e-7f));
            }
        }
        if (label_idx == mx_idx)
            acc += 1;
    }
    acc /= size;
    acc *= 100; 
    loss /= size;
    printf("Loss: %.4f, Accuracy: %.2f%%\n", loss, acc);
    free(Y_pred);
    if (save_log == true)
    {
        float data[2];
        data[0] = acc;
        data[1] = loss;
        write_log("log.txt", data, 2);
    }
}

void train(ANN *nn, int EPOCHS, int BATCH_SIZE) 
{
    float total_time = 0.0f;
    int num_batches = floor(50000 / BATCH_SIZE);
    for (int epoch = 1; epoch < EPOCHS+1; epoch++) 
    {
        GpuTimer timer;
        timer.Start();
        printf("Epoch %d/%d:\n", epoch, EPOCHS);
        cudaDeviceSynchronize();

        for (int batch = 0; batch < num_batches - 1; batch++) 
        {
            int offset = batch * BATCH_SIZE;
            forward(nn, nn->X_train + offset * 784, BATCH_SIZE);
            backward(nn, nn->X_train + offset * 784, nn->Y_train + offset * 10, BATCH_SIZE);
        }

        timer.Stop();
		float time = timer.Elapsed();
		printf("Time: %f ms\n", time);
        total_time += time;

        printf("Train: ");
        eval(nn, 0, nn->X_train, nn->Y_train, false);

        printf("Valid: ");
        eval(nn, 1, nn->X_valid, nn->Y_valid, false);
            
        printf("\n");
    }
    
    printf("Finished training\nTest: ");
    total_time /= EPOCHS;
    eval(nn, 1, nn->X_test, nn->Y_test, true); // Save log
    printf("\nAverage time per epoch: %f ms\n", total_time);
    float data[1];
    data[0] = total_time;
    write_log("log.txt", data, 1, true);
}



void readData(const char* filename, float* X, float* Y, int size)
{
    FILE* file = fopen(filename, "r");
    float val;
    int label;
    for (int i = 0; i < size; i++) 
    {
        for (int j = 0; j < 784; j++) 
        {
            fscanf(file, "%f,", &val);
            X[i * 784 + j] = val / 255; 
        }        
        for (int j = 0; j < 10; ++j) 
            Y[i * 10 + j] = 0;
        fscanf(file, "%d\n", &label);
        Y[i * 10 + label] = 1;
    }
    fclose(file);
}

int main(int argc, char ** argv)
{
    srand(42);
    int BATCH_SIZE = atoi(argv[1]);
    int EPOCHS = atoi(argv[2]);
    printf("Version: v0 (CPU)\n");

    ANN nn;
    initANN(&nn, BATCH_SIZE);

    readData("..//train.txt", nn.X_train, nn.Y_train, 50000); 
    readData("..//valid.txt", nn.X_valid, nn.Y_valid, 10000);
    readData("..//test.txt",  nn.X_test,  nn.Y_test,  10000);

    train(&nn, EPOCHS, BATCH_SIZE);

    free(nn.W1);
    free(nn.W2);
    free(nn.W3);
    free(nn.b1);
    free(nn.b2);
    free(nn.b3);
    free(nn.Z1);
    free(nn.Z2);
    free(nn.Z3);
    free(nn.A1);
    free(nn.A2);

    free(nn.d_W1);
    free(nn.d_W2);
    free(nn.d_W3);
    free(nn.d_b1);
    free(nn.d_b2);
    free(nn.d_b3);
    free(nn.d_Z1);
    free(nn.d_Z2);
    free(nn.d_Z3);
    free(nn.d_A1);
    free(nn.d_A2);

    free(nn.X_train);
    free(nn.Y_train);
    free(nn.X_valid);
    free(nn.Y_valid);
    free(nn.X_test);
    free(nn.Y_test);
    free(nn.Y_pred);

    return 0;
}