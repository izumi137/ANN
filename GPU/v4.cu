#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>



typedef struct {
    __half *W1, *W2, *W3;
    __half *d_W1, *d_W2, *d_W3;

    __half *b1, *b2, *b3;
    __half *d_b1, *d_b2, *d_b3;

    __half *Z1, *Z2, *Z3, *A1, *A2;
    __half *d_Z1, *d_Z2, *d_Z3, *d_A1, *d_A2;
    
    __half *Y_pred;

    __half *X_train, *Y_train;
    __half *X_valid, *Y_valid;
    __half *X_test,  *Y_test;
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

#define CHECK(call)\
{\
	const cudaError_t error = call;\
	if (error != cudaSuccess)\
	{\
		fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);\
		fprintf(stderr, "code: %d, reason: %s\n", error,\
				cudaGetErrorString(error));\
		exit(EXIT_FAILURE);\
	}\
}


// C(mxk) = A(mxn) @ B(nxk)
__global__ void matMulAB(__half *C, __half *A, __half *B, int m, int n, int k)
{
    // Block and thread indices
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int TILE_WIDTH = blockDim.x;

    extern __shared__ __half sharedMem[];
    __half* tileA = sharedMem;                                  
    __half* tileB = tileA + TILE_WIDTH * TILE_WIDTH;     

    __half sum = __float2half(0.0f);

    // Iterate over tiles
    for (int t = 0; t < (n + TILE_WIDTH - 1) / TILE_WIDTH; ++t)
    {
        // Load A and B to shared memory
        if (row < m && (t * TILE_WIDTH + threadIdx.x) < n)
            tileA[threadIdx.y * TILE_WIDTH + threadIdx.x] = A[row * n + t * TILE_WIDTH + threadIdx.x];
        else
            tileA[threadIdx.y * TILE_WIDTH + threadIdx.x] = __float2half(0.0f);

        if (col < k && (t * blockDim.y + threadIdx.y) < n)
            tileB[threadIdx.y * TILE_WIDTH + threadIdx.x] = B[(t * blockDim.y + threadIdx.y) * k + col];
        else
            tileB[threadIdx.y * TILE_WIDTH + threadIdx.x] = __float2half(0.0f);

        __syncthreads();

        // Multiply the tiles and accumulate the result
        for (int i = 0; i < TILE_WIDTH; ++i)
            sum = __hadd(sum, __hmul(tileA[threadIdx.y * TILE_WIDTH + i], tileB[i * TILE_WIDTH + threadIdx.x]));

        __syncthreads();
    }

    if (row < m && col < k)
        C[row * k + col] = sum;
}

// C(mxk) = A(nxm)^T @ B(nxk)
__global__ void matMulATB(__half *C, __half *A, __half *B, int m, int n, int k)
{
    // Block and thread indices
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int TILE_WIDTH = blockDim.x;

    extern __shared__ __half sharedMem[];
    __half* tileA = sharedMem;                           
    __half* tileB = tileA + TILE_WIDTH * TILE_WIDTH;     

    __half sum = __float2half(0.0f);

    // Loop over tiles
    for (int t = 0; t < (n + TILE_WIDTH - 1) / TILE_WIDTH; ++t)
    {
        // Load A and B to shared memory
        if (row < m && (t * TILE_WIDTH + threadIdx.y) < n)
            tileA[threadIdx.x * TILE_WIDTH + threadIdx.y] = A[(t * TILE_WIDTH + threadIdx.x) * m + row];
        else
            tileA[threadIdx.x * TILE_WIDTH + threadIdx.y] = __float2half(0.0f);

        if (col < k && (t * TILE_WIDTH + threadIdx.y) < n)
            tileB[threadIdx.x * TILE_WIDTH + threadIdx.y] = B[(t * TILE_WIDTH + threadIdx.y) * k + col];
        else
            tileB[threadIdx.x * TILE_WIDTH + threadIdx.y] = __float2half(0.0f);

        __syncthreads();

        // Multiply the tiles and accumulate the result
        for (int i = 0; i < TILE_WIDTH; ++i)
        {
            sum = __hadd(sum, __hmul(tileA[i * blockDim.y + threadIdx.y], tileB[threadIdx.x * blockDim.y + i]));
        }

        __syncthreads();
    }

    if (row < m && col < k)
        C[row * k + col] = sum;
}

// C(mxk) = A(mxn) @ B(kxn)^T
__global__ void matMulABT(__half *C, __half *A, __half *B, int m, int n, int k)
{
    // Block and thread indices
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int TILE_WIDTH = blockDim.x;

    extern __shared__ __half sharedMem[];
    __half* tileA = sharedMem;                            // Tile for matrix A
    __half* tileB = tileA + TILE_WIDTH * TILE_WIDTH;      // Tile for transposed matrix B

    // Accumulate result for this thread
    __half sum = __float2half(0.0f);

    // Loop over tiles
    for (int t = 0; t < (n + TILE_WIDTH - 1) / TILE_WIDTH; ++t)
    {
        // Load a tile of A into shared memory
        if (row < m && (t * TILE_WIDTH + threadIdx.x) < n)
            tileA[threadIdx.y * TILE_WIDTH + threadIdx.x] = A[row * n + t * TILE_WIDTH + threadIdx.x];
        else
            tileA[threadIdx.y * TILE_WIDTH + threadIdx.x] = __float2half(0.0f);

        // Load a tile of B (transposed) into shared memory
        if (col < k && (t * TILE_WIDTH + threadIdx.x) < n)
            tileB[threadIdx.y * TILE_WIDTH + threadIdx.x] = B[col * n + t * TILE_WIDTH + threadIdx.y];
            
        else
            tileB[threadIdx.y * TILE_WIDTH + threadIdx.x] = __float2half(0.0f);

        __syncthreads();

        // Multiply the tiles and accumulate the result
        for (int i = 0; i < TILE_WIDTH; ++i)
        {
            sum = __hadd(sum, __hmul(tileA[threadIdx.y * TILE_WIDTH + i], tileB[threadIdx.x + i * TILE_WIDTH]));
        }

        __syncthreads();
    }

    // Write the result back to C
    if (row < m && col < k)
        C[row * k + col] = sum;
}

// Z(mxk) + b(1xk)
__global__ void addBias(__half *Z, __half *b, int m, int k) 
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < k)
        Z[row * k + col] = __hadd(b[col], Z[row * k + col]);
}

// C(mxk) = A(mxk) - B(mxk)
__global__ void matSub(__half *C, __half *A, __half *B, int m, int k)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < k)
    {
        int idx = row * k + col;
        C[idx] = __hsub(A[idx], B[idx]);
    }
}

// relu(A(mxk)) = max(Z(mxk), 0)
__global__ void relu(__half *A, __half *Z, int m, int k) 
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < k)
    {
        int idx = row * k + col;
        A[idx] = __hmax(__float2half(0.0f), Z[idx]);
    }
}

// drelu(d_Z(mxk)) = d_A*(Z > 0) 
__global__ void drelu(__half *d_Z, __half *d_A, __half *Z, int m, int k) 
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < k)
    {
        int idx = row * k + col;
        d_Z[idx] = __hmul(d_A[idx], (__hgt(Z[idx], __float2half(0.0f)) ? __float2half(1.0f) : __float2half(0.0f)));
    }
}

// A(BATCHSIZExLENGTH) = softmax(Z(BATCH_SIZExLENGTH))
__global__ void softmax(__half *A, __half *Z, int BATCH_SIZE, int length) 
{
    int b = blockIdx.x * blockDim.x + threadIdx.x;

    if (b < BATCH_SIZE) 
    {
        __half mx = Z[b * length];
        for (int i = 1; i < length; ++i) 
            mx = __hmax(mx, Z[b * length + i]);  

        __half sum = __float2half(0.0f);  
        for (int i = 0; i < length; ++i) 
        {
            A[b * length + i] = expf(Z[b * length + i] - mx); 
            sum = __hadd(sum, A[b * length + i]);  
        }

        const __half epsilon = __float2half(1e-7f);  
        for (int i = 0; i < length; ++i) 
            A[b * length + i] = __hdiv(A[b * length + i], __hmax(sum, epsilon)); 
    }
}

// d_b = sumBatch(d_Z) = sum_batch(BATCHSIZExLENGTH) = (1xLENGTH)
__global__ void sumBatch(__half *d_b, __half *d_Z, int BATCH_SIZE, int length)
{
    int l = blockIdx.x * blockDim.x + threadIdx.x;
    if (l < length) 
    {
        __half sum = __float2half(0.0f);  
        for (int i = 0; i < BATCH_SIZE; ++i)
            sum = __hadd(sum, d_Z[i * length + l]); 

        d_b[l] = sum;
    }
}

// W(mxk) -= LR * d_W(mxk)
__global__ void updateWeight2D(__half *W, __half *d_W, int m, int k, __half LEARNING_RATE)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < k)
    {
        int idx = row * k + col;
        W[idx] = __hsub(W[idx], __hmul(LEARNING_RATE, d_W[idx]));
    }
}

// b(1xk) -= LR * d_b(1xk)
__global__ void updateWeight1D(__half *b, __half *d_b, int k, __half LEARNING_RATE)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < k)
        b[idx] = __hsub(b[idx], __hmul(LEARNING_RATE, d_b[idx]));
}

// Initialize weight matrix W (size mxk)
void initWeight(__half *W, int m, int k) {
    float scale = __float2half(sqrtf(2.0f / k)); 
    for (int i = 0; i < m * k; i++) {
        float rand_val = static_cast<float>(rand()) / RAND_MAX; 
        __half uniform = __float2half(rand_val * 2.0f * scale - scale); 
        W[i] = uniform;
    }
}


void initANN(ANN *nn, __half *X_train, __half *Y_train, __half *X_valid, __half *Y_valid, __half *X_test, __half *Y_test, int BATCH_SIZE)
{
    __half *W1, *W2, *W3;

    W1 = (__half*)malloc(128 * 784 * sizeof(__half));
    W2 = (__half*)malloc(128 * 128 * sizeof(__half));
    W3 = (__half*)malloc(10  * 128 * sizeof(__half));

    initWeight(W1, 128, 784);
    initWeight(W2, 128, 128);
    initWeight(W3, 10 , 128);

    CHECK(cudaMalloc(&nn->Y_pred, 50000 * 10  * sizeof(__half)));

    CHECK(cudaMalloc(&nn->Z1, 50000 * 128 * sizeof(__half)));
    CHECK(cudaMalloc(&nn->Z2, 50000 * 128 * sizeof(__half)));
    CHECK(cudaMalloc(&nn->Z3, 50000 * 10  * sizeof(__half)));
    CHECK(cudaMalloc(&nn->A1, 50000 * 128 * sizeof(__half)));
    CHECK(cudaMalloc(&nn->A2, 50000 * 128 * sizeof(__half)));

    CHECK(cudaMalloc(&nn->d_Z1, BATCH_SIZE * 128 * sizeof(__half)));
    CHECK(cudaMalloc(&nn->d_Z2, BATCH_SIZE * 128 * sizeof(__half)));
    CHECK(cudaMalloc(&nn->d_Z3, BATCH_SIZE * 10  * sizeof(__half)));
    CHECK(cudaMalloc(&nn->d_A1, BATCH_SIZE * 128 * sizeof(__half)));
    CHECK(cudaMalloc(&nn->d_A2, BATCH_SIZE * 128 * sizeof(__half)));

    CHECK(cudaMalloc(&nn->b1, 128 * sizeof(__half)));
    CHECK(cudaMalloc(&nn->b2, 128 * sizeof(__half)));
    CHECK(cudaMalloc(&nn->b3, 10  * sizeof(__half)));

    CHECK(cudaMalloc(&nn->d_b1, 128 * sizeof(__half)));
    CHECK(cudaMalloc(&nn->d_b2, 128 * sizeof(__half)));
    CHECK(cudaMalloc(&nn->d_b3, 10  * sizeof(__half)));

    CHECK(cudaMalloc(&nn->W1, 128 * 784 * sizeof(__half)));
    CHECK(cudaMalloc(&nn->W2, 128 * 128 * sizeof(__half)));
    CHECK(cudaMalloc(&nn->W3, 10  * 128 * sizeof(__half)));

    CHECK(cudaMalloc(&nn->d_W1, 128 * 784 * sizeof(__half)));
    CHECK(cudaMalloc(&nn->d_W2, 128 * 128 * sizeof(__half)));
    CHECK(cudaMalloc(&nn->d_W3, 10  * 128 * sizeof(__half)));

    CHECK(cudaMalloc(&nn->X_train, 50000 * 784 * sizeof(__half)));
    CHECK(cudaMalloc(&nn->X_valid, 10000 * 784 * sizeof(__half)));
    CHECK(cudaMalloc(&nn->X_test,  10000 * 784 * sizeof(__half)));
    CHECK(cudaMalloc(&nn->Y_train, 50000 * 10  * sizeof(__half)));
    CHECK(cudaMalloc(&nn->Y_valid, 10000 * 10  * sizeof(__half)));
    CHECK(cudaMalloc(&nn->Y_test,  10000 * 10  * sizeof(__half)));

    CHECK(cudaMemcpy(nn->W1, W1, 128 * 784 * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(nn->W2, W2, 128 * 128 * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(nn->W3, W3, 10  * 128 * sizeof(__half), cudaMemcpyHostToDevice));

    CHECK(cudaMemcpy(nn->X_train, X_train, 50000 * 784 * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(nn->X_valid, X_valid, 10000 * 784 * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(nn->X_test,  X_test,  10000 * 784 * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(nn->Y_train, Y_train, 50000 * 10 * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(nn->Y_valid, Y_valid, 10000 * 10 * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(nn->Y_test,  Y_test,  10000 * 10 * sizeof(__half), cudaMemcpyHostToDevice));

    CHECK(cudaMemset(nn->b1, 0, 128 * sizeof(__half)));
    CHECK(cudaMemset(nn->b2, 0, 128 * sizeof(__half)));
    CHECK(cudaMemset(nn->b3, 0, 10  * sizeof(__half)));

    free(W1);
    free(W2);
    free(W3);
}


void forward(ANN *nn, __half *X, int BATCH_SIZE, dim3 bs2 = dim3(32, 32), dim3 bs1 = dim3(32))
{   
    dim3 grid1(1), grid2(1, 1);

    int sharedMemSize = (bs2.y * bs2.x * 2) * sizeof(__half);
    // Z1 = X @ W1^T + b1 = (32x784) @ (128x784)^T = (32x128) 
    grid2.y = (bs2.y + BATCH_SIZE - 1) / bs2.y;
    grid2.x = (bs2.x + 128 - 1) / bs2.x;
    matMulABT<<<grid2, bs2, sharedMemSize>>>(nn->Z1, X, nn->W1, BATCH_SIZE, 784, 128);
    addBias<<<grid2, bs2>>>(nn->Z1, nn->b1, BATCH_SIZE, 128);
    // A1 = relu(Z1) = (32x128)
    relu<<<grid2, bs2>>>(nn->A1, nn->Z1, BATCH_SIZE, 128);

    // Z2 = A1 @ W2^T + b2 = (32x128) @ (128x128)^T = (32x128)
    matMulABT<<<grid2, bs2, sharedMemSize>>>(nn->Z2, nn->A1, nn->W2, BATCH_SIZE, 128, 128);
    addBias<<<grid2, bs2>>>(nn->Z2, nn->b2, BATCH_SIZE, 128);
    // A2 = relu(Z2) = (32x128)
    relu<<<grid2, bs2>>>(nn->A2, nn->Z2, BATCH_SIZE, 128);

    // Z3 = A2 @ W3^T + b3 = (32x128) @ (10x128)^T = (32x10)
    grid2.x = (bs2.x + 10 - 1) / bs2.x;
    matMulABT<<<grid2, bs2, sharedMemSize>>>(nn->Z3, nn->A2, nn->W3, BATCH_SIZE, 128, 10);
    addBias<<<grid2, bs2>>>(nn->Z3, nn->b3, BATCH_SIZE, 10);
    // Y_pred = softmax(Z3) = (32x10)
    grid1.x = (BATCH_SIZE + bs1.x - 1) / bs1.x;
    softmax<<<grid1, bs1>>>(nn->Y_pred, nn->Z3, BATCH_SIZE, 10);
}


void backward(ANN *nn, __half *X, __half *Y_true, int BATCH_SIZE, dim3 bs2 = dim3(32, 32), dim3 bs1 = dim3(32))
{
    int sharedMemSize = (bs2.y * bs2.x * 2) * sizeof(__half);
    dim3 grid1(1), grid2(1, 1);
    // Layer: Output
    // d_Z3 = Y_pred - Y_true = (32x10)
    grid2.y = (bs2.y + BATCH_SIZE - 1) / bs2.y;
    grid2.x = (bs2.x + 10 - 1) / bs2.x;
    matSub<<<grid2, bs2>>>(nn->d_Z3, nn->Y_pred, Y_true, BATCH_SIZE, 10);
    // d_W3 = d_Z3^T @ A2 = (32x10)^T @ (32x128) = (10x128)
    grid2.y = (bs2.y + 10 - 1) / bs2.y;
    grid2.x = (bs2.x + 128 - 1) / bs2.x;
    matMulATB<<<grid2, bs2, sharedMemSize>>>(nn->d_W3, nn->d_Z3, nn->A2, 10, BATCH_SIZE, 128);
    // d_b3 = sum_batch(d_Z3) = sum_batch(32x10) = (1, 10)
    // bs1.x = 10;
    grid1.x = (bs1.x + 10 - 1) / bs1.x;
    sumBatch<<<grid1, bs1>>>(nn->d_b3, nn->d_Z3, BATCH_SIZE, 10);

    // Layer: Hidden 2
    // d_A2 = d_Z3 @ W3 = (32x10) x (10x128) = (32x128)
    grid2.y = (bs2.y + BATCH_SIZE - 1) / bs2.y;
    grid2.x = (bs2.x + 128 - 1) / bs2.x;
    matMulAB<<<grid2, bs2, sharedMemSize>>>(nn->d_A2, nn->d_Z3, nn->W3, BATCH_SIZE, 10, 128);
    // d_Z2 = d_relu(d_A2, Z2) = d_relu(32x128) = (32x128)
    drelu<<<grid2, bs2>>>(nn->d_Z2, nn->d_A2, nn->Z2, BATCH_SIZE, 128);
    // d_W2 = d_Z2^T @ A_1 = (32x128)^T @ (32x128) = (128x128)
    grid2.y = (bs2.y + 128 - 1) / bs2.y;
    // grid2.x = (bs2.x + 128 - 1) / bs2.x;
    matMulATB<<<grid2, bs2, sharedMemSize>>>(nn->d_W2, nn->d_Z2, nn->A1, 128, BATCH_SIZE, 128);
    // d_b2 = sum_batch(d_Z2) = sum_batch(32x128) = (1, 128)  
    grid1.x = (bs1.x + 128 - 1) / bs1.x;
    sumBatch<<<grid1, bs1>>>(nn->d_b2, nn->d_Z2, BATCH_SIZE, 128);

    // Layer: Hidden 1
    // d_A1 = d_Z2 @ W2 = (32x128) x (128x128) = (32x128)
    grid2.y = (bs2.y + BATCH_SIZE - 1) / bs2.y;
    matMulAB<<<grid2, bs2, sharedMemSize>>>(nn->d_A1, nn->d_Z2, nn->W2, BATCH_SIZE, 128, 128);
    // d_Z1 = d_relu(d_A1, Z1) = d_relu(32x128) = (32x128)
    drelu<<<grid2, bs2>>>(nn->d_Z1, nn->d_A1, nn->Z1, BATCH_SIZE, 128);
    // d_W1 = d_Z1^T @ X = (32x128)^T @ (32x784) = (128x784)
    grid2.y = (bs2.y + 128 - 1) / bs2.y;
    grid2.x = (bs2.x + 784 - 1) / bs2.x;
    matMulATB<<<grid2, bs2, sharedMemSize>>>(nn->d_W1, nn->d_Z1, X, 128, BATCH_SIZE, 784);
    // d_b1 = sum_batch(d_Z1) = sum_batch(32x128) = (1, 128)  
    sumBatch<<<grid1, bs1>>>(nn->d_b1, nn->d_Z1, BATCH_SIZE, 128);

    // UPDATE WEIGHTS
    float LEARNING_RATE = 0.001f;
    grid2.y = (bs2.y + 128 - 1) / bs2.y;
    grid2.x = (bs2.x + 784 - 1) / bs2.x;
    updateWeight2D<<<grid2, bs2>>>(nn->W1, nn->d_W1, 128, 784, LEARNING_RATE);
    updateWeight1D<<<grid1, bs1>>>(nn->b1, nn->d_b1, 128, LEARNING_RATE);

    grid2.x = (bs2.x + 128 - 1) / bs2.x;
    updateWeight2D<<<grid2, bs2>>>(nn->W2, nn->d_W2, 128, 128, LEARNING_RATE);
    updateWeight1D<<<grid1, bs1>>>(nn->b2, nn->d_b2, 128, LEARNING_RATE);

    grid2.y = (bs2.y + 10 - 1) / bs2.y;
    grid1.x = (bs1.x + 10 - 1) / bs1.x;
    updateWeight2D<<<grid2, bs2 >>>(nn->W3, nn->d_W3, 10, 128, LEARNING_RATE);
    updateWeight1D<<<grid1, bs1>>>(nn->b3, nn->d_b3, 10, LEARNING_RATE);
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

void eval(ANN *nn, int mode, __half* X, __half *Y_true, dim3 bs2 = dim3(32, 32), dim3 bs1 = dim3(32), bool save_log = false)
{
    int size = 10000;
    __half *Y_pred;
    if (mode == 0)
        size = 50000;

    forward(nn, X, size, bs2, bs1);

    Y_pred = (__half*)malloc(10 * size * sizeof(__half));
    
    CHECK(cudaMemcpy(Y_pred, nn->Y_pred, 10 * size * sizeof(__half), cudaMemcpyDeviceToHost));
    
    float loss = 0;
    float acc = 0;
    for (int i = 0; i < size; ++i)
    {
        __half mx = __float2half(0.0f);
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
            if (Y_true[idx] == __float2half(1.0f))
            {
                float f = __half2float(Y_pred[idx]);
                label_idx = j;
                float tmp = logf(fmaxf(f, 1e-7f));
                loss -= tmp;
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

void train(ANN *nn, int EPOCHS, int BATCH_SIZE, __half *Y_train, __half *Y_valid, __half *Y_test, dim3 bs2 = dim3(32, 32), dim3 bs1 = dim3(32)) 
{
    float total_time = 0.0f;
    int num_batches = floor(50000 / BATCH_SIZE);
    for (int epoch = 1; epoch < EPOCHS+1; epoch++) 
    {
        GpuTimer timer;
        timer.Start();
        printf("Epoch %d/%d:\n", epoch, EPOCHS);

        for (int batch = 0; batch < num_batches - 1; batch++) 
        {
            int offset = batch * BATCH_SIZE;
            forward(nn, nn->X_train + offset * 784, BATCH_SIZE, bs2, bs1);
            backward(nn, nn->X_train + offset * 784, nn->Y_train + offset * 10, BATCH_SIZE, bs2, bs1);
        }

        timer.Stop();
		float time = timer.Elapsed();
		printf("Time: %f ms\n", time);
        total_time += time;

        printf("Train: ");
        eval(nn, 0, nn->X_train, Y_train, bs2, bs1);

        printf("Valid: ");
        eval(nn, 1, nn->X_valid, Y_valid, bs2, bs1);
            

        printf("\n");
    }
    
    printf("Finished training\nTest: ");
    total_time /= EPOCHS;

    eval(nn, 1, nn->X_test, Y_test, bs2, bs1, true); // Save log
    printf("\nAverage time per epoch: %f ms\n", total_time);
    float data[1];
    data[0] = total_time;
    write_log("log.txt", data, 1, true);
}



void readData(const char* filename, __half* X, __half* Y, int size)
{
    FILE* file = fopen(filename, "r");
    float val;
    int label;
    for (int i = 0; i < size; i++) 
    {
        for (int j = 0; j < 784; j++) 
        {
            fscanf(file, "%f,", &val);
            X[i * 784 + j] = __float2half(val / 255.0f); 
        }        
        for (int j = 0; j < 10; ++j) 
            Y[i * 10 + j] = 0;
        fscanf(file, "%d\n", &label);
        Y[i * 10 + label] = __float2half(1.0f);
    }
    fclose(file);
}

int main(int argc, char ** argv)
{
    srand(42); // set seed 42
    int BATCH_SIZE = atoi(argv[1]);
    int EPOCHS = atoi(argv[2]);
    dim3 bs1(atoi(argv[3])), bs2(atoi(argv[3]), atoi(argv[3]));
    printf("Version: v4 (GPU + fp16 + shared memory matmul)\n");

    __half *X_train, *Y_train, *X_valid, *Y_valid, *X_test, *Y_test;
    X_train = (__half *)malloc(784 * 50000 * sizeof(__half));
    Y_train = (__half *)malloc(10  * 50000 * sizeof(__half));
    X_valid = (__half *)malloc(784 * 10000 * sizeof(__half));
    Y_valid = (__half *)malloc(10  * 10000 * sizeof(__half));
    X_test = (__half *)malloc(784 * 10000 * sizeof(__half));
    Y_test = (__half *)malloc(10  * 10000 * sizeof(__half));

    readData("..//train.txt", X_train, Y_train, 50000); 
    readData("..//valid.txt", X_valid, Y_valid, 10000);
    readData("..//test.txt",  X_test,  Y_test,  10000);

    ANN nn;
    initANN(&nn, X_train, Y_train, X_valid, Y_valid, X_test, Y_test, BATCH_SIZE);

    train(&nn, EPOCHS, BATCH_SIZE, Y_train, Y_valid, Y_test, bs2, bs1);

    CHECK(cudaFree(nn.W1));
    CHECK(cudaFree(nn.W2));
    CHECK(cudaFree(nn.W3));
    CHECK(cudaFree(nn.b1));
    CHECK(cudaFree(nn.b2));
    CHECK(cudaFree(nn.b3));
    CHECK(cudaFree(nn.Z1));
    CHECK(cudaFree(nn.Z2));
    CHECK(cudaFree(nn.Z3));
    CHECK(cudaFree(nn.A1));
    CHECK(cudaFree(nn.A2));

    CHECK(cudaFree(nn.d_W1));
    CHECK(cudaFree(nn.d_W2));
    CHECK(cudaFree(nn.d_W3));
    CHECK(cudaFree(nn.d_b1));
    CHECK(cudaFree(nn.d_b2));
    CHECK(cudaFree(nn.d_b3));
    CHECK(cudaFree(nn.d_Z1));
    CHECK(cudaFree(nn.d_Z2));
    CHECK(cudaFree(nn.d_Z3));
    CHECK(cudaFree(nn.d_A1));
    CHECK(cudaFree(nn.d_A2));

    CHECK(cudaFree(nn.X_train));
    CHECK(cudaFree(nn.Y_train));
    CHECK(cudaFree(nn.X_valid));
    CHECK(cudaFree(nn.Y_valid));
    CHECK(cudaFree(nn.X_test));
    CHECK(cudaFree(nn.Y_test));
    CHECK(cudaFree(nn.Y_pred));

    free(X_train);
    free(Y_train);
    free(X_valid);
    free(Y_valid);
    free(X_test);
    free(Y_test);

    return 0;
}