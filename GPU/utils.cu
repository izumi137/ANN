#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <cuda_runtime.h>


#define SMEMSIZE (TILE_SIZE * TILE_SIZE)    
#define BLOCK_SIZE 16

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
} An;

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

// (m x n) -> (n x m)
__global__ void transpose(float *iMatrix, float *oMatrix, int m, int n)
{
	__shared__ float s_blkData[BLOCK_SIZE][BLOCK_SIZE];

    // Calculate thread's row and column in the block
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Load data into shared memory
    if (row < m && col < n) {
        s_blkData[threadIdx.y][threadIdx.x] = iMatrix[row * n + col];
    }

    // Synchronize threads in the block to ensure all data is loaded
    __syncthreads();

    int oR = blockIdx.y * blockDim.y + threadIdx.y;
	int oC = blockIdx.x * blockDim.x + threadIdx.x;
    // Write the transposed data to the output matrix
    if (row < m && col < n) {
        oMatrix[oR * m + oC] = s_blkData[threadIdx.x][threadIdx.y];
    }
}


// C(mxk) = A(mxn) @ B(nxk)
__global__ void matMul(float *C, float *A, float *B, int m, int n, int k)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float tile_A[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float tile_B[BLOCK_SIZE][BLOCK_SIZE];

    float Cvalue = 0;

    
    for (int t = 0; t < (n + BLOCK_SIZE - 1) / BLOCK_SIZE + 1; ++t)
    {
        if (row < m && t * BLOCK_SIZE + threadIdx.x < n)
            tile_A[threadIdx.y][threadIdx.x] = A[row * n + t * BLOCK_SIZE + threadIdx.x];
        else
            tile_A[threadIdx.y][threadIdx.x] = 0.0f;

        if (col < k && t * BLOCK_SIZE + threadIdx.y < n)
            tile_B[threadIdx.y][threadIdx.x] = B[(t * BLOCK_SIZE + threadIdx.y) * k + col];
        else
            tile_B[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for (int i = 0; i < BLOCK_SIZE; i++)
            Cvalue += tile_A[threadIdx.y][i] * tile_B[i][threadIdx.x];

        __syncthreads();
    }

    if (row < m && col < k)
    {
        C[row * k + col] = Cvalue;
    }
}





void matrix_mul(float* in1, float* in2, float* out, int m, int n, int k,
    bool useDevice = false, dim3 blockSize = dim3(1))
{
    GpuTimer timer;
    timer.Start();
    if (useDevice == false)
    {
        // C(mxk) = A(mxn) @ B(nxk)
        for (int row = 0; row < m; row++)
        {
            for (int col = 0; col < k; col++)
            {
                float sum = 0;
                for (int i = 0; i < n; i++)
                    sum += in1[row * n + i] * in2[i * k + col];
                out[row * k + col] = sum;
            }
        }
    }
    else // Use device
    {
        // Allocate device memories
        float* d_in1, *d_in2, *d_out;
        size_t nBytes = m * n * sizeof(float);
        size_t nBytes2 = n * k * sizeof(float);
        size_t nBytes3 = m * k * sizeof(float);
        CHECK(cudaMalloc(&d_in1, nBytes));
        CHECK(cudaMalloc(&d_in2, nBytes2));
        CHECK(cudaMalloc(&d_out, nBytes3));

        // Copy data to device memories
        CHECK(cudaMemcpy(d_in1, in1, nBytes, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_in2, in2, nBytes2, cudaMemcpyHostToDevice));
        // CHECK(cudaMemset(d_out, 0, nBytes3));

        dim3 gridSize((k - 1) / blockSize.x + 1,
            (m - 1) / blockSize.y + 1);
        
		timer.Start();
        matMul<<<gridSize, blockSize>>>(d_out, d_in1, d_in2, m, n, k);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA error: %s\n", cudaGetErrorString(err));
        }
        CHECK(cudaDeviceSynchronize());

        timer.Stop();
        // Copy result from device memory
        CHECK(cudaMemcpy(out, d_out, nBytes3, cudaMemcpyDeviceToHost));

        // Free device memories
        CHECK(cudaFree(d_in1));
        CHECK(cudaFree(d_in2));
        CHECK(cudaFree(d_out));
		
		printf("Grid size: %d * %d, block size: %d * %d\n", 
					gridSize.x,gridSize.y, blockSize.x,blockSize.y);
    }
    float time = timer.Elapsed();
    printf("Processing time (%s): %f ms\n",
        useDevice == true ? "use device" : "use host", time);
}


void matrix_transpose(float* in, float* out, int m, int n,
    bool useDevice = false, dim3 blockSize = dim3(1))
{
    GpuTimer timer;
    timer.Start();
    if (useDevice == false)
    {
		timer.Start();
        for (int row = 0; row < m; row++)
        {   // (m x n) -> (n x m)
            for (int col = 0; col < n; col++)
				out[col * n + row] = in[col + row * n];            
        }
		timer.Stop();
    }
    else // Use device
    {
        // Allocate device memories
        float* d_in, *d_out;
        size_t nBytes = m * n * sizeof(float);
		CHECK(cudaMalloc(&d_in, nBytes));
        CHECK(cudaMalloc(&d_out, nBytes));

        // Copy data to device memories
        CHECK(cudaMemcpy(d_in, in, nBytes, cudaMemcpyHostToDevice));

        // TODO: Set grid size and call kernel
        dim3 gridSize((m - 1) / blockSize.x + 1,
            (n - 1) / blockSize.y + 1);
        
		timer.Start();
		// if (kernelType == 1)
		// 	transpose1<<<gridSize, blockSize>>>(d_in, d_out, w);
		// else if (kernelType == 2)
		// 	transpose2<<<gridSize, blockSize>>>(d_in, d_out, w);				
		// else if (kernelType == 3)
		// 	transpose3<<<gridSize, blockSize>>>(d_in, d_out, w);
		// else if (kernelType == 4)
		// 	transpose4<<<gridSize, blockSize>>>(d_in, d_out, w);
        transpose<<<gridSize, blockSize>>>(d_in, d_out, m, n);
		CHECK(cudaDeviceSynchronize();)
		timer.Stop();
        // Copy result from device memory
        CHECK(cudaMemcpy(out, d_out, nBytes, cudaMemcpyDeviceToHost));

        // Free device memories
        CHECK(cudaFree(d_in));
        CHECK(cudaFree(d_out));
		
		printf("Grid size: %d * %d, block size: %d * %d\n", 
					gridSize.x,gridSize.y, blockSize.x,blockSize.y);
    }
    float time = timer.Elapsed();
    printf("Processing time (%s): %f ms\n",
        useDevice == true ? "use device" : "use host", time);
}

float checkCorrectness(float * a1, float* a2, int n)
{
	float err = 0;
	for (int i = 0; i < n; i++)	
		err += abs(a1[i] - a2[i]);
	err /= n;
	return err;
}

void printDeviceInfo()
{
	cudaDeviceProp devProv;
    CHECK(cudaGetDeviceProperties(&devProv, 0));
    printf("**********GPU info**********\n");
    printf("Name: %s\n", devProv.name);
    printf("Compute capability: %d.%d\n", devProv.major, devProv.minor);
    printf("Num SMs: %d\n", devProv.multiProcessorCount);
    printf("Max num threads per SM: %d\n", devProv.maxThreadsPerMultiProcessor); 
    printf("Max num warps per SM: %d\n", devProv.maxThreadsPerMultiProcessor / devProv.warpSize);
    printf("GMEM: %lu bytes\n", devProv.totalGlobalMem);
    printf("****************************\n\n");

}

int main(int argc, char** argv)
{
	printDeviceInfo();
	
	//Declare variables
    int m = 100, n = 1002, k = 30;
    size_t nBytes2 = m * n * sizeof(float);
    size_t nBytes3 = n * k * sizeof(float);
    size_t nBytes4 = m * k * sizeof(float);

    float* x1 = (float*)malloc(nBytes2);
    float* x2 = (float*)malloc(nBytes3);
    float* y = (float*)malloc(nBytes4);
    float* y2 = (float*)malloc(nBytes4);

    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            x1[i*n + j] = static_cast <float> (j);
            printf("%f ", x1[i*n + j]);
        }
        printf("\n");
    }
        
    printf("\n");

    for (int i = 0; i < n; i++){
        for (int j = 0; j < k; j++)
        {
            x2[i*k + j] = static_cast <float> (j);
            printf("%f ", x2[i*k + j]);
        }
        printf("\n");
    }
    printf("\n");

    printf("Host Matrix Mul:\n");
    matrix_mul(x1, x2, y, m, n, k, false);
    printf("\n y =");
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < k; j++)
            printf("%f ", y[i*k + j]);
        printf("\n");
    }
    // Add vectors (on host)
	printf("\n");

	dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE); // Default
	if (argc == 3)
	{
		blockSize.x = atoi(argv[1]);
		blockSize.y = atoi(argv[2]);
	} 

	printf("Shared memory Matrix Mul:\n");
    // matrix_transpose(h_in, h_out, m, n, true,blockSize);
    matrix_mul(x1, x2, y2, m, n, k, true, blockSize);

    

	float err = checkCorrectness(y, y2, m*k);
	printf("Error between device result and host result: %f", err);	

    printf("\n y2 =");   
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < k; j++)
            printf("%f ", y2[i*k + j]);
        printf("\n");
    }


    
	
    free(x1);
    free(x2);
    free(y);
    free(y2);

    return 0;
}