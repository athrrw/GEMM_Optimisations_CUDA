#include <iostream>
#include <cuda_runtime.h>
#include <random>

const int N = 4096;


const int BM = 128;   // block tile rows
const int BN = 128;   // block tile cols
const int BK = 16;    // block tile k-depth
const int WM = 32;    // warp tile rows
const int WN = 64;    // warp tile cols
const int TM = 8;     // thread tile rows
const int TN = 8;     // thread tile cols


__global__ void kernel_warp(float* A, float* B, float* C) {

    __shared__ float sA[BK][BM];   
    __shared__ float sB[BK][BN];   

    int warpId = threadIdx.x / 32;
    int laneId = threadIdx.x % 32;  

    int warpRow = warpId / 2;        
    int warpCol = warpId % 2;        

    int threadRow = laneId / (WN / TN);   
    int threadCol = laneId % (WN / TN);


    int blockRowStart = blockIdx.y * BM;
    int blockColStart = blockIdx.x * BN;

    int warpRowStart = warpRow * WM;
    int warpColStart = warpCol * WN;

    float sum[TM][TN] = {};
    float regA[TM];
    float regB[TN];

    int num_tiles = N / BK;

    for (int t = 0; t < num_tiles; t++) {
        for (int i = 0; i < (BM * BK) / 128; i++) {
            int idx   = threadIdx.x + i * 128;
            int aRow  = idx % BM;
            int aCol  = idx / BM;  
            sA[aCol][aRow] = A[(blockRowStart + aRow) * N + t * BK + aCol];
        }

        for (int i = 0; i < (BN * BK) / 128; i++) {
            int idx  = threadIdx.x + i * 128;
            int bRow = idx / BN;    
            int bCol = idx % BN;
            sB[bRow][bCol] = B[(t * BK + bRow) * N + blockColStart + bCol];
        }

        __syncthreads();

        for (int k = 0; k < BK; k++) {
            for (int i = 0; i < TM; i++)
                regA[i] = sA[k][warpRowStart + threadRow * TM + i];

            for (int j = 0; j < TN; j++)
                regB[j] = sB[k][warpColStart + threadCol * TN + j];

            for (int i = 0; i < TM; i++)
                for (int j = 0; j < TN; j++)
                    sum[i][j] += regA[i] * regB[j];
        }

        __syncthreads();
    }

    int outRow = blockRowStart + warpRowStart + threadRow * TM;
    int outCol = blockColStart + warpColStart + threadCol * TN;

    for (int i = 0; i < TM; i++)
        for (int j = 0; j < TN; j++)
            C[(outRow + i) * N + outCol + j] = sum[i][j];
}

int main() {
    int size = N * N * sizeof(float);
    float *h_A = new float[N*N], *h_B = new float[N*N], *h_C = new float[N*N];
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size); cudaMalloc(&d_B, size); cudaMalloc(&d_C, size);

    for (int i = 0; i < N*N; i++) {
        h_A[i] = static_cast<float>(rand()) / RAND_MAX;
        h_B[i] = static_cast<float>(rand()) / RAND_MAX;
        h_C[i] = 0.0f;
    }
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    dim3 threads(128);
    dim3 blocks(N / BN, N / BM);

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);

    kernel_warp<<<blocks, threads>>>(d_A, d_B, d_C);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    double tflops = (2.0 * N * N * N) / (ms / 1000.0) / 1e12;

    cudaError_t err = cudaGetLastError();
    printf("Launch: %s\n", cudaGetErrorString(err));
    err = cudaDeviceSynchronize();
    printf("Sync:   %s\n", cudaGetErrorString(err));
    printf("Warp tiling: \n Time: %.3f ms \n TFLOPS: %.3f\n", ms, tflops);

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    delete[] h_A; 
    delete[] h_B; 
    delete[] h_C;

    cudaFree(d_A); 
    cudaFree(d_B); 
    cudaFree(d_C);

    return 0;
}