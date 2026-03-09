#include <iostream>
#include <cuda_runtime.h>
#include <random>

const int N  = 4096;
const int BM = 128;
const int BN = 128;
const int BK = 16;
const int WM = 32;
const int WN = 64;
const int TM = 8;
const int TN = 8;

__global__ void kernel_warp_vec(float* A, float* B, float* C) {

    __shared__ float sA[BK][BM];
    __shared__ float sB[BK][BN];

    int warpId   = threadIdx.x / 32;
    int laneId   = threadIdx.x % 32;
    int warpRow  = warpId / 2;
    int warpCol  = warpId % 2;
    int threadRow = laneId / (WN / TN);
    int threadCol = laneId % (WN / TN);

    int blockRowStart = blockIdx.y * BM;
    int blockColStart = blockIdx.x * BN;
    int warpRowStart  = warpRow * WM;
    int warpColStart  = warpCol * WN;

    float sum[TM][TN] = {};
    float regA[TM];
    float regB[TN];

    int num_tiles = N / BK;

    for (int t = 0; t < num_tiles; t++) {

        // Load sA with float4 (transposed into shared memory)
        // BM*BK=2048 floats, 128 threads
        for (int i = 0; i < (BM * BK) / (128 * 4); i++) {
        int loadIdx = threadIdx.x + i * 128;
        int aRow    = loadIdx / (BK / 4); 
        int aCol    = (loadIdx % (BK / 4)) * 4; 

        float4 aVec = reinterpret_cast<float4*>(
            &A[(blockRowStart + aRow) * N + t * BK + aCol])[0];

        // store transposed
        sA[aCol + 0][aRow] = aVec.x;
        sA[aCol + 1][aRow] = aVec.y;
        sA[aCol + 2][aRow] = aVec.z;
        sA[aCol + 3][aRow] = aVec.w;
        }

    for (int i = 0; i < (BN * BK) / (128 * 4); i++) {
        int loadIdx = threadIdx.x + i * 128;
        int bRow    = loadIdx / (BN / 4);
        int bCol    = (loadIdx % (BN / 4)) * 4; 

        float4 bVec = reinterpret_cast<float4*>(
            &B[(t * BK + bRow) * N + blockColStart + bCol])[0];

        sB[bRow][bCol + 0] = bVec.x;
        sB[bRow][bCol + 1] = bVec.y;
        sB[bRow][bCol + 2] = bVec.z;
        sB[bRow][bCol + 3] = bVec.w;
    }

        __syncthreads();

        //same outer product as kernel 
        //registers are filled from sA/sB then TM*TN=64 FMAs issued
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

    / --- Write with float4: each row of the TM x TN tile has TN=8 floats = 2 float4s ---
    int outRow = blockRowStart + warpRowStart + threadRow * TM;
    int outCol = blockColStart + warpColStart + threadCol * TN;

    for (int i = 0; i < TM; i++) {
        reinterpret_cast<float4*>(&C[(outRow + i) * N + outCol])[0] =
            make_float4(sum[i][0], sum[i][1], sum[i][2], sum[i][3]);
        reinterpret_cast<float4*>(&C[(outRow + i) * N + outCol + 4])[0] =
            make_float4(sum[i][4], sum[i][5], sum[i][6], sum[i][7]);
    }
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

    kernel_warp_vec<<<blocks, threads>>>(d_A, d_B, d_C);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    double tflops = (2.0 * N * N * N) / (ms / 1000.0) / 1e12;

    cudaError_t err = cudaGetLastError();
    printf("Launch: %s\n", cudaGetErrorString(err));
    err = cudaDeviceSynchronize();
    printf("Sync:   %s\n", cudaGetErrorString(err));
    printf("Warp tiling vec \n Time: %.3f ms \n TFLOPS: %.3f\n", ms, tflops);

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    delete[] h_A; delete[] h_B; delete[] h_C;
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    return 0;
}