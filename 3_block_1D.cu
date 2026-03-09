#include <iostream>
#include <cuda_runtime.h>
#include <random>

const int N = 4096;
const int TILE_SIZE = 32;  
const int TN = 4;    // each thread computes 1 row 4 cols

// Each thread computes 1 x TN = 1x4 output elements
// Block covers TILE_SIZE rows x TILE_SIZE*TN cols of C ie the block tiling
__global__ void kernel(float* A, float* B, float* C){

    __shared__ float sA[TILE_SIZE][TILE_SIZE];
    __shared__ float sB[TILE_SIZE][TILE_SIZE * TN]; 

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE * TN + threadIdx.x * TN;

    float sum[TN] = {0.0f, 0.0f, 0.0f, 0.0f};

    int num_tiles = N / TILE_SIZE;

    for(int t = 0; t < num_tiles; t++){

        //sA is still scalar loading
        sA[threadIdx.y][threadIdx.x] = A[row * N + t * TILE_SIZE + threadIdx.x];

        
        int bRow = (t * TILE_SIZE + threadIdx.y) * N;
        float4 bVec = reinterpret_cast<float4*>(&B[bRow + col])[0];
        sB[threadIdx.y][threadIdx.x * TN + 0] = bVec.x;
        sB[threadIdx.y][threadIdx.x * TN + 1] = bVec.y;
        sB[threadIdx.y][threadIdx.x * TN + 2] = bVec.z;
        sB[threadIdx.y][threadIdx.x * TN + 3] = bVec.w;

        __syncthreads();

        for(int k = 0; k < TILE_SIZE; k++){
            float a = sA[threadIdx.y][k];
            sum[0] += a * sB[k][threadIdx.x * TN + 0];
            sum[1] += a * sB[k][threadIdx.x * TN + 1];
            sum[2] += a * sB[k][threadIdx.x * TN + 2];
            sum[3] += a * sB[k][threadIdx.x * TN + 3];
        }

        __syncthreads();
    }
    reinterpret_cast<float4*>(&C[row * N + col])[0] = make_float4(sum[0], sum[1], sum[2], sum[3]);
}

int main(){
    int size = N * N * sizeof(float);
    float *h_A = new float[N*N], *h_B = new float[N*N], *h_C = new float[N*N];
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size); cudaMalloc(&d_B, size); cudaMalloc(&d_C, size);

    for(int i = 0; i < N*N; i++){
        h_A[i] = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
        h_B[i] = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
        h_C[i] = 0.0f;
    }
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks(N / (TILE_SIZE * TN), N / TILE_SIZE);

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);

    kernel<<<blocks, threads>>>(d_A, d_B, d_C);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    double gflops = (2.0 * N * N * N) / (ms) / 1e6;
    printf("1D tiling \n Time: %.3f ms \n GFLOPS: %.3f\n", ms, gflops);

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    delete[] h_A; 
    delete[] h_B; 
    delete[] h_C;

    cudaFree(d_A); 
    cudaFree(d_B); 
    cudaFree(d_C);
    return 0;
}