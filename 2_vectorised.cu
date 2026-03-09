#include <iostream>
#include <cuda_runtime.h>
#include <random>
using namespace std;

const int N = 4096;
const int TILE_SIZE = 32;

__global__ void kernel(float* A, float* B, float* C){
    __shared__ float sA[TILE_SIZE][TILE_SIZE];
    __shared__ float sB[TILE_SIZE][TILE_SIZE * 4]; 

    int col = threadIdx.x*4 + blockIdx.x * TILE_SIZE*4;
    int row = threadIdx.y + blockIdx.y * TILE_SIZE;
    float sum[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    int num_tiles = N / TILE_SIZE;

    for(int t = 0; t < num_tiles; t++){
        sA[threadIdx.y][threadIdx.x] = A[row*N + (t*TILE_SIZE + threadIdx.x)];

        int bRow = N * (t * TILE_SIZE + threadIdx.y);
        float4 bVec = reinterpret_cast<float4*>(&B[bRow + col])[0];
        sB[threadIdx.y][threadIdx.x + TILE_SIZE * 0] = bVec.x;
        sB[threadIdx.y][threadIdx.x + TILE_SIZE * 1] = bVec.y;
        sB[threadIdx.y][threadIdx.x + TILE_SIZE * 2] = bVec.z;
        sB[threadIdx.y][threadIdx.x + TILE_SIZE * 3] = bVec.w;
        __syncthreads();

        for(int i = 0; i < TILE_SIZE; i++){
            float a = sA[threadIdx.y][i];  
            sum[0] += a * sB[i][threadIdx.x];
            sum[1] += a * sB[i][threadIdx.x + TILE_SIZE];
            sum[2] += a * sB[i][threadIdx.x + TILE_SIZE * 2];
            sum[3] += a * sB[i][threadIdx.x + TILE_SIZE * 3]; 
        }
        __syncthreads();
    }

    float4 result = {sum[0], sum[1], sum[2], sum[3]};
    reinterpret_cast<float4*>(&C[row * N + col])[0] = result;
}

int main(){
    int size = N*N*sizeof(float);
    float* h_A = new float[N*N];
    float* h_B = new float[N*N];
    float* h_C = new float[N*N];
    float *d_A, *d_B, *d_C;

    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    for(int i = 0; i < N*N; i++){
        h_A[i] = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
        h_B[i] = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
        h_C[i] = 0;
    }

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks(N / (TILE_SIZE * 4), N / TILE_SIZE);  

    kernel<<<blocks, threads>>>(d_A, d_B, d_C);
    cudaError_t err = cudaGetLastError();
    printf("Launch error: %s\n", cudaGetErrorString(err));

    err = cudaDeviceSynchronize();
    printf("Sync error: %s\n", cudaGetErrorString(err));    
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return 0;
}