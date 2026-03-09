#include <iostream>
#include <cuda_runtime.h>
#include <random>

const int N  = 4096;
const int TILE_SIZE = 16;

__global__ void shared(const float* A, const float* B, float* C){
    __shared__ float sA[TILE_SIZE][TILE_SIZE];
    __shared__ float sB[TILE_SIZE][TILE_SIZE];
    float sum = 0.0f;

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    int num_tiles = N/TILE_SIZE;    
    for(int t = 0; t < num_tiles; t++){
        sA[threadIdx.y][threadIdx.x] = A[row*N + (t*TILE_SIZE + threadIdx.x)];
        sB[threadIdx.y][threadIdx.x] = B[N*(t*TILE_SIZE + threadIdx.y) + col];
        __syncthreads();
    
        for(int k = 0; k < TILE_SIZE; k++){
            sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];
            
        }
        __syncthreads();
    }
    C[row*N + col] = sum;
}

int main(){
    float* h_A = new float[N*N];
    float* h_B = new float[N*N];
    float* h_C = new float[N*N];

    float *d_A, *d_B, *d_C;
    int size = N*N*sizeof(float);
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

    dim3 threadsPerBlock(TILE_SIZE,TILE_SIZE);
    dim3 blocksPerGrid(N/TILE_SIZE, N/TILE_SIZE);

    shared <<< blocksPerGrid, threadsPerBlock >>> (d_A, d_B, d_C);

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    std::cout << "The size of matrix is: " << N << "x" << N<<std::endl;

    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    

}