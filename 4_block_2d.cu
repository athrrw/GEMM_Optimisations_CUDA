#include <iostream>
#include <cuda_runtime.h>
#include <random>

const int N         = 4096;
const int TILE_SIZE = 32;
const int TM        = 4;    // each thread, TM rows
const int TN        = 4;    // each thread, TN cols

__global__ void kernel_2d(float* A, float* B, float* C){

    __shared__ float sA[TILE_SIZE][TILE_SIZE];  
    __shared__ float sB[TILE_SIZE][TILE_SIZE];  

    int tx = threadIdx.x;  
    int ty = threadIdx.y;  

    
    int row = blockIdx.y * TILE_SIZE + ty * TM;  
    int col = blockIdx.x * TILE_SIZE + tx * TN;  

    float sum[TM][TN] = {};

    float regA[TM];
    float regB[TN];

    int num_tiles = N / TILE_SIZE;

    for(int t = 0; t < num_tiles; t++){
        for(int i = 0; i < TM; i++){
            float4 aVec = reinterpret_cast<float4*>(
                &A[(row + i) * N + t * TILE_SIZE + tx * TN])[0];
            sA[ty * TM + i][tx * TN + 0] = aVec.x;
            sA[ty * TM + i][tx * TN + 1] = aVec.y;
            sA[ty * TM + i][tx * TN + 2] = aVec.z;
            sA[ty * TM + i][tx * TN + 3] = aVec.w;
        }

        
        for(int i = 0; i < TM; i++){
            float4 bVec = reinterpret_cast<float4*>(
                &B[(t * TILE_SIZE + ty * TM + i) * N + col])[0];
            sB[ty * TM + i][tx * TN + 0] = bVec.x;
            sB[ty * TM + i][tx * TN + 1] = bVec.y;
            sB[ty * TM + i][tx * TN + 2] = bVec.z;
            sB[ty * TM + i][tx * TN + 3] = bVec.w;
        }

        __syncthreads();

        for(int k = 0; k < TILE_SIZE; k++){
            for(int i = 0; i < TM; i++){
                regA[i] = sA[ty * TM + i][k];
            }

            for(int j = 0; j < TN; j++){
                regB[j] = sB[k][tx * TN + j];
            }

            for(int i = 0; i < TM; i++){
                for(int j = 0; j < TN; j++){
                    sum[i][j] += regA[i] * regB[j];
                }
            }
        }

        __syncthreads();
    }
    for(int i = 0; i < TM; i++){
        reinterpret_cast<float4*>(&C[(row + i) * N + col])[0] =
            make_float4(sum[i][0], sum[i][1], sum[i][2], sum[i][3]);
    }
}

int main(){
    int size = N * N * sizeof(float);
    float *h_A = new float[N*N], *h_B = new float[N*N], *h_C = new float[N*N];
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size); cudaMalloc(&d_B, size); cudaMalloc(&d_C, size);

    for(int i = 0; i < N*N; i++){
        h_A[i] = static_cast<float>(rand()) / RAND_MAX;
        h_B[i] = static_cast<float>(rand()) / RAND_MAX;
        h_C[i] = 0.0f;
    }
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    
    dim3 threads(TILE_SIZE / TN, TILE_SIZE / TM);   
    dim3 blocks(N / TILE_SIZE, N / TILE_SIZE);

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);

    kernel_2d<<<blocks, threads>>>(d_A, d_B, d_C);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    double tflops = (2.0 * N * N * N) / (ms / 1000.0) / 1e12;
    printf("2D tiling \n Time: %.3f ms \n TFLOPS: %.3f\n", ms, tflops);

    cudaError_t err = cudaGetLastError();
    printf("Launch: %s\n", cudaGetErrorString(err));
    err = cudaDeviceSynchronize();
    printf("Sync:   %s\n", cudaGetErrorString(err));

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    delete[] h_A; 
    delete[] h_B; 
    delete[] h_C;

    cudaFree(d_A); 
    cudaFree(d_B); 
    cudaFree(d_C);

    return 0;
}