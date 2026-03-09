#include <cuda_runtime.h>
#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <vector>
#include <chrono>
using namespace std;
using std::cout;
using std::generate;
using std::vector;


__global__ void transpose(float* B, float* B_t, int N) {
	int row = threadIdx.y + blockDim.y * blockIdx.y;
	int col = threadIdx.x + blockDim.x * blockIdx.x;
	if (row < N && col < N) {
		B_t[col * N + row] = B[row * N + col];
	}
}
__global__ void matrixMul(float* A, float* B_t, float* C, int N) {
	int row = threadIdx.y + blockDim.y * blockIdx.y;
	int col = threadIdx.x + blockDim.x * blockIdx.x;

	float temp = 0.0f;
	if (row < N && col < N) {
		for (int i = 0; i < N; i++) {
			temp += A[i + row * N] * B_t[col + i * N];
		}
		C[row * N + col] = temp;
	}
}

int main() {
	int N = 2048;
	size_t bytes = N * N * sizeof(float);

	// host matrices
	vector<float> h_A(N * N);
	vector<float> h_B(N * N);
	vector<float> h_C(N * N);

	//malloc for host matrices
	generate(h_A.begin(), h_A.end(), []() {return static_cast<float>(rand() % 100);});
	generate(h_B.begin(), h_B.end(), []() {return static_cast<float>(rand() % 100);});

	// device vectors
	float* d_A, * d_B, * d_C, * d_Bt;
	cudaMalloc(&d_A, bytes);
	cudaMalloc(&d_B, bytes);
	cudaMalloc(&d_C, bytes);
	cudaMalloc(&d_Bt, bytes);

	cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice);

	h_A.clear();
	h_B.clear();

	dim3 grid_size((N + 15) / 16, (N + 15) / 16);
	dim3 block_size(16, 32);

	transpose << <grid_size, block_size >> > (d_B, d_Bt, N);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start, 0);
	matrixMul <<< grid_size, block_size >>> (d_A, d_Bt, d_C, N);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	float ms = 0.0f;
	float flops = 2.0f * N * N * N;
	cudaEventElapsedTime(&ms, start, stop);
	float gflops = flops / (ms * 1e6f);

	cudaMemcpy(h_C.data(), d_C, bytes, cudaMemcpyDeviceToHost);

	cout << "Time Taken for execution of " << N << "x" << N << " matrix is: " << ms << "ms. ";
	cout << "\n\nGLOPS Performance is: " << gflops;

	cudaEventDestroy(start);
	cudaEventDestroy(stop);	

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

	return 0;
}