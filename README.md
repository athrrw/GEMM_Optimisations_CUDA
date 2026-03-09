cuBLAS is Nvidia's standard go-to library for accelerating their GEMM code.
In this repository I have included my attempts of writing kernels (without using TENSOR_WMMA) to try and meet the cuBLAS performance
on my own gpu. The method I have followed is considering naive GEMM implementation as a baseline and then using NSight Compute to
find the benchmark metrics. 

GFLOPs (Giga-Float operations per second): 
-
  GFLOPs = 2 * (Matrix-Size)^3 / (1e6 * time_ms)
  (and the same for TFLOPs but using 1e9)
  <br>
  <br>
    <img width="1483" height="883" alt="gflops_chart" src="https://github.com/user-attachments/assets/c7531d22-3daa-4099-beda-cf09fd67393b" />

Time metrics: using NCU --metrics gpu__time_duration.min <kernel> 
-

1. Linear Scale (Time vs Matirx size)
   -
    <br>
    <img width="2076" height="888" alt="time_chart_linear" src="https://github.com/user-attachments/assets/523ed6ac-3e03-4861-8587-71f5f67e426c" />

2. Log Scale (Time vs Matrix size)
   -

    <br>
    <img width="1483" height="883" alt="time_chart" src="https://github.com/user-attachments/assets/1644a46c-7a0a-4b10-a78e-f851a79a61dd" />



