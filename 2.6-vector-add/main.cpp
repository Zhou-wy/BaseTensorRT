#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>

#include "vectorAdd.cuh"

int main(int argc, char const *argv[])
{
    //输入二维矩阵,单精度浮点型。
    int nx = 1 << 14;
    int ny = 1 << 14;

    float *vector_a = (float *)malloc(nx * ny * sizeof(float));
    float *vector_b = (float *)malloc(nx * ny * sizeof(float));
    float *vector_c = (float *)malloc(nx * ny * sizeof(float));
    float *vector_c_gpu = (float *)malloc(nx * ny * sizeof(float));
    initialData(vector_a, nx * ny);
    initialData(vector_b, nx * ny);

    float *vector_a_device = nullptr;
    float *vector_b_device = nullptr;
    float *vector_c_device = nullptr;

    checkRuntime(cudaMalloc(&vector_a_device, nx * ny * sizeof(float)));
    checkRuntime(cudaMalloc(&vector_b_device, nx * ny * sizeof(float)));
    checkRuntime(cudaMalloc(&vector_c_device, nx * ny * sizeof(float)));

    checkRuntime(cudaMemcpy(vector_a_device, vector_a, nx * ny * sizeof(float), cudaMemcpyHostToDevice));
    checkRuntime(cudaMemcpy(vector_b_device, vector_b, nx * ny * sizeof(float), cudaMemcpyHostToDevice));

    //测试GPU执行时间
    double gpuStart = cpuSecond();
    sumMatrix2DonGPU(vector_a_device, vector_b_device, vector_c_device, nx, ny);
    double gpuTime = cpuSecond() - gpuStart;
    printf("GPU Execution Time: %f sec\n", gpuTime);

    //在CPU上完成相同的任务
    checkRuntime(cudaMemcpy(vector_c_gpu, vector_c_device, nx * ny * sizeof(float), cudaMemcpyDeviceToHost));
    double cpuStart = cpuSecond();
    sumMatrix2DonCPU(vector_a, vector_b, vector_c, nx, ny);
    double cpuTime = cpuSecond() - cpuStart;
    printf("CPU Execution Time: %f sec\n", cpuTime);

    //检查GPU与CPU计算结果是否相同
    checkResult(vector_c, vector_c_gpu, nx * ny);

    checkRuntime(cudaFree(vector_a_device));
    checkRuntime(cudaFree(vector_b_device));
    checkRuntime(cudaFree(vector_c_device));
    free(vector_a);
    free(vector_b);
    free(vector_c);
    free(vector_c_gpu);
    return 0;
}
