#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include "vectorAdd.cuh"

double cpuSecond()
{
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1e-6);
}

void initialData(float *ip, int size)
{
    time_t t;
    srand((unsigned)time(&t));
    for (int i = 0; i < size; i++)
    {
        ip[i] = (float)(rand() & 0xffff) / 1000.0f;
    }
}

bool __check_cuda_runtime(cudaError_t code, const char *op, const char *file, int line)
{
    if (code != cudaSuccess)
    {
        const char *err_name = cudaGetErrorName(code);
        const char *err_message = cudaGetErrorString(code);
        printf("runtime error %s:%d  %s failed. \n  code = %s, message = %s\n", file, line, op, err_name, err_message);
        return false;
    }
    return true;
}

void initDevice(int devNum)
{
    int dev = devNum;
    cudaDeviceProp deviceProp;
    checkRuntime(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Using device %d: %s\n", dev, deviceProp.name);
    checkRuntime(cudaSetDevice(dev));
}

void checkResult(float *hostRef, float *gpuRef, const int N)
{
    double epsilon = 1.0E-8;
    for (int i = 0; i < N; i++)
    {
        if (abs(hostRef[i] - gpuRef[i]) > epsilon)
        {
            printf("Results don\'t match!\n");
            printf("%f(hostRef[%d] )!= %f(gpuRef[%d])\n", hostRef[i], i, gpuRef[i], i);
            return;
        }
    }
    printf("Check result success!\n");
}

//核函数，每一个线程计算矩阵中的一个元素。
__global__ void sumMatrix(const float *MatA, const float *MatB, float *MatC, int nx, int ny)
{
    int ix = threadIdx.x + blockDim.x * blockIdx.x;
    int iy = threadIdx.y + blockDim.y * blockIdx.y;
    int idx = ix + iy * ny;
    if (ix < nx && iy < ny)
    {
        MatC[idx] = MatA[idx] + MatB[idx];
    }
}

// CPU对照组，用于对比加速比
void sumMatrix2DonCPU(float *MatA, float *MatB, float *MatC, int nx, int ny)
{
    float *a = MatA;
    float *b = MatB;
    float *c = MatC;
    for (int j = 0; j < ny; j++)
    {
        for (int i = 0; i < nx; i++)
        {
            c[i] = a[i] + b[i];
        }
        c += nx;
        b += nx;
        a += nx;
    }
}

// GPU实验组，用cuda加速
void sumMatrix2DonGPU(const float *A_dev, const float *B_dev, float *C_dev, int nx, int ny)
{
    // // 如果ndata < nthreads 那block_size = ndata就够了
    // int block_size = ndata < nThreads ? ndata : nThreads;
    // // 其含义是我需要多少个blocks可以处理完所有的任务
    // int grid_size = (ndata + block_size - 1) / block_size;
    //二维线程块，32×32
    dim3 block(32, 32);
    //二维线程网格，128×128
    dim3 grid((nx - 1) / block.x + 1, (ny - 1) / block.y + 1);
    sumMatrix<<<grid, block>>>(A_dev, B_dev, C_dev, nx, ny);
    checkRuntime(cudaDeviceSynchronize());
    // cudaPeekAtLastError和cudaGetLastError都可以获取得到错误代码
    // cudaGetLastError是获取错误代码并清除掉，也就是再一次执行cudaGetLastError获取的会是success
    // 而cudaPeekAtLastError是获取当前错误，但是再一次执行cudaPeekAtLastError或者cudaGetLastErro拿到的还是那个错
    cudaError_t code = cudaPeekAtLastError();
    if (code != cudaSuccess)
    {
        const char *err_name = cudaGetErrorName(code);
        const char *err_message = cudaGetErrorString(code);
        printf("kernel error %s:%d  test_print_kernel failed. \n  code = %s, message = %s\n", __FILE__, __LINE__, err_name, err_message);
    }
}
