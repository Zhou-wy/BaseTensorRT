#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include "sharedMemory.cuh"

#define checkRuntime(op) __check_cuda_runtime((op), #op, __FILE__, __LINE__)

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

int main(int argc, char const *argv[])
{
    cudaDeviceProp prop;
    checkRuntime(cudaGetDeviceProperties(&prop, 0));

    // 通过查询maxGridSize和maxThreadsDim参数，得知能够设计的gridDims、blockDims的最大值
    // warpSize则是线程束的线程数量
    // maxThreadsPerBlock则是一个block中能够容纳的最大线程数，也就是说blockDims[0] * blockDims[1] * blockDims[2] <= maxThreadsPerBlock
    std::cout << "prop.maxGridSize = " << prop.maxGridSize[0] << "," << prop.maxGridSize[1] << "," << prop.maxGridSize[2] << std::endl;
    std::cout << "prop.maxThreadsDim = " << prop.maxThreadsDim[0] << "," << prop.maxThreadsDim[1] << "," << prop.maxThreadsDim[2] << std::endl;
    std::cout << "prop.warpSize = " << prop.warpSize << std::endl;
    std::cout << "prop.maxThreadsPerBlock = " << prop.maxThreadsPerBlock << std::endl;

    int grids[] = {1, 2, 3};               // gridDim.x  gridDim.y  gridDim.z
    int blocks[] = {1024, 1, 1};           // blockDim.x blockDim.y blockDim.z
    launch(grids, blocks);                 // grids表示的是有几个大格子，blocks表示的是每个大格子里面有多少个小格子
    checkRuntime(cudaPeekAtLastError());   // 获取错误 code 但不清楚error
    checkRuntime(cudaDeviceSynchronize()); // 进行同步，这句话以上的代码全部可以异步操作
    printf("done\n");
    return 0;
}
