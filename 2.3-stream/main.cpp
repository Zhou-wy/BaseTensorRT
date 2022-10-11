#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

#define checkRuntime(op) __check_cuda_runtime((op), #op, __FILE__, __LINE__)

bool __check_cuda_runtime(cudaError_t code, const char *op, const char *file, int line)
{

    if (code != cudaSuccess)
    {
        const char *err_name = cudaGetErrorName(code);
        const char *err_message = cudaGetErrorString(code);
        std::cout << file << ":" << line << " " << op << "failed!"
                  << "code =" << err_name << "message = " << err_message << std::endl;
        return false;
    }
    return true;
}

int main(int argc, char const *argv[])
{
    int device_id = 0;
    checkRuntime(cudaSetDevice(device_id));

    cudaStream_t stream = nullptr;
    checkRuntime(cudaStreamCreate(&stream));

    //在GPU上开辟空间
    float *memory_device = nullptr;
    checkRuntime(cudaMalloc(&memory_device, 100 * sizeof(float)));

    //在GPU上开辟空间并放数据进去，将数据复制到GPU
    float *memory_host = new float[100];
    memory_host[2] = 3.1415926;
    /*cudaMemcpyAsync：异步复制时，发出指令立即返回，并不等待复制完成！*/
    checkRuntime(cudaMemcpyAsync(memory_device, memory_host, 100 * sizeof(float), cudaMemcpyHostToDevice, stream));

    //在cpus上开辟pin memory,并将GPU上的数据复制回来
    float *memory_page_locked = nullptr;
    checkRuntime(cudaMallocHost(&memory_page_locked, 100 * sizeof(float)));
    /*cudaMemcpyAsync：异步复制时，发出指令立即返回，并不等待复制完成！*/
    checkRuntime(cudaMemcpyAsync(memory_page_locked, memory_device, sizeof(float) * 100, cudaMemcpyDeviceToHost, stream));
    /* Error: 拷贝还未完成：输出不正处 */
    std::cout << "[Error]:拷贝未完成，输出不正确! memory page locked:" << memory_page_locked[2] << std::endl;

    checkRuntime(cudaStreamSynchronize(stream)); //统一等待流队列中所有操作结束，这步最耗时间

    std::cout << "memory page locked:" << memory_page_locked[2] << std::endl;

    checkRuntime(cudaFreeHost(memory_page_locked));
    checkRuntime(cudaFree(memory_device));
    checkRuntime(cudaStreamDestroy(stream));
    delete[] memory_host;

    return 0;
}
