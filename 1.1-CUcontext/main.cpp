#include <iostream>
#include <string>
#include <cuda.h>

int main(int argc, char const *argv[])
{
    /*
    cuInit(int flags), 这里的flags目前必须给0;
        对于cuda的所有函数，必须先调用cuInit，否则其他API都会返回CUDA_ERROR_NOT_INITIALIZED
        https://docs.nvidia.com/cuda/archive/11.2.0/cuda-driver-api/group__CUDA__INITIALIZE.html
     */
    CUresult code = cuInit(0);
    if (code != CUresult::CUDA_SUCCESS)
    {
        const char *err_message = nullptr;
        cuGetErrorString(code, &err_message);
        std::cout << "Initialize failed. code = " << code << ", message =  " << err_message << std::endl;
        return -1;
    }
    /*
    测试获取当前cuda驱动的版本
    显卡、CUDA、CUDA Toolkit

     1. 显卡驱动版本，比如：Driver Version: 460.84
     2. CUDA驱动版本：比如：CUDA Version: 11.2
     3. CUDA Toolkit版本：比如自行下载时选择的10.2、11.2等；这与前两个不是一回事, CUDA Toolkit的每个版本都需要最低版本的CUDA驱动程序

     三者版本之间有依赖关系, 可参照https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html
     nvidia-smi显示的是显卡驱动版本和此驱动最高支持的CUDA驱动版本
    */
    int driver_version = 0;
    code = cuDriverGetVersion(&driver_version);
    std::cout << "CUDA Driver version is " << driver_version << std::endl;
    char device_name[100];
    CUdevice device = 0;
    code = cuDeviceGetName(device_name, sizeof(device_name), device);
    std::cout << "Device is " << device << ", name is " << device_name << std::endl;
    return 0;
}
