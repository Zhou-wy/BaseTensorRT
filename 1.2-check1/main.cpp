#include <cuda.h>
#include <iostream>

#define checkDriver(op) __check_cuda_driver((op), #op, __FILE__, __LINE__)
bool __check_cuda_driver(CUresult code, const char *op, const char *file, int line)
{
    if (code != CUresult::CUDA_SUCCESS)
    {
        const char *err_name = nullptr;
        const char *err_massage = nullptr;
        cuGetErrorName(code, &err_name);
        cuGetErrorString(code, &err_massage);
        std::cout << file << ":" << line << " " << op << "failed!"
                  << "code = " << err_name << "message = " << err_massage << std::endl;
        return false;
    }
    return true;
}

int main(int argc, char const *argv[])
{
    // 检查cuda driver的初始化
    // 实际调用的是__check_cuda_driver这个函数
    checkDriver(cuInit(0));

    // 测试获取当前cuda驱动的版本
    int driver_version = 0;
    if (!checkDriver(cuDriverGetVersion(&driver_version)))
    {
        return -1;
    }
    printf("Driver version is %d\n", driver_version);

    // 测试获取当前设备信息
    char device_name[100];
    CUdevice device = 0;
    checkDriver(cuDeviceGetName(device_name, sizeof(device_name), device));
    printf("Device %d name is %s\n", device, device_name);
    return 0;

    return 0;
}
