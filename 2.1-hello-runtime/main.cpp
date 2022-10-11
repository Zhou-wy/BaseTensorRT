#include <iostream>
#include <cuda.h>
// CUDA运行时头文件
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
    CUcontext context = nullptr;
    cuCtxGetCurrent(&context);
    std::cout << "Current context = " << context << "当前无context" << std::endl;

    // cuda runtime是以cuda为基准开发的运行时库
    // cuda runtime所使用的CUcontext是基于cuDevicePrimaryCtxRetain函数获取的
    // 即，cuDevicePrimaryCtxRetain会为每个设备关联一个context，通过cuDevicePrimaryCtxRetain函数可以获取到
    // 而context初始化的时机是懒加载模式，即当你调用一个runtime api时，会触发创建动作
    // 也因此，避免了cu驱动级别的init和destroy操作。使得api的调用更加容易

    int device_count = 0;
    checkRuntime(cudaGetDeviceCount(&device_count));
    std::cout << "evice_count = " << device_count << std::endl;

    // 取而代之，是使用setdevice来控制当前上下文，当你要使用不同设备时
    // 使用不同的device id
    // 注意，context是线程内作用的，其他线程不相关的, 一个线程一个context stack
    int device_id = 0;
    std::cout << "set current device to : " << device_id << " ,这个API依赖CUcontext,触发创建并设置" << std::endl;
    checkRuntime(cudaSetDevice(device_id));

    // 注意，是由于set device函数是“第一个执行的需要context的函数”，所以他会执行cuDevicePrimaryCtxRetain
    // 并设置当前context，这一切都是默认执行的。注意：cudaGetDeviceCount是一个不需要context的函数
    // 你可以认为绝大部分runtime api都是需要context的，所以第一个执行的cuda runtime函数，会创建context并设置上下文
    cuCtxGetCurrent(&context);
    std::cout << "SetDevice after, Current context" << context << "获取当前context" << std::endl;

    int current_device = 0;
    checkRuntime(cudaGetDevice(&current_device));
    std::cout << "current_device = " << current_device << std::endl;

    return 0;
}
