/*
 * @Descripttion:
 * @version:
 * @Author: zwy
 * @Date: 2022-09-28 09:55:16
 * @LastEditors: zwy
 * @LastEditTime: 2022-09-28 10:13:33
 */
#include "shareMemory.cuh"

int main(int argc, char const *argv[])
{
    cudaDeviceProp prop;
    checkRuntime(cudaGetDeviceProperties(&prop, 0));
    printf("prop.sharedMemPerBlock = %.2f KB\n", prop.sharedMemPerBlock / 1024.0f);
    launch();
    checkRuntime(cudaPeekAtLastError());
    checkRuntime(cudaDeviceSynchronize());
    printf("done\n");
    return 0;
}
