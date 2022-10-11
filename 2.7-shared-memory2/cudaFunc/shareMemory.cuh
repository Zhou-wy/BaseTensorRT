/*
 * @Descripttion:
 * @version:
 * @Author: zwy
 * @Date: 2022-09-28 09:55:52
 * @LastEditors: zwy
 * @LastEditTime: 2022-09-28 10:12:59
 */

#ifndef __SHARED_MEMORY_CUH
#define __SHARED_MEMORY_CUH
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
bool __check_cuda_runtime(cudaError_t code, const char *op, const char *file, int line);
#define checkRuntime(op) __check_cuda_runtime((op), #op, __FILE__, __LINE__);

void launch();
#endif