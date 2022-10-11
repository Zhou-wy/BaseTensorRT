/*
 * @description:
 * @version:
 * @Author: zwy
 * @Date: 2022-10-10 19:50:15
 * @LastEditors: zwy
 * @LastEditTime: 2022-10-10 20:44:07
 */
#include <algorithm>
#include <cuda_runtime.h>

#include "utils/cuda_tools.cuh"
#include "utils/simple_logger.hpp"
#include "Trt_Tensor.hpp"
namespace TRT
{
    int data_type_size(DataType dt)
    {
        switch (dt)
        {
        case DataType::Float:
            return sizeof(float);
        case DataType::Int32:
            return sizeof(int);
        case DataType::UInt8:
            return sizeof(uint8_t);
        default:
        {
            INFOE("Not support dtype: %d", dt);
            return -1;
        }
        }
    }
};