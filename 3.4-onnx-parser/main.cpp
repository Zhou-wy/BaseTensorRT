/*
 * @Descripttion:
 * @version:
 * @Author: zwy
 * @Date: 2022-10-02 09:36:18
 * @LastEditors: zwy
 * @LastEditTime: 2022-10-03 10:44:39
 */
#include <iostream>

// // TensorRT include
// #include <NvInfer.h>
// #include <NvInferRuntime.h>
// #include <NvOnnxParser.h>

// // cuda include
// #include <cuda.h>
// #include <cuda_runtime.h>

// personal include
#include "onnxParser.h"

int main(int argc, char const *argv[])
{
    build_model("/home/zwy/BaseTensorRT/3.4-onnx-parser/bin/demo.onnx");
    inference("/home/zwy/BaseTensorRT/3.4-onnx-parser/bin/engine.trtmodel");
    return 0;
}
