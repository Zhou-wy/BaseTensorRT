/*
 * @Descripttion:
 * @version:
 * @Author: zwy
 * @Date: 2022-10-03 11:08:56
 * @LastEditors: zwy
 * @LastEditTime: 2022-10-03 18:42:26
 */
#include <iostream>
#include <opencv2/opencv.hpp>

#include "cnnClassifier.h"

int main(int argc, char const *argv[])
{
    TRTLogger logger;
    if (!build_model("classifier.onnx", "demo.trtmodel", logger))
    {
        printf("Building tensorrt model is failed!");
        return -1;
    };
    inference("../workspace/ng.jpg", "../workspace/labels.imagenet.txt", "demo.trtmodel", logger);
    return 0;
}
