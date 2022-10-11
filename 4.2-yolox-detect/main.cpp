/*
 * @Description:
 * @version:
 * @Author: zwy
 * @Date: 2022-10-04 20:30:45
 * @LastEditors: zwy
 * @LastEditTime: 2022-10-05 09:24:14
 */
#include <iostream>
#include "yolox.h"
int main(int argc, char const *argv[])
{
    TRTLogger logger;
    if (!build_model("../workspace/yolox.trtmodel", "../workspace/yolox_nano-change.onnx", logger))
        return -1;

    cv::Mat image = cv::imread("../workspace/test.jpg");
    if (image.empty())
    {
        printf("can not find the image file!");
        return -1;
    }
    if (!inference(image, "../workspace/yolox.trtmodel", logger))
        return -1;

    return 0;
}
