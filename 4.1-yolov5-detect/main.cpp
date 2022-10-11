/*
 * @Description:
 * @version:
 * @Author: zwy
 * @Date: 2022-10-04 09:54:55
 * @LastEditors: zwy
 * @LastEditTime: 2022-10-04 16:01:22
 */
#include <iostream>

#include "yolo.h"

int main(int argc, char const *argv[])
{
    TRTLogger logger;
    if (!build_model("../workspace/yolov5s.onnx", "../workspace/yolov5s.trtmodel", logger))
    {
        return -1;
    }
    cv::Mat src = cv::imread("../workspace/test.jpg");

    if (!inference(src, "../workspace/yolov5s.trtmodel", "../workspace/test-trt.jpg", logger, true))
    {
        return -1;
    }
    return 0;
}
