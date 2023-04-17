/*
 * @Description:
 * @version:
 * @Author: zwy
 * @Date: 2022-10-04 09:54:55
 * @LastEditors: zwy
 * @LastEditTime: 2023-04-04 13:52:15
 */
#include <iostream>

#include "yolo.h"

int main(int argc, char const *argv[])
{
    TRTLogger logger;
    if (!build_model("/home/zwy/CWorkspace/BaseTensorRT/4.1-yolov5-detect/workspace/zkView.onnx", "/home/zwy/CWorkspace/BaseTensorRT/4.1-yolov5-detect/workspace/zkView.trtmodel", logger))
    {
        return -1;
    }
    cv::Mat src = cv::imread("/home/zwy/CWorkspace/BaseTensorRT/4.1-yolov5-detect/workspace/68.JPG");

    if (!inference(src, "/home/zwy/CWorkspace/BaseTensorRT/4.1-yolov5-detect/workspace/zkView.trtmodel", "/home/zwy/CWorkspace/BaseTensorRT/4.1-yolov5-detect/workspace/68_res.JPG", logger, true))
    {
        return -1;
    }
    return 0;
}
