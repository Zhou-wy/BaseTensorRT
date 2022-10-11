/*
 * @Description:
 * @version:
 * @Author: zwy
 * @Date: 2022-10-05 09:48:44
 * @LastEditors: zwy
 * @LastEditTime: 2022-10-05 15:09:40
 */
#include <iostream>
#include "unet.h"
int main(int argc, char const *argv[])
{
    TRTLogger logger;
    cv::Mat image = cv::imread("../workspace/street.jpg");
    if (image.empty())
    {
        printf("\033[31m[error]:Could not file the image.\033[0m\n");
        return -1;
    }

    if (!build_model("../workspace/unet.trtmodel", "../workspace/unet.onnx", logger))
    {
        return -1;
    }
    inference(image, "../workspace/unet.trtmodel", "../workspace/unet-trt.jpg", logger);
    return 0;
}
