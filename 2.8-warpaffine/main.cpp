/*
 * @Descripttion:
 * @version:
 * @Author: zwy
 * @Date: 2022-09-28 10:52:26
 * @LastEditors: zwy
 * @LastEditTime: 2022-10-02 09:05:21
 */
#include <opencv2/opencv.hpp>
#include "warpaffine.cuh"

cv::Mat warpaffine_to_center_align(cv::Mat &image, const cv::Size &size)
{
    cv::Mat output(size, CV_8UC3);
    uint8_t *psrc_dev = nullptr;
    uint8_t *pdst_dev = nullptr;
    size_t src_size = image.cols * image.rows * 3;
    size_t dst_size = size.width * size.height * 3;

    //在GPU上开辟两块空间
    checkRuntime(cudaMalloc(&psrc_dev, src_size));                                    //原图大小
    checkRuntime(cudaMalloc(&pdst_dev, dst_size));                                    //输出图大小
    checkRuntime(cudaMemcpy(psrc_dev, image.data, src_size, cudaMemcpyHostToDevice)); // 搬运数据到GPU上

    warp_affine_bilinear(
        psrc_dev, image.cols * 3, image.cols, image.rows,
        pdst_dev, size.width * 3, size.width, size.height,
        114);

    // 检查核函数执行是否存在错误
    checkRuntime(cudaPeekAtLastError());
    checkRuntime(cudaMemcpy(output.data, pdst_dev, dst_size, cudaMemcpyDeviceToHost)); // 将预处理完的数据搬运回来
    checkRuntime(cudaFree(pdst_dev));
    checkRuntime(cudaFree(psrc_dev));
    return output;
}

int main(int argc, char const *argv[])
{
    cv::Mat input_image = cv::imread("/home/zwy/BaseTensorRT/2.8-warpaffine/image/lena.png");
    std::cout << input_image.cols << " " << input_image.rows << " " << std::endl;
    cv::Mat output_image = warpaffine_to_center_align(input_image, cv::Size(640, 640));
    cv::imwrite("/home/zwy/BaseTensorRT/2.8-warpaffine/image/lenaRGB2BGR.jpg", output_image);
    std::cout << "Done. save to output.jpg" << std::endl;
    return 0;
}
