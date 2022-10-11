#include "warpaffine.cuh"

bool __check_cuda_runtime(cudaError_t code, const char *op, const char *file, int line)
{
    if (code != cudaSuccess)
    {
        const char *err_name = cudaGetErrorName(code);
        const char *err_message = cudaGetErrorString(code);
        printf("runtime error %s:%d  %s failed. \n  code = %s, message = %s\n", file, line, op, err_name, err_message);
        return false;
    }
    return true;
}

__device__ void affine_project(float *matrix, int x, int y, float *proj_x, float *proj_y)
{
    // matrix
    // m0, m1, m2
    // m3, m4, m5
    *proj_x = matrix[0] * x + matrix[1] * y + matrix[2];
    *proj_y = matrix[3] * x + matrix[4] * y + matrix[5];
}

__global__ void warp_affine_bilinear_kernel(uint8_t *src, int src_line_size, int src_width, int src_height,
                                            uint8_t *dst, int dst_line_size, int dst_width, int dst_height,
                                            uint8_t fill_value, AffineMatrix matrix)
{
    int dx = blockDim.x * blockIdx.x + threadIdx.x;
    int dy = blockDim.y * blockIdx.y + threadIdx.y;
    if (dx >= dst_width || dy >= dst_height)
        return;

    float c0 = fill_value, c1 = fill_value, c2 = fill_value;
    float src_x = 0;
    float src_y = 0;
    affine_project(matrix.d2i, dx, dy, &src_x, &src_y);

    if (src_x < -1 || src_x >= src_width || src_y < -1 || src_y >= src_height)
    {
        // src_x < -1时，其高位high_x < 0，超出范围
        // src_x >= -1时，其高位high_x >= 0，存在取值
    }
    else
    {
        int y_low = floorf(src_y);
        int x_low = floorf(src_x);
        int y_high = y_low + 1;
        int x_high = x_low + 1;

        uint8_t const_values[] = {fill_value, fill_value, fill_value};
        float ly = src_y - y_low;
        float lx = src_x - x_low;
        float hy = 1 - ly;
        float hx = 1 - lx;
        float w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;
        uint8_t *v1 = const_values;
        uint8_t *v2 = const_values;
        uint8_t *v3 = const_values;
        uint8_t *v4 = const_values;
        if (y_low >= 0)
        {
            if (x_low >= 0)
                v1 = src + y_low * src_line_size + x_low * 3;

            if (x_high < src_width)
                v2 = src + y_low * src_line_size + x_high * 3;
        }

        if (y_high < src_height)
        {
            if (x_low >= 0)
                v3 = src + y_high * src_line_size + x_low * 3;

            if (x_high < src_width)
                v4 = src + y_high * src_line_size + x_high * 3;
        }

        c0 = floorf(w1 * v1[0] + w2 * v2[0] + w3 * v3[0] + w4 * v4[0] + 0.5f);
        c1 = floorf(w1 * v1[1] + w2 * v2[1] + w3 * v3[1] + w4 * v4[1] + 0.5f);
        c2 = floorf(w1 * v1[2] + w2 * v2[2] + w3 * v3[2] + w4 * v4[2] + 0.5f);
    }

    uint8_t *pdst = dst + dy * dst_line_size + dx * 3;
    // pdst[0] = c0; //R
    // pdst[1] = c1; //G
    // pdst[2] = c2; //B
    pdst[0] = c2; // B
    pdst[1] = c1; // G
    pdst[2] = c0; // R
}

void warp_affine_bilinear( // 声明
    uint8_t *src, int src_line_size, int src_width, int src_height,
    uint8_t *dst, int dst_line_size, int dst_width, int dst_height,
    uint8_t fill_value)
{
    dim3 block_size(32, 32); // blocksize最大就是1024，这里用2d来看更好理解
    dim3 grid_size((dst_width + 31) / 32, (dst_height + 31) / 32);
    AffineMatrix affine;
    affine.compute(cv::Size(src_width, src_height), cv::Size(dst_width, dst_height));
    for (int i = 0; i < 6; i++)
    {
        std::cout << *(affine.i2d + i) << " ";
        if (i % 3 ==2)
        {
            std::cout << std::endl;
        }
    }
    std::cout << std::endl;
    for (int i = 0; i < 6; i++)
    {
        std::cout << *(affine.d2i + i) << " ";
        if(i % 2 == 1){
            std::cout << std::endl;
        }
    }
    
    warp_affine_bilinear_kernel<<<grid_size, block_size, 0, nullptr>>>(
        src, src_line_size, src_width, src_height,
        dst, dst_line_size, dst_width, dst_height,
        fill_value, affine);
}
