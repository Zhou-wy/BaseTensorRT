#ifndef __VECTORADD_CUH
#define __VECTORADD_CUH
#include <sys/time.h>
#define checkRuntime(op) __check_cuda_runtime((op), #op, __FILE__, __LINE__);


void sumMatrix2DonGPU(const float *A_dev, const float *B_dev, float *C_dev, int nx, int ny);
void sumMatrix2DonCPU(float *MatA, float *MatB, float *MatC, int nx, int ny);
double cpuSecond();
void initialData(float *ip, int size);
void initDevice(int devNum);
void checkResult(float *hostRef, float *gpuRef, const int N);
bool __check_cuda_runtime(cudaError_t code, const char *op, const char *file, int line);
#endif
