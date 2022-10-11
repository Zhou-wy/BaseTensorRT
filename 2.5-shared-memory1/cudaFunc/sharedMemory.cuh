#ifndef __SHAREDMEMORY_CUH
#define __SHAREDMEMORY_CUH

#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

void launch(int *grids, int *blocks);
#endif
