# DriverAPI

## 1、驱动初始化-`cuInit`

- `cuInit`的意义是，**初始化驱动API**。如果不执行，则所有API都将返回错误，全局执行一次即可
- 没有对应的cuDestroy，不需要释放，程序销毁自动释放

```c++
/**
 * \brief Initialize the CUDA driver API
 * \param Flags - Initialization flag for CUDA.
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_INVALID_VALUE,
 * ::CUDA_ERROR_INVALID_DEVICE,
 * ::CUDA_ERROR_SYSTEM_DRIVER_MISMATCH,
 * ::CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE
 * \notefnerr
 */
CUresult CUDAAPI cuInit(unsigned int Flags);
```

## 2、返回值检查

- 正确友好的检查cuda函数的返回值，有利于程序的组织结构
- 使得代码可读性更好，错误更容易发现

使用有参宏定义检查cuda driver是否被正常初始化, 并定位程序出错的文件名、行数和错误信息。宏定义中带do...while循环可保证程序的正确性
```c++
#define checkDriver(op)          \
    do       			               \
    {  						               \
        auto code = (op);        \
        if (code != CUresult::CUDA_SUCCESS)               \
        {                                                 \
            const char *err_name = nullptr;               \
            const char *err_message = nullptr;            \
            cuGetErrorName(code, &err_name);              \
            cuGetErrorString(code, &err_message);         \
            std::cout << __FILE__ << " [line:" << __LINE__ << "] " #op << " is failed." << ", code = " << err_name << ", message = " << err_message << std::endl;\
            return -1; 	\
				}								\
    } while (0)
```


## 3、CUcontext

- context是一种上下文，关联对GPU的所有操作
- context与一块显卡关联，一个显卡可以被多个context关联
- **每个线程都有一个栈结构储存context**，栈顶是当前使用的context，对应有push、pop函数操作context的栈，所有api都以当前context为操作目标
- 如果执行任何操作你都需要传递一个device决定送到哪个设备执行，得多麻烦
![](https://blog-1300216920.cos.ap-nanjing.myqcloud.com/image-20220916135820592.png)

1、由于高频操作，是一个线程基本固定访问一个显卡不变，且只使用一个context，很少会用到多context。

2、`CreateContext`、`PushCurrent`、`PopCurrent`这种多context管理就显得麻烦，还得再简单。

3、因此推出了`cuDevicePrimaryCtxRetain`，为设备关联主context，分配、释放、设置、栈都不用你管。

4、`primaryContext`：给我设备id，给你context并设置好，此时一个显卡对应一个primary context。

5、不同线程，只要设备id一样，primary context就一样。context是线程安全的。
![](https://blog-1300216920.cos.ap-nanjing.myqcloud.com/image-20220916140118520.png)

