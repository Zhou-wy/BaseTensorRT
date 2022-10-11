# 核函数

- 1、 核函数是cuda编程的关键
- 2、通过xxx.cu创建一个cudac程序文件，并把cu交给nvcc编译，才能识别cuda语法
- 3、__global__表示为核函数，由host调用。__device__表示为设备函数，由device调用
- 4、 __host__表示为主机函数，由host调用。__shared__表示变量为共享变量
- 5、host调用核函数：function<<<gridDim, blockDim, sharedMemorySize, stream>>>(args…);
    ```c++
    Dim3 gridDim;
    DIm3 blockDim;
    //总线程数
    int nthreads = gridDim.x * gridDim.y * gridDim.z * blockDim.x * blockDim.y * blockDim.z
    //上限值
    //gridDim(21亿，65536，65536)
    //blockDim(1024，64，64)
    //blockDim.x * blockDim.y * blockDim.z <=1024
    ```
- 6、只有__global__修饰的函数才可以用<<<>>>的方式调用
- 7、调用核函数是传值的，**不能传引用**，可以传递类、结构体等，核函数可以是模板，返回值必须是void
- 8、核函数的执行，是异步的，也就是立即返回的
- 9、线程layout主要用到blockDim、gridDim
- 10、核函数内访问线程索引主要用到threadIdx、blockIdx、blockDim、gridDim这些内置变量


**核函数里面，把blockDim、gridDim看作shape，把threadIdx、blockIdx看做index，则可以按照维度高低排序看待这个信息：**

![](https://blog-1300216920.cos.ap-nanjing.myqcloud.com/20220926152215.png)