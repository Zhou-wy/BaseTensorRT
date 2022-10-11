# CUDA核函数

在GPU上执行的函数称为CUDA核函数（Kernel Function)，核函数会被GPU上多个线程执行，我们可以在核函数中获取当前线程的ID。

```c++
// CUDA核函数的定义
__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}
```
```c++
// CUDA核函数调用
addKernel<<<Dg,Db, Ns, S>>>(c, a, b);
```
- 可以总结出CUDA核函数的使用方式：

使用关键字global来标识，在CPU上调用，在GPU上执行，返回值为void
使用<<< >&gt;&gt;来指定线程索引方式

核函数相对于CPU是异步的，在核函数执行完之前就会返回，这样CPU可以不用等待核函数的完成，继续执行后续代码

核函数不支持可变参数，不支持静态变量，不支持函数指针

- 线程配置

这是刚刚接触GPU编程最为困惑的地方，到底应该如何去索引GPU的线程？首先要对GPU中的基本概念有所了解。

在Host端核函数的调用方式为：
```c++
kernel<<<Dg, Db, Ns, S>>>(param list);
```
其中，
Dg：int型或者dim3类型(x,y,z)，用于定义一个Grid中Block是如何组织的，如果是int型，则表示一维组织结构

Db：int型或者dim3类型(x,y,z)，用于定义一个Block中Thread是如何组织的，如果是int型，则表示一维组织结构

Ns：size_t类型，可缺省，默认为0； 用于设置每个block除了静态分配的共享内存外，最多能动态分配的共享内存大小，单位为byte。 0表示不需要动态分配。

S：cudaStream_t类型，可缺省，默认为0。 表示该核函数位于哪个流。
Grid/Block/Thread都是软件的组织结构，并不是硬件的，因此理论上我们可以以任意的维度（一维、二维、三维）去排列Thread；在硬件上就是一个个的SP，并没有维度这一说，只是软件上抽象成了具有维度的概念。

当使用dim3类型时，比如：
```c++
dim3 grid(3,2,1), block(4,3,1);
kernel_name<<<grid, block>>>(...);
```
表示一个Grid中有3x2x1=6个Block，在(x,y,z)三个方向上的排布方式分别是3、2、1；一个Block中有4x3x1=12个Thread，在(x,y,z)三个方向上的排布方式分别是4、3、1。

当使用int类型时，表示一维排布，比如：

```c++
kernel_name<<<5,8>>>(...);
```
表示一个Grid中有5个Block，在(x,y,z)三个方向上的排布方式分别是5、1、1；一个Block中有8个Thread，在(x,y,z)三个方向上的排布方式分别是8、1、1。

在CUDA上可以使用内置变量来获取Thread ID和Block ID：

threadIdx.[x, y, z]表示Block内Thread的编号

blockIdx.[x, y, z]表示Gird内Block的编号

blockDim.[x, y, z]表示Block的维度，也就是Block中每个方向上的Thread的数目

gridDim.[x, y, z]表示Gird的维度，也就是Grid中每个方向上Block的数目

下面我们举几个例子来说明

**一维Grid 一维Block**

kernel_name<<<4, 8>>>(...)
具体的线程索引方式如下图所示，blockIdx从0到3，threadIdx从0到7.

当我们要计算下图中红色的Thread的索引时，可以看出，它的blockIdx.x是2，threadIdx.x是1，因此它的threadId索引计算方式为：
![](https://pic2.zhimg.com/v2-d02c028a2fc90caeafabf34845ed1175_r.jpg)
```c++
int threadId = blockIdx.x * blockDim.x + threadIdx.x = 2 * 4 + 1 = 9
```

**二维Grid 二维Block**

无论是几维的，索引的原则是一样的，先求出这个Thread前面的所有Block中Thread的数量，再求出该Thread在本Block中的序号，两个相加即可。
![](https://pic4.zhimg.com/v2-0979b3112900f56b2d6b8ef498ec83c3_r.jpg)
下面为了画图方便，我们以将Block的维度设为(4,1,1)，其实是一维Block了，但计算公式是一样的：
```c++
dim grid(4,1,1), block(2,2,1);
kernel_name<<<grid, block>>>(...)
```
需要注意的是，二维排序中，Thread(0,1)表示第1行第0列的Thread，这跟我们传统中理解的横坐标和纵坐标不太一样；我们定义grid(4,2)表示第一维度有4个索引值，第二个维度有2个索引值，即2行4列

具体排列方式如下图所示，blockidx从0到3，Threadidx从(0,0)到(1,2)


Idx的具体索引方式如下公式，idx表示当前Thread在全局索引中的序号
```c++
int blockId = blockIdx.x + blockId.y * gridDim.x;
int threadId = blockId * (blockDim.x * blockDim.y) + (threadIdx.y *blockDim.x) + threadIdx.x;
```
**三维Grid 三维Block**

三维的图画起来有点复杂，我们就不画图，直接给出计算公式：
```c++
int blockId = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
int threadIc = blockId * (blockDim.x * blockDim.y * blockDim.z) 
                       + (threadIdx.z * (blockDim.x * blockDim.y)) 
                       + (threadIdx.y * blockDim.x) + threadIdx.x;   
```
三维Grid 三维Thread是参数最多的组合，对于其他维度的组合，只需要把相应的参数置1即可，下面我们将各种组合总结一下，以后大家可以直接使用：(一维的话我们默认是使用x维度，二维的话默认使用(x,y)维度)

一维Grid 一维Block
```c++
blockId = blockIdx.x 
threadId = blockIdx.x *blockDim.x + threadIdx.x
```


一维Grid 二维Block
```c++
blockId = blockIdx.x 
threadId = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x
```

一维Grid 三维Block
```c++
blockId = blockIdx.x 
threadId = blockIdx.x * blockDim.x * blockDim.y * blockDim.z 
                        + threadIdx.z * blockDim.y * blockDim.x 
                        + threadIdx.y * blockDim.x + threadIdx.x
```

二维Grid 一维Block
```c++
int blockId = blockIdx.y * gridDim.x + blockIdx.x;  
int threadId = blockId * blockDim.x + threadIdx.x;
```

二维Grid 二维Block
```c++
int blockId = blockIdx.x + blockIdx.y * gridDim.x;  
int threadId = blockId * (blockDim.x * blockDim.y) 
                       + (threadIdx.y * blockDim.x) + threadIdx.x;  
```

二维Grid 三维Block
```c++
int blockId = blockIdx.x + blockIdx.y * gridDim.x;  
int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)  
                       + (threadIdx.z * (blockDim.x * blockDim.y))  
                       + (threadIdx.y * blockDim.x) + threadIdx.x;  
```

三维Grid 一维Block
```c++
int blockId = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;  
int threadId = blockId * blockDim.x + threadIdx.x;  
```

三维Grid 二维Block
```c++
int blockId = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;  
int threadId = blockId * (blockDim.x * blockDim.y) 
                       + (threadIdx.y * blockDim.x) + threadIdx.x;  
```

三维Grid 三维Block
```c++
int blockId = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;  
int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)  
                       + (threadIdx.z * (blockDim.x * blockDim.y))  
                       + (threadIdx.y * blockDim.x) + threadIdx.x;
```