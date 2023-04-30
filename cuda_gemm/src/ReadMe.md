# Optimize CUDA Gemm

## References
1. cuda for compare https://zhuanlan.zhihu.com/p/478846788
2. deepspeed
3. tvm

## perf
1. perf 工具 https://zhuanlan.zhihu.com/p/471379451
2. nsight compute

## Bug
1. e-5
2. does not support last BLOCK not full

## CUDA
1. bbuf https://zhuanlan.zhihu.com/p/326999014
2. bainiu https://zhuanlan.zhihu.com/p/478846788
code https://github.com/tpoisonooo/how-to-optimize-gemm/tree/master/cuda
3. mega conv https://zhuanlan.zhihu.com/p/372973726
4. more deeper https://zhuanlan.zhihu.com/p/372973726
5. quat canbe ncnn
6. nchw https://mp.weixin.qq.com/s/1CToXRgyO0F8x0By31dneg
7. cuda ref2 https://zhuanlan.zhihu.com/p/518857175
8. ncnn https://zhuanlan.zhihu.com/p/457443433
9. 基础（优先） https://zhuanlan.zhihu.com/p/393636855
10. 量化 https://zhuanlan.zhihu.com/p/66958390

## some question
CPP
1. 单例模式和工厂模式
2. 指针和引用的区别
3. extern C 的作用
4. cast 
5. union 和 struct
6. std::move
7. 并行的命令和方法
8. shared_ptr weak_ptr uni_ptr
9. cuda 结构
10. shared_mm 
11. gemm 优化思路
12. 如果不遵守3/5法则会如何

1. SM SP
2. 32核心 ACE 矩阵 L-cache 线程排布 
3. 缓存 续位延迟额比例
4. shared memory
5. cuda stream
6. 广播机制
7. 硬件开发做算子
8. 自研 SIMD指令集/汇编指令/DMA/RMA/同步广播ACE/汇编指令集/IO/驱动使用/深度优化SIMD  浮点运算性能 核间通信
十几年超算 指令集 对硬件有了解
