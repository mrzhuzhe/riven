# Optimize Gemm
## Bad Case Check ASM

1. gcc -O2 -Wall -msse3 -c MMult7.c -S -o outputs/asm/MMult7.s
2. gcc -O2 -Wall -msse3 -c MMult7_bad.c -S -o outputs/asm/MMult7_bad.s

## References
1. https://github.com/tpoisonooo/how-to-optimize-gemm/
2. blis https://github.com/flame/how-to-optimize-gemm/wiki#step-by-step-optimizations
3. cuda for compare https://zhuanlan.zhihu.com/p/478846788
4. Infer engine https://github.com/zjhellofss/KuiperInfer

## perf
1. perf 工具 https://zhuanlan.zhihu.com/p/471379451 


## CUDA
1. https://zhuanlan.zhihu.com/p/326999014
2. https://zhuanlan.zhihu.com/p/478846788