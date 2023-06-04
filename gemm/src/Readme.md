# Optimize Gemm

## AVX BUgs
1. avx2 is much faster than avx 
2. 8x6 and asm is slower


## Bad Case Check ASM

1. gcc -O2 -Wall -msse3 -c MMult7.c -S -o outputs/asm/MMult7.s
2. gcc -O2 -Wall -msse3 -c MMult7_bad.c -S -o outputs/asm/MMult7_bad.s

## Debug with asm
1. compare with -g
2. objdump -d -S  outputs/MMult22_avx2.o > outputs/asm/MMult22_avx2_debug.S

## References
1. https://github.com/tpoisonooo/how-to-optimize-gemm/
2. blis https://github.com/flame/how-to-optimize-gemm/wiki#step-by-step-optimizations
3. cuda for compare https://zhuanlan.zhihu.com/p/478846788
4. Infer engine https://github.com/zjhellofss/KuiperInfer
5. a more detail [CORE] https://www.mathematik.uni-ulm.de/~lehn/apfel/sghpc/gemm/

## perf
1. perf 工具 https://zhuanlan.zhihu.com/p/471379451 
2. objdump -d ./outputs/MMult7.o >> ./outputs/asm/MMult7_2.S
3. openblas lab https://www.bilibili.com/video/BV1BY411N72y/?spm_id_from=333.788&vd_source=357616f412db6079b853b68278dc03db

## Bug
1. e-14 error https://github.com/mrzhuzhe/riven/issues/3

## CUDA
1. https://zhuanlan.zhihu.com/p/326999014
2. https://zhuanlan.zhihu.com/p/478846788