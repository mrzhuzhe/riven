# Low precision  

1. gemmlowp https://github.com/google/gemmlowp
2. survial https://zhuanlan.zhihu.com/p/66958390
3. caffe memo https://github.com/Yangqing/caffe/wiki/Convolution-in-Caffe:-a-memo
caffe conv addition https://www.zhihu.com/question/28385679/answer/44297845
4. QNNPACK https://engineering.fb.com/2018/10/29/ml-applications/qnnpack/
5. cuda convenet https://code.google.com/archive/p/cuda-convnet/
6. CS267 https://www.bilibili.com/video/BV1nA411V7iZ/?spm_id_from=333.337.search-card.all.click&vd_source=357616f412db6079b853b68278dc03db
7. cs 127 https://zhuanlan.zhihu.com/p/83067033
8. int8 quantize https://zhuanlan.zhihu.com/p/58182172

# 顺序
1. 先处理cpu卷积 了解 winogard img2col qnnpack 的具体实现
2. 在处理cpu 量化 neon 指令集
3. 处理cuda conv-net
4. 处理汇编指令加速
5. cutlass
6. quant-taichi