	.intel_syntax noprefix

	.globl _test_bandwidth
	.globl test_bandwidth
	.text
#   https://docs.oracle.com/cd/E19253-01/817-5477/eojde/index.html
_test_bandwidth:
test_bandwidth:
    xor eax, eax
    pxor xmm0, xmm0
#    lea rsi, buff  # already in
loop_start:
    addps xmm0, [rsi+4*rax]
    addps xmm0, [rsi+4*rax+16]
    addps xmm0, [rsi+4*rax+32]
    addps xmm0, [rsi+4*rax+48]
    addps xmm0, [rsi+4*rax+64]
    addps xmm0, [rsi+4*rax+80]
    addps xmm0, [rsi+4*rax+96]
    addps xmm0, [rsi+4*rax+112]
    add eax, 32
    cmp eax, edx
    jl loop_start
sum_partials:
    //movups xmm1, xmm0
    //shufps xmm1, xmm1, 0x93
    //addps xmm0, xmm1
    //shufps xmm1, xmm1, 0x93
    //addps xmm0, xmm1
    //shufps xmm1, xmm1, 0x93
    //addps xmm0, xmm1
    movups [rdi], xmm0
done:
	ret
