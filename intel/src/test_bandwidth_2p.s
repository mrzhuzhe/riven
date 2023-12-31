	.intel_syntax noprefix

	.globl _test_bandwidth_2p
	.globl test_bandwidth_2p
	.text
#   https://docs.oracle.com/cd/E19253-01/817-5477/eojde/index.html
_test_bandwidth_2p:
test_bandwidth_2p:
    xor eax, eax
    pxor xmm0, xmm0
    pxor xmm1, xmm1
#    lea rsi, buff  # already in
loop_start:
    addps xmm0, [rsi+4*rax]
    addps xmm1, [rsi+4*rax+16]
    addps xmm1, [rsi+4*rax+32]
    addps xmm0, [rsi+4*rax+48]
    addps xmm0, [rsi+4*rax+64]
    addps xmm1, [rsi+4*rax+80]
    addps xmm0, [rsi+4*rax+96]
    addps xmm1, [rsi+4*rax+112]
    add eax, 32
    cmp eax, edx
    jl loop_start
sum_partials:
    //  two method is both good
    // movaps xmm1, xmm0
    // shufps xmm1, xmm1, 0x93
    // addps xmm0, xmm1
    // shufps xmm1, xmm1, 0x93
    // addps xmm0, xmm1
    // shufps xmm1, xmm1, 0x93
    // addps xmm0, xmm1
    //  two method is both good
    addps xmm0, xmm1
   // haddps xmm0, xmm0
    // haddps xmm0, xmm0
    movaps [rdi], xmm0
done:
	ret
