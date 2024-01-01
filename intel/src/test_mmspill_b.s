	.intel_syntax noprefix

	.globl _test_mmspill_b
	.globl test_mmspill_b
	.text
_test_mmspill_b:
test_mmspill_b:
    movq xmm4, [rsp+0x18]
    mov rcx, 0x10
    movq xmm5, rcx
loop:
    mov rdx, xmm4
    movdqa xmm0, [rdx]
    movdqa xmm1, [rsp+0x20]
    pcmpeqd xmm1, xmm0
    pmovmskb eax, xmm1
    test eax, eax
    jne done
    movzx rcx, [rbx+0x60]

    padd xmm4, xmm5
    add rdi, 0x4
    movzx rdx, di
    sub rcx, 0x4
    add rsi, 0x1d0
    cmp rdx, rcx
    jle loop
done:
	ret
