	.intel_syntax noprefix

	.globl _test_mmspill
	.globl test_mmspill
	.text
_test_mmspill:
test_mmspill:
loop:
    mov rdx, [rsp+0x18]
    movdqa xmm0, [rdx]
    movdqa xmm1, [rsp+0x20]
    pcmpeqd xmm1, xmm0
    pmovmskb eax, xmm1
    test eax, eax
    jne done
    movzx rcx, [rbx+0x60]

    add qword ptr [rsp+0x18], 0x10
    add rdi, 0x4
    movzx rdx, di
    sub rcx, 0x4
    add rsi, 0x1d0
    cmp rdx, rcx
    jle loop
done:
	ret
