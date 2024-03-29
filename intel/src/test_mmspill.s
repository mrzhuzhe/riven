	.intel_syntax noprefix

	.globl _test_mmspill
	.globl test_mmspill
	.text
_test_mmspill:
test_mmspill:
loop:
    mov rdx, [rsp+0x18] # 24
    movdqa xmm0, [rdx]
    movdqa xmm1, [rsp+0x20] # 32
    pcmpeqd xmm1, xmm0  # https://mudongliang.github.io/x86/html/file_module_x86_id_234.html
    pmovmskb eax, xmm1 # https://mudongliang.github.io/x86/html/file_module_x86_id_243.html
    test eax, eax
    jne done
    movzx rcx, [rbx+0x60]    # 96

    add qword ptr [rsp+0x18], 0x10 #16
    add rdi, 0x4 #4
    movzx rdx, di   #4
    sub rcx, 0x4
    add rsi, 0x1d0 #464
    cmp rdx, rcx
    jle loop
done:
	ret
