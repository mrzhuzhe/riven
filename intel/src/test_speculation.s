	.intel_syntax noprefix

	.globl _test_speculation
	.globl test_speculation
	.text
_test_speculation:
test_speculation:
    #mov dword ptr [rax], 0  # https://stackoverflow.com/questions/2987876/what-does-dword-ptr-mean
    nullify_loop:
    mov dword ptr [eax], 0
    mov edx, dword ptr [edi]
    sub ecx, 4
    cmp dword ptr [ecx+edx], esi
    lea eax, [ecx+edx]
    jne nullify_loop
done:
	ret
