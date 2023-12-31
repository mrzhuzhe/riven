	.intel_syntax noprefix

	.globl _test_bankconflict
	.globl test_bankconflict
	.text
# rax 51711
# rsi 206844
_test_bankconflict:
test_bankconflict:
    xor rax, rax
	lea r11, [rdi]
	lea r12, [rsi]
	lea r13, [rdx]
loop:
	lea rsi, [rax*4]
	movsxd rsi, esi
	mov edi, [r11+rsi*4]
	add edi, [r12+rsi*4]
	mov r8d, [r11+rsi*4+4]
	add r8d, [r12+rsi*4+4]
	mov r9d, [r11+rsi*4+8]
	add r9d, [r12+rsi*4+8]
	mov r10d, [r11+rsi*4+12]
	add r10d, [r12+rsi*4+12]

	mov [r13+rsi*4], edi
	inc eax
	mov [r13+rsi*4+4], r8d
	mov [r13+rsi*4+8], r9d
	mov [r13+rsi*4+12], r10d
	cmp eax, ecx
	jb loop
done:
	ret
