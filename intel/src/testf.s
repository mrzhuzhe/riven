	.intel_syntax noprefix

	.globl _testf
	.globl testf
	.text
_testf:
testf:
    mov rax, 0x3f000000
#    movd xmm0, rax
	movd [rsi], xmm0	
#    mov rax, 2 	
    movd [rdi], xmm1
#    mov rax, 3 		
    movd [rcx], xmm2
#    mov rax, 4 		
    movd [rdx], xmm3	
#    mov rax, 5	
    movd [r8], xmm4
#    mov rax, 6	
    movd [r9], xmm5
done:
	ret
