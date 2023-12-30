	.intel_syntax noprefix

	.globl _testf
	.globl testf
	.text
_testf:
testf:
    mov rax, 0x3f000000
#    movd xmm0, rax
#	mov [rsi], rax	
#    mov rax, 2 	
#    mov [rdi], rax
#    mov rax, 3 		
#    mov [rcx], rax
#    mov rax, 4 		
#    mov [rdx], rax	
#    mov rax, 5	
#    mov [r8], rax
#    mov rax, 6	
    mov [r9], rax
done:
	ret
