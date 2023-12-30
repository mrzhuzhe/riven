	.intel_syntax noprefix

	.globl _test
	.globl test
	.text

_test:
test:
    mov rax, 's'
	mov [rsi], rax	
    mov rax, 'd' 	
    mov [rdi], rax
    mov rax, 'c' 		
    mov [rcx], rax
    mov rax, 'd' 		
    mov [rdx], rax	
    mov rax, '8'	
    mov [r8], rax
    mov rax, '9'	
    mov [r9], rax
done:
	ret
