	.intel_syntax noprefix

	.globl _test
	.globl test
	.text

_test:
test:
    mov rsi, 1 
	mov [rax], rsi	
    mov rsi, 2 	
    mov [rbx], rsi
    mov rsi, 3 		
    mov [rcx], rsi
    mov rsi, 4 		
    mov [rdx], rsi	
    mov rsi, 5	
    mov [r8], rsi	
done:
	ret
