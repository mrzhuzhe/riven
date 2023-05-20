
outputs/MMult22_avx2.o:     file format elf64-x86-64


Disassembly of section .text:

0000000000000000 <AddDot8x4>:
#define B(i,j) b[ (j)*ldb + (i) ]
#define C(i,j) c[ (j)*ldc + (i) ]

/* Routine for computing C = A * B + C */

void AddDot8x4(int k, double *a, int lda, double *b, int ldb, double *c, int ldc){
   0:	f3 0f 1e fa          	endbr64 
   4:	55                   	push   %rbp
   5:	48 89 e5             	mov    %rsp,%rbp
   8:	48 83 e4 e0          	and    $0xffffffffffffffe0,%rsp
   c:	48 81 ec e0 07 00 00 	sub    $0x7e0,%rsp
  13:	89 7c 24 3c          	mov    %edi,0x3c(%rsp)
  17:	48 89 74 24 30       	mov    %rsi,0x30(%rsp)
  1c:	89 54 24 38          	mov    %edx,0x38(%rsp)
  20:	48 89 4c 24 28       	mov    %rcx,0x28(%rsp)
  25:	44 89 44 24 24       	mov    %r8d,0x24(%rsp)
  2a:	4c 89 4c 24 18       	mov    %r9,0x18(%rsp)
  2f:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  36:	00 00 
  38:	48 89 84 24 d8 07 00 	mov    %rax,0x7d8(%rsp)
  3f:	00 
  40:	31 c0                	xor    %eax,%eax
}

extern __inline __m256d __attribute__((__gnu_inline__, __always_inline__, __artificial__))
_mm256_setzero_pd (void)
{
  return __extension__ (__m256d){ 0.0, 0.0, 0.0, 0.0 };
  42:	c5 f9 57 c0          	vxorpd %xmm0,%xmm0,%xmm0
    register v4df_t va0123, va4567;
    register v4df_t vb0p, vb1p, vb2p, vb3p; 

    
   
    vc00102030.v = _mm256_setzero_pd(); 
  46:	c5 fd 29 84 24 00 06 	vmovapd %ymm0,0x600(%rsp)
  4d:	00 00 
  4f:	c5 f9 57 c0          	vxorpd %xmm0,%xmm0,%xmm0
    vc01112131.v = _mm256_setzero_pd(); 
  53:	c5 fd 29 84 24 20 06 	vmovapd %ymm0,0x620(%rsp)
  5a:	00 00 
  5c:	c5 f9 57 c0          	vxorpd %xmm0,%xmm0,%xmm0
    vc02122232.v = _mm256_setzero_pd();
  60:	c5 fd 29 84 24 40 06 	vmovapd %ymm0,0x640(%rsp)
  67:	00 00 
  69:	c5 f9 57 c0          	vxorpd %xmm0,%xmm0,%xmm0
    vc03132333.v = _mm256_setzero_pd();
  6d:	c5 fd 29 84 24 60 06 	vmovapd %ymm0,0x660(%rsp)
  74:	00 00 
  76:	c5 f9 57 c0          	vxorpd %xmm0,%xmm0,%xmm0
    vc40506070.v = _mm256_setzero_pd();
  7a:	c5 fd 29 84 24 80 06 	vmovapd %ymm0,0x680(%rsp)
  81:	00 00 
  83:	c5 f9 57 c0          	vxorpd %xmm0,%xmm0,%xmm0
    vc41516171.v = _mm256_setzero_pd();
  87:	c5 fd 29 84 24 a0 06 	vmovapd %ymm0,0x6a0(%rsp)
  8e:	00 00 
  90:	c5 f9 57 c0          	vxorpd %xmm0,%xmm0,%xmm0
    vc42526272.v = _mm256_setzero_pd();
  94:	c5 fd 29 84 24 c0 06 	vmovapd %ymm0,0x6c0(%rsp)
  9b:	00 00 
  9d:	c5 f9 57 c0          	vxorpd %xmm0,%xmm0,%xmm0
    vc43536373.v = _mm256_setzero_pd();
  a1:	c5 fd 29 84 24 e0 06 	vmovapd %ymm0,0x6e0(%rsp)
  a8:	00 00 

    for ( p=0; p<k; p++ ){
  aa:	c7 44 24 4c 00 00 00 	movl   $0x0,0x4c(%rsp)
  b1:	00 
  b2:	e9 31 03 00 00       	jmpq   3e8 <AddDot8x4+0x3e8>
  b7:	48 8b 44 24 30       	mov    0x30(%rsp),%rax
  bc:	48 89 44 24 78       	mov    %rax,0x78(%rsp)
  return *(__m256d *)__P;
  c1:	48 8b 44 24 78       	mov    0x78(%rsp),%rax
  c6:	c5 fd 28 00          	vmovapd (%rax),%ymm0

      //  数据需要对其
      //  https://stackoverflow.com/questions/33373318/avx-segmentation-fault-on-linux
      // unalign will suddenly fail but align will be a little slower
      va0123.v = _mm256_load_pd((double *) a);      
  ca:	c5 fd 29 84 24 00 07 	vmovapd %ymm0,0x700(%rsp)
  d1:	00 00 
      va4567.v = _mm256_load_pd((double *) a+4);
  d3:	48 8b 44 24 30       	mov    0x30(%rsp),%rax
  d8:	48 83 c0 20          	add    $0x20,%rax
  dc:	48 89 44 24 70       	mov    %rax,0x70(%rsp)
  e1:	48 8b 44 24 70       	mov    0x70(%rsp),%rax
  e6:	c5 fd 28 00          	vmovapd (%rax),%ymm0
  ea:	c5 fd 29 84 24 20 07 	vmovapd %ymm0,0x720(%rsp)
  f1:	00 00 

      //va0123.v = _mm256_loadu_pd((double *) a);      
      //va4567.v = _mm256_loadu_pd((double *) a+4);
      
      a += 8;
  f3:	48 83 44 24 30 40    	addq   $0x40,0x30(%rsp)
  f9:	48 8b 44 24 28       	mov    0x28(%rsp),%rax
  fe:	48 89 44 24 68       	mov    %rax,0x68(%rsp)
  return (__m256d) __builtin_ia32_vbroadcastsd256 (__X);
 103:	48 8b 44 24 68       	mov    0x68(%rsp),%rax
 108:	c4 e2 7d 19 00       	vbroadcastsd (%rax),%ymm0
 10d:	90                   	nop

      vb0p.v = _mm256_broadcast_sd((double *) b);
 10e:	c5 fd 29 84 24 40 07 	vmovapd %ymm0,0x740(%rsp)
 115:	00 00 
      vb1p.v = _mm256_broadcast_sd((double *) (b + 1));
 117:	48 8b 44 24 28       	mov    0x28(%rsp),%rax
 11c:	48 83 c0 08          	add    $0x8,%rax
 120:	48 89 44 24 60       	mov    %rax,0x60(%rsp)
 125:	48 8b 44 24 60       	mov    0x60(%rsp),%rax
 12a:	c4 e2 7d 19 00       	vbroadcastsd (%rax),%ymm0
 12f:	90                   	nop
 130:	c5 fd 29 84 24 60 07 	vmovapd %ymm0,0x760(%rsp)
 137:	00 00 
      vb2p.v = _mm256_broadcast_sd((double *) (b + 2));
 139:	48 8b 44 24 28       	mov    0x28(%rsp),%rax
 13e:	48 83 c0 10          	add    $0x10,%rax
 142:	48 89 44 24 58       	mov    %rax,0x58(%rsp)
 147:	48 8b 44 24 58       	mov    0x58(%rsp),%rax
 14c:	c4 e2 7d 19 00       	vbroadcastsd (%rax),%ymm0
 151:	90                   	nop
 152:	c5 fd 29 84 24 80 07 	vmovapd %ymm0,0x780(%rsp)
 159:	00 00 
      vb3p.v = _mm256_broadcast_sd((double *) (b + 3));
 15b:	48 8b 44 24 28       	mov    0x28(%rsp),%rax
 160:	48 83 c0 18          	add    $0x18,%rax
 164:	48 89 44 24 50       	mov    %rax,0x50(%rsp)
 169:	48 8b 44 24 50       	mov    0x50(%rsp),%rax
 16e:	c4 e2 7d 19 00       	vbroadcastsd (%rax),%ymm0
 173:	90                   	nop
 174:	c5 fd 29 84 24 a0 07 	vmovapd %ymm0,0x7a0(%rsp)
 17b:	00 00 

      //vc00102030.v += va0123.v * vb0p.v;                    
      vc00102030.v += _mm256_mul_pd(va0123.v, vb0p.v);     
 17d:	c5 fd 28 84 24 40 07 	vmovapd 0x740(%rsp),%ymm0
 184:	00 00 
 186:	c5 fd 28 8c 24 00 07 	vmovapd 0x700(%rsp),%ymm1
 18d:	00 00 
 18f:	c5 fd 29 8c 24 c0 02 	vmovapd %ymm1,0x2c0(%rsp)
 196:	00 00 
 198:	c5 fd 29 84 24 e0 02 	vmovapd %ymm0,0x2e0(%rsp)
 19f:	00 00 
  return (__m256d) ((__v4df)__A * (__v4df)__B);
 1a1:	c5 fd 28 84 24 c0 02 	vmovapd 0x2c0(%rsp),%ymm0
 1a8:	00 00 
 1aa:	c5 fd 59 8c 24 e0 02 	vmulpd 0x2e0(%rsp),%ymm0,%ymm1
 1b1:	00 00 
 1b3:	c5 fd 28 84 24 00 06 	vmovapd 0x600(%rsp),%ymm0
 1ba:	00 00 
 1bc:	c5 f5 58 c0          	vaddpd %ymm0,%ymm1,%ymm0
 1c0:	c5 fd 29 84 24 00 06 	vmovapd %ymm0,0x600(%rsp)
 1c7:	00 00 
      vc01112131.v += _mm256_mul_pd(va0123.v, vb1p.v);          
 1c9:	c5 fd 28 84 24 60 07 	vmovapd 0x760(%rsp),%ymm0
 1d0:	00 00 
 1d2:	c5 fd 28 8c 24 00 07 	vmovapd 0x700(%rsp),%ymm1
 1d9:	00 00 
 1db:	c5 fd 29 8c 24 80 02 	vmovapd %ymm1,0x280(%rsp)
 1e2:	00 00 
 1e4:	c5 fd 29 84 24 a0 02 	vmovapd %ymm0,0x2a0(%rsp)
 1eb:	00 00 
 1ed:	c5 fd 28 84 24 80 02 	vmovapd 0x280(%rsp),%ymm0
 1f4:	00 00 
 1f6:	c5 fd 59 8c 24 a0 02 	vmulpd 0x2a0(%rsp),%ymm0,%ymm1
 1fd:	00 00 
 1ff:	c5 fd 28 84 24 20 06 	vmovapd 0x620(%rsp),%ymm0
 206:	00 00 
 208:	c5 f5 58 c0          	vaddpd %ymm0,%ymm1,%ymm0
 20c:	c5 fd 29 84 24 20 06 	vmovapd %ymm0,0x620(%rsp)
 213:	00 00 
      vc02122232.v += _mm256_mul_pd(va0123.v, vb2p.v); 
 215:	c5 fd 28 84 24 80 07 	vmovapd 0x780(%rsp),%ymm0
 21c:	00 00 
 21e:	c5 fd 28 8c 24 00 07 	vmovapd 0x700(%rsp),%ymm1
 225:	00 00 
 227:	c5 fd 29 8c 24 40 02 	vmovapd %ymm1,0x240(%rsp)
 22e:	00 00 
 230:	c5 fd 29 84 24 60 02 	vmovapd %ymm0,0x260(%rsp)
 237:	00 00 
 239:	c5 fd 28 84 24 40 02 	vmovapd 0x240(%rsp),%ymm0
 240:	00 00 
 242:	c5 fd 59 8c 24 60 02 	vmulpd 0x260(%rsp),%ymm0,%ymm1
 249:	00 00 
 24b:	c5 fd 28 84 24 40 06 	vmovapd 0x640(%rsp),%ymm0
 252:	00 00 
 254:	c5 f5 58 c0          	vaddpd %ymm0,%ymm1,%ymm0
 258:	c5 fd 29 84 24 40 06 	vmovapd %ymm0,0x640(%rsp)
 25f:	00 00 
      vc03132333.v += _mm256_mul_pd(va0123.v, vb3p.v);   
 261:	c5 fd 28 84 24 a0 07 	vmovapd 0x7a0(%rsp),%ymm0
 268:	00 00 
 26a:	c5 fd 28 8c 24 00 07 	vmovapd 0x700(%rsp),%ymm1
 271:	00 00 
 273:	c5 fd 29 8c 24 00 02 	vmovapd %ymm1,0x200(%rsp)
 27a:	00 00 
 27c:	c5 fd 29 84 24 20 02 	vmovapd %ymm0,0x220(%rsp)
 283:	00 00 
 285:	c5 fd 28 84 24 00 02 	vmovapd 0x200(%rsp),%ymm0
 28c:	00 00 
 28e:	c5 fd 59 8c 24 20 02 	vmulpd 0x220(%rsp),%ymm0,%ymm1
 295:	00 00 
 297:	c5 fd 28 84 24 60 06 	vmovapd 0x660(%rsp),%ymm0
 29e:	00 00 
 2a0:	c5 f5 58 c0          	vaddpd %ymm0,%ymm1,%ymm0
 2a4:	c5 fd 29 84 24 60 06 	vmovapd %ymm0,0x660(%rsp)
 2ab:	00 00 
      //  _mm256_mul_pd _mm256_add_pd
      vc40506070.v += _mm256_mul_pd(va4567.v, vb0p.v);  
 2ad:	c5 fd 28 84 24 40 07 	vmovapd 0x740(%rsp),%ymm0
 2b4:	00 00 
 2b6:	c5 fd 28 8c 24 20 07 	vmovapd 0x720(%rsp),%ymm1
 2bd:	00 00 
 2bf:	c5 fd 29 8c 24 c0 01 	vmovapd %ymm1,0x1c0(%rsp)
 2c6:	00 00 
 2c8:	c5 fd 29 84 24 e0 01 	vmovapd %ymm0,0x1e0(%rsp)
 2cf:	00 00 
 2d1:	c5 fd 28 84 24 c0 01 	vmovapd 0x1c0(%rsp),%ymm0
 2d8:	00 00 
 2da:	c5 fd 59 8c 24 e0 01 	vmulpd 0x1e0(%rsp),%ymm0,%ymm1
 2e1:	00 00 
 2e3:	c5 fd 28 84 24 80 06 	vmovapd 0x680(%rsp),%ymm0
 2ea:	00 00 
 2ec:	c5 f5 58 c0          	vaddpd %ymm0,%ymm1,%ymm0
 2f0:	c5 fd 29 84 24 80 06 	vmovapd %ymm0,0x680(%rsp)
 2f7:	00 00 
      vc41516171.v += _mm256_mul_pd(va4567.v, vb1p.v);  
 2f9:	c5 fd 28 84 24 60 07 	vmovapd 0x760(%rsp),%ymm0
 300:	00 00 
 302:	c5 fd 28 8c 24 20 07 	vmovapd 0x720(%rsp),%ymm1
 309:	00 00 
 30b:	c5 fd 29 8c 24 80 01 	vmovapd %ymm1,0x180(%rsp)
 312:	00 00 
 314:	c5 fd 29 84 24 a0 01 	vmovapd %ymm0,0x1a0(%rsp)
 31b:	00 00 
 31d:	c5 fd 28 84 24 80 01 	vmovapd 0x180(%rsp),%ymm0
 324:	00 00 
 326:	c5 fd 59 8c 24 a0 01 	vmulpd 0x1a0(%rsp),%ymm0,%ymm1
 32d:	00 00 
 32f:	c5 fd 28 84 24 a0 06 	vmovapd 0x6a0(%rsp),%ymm0
 336:	00 00 
 338:	c5 f5 58 c0          	vaddpd %ymm0,%ymm1,%ymm0
 33c:	c5 fd 29 84 24 a0 06 	vmovapd %ymm0,0x6a0(%rsp)
 343:	00 00 
      vc42526272.v += _mm256_mul_pd(va4567.v, vb2p.v);  
 345:	c5 fd 28 84 24 80 07 	vmovapd 0x780(%rsp),%ymm0
 34c:	00 00 
 34e:	c5 fd 28 8c 24 20 07 	vmovapd 0x720(%rsp),%ymm1
 355:	00 00 
 357:	c5 fd 29 8c 24 40 01 	vmovapd %ymm1,0x140(%rsp)
 35e:	00 00 
 360:	c5 fd 29 84 24 60 01 	vmovapd %ymm0,0x160(%rsp)
 367:	00 00 
 369:	c5 fd 28 84 24 40 01 	vmovapd 0x140(%rsp),%ymm0
 370:	00 00 
 372:	c5 fd 59 8c 24 60 01 	vmulpd 0x160(%rsp),%ymm0,%ymm1
 379:	00 00 
 37b:	c5 fd 28 84 24 c0 06 	vmovapd 0x6c0(%rsp),%ymm0
 382:	00 00 
 384:	c5 f5 58 c0          	vaddpd %ymm0,%ymm1,%ymm0
 388:	c5 fd 29 84 24 c0 06 	vmovapd %ymm0,0x6c0(%rsp)
 38f:	00 00 
      vc43536373.v += _mm256_mul_pd(va4567.v, vb3p.v);     
 391:	c5 fd 28 84 24 a0 07 	vmovapd 0x7a0(%rsp),%ymm0
 398:	00 00 
 39a:	c5 fd 28 8c 24 20 07 	vmovapd 0x720(%rsp),%ymm1
 3a1:	00 00 
 3a3:	c5 fd 29 8c 24 00 01 	vmovapd %ymm1,0x100(%rsp)
 3aa:	00 00 
 3ac:	c5 fd 29 84 24 20 01 	vmovapd %ymm0,0x120(%rsp)
 3b3:	00 00 
 3b5:	c5 fd 28 84 24 00 01 	vmovapd 0x100(%rsp),%ymm0
 3bc:	00 00 
 3be:	c5 fd 59 8c 24 20 01 	vmulpd 0x120(%rsp),%ymm0,%ymm1
 3c5:	00 00 
 3c7:	c5 fd 28 84 24 e0 06 	vmovapd 0x6e0(%rsp),%ymm0
 3ce:	00 00 
 3d0:	c5 f5 58 c0          	vaddpd %ymm0,%ymm1,%ymm0
 3d4:	c5 fd 29 84 24 e0 06 	vmovapd %ymm0,0x6e0(%rsp)
 3db:	00 00 

      b += 4;
 3dd:	48 83 44 24 28 20    	addq   $0x20,0x28(%rsp)
    for ( p=0; p<k; p++ ){
 3e3:	83 44 24 4c 01       	addl   $0x1,0x4c(%rsp)
 3e8:	8b 44 24 4c          	mov    0x4c(%rsp),%eax
 3ec:	3b 44 24 3c          	cmp    0x3c(%rsp),%eax
 3f0:	0f 8c c1 fc ff ff    	jl     b7 <AddDot8x4+0xb7>

    }

    //
    _mm256_store_pd(c, _mm256_add_pd(_mm256_load_pd(c), vc00102030.v));
 3f6:	c5 fd 28 84 24 00 06 	vmovapd 0x600(%rsp),%ymm0
 3fd:	00 00 
 3ff:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
 404:	48 89 84 24 f8 00 00 	mov    %rax,0xf8(%rsp)
 40b:	00 
  return *(__m256d *)__P;
 40c:	48 8b 84 24 f8 00 00 	mov    0xf8(%rsp),%rax
 413:	00 
 414:	c5 fd 28 08          	vmovapd (%rax),%ymm1
 418:	c5 fd 29 8c 24 c0 05 	vmovapd %ymm1,0x5c0(%rsp)
 41f:	00 00 
 421:	c5 fd 29 84 24 e0 05 	vmovapd %ymm0,0x5e0(%rsp)
 428:	00 00 
  return (__m256d) ((__v4df)__A + (__v4df)__B);
 42a:	c5 fd 28 84 24 c0 05 	vmovapd 0x5c0(%rsp),%ymm0
 431:	00 00 
 433:	c5 fd 58 84 24 e0 05 	vaddpd 0x5e0(%rsp),%ymm0,%ymm0
 43a:	00 00 
 43c:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
 441:	48 89 84 24 f0 00 00 	mov    %rax,0xf0(%rsp)
 448:	00 
 449:	c5 fd 29 84 24 a0 05 	vmovapd %ymm0,0x5a0(%rsp)
 450:	00 00 
  *(__m256d *)__P = __A;
 452:	48 8b 84 24 f0 00 00 	mov    0xf0(%rsp),%rax
 459:	00 
 45a:	c5 fd 28 84 24 a0 05 	vmovapd 0x5a0(%rsp),%ymm0
 461:	00 00 
 463:	c5 fd 29 00          	vmovapd %ymm0,(%rax)
}
 467:	90                   	nop
    _mm256_store_pd((c + ldc), _mm256_add_pd(_mm256_load_pd((c + ldc)), vc01112131.v));
 468:	c5 fd 28 84 24 20 06 	vmovapd 0x620(%rsp),%ymm0
 46f:	00 00 
 471:	8b 45 10             	mov    0x10(%rbp),%eax
 474:	48 98                	cltq   
 476:	48 8d 14 c5 00 00 00 	lea    0x0(,%rax,8),%rdx
 47d:	00 
 47e:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
 483:	48 01 d0             	add    %rdx,%rax
 486:	48 89 84 24 e8 00 00 	mov    %rax,0xe8(%rsp)
 48d:	00 
  return *(__m256d *)__P;
 48e:	48 8b 84 24 e8 00 00 	mov    0xe8(%rsp),%rax
 495:	00 
 496:	c5 fd 28 08          	vmovapd (%rax),%ymm1
 49a:	c5 fd 29 8c 24 60 05 	vmovapd %ymm1,0x560(%rsp)
 4a1:	00 00 
 4a3:	c5 fd 29 84 24 80 05 	vmovapd %ymm0,0x580(%rsp)
 4aa:	00 00 
  return (__m256d) ((__v4df)__A + (__v4df)__B);
 4ac:	c5 fd 28 84 24 60 05 	vmovapd 0x560(%rsp),%ymm0
 4b3:	00 00 
 4b5:	c5 fd 58 84 24 80 05 	vaddpd 0x580(%rsp),%ymm0,%ymm0
 4bc:	00 00 
 4be:	8b 45 10             	mov    0x10(%rbp),%eax
 4c1:	48 98                	cltq   
 4c3:	48 8d 14 c5 00 00 00 	lea    0x0(,%rax,8),%rdx
 4ca:	00 
 4cb:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
 4d0:	48 01 d0             	add    %rdx,%rax
 4d3:	48 89 84 24 e0 00 00 	mov    %rax,0xe0(%rsp)
 4da:	00 
 4db:	c5 fd 29 84 24 40 05 	vmovapd %ymm0,0x540(%rsp)
 4e2:	00 00 
  *(__m256d *)__P = __A;
 4e4:	48 8b 84 24 e0 00 00 	mov    0xe0(%rsp),%rax
 4eb:	00 
 4ec:	c5 fd 28 84 24 40 05 	vmovapd 0x540(%rsp),%ymm0
 4f3:	00 00 
 4f5:	c5 fd 29 00          	vmovapd %ymm0,(%rax)
}
 4f9:	90                   	nop
    _mm256_store_pd((c + ldc * 2), _mm256_add_pd(_mm256_load_pd((c + ldc * 2)), vc02122232.v));
 4fa:	c5 fd 28 84 24 40 06 	vmovapd 0x640(%rsp),%ymm0
 501:	00 00 
 503:	8b 45 10             	mov    0x10(%rbp),%eax
 506:	01 c0                	add    %eax,%eax
 508:	48 98                	cltq   
 50a:	48 8d 14 c5 00 00 00 	lea    0x0(,%rax,8),%rdx
 511:	00 
 512:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
 517:	48 01 d0             	add    %rdx,%rax
 51a:	48 89 84 24 d8 00 00 	mov    %rax,0xd8(%rsp)
 521:	00 
  return *(__m256d *)__P;
 522:	48 8b 84 24 d8 00 00 	mov    0xd8(%rsp),%rax
 529:	00 
 52a:	c5 fd 28 08          	vmovapd (%rax),%ymm1
 52e:	c5 fd 29 8c 24 00 05 	vmovapd %ymm1,0x500(%rsp)
 535:	00 00 
 537:	c5 fd 29 84 24 20 05 	vmovapd %ymm0,0x520(%rsp)
 53e:	00 00 
  return (__m256d) ((__v4df)__A + (__v4df)__B);
 540:	c5 fd 28 84 24 00 05 	vmovapd 0x500(%rsp),%ymm0
 547:	00 00 
 549:	c5 fd 58 84 24 20 05 	vaddpd 0x520(%rsp),%ymm0,%ymm0
 550:	00 00 
 552:	8b 45 10             	mov    0x10(%rbp),%eax
 555:	01 c0                	add    %eax,%eax
 557:	48 98                	cltq   
 559:	48 8d 14 c5 00 00 00 	lea    0x0(,%rax,8),%rdx
 560:	00 
 561:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
 566:	48 01 d0             	add    %rdx,%rax
 569:	48 89 84 24 d0 00 00 	mov    %rax,0xd0(%rsp)
 570:	00 
 571:	c5 fd 29 84 24 e0 04 	vmovapd %ymm0,0x4e0(%rsp)
 578:	00 00 
  *(__m256d *)__P = __A;
 57a:	48 8b 84 24 d0 00 00 	mov    0xd0(%rsp),%rax
 581:	00 
 582:	c5 fd 28 84 24 e0 04 	vmovapd 0x4e0(%rsp),%ymm0
 589:	00 00 
 58b:	c5 fd 29 00          	vmovapd %ymm0,(%rax)
}
 58f:	90                   	nop
    _mm256_store_pd((c + ldc * 3), _mm256_add_pd(_mm256_load_pd((c + ldc * 3)), vc03132333.v));
 590:	c5 fd 28 84 24 60 06 	vmovapd 0x660(%rsp),%ymm0
 597:	00 00 
 599:	8b 55 10             	mov    0x10(%rbp),%edx
 59c:	89 d0                	mov    %edx,%eax
 59e:	01 c0                	add    %eax,%eax
 5a0:	01 d0                	add    %edx,%eax
 5a2:	48 98                	cltq   
 5a4:	48 8d 14 c5 00 00 00 	lea    0x0(,%rax,8),%rdx
 5ab:	00 
 5ac:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
 5b1:	48 01 d0             	add    %rdx,%rax
 5b4:	48 89 84 24 c8 00 00 	mov    %rax,0xc8(%rsp)
 5bb:	00 
  return *(__m256d *)__P;
 5bc:	48 8b 84 24 c8 00 00 	mov    0xc8(%rsp),%rax
 5c3:	00 
 5c4:	c5 fd 28 08          	vmovapd (%rax),%ymm1
 5c8:	c5 fd 29 8c 24 a0 04 	vmovapd %ymm1,0x4a0(%rsp)
 5cf:	00 00 
 5d1:	c5 fd 29 84 24 c0 04 	vmovapd %ymm0,0x4c0(%rsp)
 5d8:	00 00 
  return (__m256d) ((__v4df)__A + (__v4df)__B);
 5da:	c5 fd 28 84 24 a0 04 	vmovapd 0x4a0(%rsp),%ymm0
 5e1:	00 00 
 5e3:	c5 fd 58 84 24 c0 04 	vaddpd 0x4c0(%rsp),%ymm0,%ymm0
 5ea:	00 00 
 5ec:	8b 55 10             	mov    0x10(%rbp),%edx
 5ef:	89 d0                	mov    %edx,%eax
 5f1:	01 c0                	add    %eax,%eax
 5f3:	01 d0                	add    %edx,%eax
 5f5:	48 98                	cltq   
 5f7:	48 8d 14 c5 00 00 00 	lea    0x0(,%rax,8),%rdx
 5fe:	00 
 5ff:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
 604:	48 01 d0             	add    %rdx,%rax
 607:	48 89 84 24 c0 00 00 	mov    %rax,0xc0(%rsp)
 60e:	00 
 60f:	c5 fd 29 84 24 80 04 	vmovapd %ymm0,0x480(%rsp)
 616:	00 00 
  *(__m256d *)__P = __A;
 618:	48 8b 84 24 c0 00 00 	mov    0xc0(%rsp),%rax
 61f:	00 
 620:	c5 fd 28 84 24 80 04 	vmovapd 0x480(%rsp),%ymm0
 627:	00 00 
 629:	c5 fd 29 00          	vmovapd %ymm0,(%rax)
}
 62d:	90                   	nop
    

    //
    _mm256_store_pd((c + 4), _mm256_add_pd(_mm256_load_pd((c + 4)), vc40506070.v));
 62e:	c5 fd 28 84 24 80 06 	vmovapd 0x680(%rsp),%ymm0
 635:	00 00 
 637:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
 63c:	48 83 c0 20          	add    $0x20,%rax
 640:	48 89 84 24 b8 00 00 	mov    %rax,0xb8(%rsp)
 647:	00 
  return *(__m256d *)__P;
 648:	48 8b 84 24 b8 00 00 	mov    0xb8(%rsp),%rax
 64f:	00 
 650:	c5 fd 28 08          	vmovapd (%rax),%ymm1
 654:	c5 fd 29 8c 24 40 04 	vmovapd %ymm1,0x440(%rsp)
 65b:	00 00 
 65d:	c5 fd 29 84 24 60 04 	vmovapd %ymm0,0x460(%rsp)
 664:	00 00 
  return (__m256d) ((__v4df)__A + (__v4df)__B);
 666:	c5 fd 28 84 24 40 04 	vmovapd 0x440(%rsp),%ymm0
 66d:	00 00 
 66f:	c5 fd 58 84 24 60 04 	vaddpd 0x460(%rsp),%ymm0,%ymm0
 676:	00 00 
 678:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
 67d:	48 83 c0 20          	add    $0x20,%rax
 681:	48 89 84 24 b0 00 00 	mov    %rax,0xb0(%rsp)
 688:	00 
 689:	c5 fd 29 84 24 20 04 	vmovapd %ymm0,0x420(%rsp)
 690:	00 00 
  *(__m256d *)__P = __A;
 692:	48 8b 84 24 b0 00 00 	mov    0xb0(%rsp),%rax
 699:	00 
 69a:	c5 fd 28 84 24 20 04 	vmovapd 0x420(%rsp),%ymm0
 6a1:	00 00 
 6a3:	c5 fd 29 00          	vmovapd %ymm0,(%rax)
}
 6a7:	90                   	nop
    _mm256_store_pd((c + ldc + 4), _mm256_add_pd(_mm256_load_pd((c + ldc + 4)), vc41516171.v));
 6a8:	c5 fd 28 84 24 a0 06 	vmovapd 0x6a0(%rsp),%ymm0
 6af:	00 00 
 6b1:	8b 45 10             	mov    0x10(%rbp),%eax
 6b4:	48 98                	cltq   
 6b6:	48 83 c0 04          	add    $0x4,%rax
 6ba:	48 8d 14 c5 00 00 00 	lea    0x0(,%rax,8),%rdx
 6c1:	00 
 6c2:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
 6c7:	48 01 d0             	add    %rdx,%rax
 6ca:	48 89 84 24 a8 00 00 	mov    %rax,0xa8(%rsp)
 6d1:	00 
  return *(__m256d *)__P;
 6d2:	48 8b 84 24 a8 00 00 	mov    0xa8(%rsp),%rax
 6d9:	00 
 6da:	c5 fd 28 08          	vmovapd (%rax),%ymm1
 6de:	c5 fd 29 8c 24 e0 03 	vmovapd %ymm1,0x3e0(%rsp)
 6e5:	00 00 
 6e7:	c5 fd 29 84 24 00 04 	vmovapd %ymm0,0x400(%rsp)
 6ee:	00 00 
  return (__m256d) ((__v4df)__A + (__v4df)__B);
 6f0:	c5 fd 28 84 24 e0 03 	vmovapd 0x3e0(%rsp),%ymm0
 6f7:	00 00 
 6f9:	c5 fd 58 84 24 00 04 	vaddpd 0x400(%rsp),%ymm0,%ymm0
 700:	00 00 
 702:	8b 45 10             	mov    0x10(%rbp),%eax
 705:	48 98                	cltq   
 707:	48 83 c0 04          	add    $0x4,%rax
 70b:	48 8d 14 c5 00 00 00 	lea    0x0(,%rax,8),%rdx
 712:	00 
 713:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
 718:	48 01 d0             	add    %rdx,%rax
 71b:	48 89 84 24 a0 00 00 	mov    %rax,0xa0(%rsp)
 722:	00 
 723:	c5 fd 29 84 24 c0 03 	vmovapd %ymm0,0x3c0(%rsp)
 72a:	00 00 
  *(__m256d *)__P = __A;
 72c:	48 8b 84 24 a0 00 00 	mov    0xa0(%rsp),%rax
 733:	00 
 734:	c5 fd 28 84 24 c0 03 	vmovapd 0x3c0(%rsp),%ymm0
 73b:	00 00 
 73d:	c5 fd 29 00          	vmovapd %ymm0,(%rax)
}
 741:	90                   	nop
    _mm256_store_pd((c + ldc * 2 + 4), _mm256_add_pd(_mm256_load_pd((c + ldc * 2 + 4)), vc42526272.v));
 742:	c5 fd 28 84 24 c0 06 	vmovapd 0x6c0(%rsp),%ymm0
 749:	00 00 
 74b:	8b 45 10             	mov    0x10(%rbp),%eax
 74e:	01 c0                	add    %eax,%eax
 750:	48 98                	cltq   
 752:	48 83 c0 04          	add    $0x4,%rax
 756:	48 8d 14 c5 00 00 00 	lea    0x0(,%rax,8),%rdx
 75d:	00 
 75e:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
 763:	48 01 d0             	add    %rdx,%rax
 766:	48 89 84 24 98 00 00 	mov    %rax,0x98(%rsp)
 76d:	00 
  return *(__m256d *)__P;
 76e:	48 8b 84 24 98 00 00 	mov    0x98(%rsp),%rax
 775:	00 
 776:	c5 fd 28 08          	vmovapd (%rax),%ymm1
 77a:	c5 fd 29 8c 24 80 03 	vmovapd %ymm1,0x380(%rsp)
 781:	00 00 
 783:	c5 fd 29 84 24 a0 03 	vmovapd %ymm0,0x3a0(%rsp)
 78a:	00 00 
  return (__m256d) ((__v4df)__A + (__v4df)__B);
 78c:	c5 fd 28 84 24 80 03 	vmovapd 0x380(%rsp),%ymm0
 793:	00 00 
 795:	c5 fd 58 84 24 a0 03 	vaddpd 0x3a0(%rsp),%ymm0,%ymm0
 79c:	00 00 
 79e:	8b 45 10             	mov    0x10(%rbp),%eax
 7a1:	01 c0                	add    %eax,%eax
 7a3:	48 98                	cltq   
 7a5:	48 83 c0 04          	add    $0x4,%rax
 7a9:	48 8d 14 c5 00 00 00 	lea    0x0(,%rax,8),%rdx
 7b0:	00 
 7b1:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
 7b6:	48 01 d0             	add    %rdx,%rax
 7b9:	48 89 84 24 90 00 00 	mov    %rax,0x90(%rsp)
 7c0:	00 
 7c1:	c5 fd 29 84 24 60 03 	vmovapd %ymm0,0x360(%rsp)
 7c8:	00 00 
  *(__m256d *)__P = __A;
 7ca:	48 8b 84 24 90 00 00 	mov    0x90(%rsp),%rax
 7d1:	00 
 7d2:	c5 fd 28 84 24 60 03 	vmovapd 0x360(%rsp),%ymm0
 7d9:	00 00 
 7db:	c5 fd 29 00          	vmovapd %ymm0,(%rax)
}
 7df:	90                   	nop
    _mm256_store_pd((c + ldc * 3 + 4), _mm256_add_pd(_mm256_load_pd((c + ldc * 3 + 4)), vc43536373.v));
 7e0:	c5 fd 28 84 24 e0 06 	vmovapd 0x6e0(%rsp),%ymm0
 7e7:	00 00 
 7e9:	8b 55 10             	mov    0x10(%rbp),%edx
 7ec:	89 d0                	mov    %edx,%eax
 7ee:	01 c0                	add    %eax,%eax
 7f0:	01 d0                	add    %edx,%eax
 7f2:	48 98                	cltq   
 7f4:	48 83 c0 04          	add    $0x4,%rax
 7f8:	48 8d 14 c5 00 00 00 	lea    0x0(,%rax,8),%rdx
 7ff:	00 
 800:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
 805:	48 01 d0             	add    %rdx,%rax
 808:	48 89 84 24 88 00 00 	mov    %rax,0x88(%rsp)
 80f:	00 
  return *(__m256d *)__P;
 810:	48 8b 84 24 88 00 00 	mov    0x88(%rsp),%rax
 817:	00 
 818:	c5 fd 28 08          	vmovapd (%rax),%ymm1
 81c:	c5 fd 29 8c 24 20 03 	vmovapd %ymm1,0x320(%rsp)
 823:	00 00 
 825:	c5 fd 29 84 24 40 03 	vmovapd %ymm0,0x340(%rsp)
 82c:	00 00 
  return (__m256d) ((__v4df)__A + (__v4df)__B);
 82e:	c5 fd 28 84 24 20 03 	vmovapd 0x320(%rsp),%ymm0
 835:	00 00 
 837:	c5 fd 58 84 24 40 03 	vaddpd 0x340(%rsp),%ymm0,%ymm0
 83e:	00 00 
 840:	8b 55 10             	mov    0x10(%rbp),%edx
 843:	89 d0                	mov    %edx,%eax
 845:	01 c0                	add    %eax,%eax
 847:	01 d0                	add    %edx,%eax
 849:	48 98                	cltq   
 84b:	48 83 c0 04          	add    $0x4,%rax
 84f:	48 8d 14 c5 00 00 00 	lea    0x0(,%rax,8),%rdx
 856:	00 
 857:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
 85c:	48 01 d0             	add    %rdx,%rax
 85f:	48 89 84 24 80 00 00 	mov    %rax,0x80(%rsp)
 866:	00 
 867:	c5 fd 29 84 24 00 03 	vmovapd %ymm0,0x300(%rsp)
 86e:	00 00 
  *(__m256d *)__P = __A;
 870:	48 8b 84 24 80 00 00 	mov    0x80(%rsp),%rax
 877:	00 
 878:	c5 fd 28 84 24 00 03 	vmovapd 0x300(%rsp),%ymm0
 87f:	00 00 
 881:	c5 fd 29 00          	vmovapd %ymm0,(%rax)
}
 885:	90                   	nop

         

};
 886:	90                   	nop
 887:	48 8b 84 24 d8 07 00 	mov    0x7d8(%rsp),%rax
 88e:	00 
 88f:	64 48 33 04 25 28 00 	xor    %fs:0x28,%rax
 896:	00 00 
 898:	74 05                	je     89f <AddDot8x4+0x89f>
 89a:	e8 00 00 00 00       	callq  89f <AddDot8x4+0x89f>
 89f:	c9                   	leaveq 
 8a0:	c3                   	retq   

00000000000008a1 <PackMatrixA>:


void PackMatrixA(int k, double *a, int lda, double *a_to){
 8a1:	f3 0f 1e fa          	endbr64 
 8a5:	55                   	push   %rbp
 8a6:	48 89 e5             	mov    %rsp,%rbp
 8a9:	89 7d ec             	mov    %edi,-0x14(%rbp)
 8ac:	48 89 75 e0          	mov    %rsi,-0x20(%rbp)
 8b0:	89 55 e8             	mov    %edx,-0x18(%rbp)
 8b3:	48 89 4d d8          	mov    %rcx,-0x28(%rbp)
  int j;
  for (j=0; j <k ; j++){
 8b7:	c7 45 f4 00 00 00 00 	movl   $0x0,-0xc(%rbp)
 8be:	e9 e7 00 00 00       	jmpq   9aa <PackMatrixA+0x109>
    double *a_ij_pntr = &A(0, j);
 8c3:	8b 45 f4             	mov    -0xc(%rbp),%eax
 8c6:	0f af 45 e8          	imul   -0x18(%rbp),%eax
 8ca:	48 98                	cltq   
 8cc:	48 8d 14 c5 00 00 00 	lea    0x0(,%rax,8),%rdx
 8d3:	00 
 8d4:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
 8d8:	48 01 d0             	add    %rdx,%rax
 8db:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
    *a_to++ = *a_ij_pntr;
 8df:	48 8b 45 d8          	mov    -0x28(%rbp),%rax
 8e3:	48 8d 50 08          	lea    0x8(%rax),%rdx
 8e7:	48 89 55 d8          	mov    %rdx,-0x28(%rbp)
 8eb:	48 8b 55 f8          	mov    -0x8(%rbp),%rdx
 8ef:	c5 fb 10 02          	vmovsd (%rdx),%xmm0
 8f3:	c5 fb 11 00          	vmovsd %xmm0,(%rax)
    *a_to++ = *(a_ij_pntr+1);
 8f7:	48 8b 45 d8          	mov    -0x28(%rbp),%rax
 8fb:	48 8d 50 08          	lea    0x8(%rax),%rdx
 8ff:	48 89 55 d8          	mov    %rdx,-0x28(%rbp)
 903:	48 8b 55 f8          	mov    -0x8(%rbp),%rdx
 907:	c5 fb 10 42 08       	vmovsd 0x8(%rdx),%xmm0
 90c:	c5 fb 11 00          	vmovsd %xmm0,(%rax)
    *a_to++ = *(a_ij_pntr+2);
 910:	48 8b 45 d8          	mov    -0x28(%rbp),%rax
 914:	48 8d 50 08          	lea    0x8(%rax),%rdx
 918:	48 89 55 d8          	mov    %rdx,-0x28(%rbp)
 91c:	48 8b 55 f8          	mov    -0x8(%rbp),%rdx
 920:	c5 fb 10 42 10       	vmovsd 0x10(%rdx),%xmm0
 925:	c5 fb 11 00          	vmovsd %xmm0,(%rax)
    *a_to++ = *(a_ij_pntr+3);
 929:	48 8b 45 d8          	mov    -0x28(%rbp),%rax
 92d:	48 8d 50 08          	lea    0x8(%rax),%rdx
 931:	48 89 55 d8          	mov    %rdx,-0x28(%rbp)
 935:	48 8b 55 f8          	mov    -0x8(%rbp),%rdx
 939:	c5 fb 10 42 18       	vmovsd 0x18(%rdx),%xmm0
 93e:	c5 fb 11 00          	vmovsd %xmm0,(%rax)
    *a_to++ = *(a_ij_pntr+4);
 942:	48 8b 45 d8          	mov    -0x28(%rbp),%rax
 946:	48 8d 50 08          	lea    0x8(%rax),%rdx
 94a:	48 89 55 d8          	mov    %rdx,-0x28(%rbp)
 94e:	48 8b 55 f8          	mov    -0x8(%rbp),%rdx
 952:	c5 fb 10 42 20       	vmovsd 0x20(%rdx),%xmm0
 957:	c5 fb 11 00          	vmovsd %xmm0,(%rax)
    *a_to++ = *(a_ij_pntr+5);
 95b:	48 8b 45 d8          	mov    -0x28(%rbp),%rax
 95f:	48 8d 50 08          	lea    0x8(%rax),%rdx
 963:	48 89 55 d8          	mov    %rdx,-0x28(%rbp)
 967:	48 8b 55 f8          	mov    -0x8(%rbp),%rdx
 96b:	c5 fb 10 42 28       	vmovsd 0x28(%rdx),%xmm0
 970:	c5 fb 11 00          	vmovsd %xmm0,(%rax)
    *a_to++ = *(a_ij_pntr+6);
 974:	48 8b 45 d8          	mov    -0x28(%rbp),%rax
 978:	48 8d 50 08          	lea    0x8(%rax),%rdx
 97c:	48 89 55 d8          	mov    %rdx,-0x28(%rbp)
 980:	48 8b 55 f8          	mov    -0x8(%rbp),%rdx
 984:	c5 fb 10 42 30       	vmovsd 0x30(%rdx),%xmm0
 989:	c5 fb 11 00          	vmovsd %xmm0,(%rax)
    *a_to++ = *(a_ij_pntr+7);
 98d:	48 8b 45 d8          	mov    -0x28(%rbp),%rax
 991:	48 8d 50 08          	lea    0x8(%rax),%rdx
 995:	48 89 55 d8          	mov    %rdx,-0x28(%rbp)
 999:	48 8b 55 f8          	mov    -0x8(%rbp),%rdx
 99d:	c5 fb 10 42 38       	vmovsd 0x38(%rdx),%xmm0
 9a2:	c5 fb 11 00          	vmovsd %xmm0,(%rax)
  for (j=0; j <k ; j++){
 9a6:	83 45 f4 01          	addl   $0x1,-0xc(%rbp)
 9aa:	8b 45 f4             	mov    -0xc(%rbp),%eax
 9ad:	3b 45 ec             	cmp    -0x14(%rbp),%eax
 9b0:	0f 8c 0d ff ff ff    	jl     8c3 <PackMatrixA+0x22>
  }
}
 9b6:	90                   	nop
 9b7:	90                   	nop
 9b8:	5d                   	pop    %rbp
 9b9:	c3                   	retq   

00000000000009ba <PackMatrixB>:


void PackMatrixB(int k, double *b, int ldb, double *b_to){
 9ba:	f3 0f 1e fa          	endbr64 
 9be:	55                   	push   %rbp
 9bf:	48 89 e5             	mov    %rsp,%rbp
 9c2:	89 7d cc             	mov    %edi,-0x34(%rbp)
 9c5:	48 89 75 c0          	mov    %rsi,-0x40(%rbp)
 9c9:	89 55 c8             	mov    %edx,-0x38(%rbp)
 9cc:	48 89 4d b8          	mov    %rcx,-0x48(%rbp)
  int i;
  double 
    *b0p = &B(0, 0), *b1p = &B(0,1),
 9d0:	48 8b 45 c0          	mov    -0x40(%rbp),%rax
 9d4:	48 89 45 e0          	mov    %rax,-0x20(%rbp)
 9d8:	8b 45 c8             	mov    -0x38(%rbp),%eax
 9db:	48 98                	cltq   
 9dd:	48 8d 14 c5 00 00 00 	lea    0x0(,%rax,8),%rdx
 9e4:	00 
 9e5:	48 8b 45 c0          	mov    -0x40(%rbp),%rax
 9e9:	48 01 d0             	add    %rdx,%rax
 9ec:	48 89 45 e8          	mov    %rax,-0x18(%rbp)
    *b2p = &B(0, 2), *b3p = &B(0,3)
 9f0:	8b 45 c8             	mov    -0x38(%rbp),%eax
 9f3:	01 c0                	add    %eax,%eax
 9f5:	48 98                	cltq   
 9f7:	48 8d 14 c5 00 00 00 	lea    0x0(,%rax,8),%rdx
 9fe:	00 
 9ff:	48 8b 45 c0          	mov    -0x40(%rbp),%rax
 a03:	48 01 d0             	add    %rdx,%rax
 a06:	48 89 45 f0          	mov    %rax,-0x10(%rbp)
 a0a:	8b 55 c8             	mov    -0x38(%rbp),%edx
 a0d:	89 d0                	mov    %edx,%eax
 a0f:	01 c0                	add    %eax,%eax
 a11:	01 d0                	add    %edx,%eax
 a13:	48 98                	cltq   
 a15:	48 8d 14 c5 00 00 00 	lea    0x0(,%rax,8),%rdx
 a1c:	00 
 a1d:	48 8b 45 c0          	mov    -0x40(%rbp),%rax
 a21:	48 01 d0             	add    %rdx,%rax
 a24:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
    *b4p = &B(0, 4), *b5p = &B(0,5),
    *b6p = &B(0, 6), *b7p = &B(0,7)
    */
   ; 

  for (i=0; i < k ; i++){    
 a28:	c7 45 dc 00 00 00 00 	movl   $0x0,-0x24(%rbp)
 a2f:	e9 84 00 00 00       	jmpq   ab8 <PackMatrixB+0xfe>
    *b_to++ = *(b0p++);
 a34:	48 8b 55 e0          	mov    -0x20(%rbp),%rdx
 a38:	48 8d 42 08          	lea    0x8(%rdx),%rax
 a3c:	48 89 45 e0          	mov    %rax,-0x20(%rbp)
 a40:	48 8b 45 b8          	mov    -0x48(%rbp),%rax
 a44:	48 8d 48 08          	lea    0x8(%rax),%rcx
 a48:	48 89 4d b8          	mov    %rcx,-0x48(%rbp)
 a4c:	c5 fb 10 02          	vmovsd (%rdx),%xmm0
 a50:	c5 fb 11 00          	vmovsd %xmm0,(%rax)
    *b_to++ = *(b1p++);
 a54:	48 8b 55 e8          	mov    -0x18(%rbp),%rdx
 a58:	48 8d 42 08          	lea    0x8(%rdx),%rax
 a5c:	48 89 45 e8          	mov    %rax,-0x18(%rbp)
 a60:	48 8b 45 b8          	mov    -0x48(%rbp),%rax
 a64:	48 8d 48 08          	lea    0x8(%rax),%rcx
 a68:	48 89 4d b8          	mov    %rcx,-0x48(%rbp)
 a6c:	c5 fb 10 02          	vmovsd (%rdx),%xmm0
 a70:	c5 fb 11 00          	vmovsd %xmm0,(%rax)
    *b_to++ = *(b2p++);
 a74:	48 8b 55 f0          	mov    -0x10(%rbp),%rdx
 a78:	48 8d 42 08          	lea    0x8(%rdx),%rax
 a7c:	48 89 45 f0          	mov    %rax,-0x10(%rbp)
 a80:	48 8b 45 b8          	mov    -0x48(%rbp),%rax
 a84:	48 8d 48 08          	lea    0x8(%rax),%rcx
 a88:	48 89 4d b8          	mov    %rcx,-0x48(%rbp)
 a8c:	c5 fb 10 02          	vmovsd (%rdx),%xmm0
 a90:	c5 fb 11 00          	vmovsd %xmm0,(%rax)
    *b_to++ = *(b3p++);
 a94:	48 8b 55 f8          	mov    -0x8(%rbp),%rdx
 a98:	48 8d 42 08          	lea    0x8(%rdx),%rax
 a9c:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
 aa0:	48 8b 45 b8          	mov    -0x48(%rbp),%rax
 aa4:	48 8d 48 08          	lea    0x8(%rax),%rcx
 aa8:	48 89 4d b8          	mov    %rcx,-0x48(%rbp)
 aac:	c5 fb 10 02          	vmovsd (%rdx),%xmm0
 ab0:	c5 fb 11 00          	vmovsd %xmm0,(%rax)
  for (i=0; i < k ; i++){    
 ab4:	83 45 dc 01          	addl   $0x1,-0x24(%rbp)
 ab8:	8b 45 dc             	mov    -0x24(%rbp),%eax
 abb:	3b 45 cc             	cmp    -0x34(%rbp),%eax
 abe:	0f 8c 70 ff ff ff    	jl     a34 <PackMatrixB+0x7a>
    *b_to++ = *(b5p++);
    *b_to++ = *(b6p++);
    *b_to++ = *(b7p++);
    */
  }
}
 ac4:	90                   	nop
 ac5:	90                   	nop
 ac6:	5d                   	pop    %rbp
 ac7:	c3                   	retq   

0000000000000ac8 <Innerkernel>:

void Innerkernel(int m, int n, int k, double *a, int lda, double *b, int ldb, double *c, int ldc, int first_time)
{
 ac8:	f3 0f 1e fa          	endbr64 
 acc:	55                   	push   %rbp
 acd:	48 89 e5             	mov    %rsp,%rbp
 ad0:	41 55                	push   %r13
 ad2:	41 54                	push   %r12
 ad4:	53                   	push   %rbx
 ad5:	48 83 ec 58          	sub    $0x58,%rsp
 ad9:	89 7d bc             	mov    %edi,-0x44(%rbp)
 adc:	89 75 b8             	mov    %esi,-0x48(%rbp)
 adf:	89 55 b4             	mov    %edx,-0x4c(%rbp)
 ae2:	48 89 4d a8          	mov    %rcx,-0x58(%rbp)
 ae6:	44 89 45 b0          	mov    %r8d,-0x50(%rbp)
 aea:	4c 89 4d a0          	mov    %r9,-0x60(%rbp)
 aee:	48 8b 45 18          	mov    0x18(%rbp),%rax
 af2:	48 89 45 98          	mov    %rax,-0x68(%rbp)
 af6:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
 afd:	00 00 
 aff:	48 89 45 d8          	mov    %rax,-0x28(%rbp)
 b03:	31 c0                	xor    %eax,%eax
 b05:	48 89 e0             	mov    %rsp,%rax
 b08:	48 89 c3             	mov    %rax,%rbx
  int i, j;
  // todo this part need align and malloc
  double packedA[m*k] __attribute__ ((aligned (64)));
 b0b:	8b 45 bc             	mov    -0x44(%rbp),%eax
 b0e:	0f af 45 b4          	imul   -0x4c(%rbp),%eax
 b12:	48 63 d0             	movslq %eax,%rdx
 b15:	48 83 ea 01          	sub    $0x1,%rdx
 b19:	48 89 55 c8          	mov    %rdx,-0x38(%rbp)
 b1d:	48 63 d0             	movslq %eax,%rdx
 b20:	49 89 d4             	mov    %rdx,%r12
 b23:	41 bd 00 00 00 00    	mov    $0x0,%r13d
 b29:	48 63 d0             	movslq %eax,%rdx
 b2c:	49 89 d2             	mov    %rdx,%r10
 b2f:	41 bb 00 00 00 00    	mov    $0x0,%r11d
 b35:	48 98                	cltq   
 b37:	48 c1 e0 03          	shl    $0x3,%rax
 b3b:	48 8d 50 38          	lea    0x38(%rax),%rdx
 b3f:	b8 10 00 00 00       	mov    $0x10,%eax
 b44:	48 83 e8 01          	sub    $0x1,%rax
 b48:	48 01 d0             	add    %rdx,%rax
 b4b:	bf 10 00 00 00       	mov    $0x10,%edi
 b50:	ba 00 00 00 00       	mov    $0x0,%edx
 b55:	48 f7 f7             	div    %rdi
 b58:	48 6b c0 10          	imul   $0x10,%rax,%rax
 b5c:	48 89 c2             	mov    %rax,%rdx
 b5f:	48 81 e2 00 f0 ff ff 	and    $0xfffffffffffff000,%rdx
 b66:	48 89 e7             	mov    %rsp,%rdi
 b69:	48 29 d7             	sub    %rdx,%rdi
 b6c:	48 89 fa             	mov    %rdi,%rdx
 b6f:	48 39 d4             	cmp    %rdx,%rsp
 b72:	74 12                	je     b86 <Innerkernel+0xbe>
 b74:	48 81 ec 00 10 00 00 	sub    $0x1000,%rsp
 b7b:	48 83 8c 24 f8 0f 00 	orq    $0x0,0xff8(%rsp)
 b82:	00 00 
 b84:	eb e9                	jmp    b6f <Innerkernel+0xa7>
 b86:	48 89 c2             	mov    %rax,%rdx
 b89:	81 e2 ff 0f 00 00    	and    $0xfff,%edx
 b8f:	48 29 d4             	sub    %rdx,%rsp
 b92:	48 89 c2             	mov    %rax,%rdx
 b95:	81 e2 ff 0f 00 00    	and    $0xfff,%edx
 b9b:	48 85 d2             	test   %rdx,%rdx
 b9e:	74 10                	je     bb0 <Innerkernel+0xe8>
 ba0:	25 ff 0f 00 00       	and    $0xfff,%eax
 ba5:	48 83 e8 08          	sub    $0x8,%rax
 ba9:	48 01 e0             	add    %rsp,%rax
 bac:	48 83 08 00          	orq    $0x0,(%rax)
 bb0:	48 89 e0             	mov    %rsp,%rax
 bb3:	48 83 c0 3f          	add    $0x3f,%rax
 bb7:	48 c1 e8 06          	shr    $0x6,%rax
 bbb:	48 c1 e0 06          	shl    $0x6,%rax
 bbf:	48 89 45 d0          	mov    %rax,-0x30(%rbp)
  static double packedB[kc*nb] __attribute__ ((aligned (64)));
  for (j = 0; j < n; j+=4){
 bc3:	c7 45 c0 00 00 00 00 	movl   $0x0,-0x40(%rbp)
 bca:	e9 26 01 00 00       	jmpq   cf5 <Innerkernel+0x22d>
    if ( first_time ) {
 bcf:	83 7d 28 00          	cmpl   $0x0,0x28(%rbp)
 bd3:	74 44                	je     c19 <Innerkernel+0x151>
      PackMatrixB(k, &B(0, j), ldb, &packedB[j*k]);
 bd5:	8b 45 c0             	mov    -0x40(%rbp),%eax
 bd8:	0f af 45 b4          	imul   -0x4c(%rbp),%eax
 bdc:	48 98                	cltq   
 bde:	48 8d 14 c5 00 00 00 	lea    0x0(,%rax,8),%rdx
 be5:	00 
 be6:	48 8d 05 00 00 00 00 	lea    0x0(%rip),%rax        # bed <Innerkernel+0x125>
 bed:	48 01 c2             	add    %rax,%rdx
 bf0:	8b 45 c0             	mov    -0x40(%rbp),%eax
 bf3:	0f af 45 10          	imul   0x10(%rbp),%eax
 bf7:	48 98                	cltq   
 bf9:	48 8d 0c c5 00 00 00 	lea    0x0(,%rax,8),%rcx
 c00:	00 
 c01:	48 8b 45 a0          	mov    -0x60(%rbp),%rax
 c05:	48 8d 34 01          	lea    (%rcx,%rax,1),%rsi
 c09:	8b 45 b4             	mov    -0x4c(%rbp),%eax
 c0c:	48 89 d1             	mov    %rdx,%rcx
 c0f:	8b 55 10             	mov    0x10(%rbp),%edx
 c12:	89 c7                	mov    %eax,%edi
 c14:	e8 00 00 00 00       	callq  c19 <Innerkernel+0x151>
    }
    
    for (i = 0; i < m; i +=8){
 c19:	c7 45 c4 00 00 00 00 	movl   $0x0,-0x3c(%rbp)
 c20:	e9 c0 00 00 00       	jmpq   ce5 <Innerkernel+0x21d>
      if ( j == 0 ) PackMatrixA(k, &A(i, 0), lda, &packedA[i*k]);
 c25:	83 7d c0 00          	cmpl   $0x0,-0x40(%rbp)
 c29:	75 3b                	jne    c66 <Innerkernel+0x19e>
 c2b:	8b 45 c4             	mov    -0x3c(%rbp),%eax
 c2e:	0f af 45 b4          	imul   -0x4c(%rbp),%eax
 c32:	48 98                	cltq   
 c34:	48 8d 14 c5 00 00 00 	lea    0x0(,%rax,8),%rdx
 c3b:	00 
 c3c:	48 8b 45 d0          	mov    -0x30(%rbp),%rax
 c40:	48 8d 0c 02          	lea    (%rdx,%rax,1),%rcx
 c44:	8b 45 c4             	mov    -0x3c(%rbp),%eax
 c47:	48 98                	cltq   
 c49:	48 8d 14 c5 00 00 00 	lea    0x0(,%rax,8),%rdx
 c50:	00 
 c51:	48 8b 45 a8          	mov    -0x58(%rbp),%rax
 c55:	48 8d 34 02          	lea    (%rdx,%rax,1),%rsi
 c59:	8b 55 b0             	mov    -0x50(%rbp),%edx
 c5c:	8b 45 b4             	mov    -0x4c(%rbp),%eax
 c5f:	89 c7                	mov    %eax,%edi
 c61:	e8 00 00 00 00       	callq  c66 <Innerkernel+0x19e>
      AddDot8x4(k, &packedA[i*k], 8, &packedB[j*k], k , &C( i,j ), ldc);
 c66:	8b 45 c0             	mov    -0x40(%rbp),%eax
 c69:	0f af 45 20          	imul   0x20(%rbp),%eax
 c6d:	89 c2                	mov    %eax,%edx
 c6f:	8b 45 c4             	mov    -0x3c(%rbp),%eax
 c72:	01 d0                	add    %edx,%eax
 c74:	48 98                	cltq   
 c76:	48 8d 14 c5 00 00 00 	lea    0x0(,%rax,8),%rdx
 c7d:	00 
 c7e:	48 8b 45 98          	mov    -0x68(%rbp),%rax
 c82:	4c 8d 04 02          	lea    (%rdx,%rax,1),%r8
 c86:	8b 45 c0             	mov    -0x40(%rbp),%eax
 c89:	0f af 45 b4          	imul   -0x4c(%rbp),%eax
 c8d:	48 98                	cltq   
 c8f:	48 8d 14 c5 00 00 00 	lea    0x0(,%rax,8),%rdx
 c96:	00 
 c97:	48 8d 05 00 00 00 00 	lea    0x0(%rip),%rax        # c9e <Innerkernel+0x1d6>
 c9e:	48 01 c2             	add    %rax,%rdx
 ca1:	8b 45 c4             	mov    -0x3c(%rbp),%eax
 ca4:	0f af 45 b4          	imul   -0x4c(%rbp),%eax
 ca8:	48 98                	cltq   
 caa:	48 8d 0c c5 00 00 00 	lea    0x0(,%rax,8),%rcx
 cb1:	00 
 cb2:	48 8b 45 d0          	mov    -0x30(%rbp),%rax
 cb6:	48 8d 34 01          	lea    (%rcx,%rax,1),%rsi
 cba:	8b 7d b4             	mov    -0x4c(%rbp),%edi
 cbd:	8b 45 b4             	mov    -0x4c(%rbp),%eax
 cc0:	48 83 ec 08          	sub    $0x8,%rsp
 cc4:	8b 4d 20             	mov    0x20(%rbp),%ecx
 cc7:	51                   	push   %rcx
 cc8:	4d 89 c1             	mov    %r8,%r9
 ccb:	41 89 f8             	mov    %edi,%r8d
 cce:	48 89 d1             	mov    %rdx,%rcx
 cd1:	ba 08 00 00 00       	mov    $0x8,%edx
 cd6:	89 c7                	mov    %eax,%edi
 cd8:	e8 00 00 00 00       	callq  cdd <Innerkernel+0x215>
 cdd:	48 83 c4 10          	add    $0x10,%rsp
    for (i = 0; i < m; i +=8){
 ce1:	83 45 c4 08          	addl   $0x8,-0x3c(%rbp)
 ce5:	8b 45 c4             	mov    -0x3c(%rbp),%eax
 ce8:	3b 45 bc             	cmp    -0x44(%rbp),%eax
 ceb:	0f 8c 34 ff ff ff    	jl     c25 <Innerkernel+0x15d>
  for (j = 0; j < n; j+=4){
 cf1:	83 45 c0 04          	addl   $0x4,-0x40(%rbp)
 cf5:	8b 45 c0             	mov    -0x40(%rbp),%eax
 cf8:	3b 45 b8             	cmp    -0x48(%rbp),%eax
 cfb:	0f 8c ce fe ff ff    	jl     bcf <Innerkernel+0x107>
 d01:	48 89 dc             	mov    %rbx,%rsp
    }
  }
}
 d04:	90                   	nop
 d05:	48 8b 45 d8          	mov    -0x28(%rbp),%rax
 d09:	64 48 33 04 25 28 00 	xor    %fs:0x28,%rax
 d10:	00 00 
 d12:	74 05                	je     d19 <Innerkernel+0x251>
 d14:	e8 00 00 00 00       	callq  d19 <Innerkernel+0x251>
 d19:	48 8d 65 e8          	lea    -0x18(%rbp),%rsp
 d1d:	5b                   	pop    %rbx
 d1e:	41 5c                	pop    %r12
 d20:	41 5d                	pop    %r13
 d22:	5d                   	pop    %rbp
 d23:	c3                   	retq   

0000000000000d24 <MY_MMult>:

void MY_MMult( int m, int n, int k, double *a, int lda, 
                                    double *b, int ldb,
                                    double *c, int ldc )
{
 d24:	f3 0f 1e fa          	endbr64 
 d28:	55                   	push   %rbp
 d29:	48 89 e5             	mov    %rsp,%rbp
 d2c:	48 83 ec 30          	sub    $0x30,%rsp
 d30:	89 7d ec             	mov    %edi,-0x14(%rbp)
 d33:	89 75 e8             	mov    %esi,-0x18(%rbp)
 d36:	89 55 e4             	mov    %edx,-0x1c(%rbp)
 d39:	48 89 4d d8          	mov    %rcx,-0x28(%rbp)
 d3d:	44 89 45 e0          	mov    %r8d,-0x20(%rbp)
 d41:	4c 89 4d d0          	mov    %r9,-0x30(%rbp)
  int i, j, jb, ib;

  for ( j=0; j< k ; j+=kc ){        /* Loop over the columns of C */
 d45:	c7 45 f4 00 00 00 00 	movl   $0x0,-0xc(%rbp)
 d4c:	e9 c9 00 00 00       	jmpq   e1a <MY_MMult+0xf6>
    jb = min(k-j, kc);
 d51:	8b 45 e4             	mov    -0x1c(%rbp),%eax
 d54:	2b 45 f4             	sub    -0xc(%rbp),%eax
 d57:	ba 80 00 00 00       	mov    $0x80,%edx
 d5c:	3d 80 00 00 00       	cmp    $0x80,%eax
 d61:	0f 4f c2             	cmovg  %edx,%eax
 d64:	89 45 f8             	mov    %eax,-0x8(%rbp)
    for ( i=0; i<m; i+=mc ){        /* Loop over the rows of C */
 d67:	c7 45 f0 00 00 00 00 	movl   $0x0,-0x10(%rbp)
 d6e:	e9 97 00 00 00       	jmpq   e0a <MY_MMult+0xe6>
      /* Update the C( i,j ) with the inner product of the ith row of A
	 and the jth column of B */
      ib = min(m-i, mc);
 d73:	8b 45 ec             	mov    -0x14(%rbp),%eax
 d76:	2b 45 f0             	sub    -0x10(%rbp),%eax
 d79:	ba 00 01 00 00       	mov    $0x100,%edx
 d7e:	3d 00 01 00 00       	cmp    $0x100,%eax
 d83:	0f 4f c2             	cmovg  %edx,%eax
 d86:	89 45 fc             	mov    %eax,-0x4(%rbp)
      Innerkernel(ib, n, jb, &A(i, j), lda, &B(j,0), ldb, &C(i, 0), ldc, i==0);
 d89:	83 7d f0 00          	cmpl   $0x0,-0x10(%rbp)
 d8d:	0f 94 c0             	sete   %al
 d90:	0f b6 f8             	movzbl %al,%edi
 d93:	8b 45 f0             	mov    -0x10(%rbp),%eax
 d96:	48 98                	cltq   
 d98:	48 8d 14 c5 00 00 00 	lea    0x0(,%rax,8),%rdx
 d9f:	00 
 da0:	48 8b 45 18          	mov    0x18(%rbp),%rax
 da4:	4c 8d 04 02          	lea    (%rdx,%rax,1),%r8
 da8:	8b 45 f4             	mov    -0xc(%rbp),%eax
 dab:	48 98                	cltq   
 dad:	48 8d 14 c5 00 00 00 	lea    0x0(,%rax,8),%rdx
 db4:	00 
 db5:	48 8b 45 d0          	mov    -0x30(%rbp),%rax
 db9:	4c 8d 0c 02          	lea    (%rdx,%rax,1),%r9
 dbd:	8b 45 f4             	mov    -0xc(%rbp),%eax
 dc0:	0f af 45 e0          	imul   -0x20(%rbp),%eax
 dc4:	89 c2                	mov    %eax,%edx
 dc6:	8b 45 f0             	mov    -0x10(%rbp),%eax
 dc9:	01 d0                	add    %edx,%eax
 dcb:	48 98                	cltq   
 dcd:	48 8d 14 c5 00 00 00 	lea    0x0(,%rax,8),%rdx
 dd4:	00 
 dd5:	48 8b 45 d8          	mov    -0x28(%rbp),%rax
 dd9:	48 8d 0c 02          	lea    (%rdx,%rax,1),%rcx
 ddd:	44 8b 55 e0          	mov    -0x20(%rbp),%r10d
 de1:	8b 55 f8             	mov    -0x8(%rbp),%edx
 de4:	8b 75 e8             	mov    -0x18(%rbp),%esi
 de7:	8b 45 fc             	mov    -0x4(%rbp),%eax
 dea:	57                   	push   %rdi
 deb:	8b 7d 20             	mov    0x20(%rbp),%edi
 dee:	57                   	push   %rdi
 def:	41 50                	push   %r8
 df1:	8b 7d 10             	mov    0x10(%rbp),%edi
 df4:	57                   	push   %rdi
 df5:	45 89 d0             	mov    %r10d,%r8d
 df8:	89 c7                	mov    %eax,%edi
 dfa:	e8 00 00 00 00       	callq  dff <MY_MMult+0xdb>
 dff:	48 83 c4 20          	add    $0x20,%rsp
    for ( i=0; i<m; i+=mc ){        /* Loop over the rows of C */
 e03:	81 45 f0 00 01 00 00 	addl   $0x100,-0x10(%rbp)
 e0a:	8b 45 f0             	mov    -0x10(%rbp),%eax
 e0d:	3b 45 ec             	cmp    -0x14(%rbp),%eax
 e10:	0f 8c 5d ff ff ff    	jl     d73 <MY_MMult+0x4f>
  for ( j=0; j< k ; j+=kc ){        /* Loop over the columns of C */
 e16:	83 6d f4 80          	subl   $0xffffff80,-0xc(%rbp)
 e1a:	8b 45 f4             	mov    -0xc(%rbp),%eax
 e1d:	3b 45 e4             	cmp    -0x1c(%rbp),%eax
 e20:	0f 8c 2b ff ff ff    	jl     d51 <MY_MMult+0x2d>
            
    }
  }
 e26:	90                   	nop
 e27:	90                   	nop
 e28:	c9                   	leaveq 
 e29:	c3                   	retq   
