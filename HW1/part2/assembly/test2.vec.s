	.text
	.file	"test2.cpp"
	.section	.rodata.cst8,"aM",@progbits,8
	.p2align	3                               # -- Begin function _Z5test2PfS_S_i
.LCPI0_0:
	.quad	0x3e112e0be826d695              # double 1.0000000000000001E-9
	.text
	.globl	_Z5test2PfS_S_i
	.p2align	4, 0x90
	.type	_Z5test2PfS_S_i,@function
_Z5test2PfS_S_i:                        # @_Z5test2PfS_S_i
	.cfi_startproc
# %bb.0:
	pushq	%r15
	.cfi_def_cfa_offset 16
	pushq	%r14
	.cfi_def_cfa_offset 24
	pushq	%r13
	.cfi_def_cfa_offset 32
	pushq	%r12
	.cfi_def_cfa_offset 40
	pushq	%rbx
	.cfi_def_cfa_offset 48
	subq	$32, %rsp
	.cfi_def_cfa_offset 80
	.cfi_offset %rbx, -48
	.cfi_offset %r12, -40
	.cfi_offset %r13, -32
	.cfi_offset %r14, -24
	.cfi_offset %r15, -16
	movq	%rdx, %r14
	movq	%rsi, %r15
	movq	%rdi, %rbx
	leaq	8(%rsp), %rsi
	movl	$1, %edi
	callq	clock_gettime
	testl	%eax, %eax
	jne	.LBB0_7
# %bb.1:
	movq	8(%rsp), %r13
	movq	16(%rsp), %r12
	xorl	%eax, %eax
	.p2align	4, 0x90
.LBB0_2:                                # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_3 Depth 2
	xorl	%ecx, %ecx
	.p2align	4, 0x90
.LBB0_3:                                #   Parent Loop BB0_2 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	movaps	(%r15,%rcx,4), %xmm0
	movaps	16(%r15,%rcx,4), %xmm1
	maxps	(%rbx,%rcx,4), %xmm0
	maxps	16(%rbx,%rcx,4), %xmm1
	movups	%xmm0, (%r14,%rcx,4)
	movups	%xmm1, 16(%r14,%rcx,4)
	movaps	32(%r15,%rcx,4), %xmm0
	movaps	48(%r15,%rcx,4), %xmm1
	maxps	32(%rbx,%rcx,4), %xmm0
	maxps	48(%rbx,%rcx,4), %xmm1
	movups	%xmm0, 32(%r14,%rcx,4)
	movups	%xmm1, 48(%r14,%rcx,4)
	addq	$16, %rcx
	cmpq	$1024, %rcx                     # imm = 0x400
	jne	.LBB0_3
# %bb.4:                                #   in Loop: Header=BB0_2 Depth=1
	addl	$1, %eax
	cmpl	$20000000, %eax                 # imm = 0x1312D00
	jne	.LBB0_2
# %bb.5:
	leaq	8(%rsp), %rsi
	movl	$1, %edi
	callq	clock_gettime
	testl	%eax, %eax
	jne	.LBB0_7
# %bb.6:
	movq	8(%rsp), %rax
	subq	%r13, %rax
	movq	16(%rsp), %rcx
	subq	%r12, %rcx
	xorps	%xmm0, %xmm0
	cvtsi2sd	%rax, %xmm0
	xorps	%xmm1, %xmm1
	cvtsi2sd	%rcx, %xmm1
	mulsd	.LCPI0_0(%rip), %xmm1
	addsd	%xmm0, %xmm1
	movsd	%xmm1, 24(%rsp)                 # 8-byte Spill
	movl	$_ZSt4cout, %edi
	movl	$.L.str, %esi
	movl	$47, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	movl	$_ZSt4cout, %edi
	movsd	24(%rsp), %xmm0                 # 8-byte Reload
                                        # xmm0 = mem[0],zero
	callq	_ZNSo9_M_insertIdEERSoT_
	movq	%rax, %rbx
	movl	$.L.str.1, %esi
	movl	$8, %edx
	movq	%rax, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	movq	%rbx, %rdi
	movl	$1024, %esi                     # imm = 0x400
	callq	_ZNSolsEi
	movq	%rax, %rbx
	movl	$.L.str.2, %esi
	movl	$5, %edx
	movq	%rax, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	movq	%rbx, %rdi
	movl	$20000000, %esi                 # imm = 0x1312D00
	callq	_ZNSolsEi
	movl	$.L.str.3, %esi
	movl	$2, %edx
	movq	%rax, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	addq	$32, %rsp
	.cfi_def_cfa_offset 48
	popq	%rbx
	.cfi_def_cfa_offset 40
	popq	%r12
	.cfi_def_cfa_offset 32
	popq	%r13
	.cfi_def_cfa_offset 24
	popq	%r14
	.cfi_def_cfa_offset 16
	popq	%r15
	.cfi_def_cfa_offset 8
	retq
.LBB0_7:
	.cfi_def_cfa_offset 80
	movl	$.L.str.4, %edi
	movl	$.L.str.5, %esi
	movl	$.L__PRETTY_FUNCTION__._ZL7gettimev, %ecx
	movl	$75, %edx
	callq	__assert_fail
.Lfunc_end0:
	.size	_Z5test2PfS_S_i, .Lfunc_end0-_Z5test2PfS_S_i
	.cfi_endproc
                                        # -- End function
	.section	.text.startup,"ax",@progbits
	.p2align	4, 0x90                         # -- Begin function _GLOBAL__sub_I_test2.cpp
	.type	_GLOBAL__sub_I_test2.cpp,@function
_GLOBAL__sub_I_test2.cpp:               # @_GLOBAL__sub_I_test2.cpp
	.cfi_startproc
# %bb.0:
	pushq	%rax
	.cfi_def_cfa_offset 16
	movl	$_ZStL8__ioinit, %edi
	callq	_ZNSt8ios_base4InitC1Ev
	movl	$_ZNSt8ios_base4InitD1Ev, %edi
	movl	$_ZStL8__ioinit, %esi
	movl	$__dso_handle, %edx
	popq	%rax
	.cfi_def_cfa_offset 8
	jmp	__cxa_atexit                    # TAILCALL
.Lfunc_end1:
	.size	_GLOBAL__sub_I_test2.cpp, .Lfunc_end1-_GLOBAL__sub_I_test2.cpp
	.cfi_endproc
                                        # -- End function
	.type	_ZStL8__ioinit,@object          # @_ZStL8__ioinit
	.local	_ZStL8__ioinit
	.comm	_ZStL8__ioinit,1,1
	.hidden	__dso_handle
	.type	.L.str,@object                  # @.str
	.section	.rodata.str1.1,"aMS",@progbits,1
.L.str:
	.asciz	"Elapsed execution time of the loop in test2():\n"
	.size	.L.str, 48

	.type	.L.str.1,@object                # @.str.1
.L.str.1:
	.asciz	"sec (N: "
	.size	.L.str.1, 9

	.type	.L.str.2,@object                # @.str.2
.L.str.2:
	.asciz	", I: "
	.size	.L.str.2, 6

	.type	.L.str.3,@object                # @.str.3
.L.str.3:
	.asciz	")\n"
	.size	.L.str.3, 3

	.type	.L.str.4,@object                # @.str.4
.L.str.4:
	.asciz	"r == 0"
	.size	.L.str.4, 7

	.type	.L.str.5,@object                # @.str.5
.L.str.5:
	.asciz	"./fasttime.h"
	.size	.L.str.5, 13

	.type	.L__PRETTY_FUNCTION__._ZL7gettimev,@object # @__PRETTY_FUNCTION__._ZL7gettimev
.L__PRETTY_FUNCTION__._ZL7gettimev:
	.asciz	"fasttime_t gettime()"
	.size	.L__PRETTY_FUNCTION__._ZL7gettimev, 21

	.section	.init_array,"aw",@init_array
	.p2align	3
	.quad	_GLOBAL__sub_I_test2.cpp
	.ident	"Ubuntu clang version 11.0.0-++20200925013337+81eb1c1fa75-1~exp1~20200925114015.107"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym _GLOBAL__sub_I_test2.cpp
	.addrsig_sym _ZStL8__ioinit
	.addrsig_sym __dso_handle
	.addrsig_sym _ZSt4cout
