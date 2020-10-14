	.text
	.file	"test1.cpp"
	.section	.rodata.cst8,"aM",@progbits,8
	.p2align	3                               # -- Begin function _Z5test1PfS_S_i
.LCPI0_0:
	.quad	0x3e112e0be826d695              # double 1.0000000000000001E-9
	.text
	.globl	_Z5test1PfS_S_i
	.p2align	4, 0x90
	.type	_Z5test1PfS_S_i,@function
_Z5test1PfS_S_i:                        # @_Z5test1PfS_S_i
.Lfunc_begin0:
	.cfi_startproc
	.cfi_personality 3, __gxx_personality_v0
	.cfi_lsda 3, .Lexception0
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
	subq	$544, %rsp                      # imm = 0x220
	.cfi_def_cfa_offset 592
	.cfi_offset %rbx, -48
	.cfi_offset %r12, -40
	.cfi_offset %r13, -32
	.cfi_offset %r14, -24
	.cfi_offset %r15, -16
	movq	%rdx, %r14
	movq	%rsi, %r15
	movq	%rdi, %rbx
	leaq	16(%rsp), %rsi
	movl	$1, %edi
	callq	clock_gettime
	testl	%eax, %eax
	jne	.LBB0_18
# %bb.1:
	movq	16(%rsp), %r13
	movq	24(%rsp), %r12
	xorl	%eax, %eax
	.p2align	4, 0x90
.LBB0_2:                                # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_3 Depth 2
	xorl	%ecx, %ecx
	.p2align	4, 0x90
.LBB0_3:                                #   Parent Loop BB0_2 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	movaps	(%rbx,%rcx,4), %xmm0
	movaps	16(%rbx,%rcx,4), %xmm1
	addps	(%r15,%rcx,4), %xmm0
	addps	16(%r15,%rcx,4), %xmm1
	movaps	%xmm0, (%r14,%rcx,4)
	movaps	%xmm1, 16(%r14,%rcx,4)
	movaps	32(%rbx,%rcx,4), %xmm0
	movaps	48(%rbx,%rcx,4), %xmm1
	addps	32(%r15,%rcx,4), %xmm0
	addps	48(%r15,%rcx,4), %xmm1
	movaps	%xmm0, 32(%r14,%rcx,4)
	movaps	%xmm1, 48(%r14,%rcx,4)
	addq	$16, %rcx
	cmpq	$1024, %rcx                     # imm = 0x400
	jne	.LBB0_3
# %bb.4:                                #   in Loop: Header=BB0_2 Depth=1
	addl	$1, %eax
	cmpl	$20000000, %eax                 # imm = 0x1312D00
	jne	.LBB0_2
# %bb.5:
	leaq	16(%rsp), %rsi
	movl	$1, %edi
	callq	clock_gettime
	testl	%eax, %eax
	jne	.LBB0_18
# %bb.6:
	movq	16(%rsp), %rax
	subq	%r13, %rax
	movq	24(%rsp), %rcx
	subq	%r12, %rcx
	xorps	%xmm0, %xmm0
	cvtsi2sd	%rax, %xmm0
	xorps	%xmm1, %xmm1
	cvtsi2sd	%rcx, %xmm1
	mulsd	.LCPI0_0(%rip), %xmm1
	addsd	%xmm0, %xmm1
	movsd	%xmm1, 8(%rsp)                  # 8-byte Spill
	movl	$_ZSt4cout, %edi
	movl	$.L.str, %esi
	movl	$47, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	movl	$_ZSt4cout, %edi
	movsd	8(%rsp), %xmm0                  # 8-byte Reload
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
	leaq	16(%rsp), %rdi
	movl	$.L.str.4, %esi
	movl	$17, %edx
	callq	_ZNSt13basic_fstreamIcSt11char_traitsIcEEC1EPKcSt13_Ios_Openmode
	leaq	32(%rsp), %rdi
.Ltmp0:
	movsd	8(%rsp), %xmm0                  # 8-byte Reload
                                        # xmm0 = mem[0],zero
	callq	_ZNSo9_M_insertIdEERSoT_
.Ltmp1:
# %bb.7:
	movq	%rax, %r14
	movq	(%rax), %rax
	movq	-24(%rax), %rax
	movq	240(%r14,%rax), %rbx
	testq	%rbx, %rbx
	je	.LBB0_8
# %bb.10:
	cmpb	$0, 56(%rbx)
	je	.LBB0_12
# %bb.11:
	movb	67(%rbx), %al
	jmp	.LBB0_14
.LBB0_12:
.Ltmp2:
	movq	%rbx, %rdi
	callq	_ZNKSt5ctypeIcE13_M_widen_initEv
.Ltmp3:
# %bb.13:
	movq	(%rbx), %rax
.Ltmp4:
	movq	%rbx, %rdi
	movl	$10, %esi
	callq	*48(%rax)
.Ltmp5:
.LBB0_14:
.Ltmp6:
	movsbl	%al, %esi
	movq	%r14, %rdi
	callq	_ZNSo3putEc
.Ltmp7:
# %bb.15:
.Ltmp8:
	movq	%rax, %rdi
	callq	_ZNSo5flushEv
.Ltmp9:
# %bb.16:
	movq	_ZTTSt13basic_fstreamIcSt11char_traitsIcEE(%rip), %rax
	movq	%rax, 16(%rsp)
	movq	_ZTTSt13basic_fstreamIcSt11char_traitsIcEE+64(%rip), %rcx
	movq	-24(%rax), %rax
	movq	%rcx, 16(%rsp,%rax)
	movq	_ZTTSt13basic_fstreamIcSt11char_traitsIcEE+72(%rip), %rax
	movq	%rax, 32(%rsp)
	leaq	40(%rsp), %rdi
	callq	_ZNSt13basic_filebufIcSt11char_traitsIcEED2Ev
	movq	_ZTTSt13basic_fstreamIcSt11char_traitsIcEE+16(%rip), %rax
	movq	%rax, 16(%rsp)
	movq	_ZTTSt13basic_fstreamIcSt11char_traitsIcEE+24(%rip), %rcx
	movq	-24(%rax), %rax
	movq	%rcx, 16(%rsp,%rax)
	movq	$0, 24(%rsp)
	leaq	280(%rsp), %rdi
	callq	_ZNSt8ios_baseD2Ev
	addq	$544, %rsp                      # imm = 0x220
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
.LBB0_18:
	.cfi_def_cfa_offset 592
	movl	$.L.str.5, %edi
	movl	$.L.str.6, %esi
	movl	$.L__PRETTY_FUNCTION__._ZL7gettimev, %ecx
	movl	$75, %edx
	callq	__assert_fail
.LBB0_8:
.Ltmp10:
	callq	_ZSt16__throw_bad_castv
.Ltmp11:
# %bb.9:
.LBB0_17:
.Ltmp12:
	movq	%rax, %rbx
	movq	_ZTTSt13basic_fstreamIcSt11char_traitsIcEE(%rip), %rax
	movq	%rax, 16(%rsp)
	movq	_ZTTSt13basic_fstreamIcSt11char_traitsIcEE+64(%rip), %rcx
	movq	-24(%rax), %rax
	movq	%rcx, 16(%rsp,%rax)
	movq	_ZTTSt13basic_fstreamIcSt11char_traitsIcEE+72(%rip), %rax
	movq	%rax, 32(%rsp)
	leaq	40(%rsp), %rdi
	callq	_ZNSt13basic_filebufIcSt11char_traitsIcEED2Ev
	movq	_ZTTSt13basic_fstreamIcSt11char_traitsIcEE+16(%rip), %rax
	movq	%rax, 16(%rsp)
	movq	_ZTTSt13basic_fstreamIcSt11char_traitsIcEE+24(%rip), %rcx
	movq	-24(%rax), %rax
	movq	%rcx, 16(%rsp,%rax)
	movq	$0, 24(%rsp)
	leaq	280(%rsp), %rdi
	callq	_ZNSt8ios_baseD2Ev
	movq	%rbx, %rdi
	callq	_Unwind_Resume
.Lfunc_end0:
	.size	_Z5test1PfS_S_i, .Lfunc_end0-_Z5test1PfS_S_i
	.cfi_endproc
	.section	.gcc_except_table,"a",@progbits
	.p2align	2
GCC_except_table0:
.Lexception0:
	.byte	255                             # @LPStart Encoding = omit
	.byte	255                             # @TType Encoding = omit
	.byte	1                               # Call site Encoding = uleb128
	.uleb128 .Lcst_end0-.Lcst_begin0
.Lcst_begin0:
	.uleb128 .Lfunc_begin0-.Lfunc_begin0    # >> Call Site 1 <<
	.uleb128 .Ltmp0-.Lfunc_begin0           #   Call between .Lfunc_begin0 and .Ltmp0
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp0-.Lfunc_begin0           # >> Call Site 2 <<
	.uleb128 .Ltmp11-.Ltmp0                 #   Call between .Ltmp0 and .Ltmp11
	.uleb128 .Ltmp12-.Lfunc_begin0          #     jumps to .Ltmp12
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp11-.Lfunc_begin0          # >> Call Site 3 <<
	.uleb128 .Lfunc_end0-.Ltmp11            #   Call between .Ltmp11 and .Lfunc_end0
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
.Lcst_end0:
	.p2align	2
                                        # -- End function
	.section	.text.startup,"ax",@progbits
	.p2align	4, 0x90                         # -- Begin function _GLOBAL__sub_I_test1.cpp
	.type	_GLOBAL__sub_I_test1.cpp,@function
_GLOBAL__sub_I_test1.cpp:               # @_GLOBAL__sub_I_test1.cpp
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
	.size	_GLOBAL__sub_I_test1.cpp, .Lfunc_end1-_GLOBAL__sub_I_test1.cpp
	.cfi_endproc
                                        # -- End function
	.type	_ZStL8__ioinit,@object          # @_ZStL8__ioinit
	.local	_ZStL8__ioinit
	.comm	_ZStL8__ioinit,1,1
	.hidden	__dso_handle
	.type	.L.str,@object                  # @.str
	.section	.rodata.str1.1,"aMS",@progbits,1
.L.str:
	.asciz	"Elapsed execution time of the loop in test1():\n"
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
	.asciz	"case3.txt"
	.size	.L.str.4, 10

	.type	.L.str.5,@object                # @.str.5
.L.str.5:
	.asciz	"r == 0"
	.size	.L.str.5, 7

	.type	.L.str.6,@object                # @.str.6
.L.str.6:
	.asciz	"./fasttime.h"
	.size	.L.str.6, 13

	.type	.L__PRETTY_FUNCTION__._ZL7gettimev,@object # @__PRETTY_FUNCTION__._ZL7gettimev
.L__PRETTY_FUNCTION__._ZL7gettimev:
	.asciz	"fasttime_t gettime()"
	.size	.L__PRETTY_FUNCTION__._ZL7gettimev, 21

	.section	.init_array,"aw",@init_array
	.p2align	3
	.quad	_GLOBAL__sub_I_test1.cpp
	.ident	"Ubuntu clang version 11.0.0-++20200925013337+81eb1c1fa75-1~exp1~20200925114015.107"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __gxx_personality_v0
	.addrsig_sym _GLOBAL__sub_I_test1.cpp
	.addrsig_sym _Unwind_Resume
	.addrsig_sym _ZStL8__ioinit
	.addrsig_sym __dso_handle
	.addrsig_sym _ZSt4cout
