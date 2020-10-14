	.text
	.file	"main.cpp"
	.section	.rodata.cst4,"aM",@progbits,4
	.p2align	2                               # -- Begin function main
.LCPI0_0:
	.long	0x40800000                      # float 4
.LCPI0_1:
	.long	0x30000000                      # float 4.65661287E-10
.LCPI0_2:
	.long	0xbf800000                      # float -1
	.section	.rodata.cst8,"aM",@progbits,8
	.p2align	3
.LCPI0_3:
	.quad	0x4010000000000000              # double 4
.LCPI0_4:
	.quad	0x41dfffffffc00000              # double 2147483647
.LCPI0_5:
	.quad	0xbff0000000000000              # double -1
	.text
	.globl	main
	.p2align	4, 0x90
	.type	main,@function
main:                                   # @main
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	pushq	%r15
	.cfi_def_cfa_offset 24
	pushq	%r14
	.cfi_def_cfa_offset 32
	pushq	%r13
	.cfi_def_cfa_offset 40
	pushq	%r12
	.cfi_def_cfa_offset 48
	pushq	%rbx
	.cfi_def_cfa_offset 56
	subq	$24, %rsp
	.cfi_def_cfa_offset 80
	.cfi_offset %rbx, -56
	.cfi_offset %r12, -48
	.cfi_offset %r13, -40
	.cfi_offset %r14, -32
	.cfi_offset %r15, -24
	.cfi_offset %rbp, -16
	movq	%rsi, %r15
	movl	%edi, %ebp
	movl	$1024, %ebx                     # imm = 0x400
	movl	$1, %r14d
	.p2align	4, 0x90
.LBB0_1:                                # =>This Inner Loop Header: Depth=1
	movl	$.L.str.3, %edx
	movl	$_ZZ4mainE12long_options, %ecx
	movl	%ebp, %edi
	movq	%r15, %rsi
	xorl	%r8d, %r8d
	callq	getopt_long
	cmpl	$115, %eax
	jne	.LBB0_2
# %bb.6:                                #   in Loop: Header=BB0_1 Depth=1
	movq	optarg(%rip), %rdi
	xorl	%esi, %esi
	movl	$10, %edx
	callq	strtol
	movq	%rax, %rbx
	testl	%ebx, %ebx
	jg	.LBB0_1
	jmp	.LBB0_7
.LBB0_2:                                #   in Loop: Header=BB0_1 Depth=1
	cmpl	$-1, %eax
	je	.LBB0_10
# %bb.3:                                #   in Loop: Header=BB0_1 Depth=1
	cmpl	$116, %eax
	jne	.LBB0_9
# %bb.4:                                #   in Loop: Header=BB0_1 Depth=1
	movq	optarg(%rip), %rdi
	xorl	%esi, %esi
	movl	$10, %edx
	callq	strtol
	movq	%rax, %r14
	addl	$-1, %eax
	cmpl	$3, %eax
	jb	.LBB0_1
# %bb.5:
	movl	$_ZSt4cout, %edi
	movl	$.L.str.6, %esi
	movl	$11, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	movl	$_ZSt4cout, %edi
	movl	%r14d, %esi
	callq	_ZNSolsEi
	movl	$.L.str.7, %esi
	movl	$21, %edx
	jmp	.LBB0_8
.LBB0_7:
	movl	$_ZSt4cout, %edi
	movl	$.L.str.4, %esi
	movl	$30, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	movl	$_ZSt4cout, %edi
	movl	%ebx, %esi
	callq	_ZNSolsEi
	movl	$.L.str.5, %esi
	movl	$7, %edx
.LBB0_8:
	movq	%rax, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	movl	$-1, %eax
	jmp	.LBB0_20
.LBB0_10:
	movq	%r14, 16(%rsp)                  # 8-byte Spill
	movslq	%ebx, %r15
	movl	$4, %ecx
	movq	%r15, %rax
	mulq	%rcx
	movq	%rax, %rbp
	movq	$-1, %r14
	cmovoq	%r14, %rbp
	movl	$32, %esi
	movq	%rbp, %rdi
	callq	_ZnamSt11align_val_t
	movq	%rax, 8(%rsp)                   # 8-byte Spill
	movl	$32, %esi
	movq	%rbp, %rdi
	callq	_ZnamSt11align_val_t
	movq	%rax, %r13
	movl	$8, %ecx
	movq	%r15, %rax
	mulq	%rcx
	cmovnoq	%rax, %r14
	movl	$32, %esi
	movq	%r14, %rdi
	callq	_ZnamSt11align_val_t
	movq	%rax, %r14
	movl	$32, %esi
	movq	%rbp, %rdi
	callq	_ZnamSt11align_val_t
	movq	%rax, %r12
	testl	%r15d, %r15d
	je	.LBB0_13
# %bb.11:
	movl	%ebx, %ebp
	xorl	%r15d, %r15d
	.p2align	4, 0x90
.LBB0_12:                               # =>This Inner Loop Header: Depth=1
	callq	rand
	xorps	%xmm0, %xmm0
	cvtsi2ss	%eax, %xmm0
	movss	.LCPI0_0(%rip), %xmm1           # xmm1 = mem[0],zero,zero,zero
	mulss	%xmm1, %xmm0
	movss	.LCPI0_1(%rip), %xmm1           # xmm1 = mem[0],zero,zero,zero
	mulss	%xmm1, %xmm0
	movss	.LCPI0_2(%rip), %xmm1           # xmm1 = mem[0],zero,zero,zero
	addss	%xmm1, %xmm0
	movq	8(%rsp), %rax                   # 8-byte Reload
	movss	%xmm0, (%rax,%r15,4)
	callq	rand
	xorps	%xmm0, %xmm0
	cvtsi2ss	%eax, %xmm0
	mulss	.LCPI0_0(%rip), %xmm0
	mulss	.LCPI0_1(%rip), %xmm0
	addss	.LCPI0_2(%rip), %xmm0
	movss	%xmm0, (%r13,%r15,4)
	callq	rand
	xorps	%xmm0, %xmm0
	cvtsi2sd	%eax, %xmm0
	mulsd	.LCPI0_3(%rip), %xmm0
	divsd	.LCPI0_4(%rip), %xmm0
	addsd	.LCPI0_5(%rip), %xmm0
	movsd	%xmm0, (%r14,%r15,8)
	movl	$0, (%r12,%r15,4)
	addq	$1, %r15
	cmpq	%r15, %rbp
	jne	.LBB0_12
.LBB0_13:
	movq	%r12, (%rsp)                    # 8-byte Spill
	movq	8(%rsp), %r15                   # 8-byte Reload
	movl	$_ZSt4cout, %edi
	movl	$.L.str.8, %esi
	movl	$12, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	movl	$_ZSt4cout, %edi
	movq	16(%rsp), %rbp                  # 8-byte Reload
	movl	%ebp, %esi
	callq	_ZNSolsEi
	movl	$.L.str.9, %esi
	movl	$6, %edx
	movq	%rax, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	cmpl	$3, %ebp
	je	.LBB0_18
# %bb.14:
	cmpl	$2, %ebp
	je	.LBB0_17
# %bb.15:
	cmpl	$1, %ebp
	movq	(%rsp), %rbp                    # 8-byte Reload
	jne	.LBB0_19
# %bb.16:
	movq	%r15, %rdi
	movq	%r13, %rsi
	movq	%rbp, %rdx
	movl	%ebx, %ecx
	callq	_Z5test1PfS_S_i
	jmp	.LBB0_19
.LBB0_9:
	movq	(%r15), %rsi
	movl	$.L.str.10, %edi
	xorl	%eax, %eax
	callq	printf
	movl	$.Lstr, %edi
	callq	puts
	movl	$.Lstr.15, %edi
	callq	puts
	movl	$.Lstr.16, %edi
	callq	puts
	movl	$.Lstr.17, %edi
	callq	puts
	movl	$1, %eax
	jmp	.LBB0_20
.LBB0_18:
	movq	%r14, %rdi
	movl	%ebx, %esi
	callq	_Z5test3Pdi
	movq	(%rsp), %rbp                    # 8-byte Reload
	jmp	.LBB0_19
.LBB0_17:
	movq	%r15, %rdi
	movq	%r13, %rsi
	movq	(%rsp), %rbp                    # 8-byte Reload
	movq	%rbp, %rdx
	movl	%ebx, %ecx
	callq	_Z5test2PfS_S_i
.LBB0_19:
	movq	%r15, %rdi
	callq	_ZdaPv
	movq	%r13, %rdi
	callq	_ZdaPv
	movq	%r14, %rdi
	callq	_ZdaPv
	movq	%rbp, %rdi
	callq	_ZdaPv
	xorl	%eax, %eax
.LBB0_20:
	addq	$24, %rsp
	.cfi_def_cfa_offset 56
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%r12
	.cfi_def_cfa_offset 40
	popq	%r13
	.cfi_def_cfa_offset 32
	popq	%r14
	.cfi_def_cfa_offset 24
	popq	%r15
	.cfi_def_cfa_offset 16
	popq	%rbp
	.cfi_def_cfa_offset 8
	retq
.Lfunc_end0:
	.size	main, .Lfunc_end0-main
	.cfi_endproc
                                        # -- End function
	.globl	_Z5usagePKc                     # -- Begin function _Z5usagePKc
	.p2align	4, 0x90
	.type	_Z5usagePKc,@function
_Z5usagePKc:                            # @_Z5usagePKc
	.cfi_startproc
# %bb.0:
	pushq	%rax
	.cfi_def_cfa_offset 16
	movq	%rdi, %rsi
	movl	$.L.str.10, %edi
	xorl	%eax, %eax
	callq	printf
	movl	$.Lstr, %edi
	callq	puts
	movl	$.Lstr.15, %edi
	callq	puts
	movl	$.Lstr.16, %edi
	callq	puts
	movl	$.Lstr.17, %edi
	popq	%rax
	.cfi_def_cfa_offset 8
	jmp	puts                            # TAILCALL
.Lfunc_end1:
	.size	_Z5usagePKc, .Lfunc_end1-_Z5usagePKc
	.cfi_endproc
                                        # -- End function
	.section	.rodata.cst4,"aM",@progbits,4
	.p2align	2                               # -- Begin function _Z9initValuePfS_PdS_j
.LCPI2_0:
	.long	0x40800000                      # float 4
.LCPI2_1:
	.long	0x30000000                      # float 4.65661287E-10
.LCPI2_2:
	.long	0xbf800000                      # float -1
	.section	.rodata.cst8,"aM",@progbits,8
	.p2align	3
.LCPI2_3:
	.quad	0x4010000000000000              # double 4
.LCPI2_4:
	.quad	0x41dfffffffc00000              # double 2147483647
.LCPI2_5:
	.quad	0xbff0000000000000              # double -1
	.text
	.globl	_Z9initValuePfS_PdS_j
	.p2align	4, 0x90
	.type	_Z9initValuePfS_PdS_j,@function
_Z9initValuePfS_PdS_j:                  # @_Z9initValuePfS_PdS_j
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	pushq	%r15
	.cfi_def_cfa_offset 24
	pushq	%r14
	.cfi_def_cfa_offset 32
	pushq	%r13
	.cfi_def_cfa_offset 40
	pushq	%r12
	.cfi_def_cfa_offset 48
	pushq	%rbx
	.cfi_def_cfa_offset 56
	pushq	%rax
	.cfi_def_cfa_offset 64
	.cfi_offset %rbx, -56
	.cfi_offset %r12, -48
	.cfi_offset %r13, -40
	.cfi_offset %r14, -32
	.cfi_offset %r15, -24
	.cfi_offset %rbp, -16
	testl	%r8d, %r8d
	je	.LBB2_3
# %bb.1:
	movq	%rcx, %r14
	movq	%rdx, %r15
	movq	%rsi, %r12
	movq	%rdi, %r13
	movl	%r8d, %ebx
	xorl	%ebp, %ebp
	.p2align	4, 0x90
.LBB2_2:                                # =>This Inner Loop Header: Depth=1
	callq	rand
	xorps	%xmm0, %xmm0
	cvtsi2ss	%eax, %xmm0
	movss	.LCPI2_0(%rip), %xmm1           # xmm1 = mem[0],zero,zero,zero
	mulss	%xmm1, %xmm0
	movss	.LCPI2_1(%rip), %xmm1           # xmm1 = mem[0],zero,zero,zero
	mulss	%xmm1, %xmm0
	movss	.LCPI2_2(%rip), %xmm1           # xmm1 = mem[0],zero,zero,zero
	addss	%xmm1, %xmm0
	movss	%xmm0, (%r13,%rbp,4)
	callq	rand
	xorps	%xmm0, %xmm0
	cvtsi2ss	%eax, %xmm0
	mulss	.LCPI2_0(%rip), %xmm0
	mulss	.LCPI2_1(%rip), %xmm0
	addss	.LCPI2_2(%rip), %xmm0
	movss	%xmm0, (%r12,%rbp,4)
	callq	rand
	xorps	%xmm0, %xmm0
	cvtsi2sd	%eax, %xmm0
	mulsd	.LCPI2_3(%rip), %xmm0
	divsd	.LCPI2_4(%rip), %xmm0
	addsd	.LCPI2_5(%rip), %xmm0
	movsd	%xmm0, (%r15,%rbp,8)
	movl	$0, (%r14,%rbp,4)
	addq	$1, %rbp
	cmpq	%rbp, %rbx
	jne	.LBB2_2
.LBB2_3:
	addq	$8, %rsp
	.cfi_def_cfa_offset 56
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%r12
	.cfi_def_cfa_offset 40
	popq	%r13
	.cfi_def_cfa_offset 32
	popq	%r14
	.cfi_def_cfa_offset 24
	popq	%r15
	.cfi_def_cfa_offset 16
	popq	%rbp
	.cfi_def_cfa_offset 8
	retq
.Lfunc_end2:
	.size	_Z9initValuePfS_PdS_j, .Lfunc_end2-_Z9initValuePfS_PdS_j
	.cfi_endproc
                                        # -- End function
	.section	.text.startup,"ax",@progbits
	.p2align	4, 0x90                         # -- Begin function _GLOBAL__sub_I_main.cpp
	.type	_GLOBAL__sub_I_main.cpp,@function
_GLOBAL__sub_I_main.cpp:                # @_GLOBAL__sub_I_main.cpp
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
.Lfunc_end3:
	.size	_GLOBAL__sub_I_main.cpp, .Lfunc_end3-_GLOBAL__sub_I_main.cpp
	.cfi_endproc
                                        # -- End function
	.type	_ZStL8__ioinit,@object          # @_ZStL8__ioinit
	.local	_ZStL8__ioinit
	.comm	_ZStL8__ioinit,1,1
	.hidden	__dso_handle
	.type	_ZZ4mainE12long_options,@object # @_ZZ4mainE12long_options
	.data
	.p2align	4
_ZZ4mainE12long_options:
	.quad	.L.str
	.long	1                               # 0x1
	.zero	4
	.quad	0
	.long	115                             # 0x73
	.zero	4
	.quad	.L.str.1
	.long	1                               # 0x1
	.zero	4
	.quad	0
	.long	116                             # 0x74
	.zero	4
	.quad	.L.str.2
	.long	0                               # 0x0
	.zero	4
	.quad	0
	.long	63                              # 0x3f
	.zero	4
	.zero	32
	.size	_ZZ4mainE12long_options, 128

	.type	.L.str,@object                  # @.str
	.section	.rodata.str1.1,"aMS",@progbits,1
.L.str:
	.asciz	"size"
	.size	.L.str, 5

	.type	.L.str.1,@object                # @.str.1
.L.str.1:
	.asciz	"test"
	.size	.L.str.1, 5

	.type	.L.str.2,@object                # @.str.2
.L.str.2:
	.asciz	"help"
	.size	.L.str.2, 5

	.type	.L.str.3,@object                # @.str.3
.L.str.3:
	.asciz	"st:?"
	.size	.L.str.3, 5

	.type	.L.str.4,@object                # @.str.4
.L.str.4:
	.asciz	"Error: Workload size is set to"
	.size	.L.str.4, 31

	.type	.L.str.5,@object                # @.str.5
.L.str.5:
	.asciz	" (<0).\n"
	.size	.L.str.5, 8

	.type	.L.str.6,@object                # @.str.6
.L.str.6:
	.asciz	"Error: test"
	.size	.L.str.6, 12

	.type	.L.str.7,@object                # @.str.7
.L.str.7:
	.asciz	"() is not available.\n"
	.size	.L.str.7, 22

	.type	.L.str.8,@object                # @.str.8
.L.str.8:
	.asciz	"Running test"
	.size	.L.str.8, 13

	.type	.L.str.9,@object                # @.str.9
.L.str.9:
	.asciz	"()...\n"
	.size	.L.str.9, 7

	.type	.L.str.10,@object               # @.str.10
.L.str.10:
	.asciz	"Usage: %s [options]\n"
	.size	.L.str.10, 21

	.section	.init_array,"aw",@init_array
	.p2align	3
	.quad	_GLOBAL__sub_I_main.cpp
	.type	.Lstr,@object                   # @str
	.section	.rodata.str1.1,"aMS",@progbits,1
.Lstr:
	.asciz	"Program Options:"
	.size	.Lstr, 17

	.type	.Lstr.15,@object                # @str.15
.Lstr.15:
	.asciz	"  -s  --size <N>     Use workload size N (Default = 1024)"
	.size	.Lstr.15, 58

	.type	.Lstr.16,@object                # @str.16
.Lstr.16:
	.asciz	"  -t  --test <N>     Just run the testN function (Default = 1)"
	.size	.Lstr.16, 63

	.type	.Lstr.17,@object                # @str.17
.Lstr.17:
	.asciz	"  -h  --help         This message"
	.size	.Lstr.17, 34

	.ident	"Ubuntu clang version 11.0.0-++20200925013337+81eb1c1fa75-1~exp1~20200925114015.107"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym _GLOBAL__sub_I_main.cpp
	.addrsig_sym _ZStL8__ioinit
	.addrsig_sym __dso_handle
	.addrsig_sym _ZZ4mainE12long_options
	.addrsig_sym _ZSt4cout
