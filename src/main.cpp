#include <stdint.h>
#include <stdlib.h>
#include <iostream>
#include <utility>

static inline size_t var_1  = 1;
static inline size_t var_2  = 2;
static inline size_t var_3  = 3;
static inline size_t var_4  = 4;
static inline size_t var_5  = 5;

auto test(size_t dispatch_code) noexcept -> size_t{

    switch (dispatch_code){
        case 1:
            return var_1 + 2;
        case 2:
            return var_2 + 2;
        case 3:
            return var_3 + 3;
        case 4:
            return var_4 + 4;
        case 5:
            return var_5 + 5;
        default:
            std::unreachable();
    }
} 

int main(){

    size_t dispatch_code{};
    std::cin >> dispatch_code;
    std::cin >> var_1;
    std::cin >> var_2;
    std::cin >> var_3;
    std::cin >> var_4;
    std::cin >> var_5;
    std::cout << test(dispatch_code);
}

/*
clang generated

test(unsigned int):
        mvn     r1, #3
        add     r0, r1, r0, lsl #2
        ldr     r1, .LCPI0_0
.LPC0_0:
        add     r1, pc, r1
        ldr     r1, [r1, r0]
        ldr     r2, .LCPI0_1
.LPC0_1:
        add     r2, pc, r2
        ldr     r0, [r2, r0]
        ldr     r1, [r1]
        add     r0, r1, r0
        bx      lr
.LCPI0_0:
        .long   .Lswitch.table.main-(.LPC0_0+8)
.LCPI0_1:
        .long   .Lswitch.table.main.2-(.LPC0_1+8)

main:
        push    {r4, r5, r11, lr}
        sub     sp, sp, #8
        mov     r0, #0
        str     r0, [sp, #4]
        ldr     r4, .LCPI1_0
        add     r1, sp, #4
.LPC1_0:
        ldr     r4, [pc, r4]
        mov     r0, r4
        bl      std::istream& std::istream::_M_extract<unsigned int>(unsigned int&)
        ldr     r5, .LCPI1_1
        mov     r0, r4
.LPC1_1:
        add     r5, pc, r5
        mov     r1, r5
        bl      std::istream& std::istream::_M_extract<unsigned int>(unsigned int&)
        add     r1, r5, #4
        mov     r0, r4
        bl      std::istream& std::istream::_M_extract<unsigned int>(unsigned int&)
        add     r1, r5, #8
        mov     r0, r4
        bl      std::istream& std::istream::_M_extract<unsigned int>(unsigned int&)
        add     r1, r5, #12
        mov     r0, r4
        bl      std::istream& std::istream::_M_extract<unsigned int>(unsigned int&)
        add     r1, r5, #16
        mov     r0, r4
        bl      std::istream& std::istream::_M_extract<unsigned int>(unsigned int&)
        ldr     r0, [sp, #4]
        mvn     r1, #3
        add     r0, r1, r0, lsl #2
        ldr     r1, .LCPI1_2
.LPC1_2:
        add     r1, pc, r1
        ldr     r1, [r1, r0]
        ldr     r2, .LCPI1_3
.LPC1_3:
        add     r2, pc, r2
        ldr     r0, [r2, r0]
        ldr     r1, [r1]
        add     r1, r1, r0
        ldr     r0, .LCPI1_4
.LPC1_4:
        ldr     r0, [pc, r0]
        bl      std::ostream& std::ostream::_M_insert<unsigned long>(unsigned long)
        mov     r0, #0
        add     sp, sp, #8
        pop     {r4, r5, r11, pc}
.LCPI1_0:
.Ltmp26:
        .long   _ZSt3cin(GOT_PREL)-((.LPC1_0+8)-.Ltmp26)
.LCPI1_1:
        .long   .L_MergedGlobals-(.LPC1_1+8)
.LCPI1_2:
        .long   .Lswitch.table.main-(.LPC1_2+8)
.LCPI1_3:
        .long   .Lswitch.table.main.2-(.LPC1_3+8)
.LCPI1_4:
.Ltmp27:
        .long   _ZSt4cout(GOT_PREL)-((.LPC1_4+8)-.Ltmp27)

.Lswitch.table.main:
        .long   .L_MergedGlobals
        .long   .L_MergedGlobals+4
        .long   .L_MergedGlobals+8
        .long   .L_MergedGlobals+12
        .long   .L_MergedGlobals+16

.Lswitch.table.main.2:
        .long   2
        .long   2
        .long   3
        .long   4
        .long   5

.L_MergedGlobals:
        .long   1
        .long   2
        .long   3
        .long   4
        .long   5
*/

/*

g++-14 -std=c++23 -O3 generated

test(unsigned long):
        cmp     x0, 3
        beq     .L2
        bls     .L12
        cmp     x0, 4
        beq     .L6
        adrp    x0, .LANCHOR0+32
        ldr     x0, [x0, #:lo12:.LANCHOR0+32]
        add     x0, x0, 5
        ret
.L12:
        cmp     x0, 1
        beq     .L4
        adrp    x0, .LANCHOR0+8
        ldr     x0, [x0, #:lo12:.LANCHOR0+8]
        add     x0, x0, 2
        ret
.L4:
        adrp    x0, .LANCHOR0
        ldr     x0, [x0, #:lo12:.LANCHOR0]
        add     x0, x0, 2
        ret
.L2:
        adrp    x0, .LANCHOR0+16
        ldr     x0, [x0, #:lo12:.LANCHOR0+16]
        add     x0, x0, 3
        ret
.L6:
        adrp    x0, .LANCHOR0+24
        ldr     x0, [x0, #:lo12:.LANCHOR0+24]
        add     x0, x0, 4
        ret
main:
        stp     x29, x30, [sp, -48]!
        mov     x29, sp
        add     x1, sp, 40
        stp     x19, x20, [sp, 16]
        adrp    x19, _ZSt3cin
        add     x19, x19, :lo12:_ZSt3cin
        mov     x0, x19
        adrp    x20, .LANCHOR0
        add     x20, x20, :lo12:.LANCHOR0
        str     xzr, [sp, 40]
        bl      std::basic_istream<char, std::char_traits<char> >& std::basic_istream<char, std::char_traits<char> >::_M_extract<unsigned long>(unsigned long&)
        mov     x1, x20
        mov     x0, x19
        bl      std::basic_istream<char, std::char_traits<char> >& std::basic_istream<char, std::char_traits<char> >::_M_extract<unsigned long>(unsigned long&)
        add     x1, x20, 8
        mov     x0, x19
        bl      std::basic_istream<char, std::char_traits<char> >& std::basic_istream<char, std::char_traits<char> >::_M_extract<unsigned long>(unsigned long&)
        add     x1, x20, 16
        mov     x0, x19
        bl      std::basic_istream<char, std::char_traits<char> >& std::basic_istream<char, std::char_traits<char> >::_M_extract<unsigned long>(unsigned long&)
        add     x1, x20, 24
        mov     x0, x19
        bl      std::basic_istream<char, std::char_traits<char> >& std::basic_istream<char, std::char_traits<char> >::_M_extract<unsigned long>(unsigned long&)
        add     x1, x20, 32
        mov     x0, x19
        bl      std::basic_istream<char, std::char_traits<char> >& std::basic_istream<char, std::char_traits<char> >::_M_extract<unsigned long>(unsigned long&)
        ldr     x0, [sp, 40]
        bl      test(unsigned long)
        mov     x1, x0
        adrp    x0, _ZSt4cout
        add     x0, x0, :lo12:_ZSt4cout
        bl      std::basic_ostream<char, std::char_traits<char> >& std::basic_ostream<char, std::char_traits<char> >::_M_insert<unsigned long>(unsigned long)
        mov     w0, 0
        ldp     x19, x20, [sp, 16]
        ldp     x29, x30, [sp], 48
        ret
*/