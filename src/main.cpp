#define DEBUG_MODE_FLAG true

#include "stdx.h"

int main(){

    //let's say assume that we are in a seq_cst transaction block
    //if a function is inlinable - fine
    //if a function is not inlinable (for whatever reason) - if the function is stateless - such that its computation result is solely depended on the arguments - fine
    //                                                     - if the function is stateful - such that its computation result is not solely depended on the arguments and such dependencies are concurrently mutable - stdx::seq_cst_guard() is required
    
    //<seq_cst> <transaction_block> <seq_cst> - first and last <seq_cst> have to have lower scope idx - w.r.t. transaction block - this is gcc implementation - I dont know about other compilers
    //always use seq_cst - especially std::atomic_signal_fence(std::memory_order_seq_cst) - it works - and it does not produce any extra cpu instructions
    //always put your concurrent memory_transaction inside a stdx::lock_guard() block - always use std::mutex or std::atomic_flag for mutual exclusion, you don't want that atomic_operations or lock-free programming - trust me - it's a crippled child of std::mutex and std::atomic_flag
    //not because you won't implement it logically correct (or std-correct) - but you won't implement it compilerly correct
    //it's hard to program in C++ - so better to stick to the ways that actually works
    //most importantly, don't try to be smart - chances are 99% of the time you don't know shit
}