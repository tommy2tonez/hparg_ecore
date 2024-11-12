#define DEBUG_MODE_FLAG true

// #include "stdx.h"

#include <atomic>
#include <mutex>

    template <class Lock>
    class xlock_guard{};

    template <>
    class xlock_guard<std::atomic_flag>{

        private:

            std::atomic_flag * volatile mtx; 

        public:

            __attribute__((always_inline)) xlock_guard(std::atomic_flag& mtx) noexcept: mtx(&mtx){

                this->mtx->test_and_set();
                std::atomic_thread_fence(std::memory_order_seq_cst);
           }

            xlock_guard(const xlock_guard&) = delete;
            xlock_guard(xlock_guard&&) = delete;

            __attribute__((always_inline)) ~xlock_guard() noexcept{

                std::atomic_thread_fence(std::memory_order_seq_cst);
                this->mtx->clear();
            }

            xlock_guard& operator =(const xlock_guard&) = delete;
            xlock_guard& operator =(xlock_guard&&) = delete;
    };

    template <>
    class xlock_guard<std::mutex>{

        private:

            std::mutex * volatile mtx;
        
        public:

            __attribute__((always_inline)) xlock_guard(std::mutex& mtx) noexcept: mtx(&mtx){

                this->mtx->lock();
                std::atomic_thread_fence(std::memory_order_seq_cst);
            }

            xlock_guard(const xlock_guard&) = delete;
            xlock_guard(xlock_guard&&) = delete;

            __attribute__((always_inline)) ~xlock_guard() noexcept{

                std::atomic_thread_fence(std::memory_order_seq_cst);
                this->mtx->unlock();
            }

            xlock_guard& operator =(const xlock_guard&) = delete;
            xlock_guard& operator =(xlock_guard&&) = delete;
    };

int main(){

    std::mutex lck{};
    xlock_guard<std::mutex> lck_grd(lck);

    {
        //transaction goes here
    }

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