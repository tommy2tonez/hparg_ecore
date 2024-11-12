#define DEBUG_MODE_FLAG true

// #include "stdx.h"

#include <atomic>
#include <mutex>

    struct polymorphic_launderer{
        virtual auto ptr() volatile noexcept -> void * = 0;
    };

    template <class T>
    struct launderer{}; 

    template <>
    struct launderer<uint8_t>: polymorphic_launderer{
        uint8_t * volatile value;

        virtual auto ptr() volatile noexcept -> void *{
            return value;
        }
    };

    template <>
    struct launderer<uint16_t>: polymorphic_launderer{
        uint16_t * volatile value;

        virtual auto ptr() volatile noexcept -> void *{
            return value;
        }
    };

    template <>
    struct launderer<uint32_t>: polymorphic_launderer{
        uint32_t * volatile value;

        virtual auto ptr() volatile noexcept -> void *{
            return value;
        }
    };

    template <>
    struct launderer<uint64_t>: polymorphic_launderer{
        uint64_t * volatile value;

        virtual auto ptr() volatile noexcept -> void *{
            return value;
        }
    };

    template <>
    struct launderer<int8_t>: polymorphic_launderer{
        int8_t * volatile value;

        virtual auto ptr() volatile noexcept -> void *{
            return value;
        }
    };

    template <>
    struct launderer<int16_t>: polymorphic_launderer{
        int16_t * volatile value;

        virtual auto ptr() volatile noexcept -> void *{
            return value;
        }
    };

    template <>
    struct launderer<int32_t>: polymorphic_launderer{
        int32_t * volatile value;

        virtual auto ptr() volatile noexcept -> void *{
            return value;
        }
    };

    template <>
    struct launderer<int64_t>: polymorphic_launderer{
        int64_t * volatile value;

        virtual auto ptr() volatile noexcept -> void *{
            return value;
        }
    };

    template <>
    struct launderer<float>: polymorphic_launderer{
        float * volatile value;

        virtual auto ptr() volatile noexcept -> void *{
            return value;
        }
    };

    template <>
    struct launderer<double>: polymorphic_launderer{
        double * volatile value;

        virtual auto ptr() volatile noexcept -> void *{
            return value;
        }
    };

    template <>
    struct launderer<void>: polymorphic_launderer{
        void * volatile value;

        virtual auto ptr() volatile noexcept -> void *{
            return value;
        }
    };

    template <class T, class T1>
    inline __attribute__((always_inline)) auto launder_pointer(T1 * volatile ptr) noexcept -> T *{

        volatile launderer<T1> launder_machine;
        launder_machine.value = ptr;
        std::atomic_signal_fence(std::memory_order_seq_cst);
        void * clean_ptr = std::launder(static_cast<volatile polymorphic_launderer *>(&launder_machine))->ptr(); 

        return static_cast<T *>(clean_ptr);
    }

int main(){

    uint32_t * ptr{};
    auto other = launder_pointer<int>(ptr);
    // std::mutex lck{};
    // xlock_guard<std::mutex> lck_grd(lck);

    // {
    //     //transaction goes here
    // }

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