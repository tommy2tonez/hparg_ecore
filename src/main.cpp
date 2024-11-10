#include <stdint.h>
#include <stdlib.h>
#include <iostream>
#include <utility>
#include <memory>
#include <cstring>
#include <atomic>
#include <thread>
#include <mutex>

static inline std::atomic<int *> my_memory_barrier{};
static inline std::mutex mtx{};
static inline int other = 2;

void test(){

    // auto lck_grd = std::lock_guard<std::mutex>(mtx);
    my_memory_barrier.exchange(&other);
}

    template <class T>
    inline auto launder_pointer(void * ptr) noexcept -> T *{

        std::atomic_signal_fence(std::memory_order_seq_cst);
        return static_cast<T *>(*std::launder(&ptr));
    }

    template <class T>
    inline auto launder_pointer(const void * ptr) noexcept -> const T *{

        std::atomic_signal_fence(std::memory_order_seq_cst);
        return static_cast<const T *>(*std::launder(&ptr));
    }


int main(){

    int inp{};
    std::cin >> inp;
    int new_rs{};
    
    std::memcpy(&new_rs, launder_pointer<int>(&inp), sizeof(int)); //this is a very fragile laundering - the ONLY compiler that's actually being reasonable is gcc - yeah - surprise

    int total = inp * new_rs;
    std::cout << total;
}