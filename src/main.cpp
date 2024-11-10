#include <stdint.h>
#include <stdlib.h>
#include <iostream>
#include <utility>
#include <memory>
#include <cstring>
#include <atomic>
#include <thread>
#include <mutex>


template <class T, std::enable_if_t<std::is_fundamental_v<T>, bool> = true>
inline auto launder_pointer(void * volatile ptr) noexcept -> T *{

    std::atomic_signal_fence(std::memory_order_seq_cst);
    return static_cast<T *>(*std::launder(&ptr));
}

template <class T, std::enable_if_t<std::is_fundamental_v<T>, bool> = true>
inline auto launder_pointer(const void * volatile ptr) noexcept -> const T *{

    std::atomic_signal_fence(std::memory_order_seq_cst);
    return static_cast<const T *>(*std::launder(&ptr));
}

int main(){

    int inp{};
    std::cin >> inp;
    int new_rs{};

    std::memcpy(&new_rs, launder_pointer<int>(static_cast<const int *>(&inp)), sizeof(int)); //this is a very fragile laundering - the ONLY compiler that's actually being reasonable is gcc - yeah - surprise

    int total = inp * new_rs;
    std::cout << total;
}