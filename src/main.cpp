#include <iostream>
#include <type_traits>
#include <utility>
#include <tuple>
// #include "network_memlock.h"
#include <fstream>
#include <memory>

template <class T>
struct type_reduction{
    using type = T;
};

template <size_t VALUE>
struct type_reduction<std::integral_constant<size_t, VALUE>>{
    static inline constexpr size_t type = VALUE;
};

template <size_t First, size_t Second>
struct Bar{};

template <class ...Args>
struct tags{};

template <class ...Args>
void foo(tags<Args...>){

    Bar<typename type_reduction<Args>::type...>{};
}

int main(){

    foo(tags<std::integral_constant<size_t, 2>, std::integral_constant<size_t, 1>>{});
    // if (int i = 0; i){
        
    // }

    // std::cout << istr << std::endl;
}