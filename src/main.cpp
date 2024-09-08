#include <unistd.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <stdio.h>
#include <iostream>
#include <tuple>

template <class ...Args>
void bar(Args ...){

}

template <class ...Args, class ...AArgs>
void foo(std::tuple<Args...>, std::tuple<AArgs...>){

    bar(Args()..., AArgs()...);
}

int main(){
    
    foo(std::tuple<int, int>{}, std::tuple<int, int>{});
}