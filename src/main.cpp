#include <type_traits>
#include <tuple>
#include <optional>
#include <iostream>
#include <memory>

void fooo(int *){

}

auto make(){

    
    int * i = new int;
    return std::unique_ptr<int, void (*)(int *)>(i, fooo);
}
int main(){

    decltype(fooo) obj = fooo;

    auto ptr = make();
    auto ptr2 = std::move(ptr);
    // std::unique_ptr<int, decltype(foo)> handle;
}