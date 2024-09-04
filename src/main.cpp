#include <type_traits>
#include <tuple>
#include <optional>
#include <iostream>
#include <memory>

struct Bar{
    int i;
};

int main(){

    std::optional<Bar> bar{};

    bar->i;
    // std::unique_ptr<int, decltype(foo)> handle;
}