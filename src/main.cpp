#define DEBUG_MODE_FLAG false
#include <unordered_set>
#include <atomic>
#include <memory>
#include <chrono>

template <class ...Args>
void foo(Args ...args){

    std::unordered_set<size_t> set{};

    (set.insert(args), ...);
    bool flag = (set.contains(args) || ...);
}

int main(){

}