#define DEBUG_MODE_FLAG false
#include <unordered_set>
#include <atomic>
#include <memory>
#include <chrono>
#include <iterator>

template <class ...Args>
void foo(Args ...args){

    std::unordered_set<size_t> set{};

    (set.insert(args), ...);
    bool flag = (set.contains(args) || ...);
}

int main(){

    size_t a = 0;
    std::move_iterator<size_t *> b = std::make_move_iterator(&a);
    size_t c = b[0];

    //we programmers only need one Hello
}