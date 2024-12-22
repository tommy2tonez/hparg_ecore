#define DEBUG_MODE_FLAG false
#include <unordered_set>

template <class ...Args>
void foo(Args ...args){

    std::unordered_set<size_t> set{};

    (set.insert(args), ...);
    bool flag = (set.contains(args) || ...);
}

int main(){

    foo(size_t{1}, size_t{2}, size_t{3});
    //alrights - let's have a demo for 10**6 x 10**6 linear in a week

}