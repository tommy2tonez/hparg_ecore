#define DEBUG_MODE_FLAG false
#include <unordered_set>

template <class ...Args>
void foo(Args ...args){

    std::unordered_set<size_t> set{};

    (set.insert(args), ...);
    bool flag = (set.contains(args) || ...);
}

int main(){

    //alrights fellas - we implemented relaxed memory ordering on memregions, we want to use this in conjunction with _mm_pause() because that's a virtue - which is the BASE of every uma_ptr_t operations - so there would be no issue with spinlocks destroying hardware performance
    //we've reached 5MBs of raw code fellas - that's a major milestone
}