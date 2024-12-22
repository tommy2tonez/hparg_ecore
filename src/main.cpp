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
    //alrights - if yall have any pull requests regarding the core - now is the time to speak - otherwise we are doing N = 10^6 x N = 10^6 matmul next week by using pair accum + pair - the space complexity is 10^12 - time complexity is 10 ** 18
    //we are hitting that 500TBs cuda flops/ core*s + 12GBs host flops/ core*s - full network bandwidth 32 concurrent outbound sockets - AWS test 
    //we got 1 pull request of making memlock_reference -> relaxed + memlock_reference on all the mutating regions 
}