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

    //today I'm going to write a concurrency tutorial ONCE, for all the noobs out there to understand memory orderings, which you guys have misunderstood your entire life
    //every function in C++ is eventually relaxed, if it is not relaxed, then it is wrong (we aren't talking about __force_inline__ which is not a function)

    //consider this program

    int a = 1;
    int b = 2;
    int c = a + b;

    //consider this program

    std::atomic<int> a = 1;
    std::atomic<int> b = 2;
    std::atomic<int> c = a.load(std::memory_order_relaxed) + b.load(std::memory_order_relaxed);

    //consider this program

    std::atomic<int> a = 1;
    int b = 2;
    int c = a.load(std::memory_order_relaxed) + b;

    //these three programs are equivalent

    //we only use memory ordering IF the atomic variable denotes the beginning of a concurrent transaction, marked by acquire, and end the tranasaction with memory order release
    //the only defined use case of memory ordering is precisely that - for concurrent transaction, serialized accesses 
    //when use memory ordering - there are two things we need to consider - first, the hardware memory ordering, and the compilation ordering
    //std::memory_order_acquire and std::memory_order_release at the beginning and the end of the transaction is a sufficient hardware instruction but not complication instruction
    //in order for that to be a sufficient complication instruction - either std::atomic_signal_fence(std::memory_order_acquire) and std::atomic_signal_fence(std::memory_order_release) must be in the same scope of the transaction - or the atomic variable memory orderings must be in the same scope of the instructions
}