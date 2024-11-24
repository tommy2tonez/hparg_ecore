#define DEBUG_MODE_FLAG true 

#include "network_memlock.h"

int main(){

    //most of the time - you MUST use mutual exclusive lock to do current transactions - because the current STD does not guarantee defined behaviors for your program if you decide to "invent your own atomic operations" - it's silly - the entire memory ordering thing - like it's written by some high-schooler on heroin or sth
    //and your mutual exclusive lock operation (try_lock, acquire_lock, release_lock) is relaxed with respect to the calling function
    //and post the mutual excluive lock - you want std::atomic_thread_fence(std::memory_order_seq_cst)
    //and pre releasing the lock - you also want std::atomic_thread_fence(std::memory_order_seq_cst)
    //and the std::atomic_thread_fence() must be able to see (force_inline) your concurrent block transaction - otherwise - you are, again, not protected by the compiler-std, but protected by the std-std
    //this is what 20 years of atomic operations summarized for you guys - yeah - it's that - implement things that is proven to work
    //you will be surprised by the number of softwares that invoke undefined behaviors by "simply trying to be smart" - even guys at big techs
    //you are smart, you adhere to the std-std - but there is more to life than that - there is also a compiler-std and it has a completely different set of rules and duct tapes
}
