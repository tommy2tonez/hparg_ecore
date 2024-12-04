#include <iostream>
#include "dg_map_variants.h"
#include "dense_hash_map/dense_hash_map.hpp"
#include <random>
#include <utility>
#include <algorithm>
#include <chrono>
#include <vector>
#include <functional>
#include <unordered_map>
#include "test_map.h"

struct NullKeyGen{

    constexpr auto operator()() -> std::pair<size_t,size_t>{
        return {std::numeric_limits<size_t>::max(), {}};
    }
};

int main(){

    using namespace std::chrono;

    //alright guys - let's talk about financial transactions
    //assume, for now, that we have a radix tree of N descendants
    //every guy in the tree has his own version of correct financial graph
    //everytime we do a transaction - we send the event to a random guy in the tree
    //so graph synchronization is done by dimensional reduction, from descendants to ancestors
    //for N, set_or(graphs) - produce correct result - push upward to ancestor - otherwise, resolute and intersect all the ideas from all the N graphs to get the correct graph - push to ancestor
    //assume we are at root, root is where all the "financial_graph_variants" are resolved - and now we want to push our "newly_synchronous_graph" to all descedants - and now we have a financial system that's woking - fine

    //what does this radix tree have anything to do with synchronous brain?

    //assume this cycle of synchronous brain, a -> b -> c -> d -> a
    //assume another cycle of synchronous brain, b -> q -> w -> e -> r -> t -> y -> b

    //let a -> b -> c -> d -> a be a radix tree of N = 3, and root = a 
    //let b -> q -> w -> e -> r -> t -> y -> b be a radix tree of N = 6, and root = b

    //so a transaction - accorading to our theorem, must be completed within 1 global cycle of synchronous brain
    //we are assuming a functioning, working brain

    //our synchronous brain set up is very simple, we have rings of tiles
    //each ring of tile is actually a tile of different versions
    //WLOG, without synchronous brain, we have a brain B = set(logit_1, logit_2, logit_3, logit_4)

    //with synchronous brain, we have 2 brains B    = set(logit1_1, logit1_2, logit1_3, logit1_4)
    //                                         B1   = set(logit2_1, logit2_2, logit2_3, logit2_4)

    //ring1 = (logit1_1, logit2_1)
    //ring2 = (logit1_2, logit2_2)
    //etc.

    //backprop is the main way of communication - it's exactly how our brain works - we see things (inputs) - we correct things (backprop) and we acknowledge things

    //okay, so this is not just about finanical transactions but probably about every type of transactions
    //the idea is very simple - but to put this to practice - I'm saying probably 3-5 years
    //we need to have a working brain - we need to "battle test" the brain - to make sure that it is very very accurate
    //we need a system that runs on every device
    //and we need to get the engineering right
}

