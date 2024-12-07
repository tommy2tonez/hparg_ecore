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

    //alrights - I've been thinking about pathing problems - and model training problems
    //there are two types of model trainers:
    //the loss-function absolutists and the model absolutists
    //I'm the latter - I think (hint - I dont think - I know) a good balanced, symmetric model will cancel out the loss_rate benefits from the dynamic loss function - and <loss_functor> methods like AdamW, SGD, cosine anneal, etc. aren't quantifiable

    //there is no good answer in pathing - the best we can do is have a small neural networks of size 20-30 hops - then train them concurrently - pick the one that has the least loss_rate and validation_rate (compared to the same weight models)
    //what we wanna do is to saturate the model - then we gonna unlink the logit tiles - by using detach_clone tiles - and we rinse and repeats   
    //so f(x) -> y, and F(f(x), g(x)) -> y is the answer to the rotor | gear problem which states that the deeper the neural network - the harder it is to train the neural network
    //we'll build the ring of compression

    //ring of compression is easy f(g(x)) - > x - 20 hops (dijkstra + path + etc. to permute all possible transformation paths - pacm | uacm | pair | mono)
    //f'(g'(g(x))) -> g(x) - 20 hops
    //so on and so forth
    //we want to train all the networks CONCURRENTLY (I'm talking about 50 TBs concurrent training - in one computation node - or petabytes of concurrent training in multiple nodes) - and extract the derivative of gradients
}