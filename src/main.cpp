#define DEBUG_MODE_FLAG true
#define STRONG_MEMORY_ORDERING_FLAG true

// #include "network_memlock.h"
// #include "network_memlock_proxyspin.h"
#include <atomic>
#include <random>
#include <memory>
#include <functional>
#include <utility>
#include <algorithm>
#include <vector>
#include <chrono>
#include <iostream>
#include "assert.h"
// #include "stdx.h"
#include "sort_variants.h"

//let's strategize
//what do we have

//collapse the wall (cand reduction optimization)
//block quicksort (branching optimization)
//insertion iteration up to n flops sort
//find minimum pivot greater than (by leveraging the back index of the insertion sort)
//pivoting the remaining array + continue the quicksort

template <class Task>
auto timeit(Task task) -> size_t{

    auto then = std::chrono::high_resolution_clock::now();
    task();
    auto now = std::chrono::high_resolution_clock::now();

    return std::chrono::duration_cast<std::chrono::milliseconds>(now - then).count();
} 

int main(){

    const size_t SZ = size_t{1} << 26;

    std::vector<uint32_t> vec(SZ);
    std::generate(vec.begin(), vec.end(), std::bind(std::uniform_int_distribution<uint32_t>{}, std::mt19937{}));

    std::vector<uint32_t> vec2 = vec;
    std::vector<uint32_t> vec3 = vec;

    std::cout << "<insertion_sort_1>" << timeit([&]{std::sort(vec.data(), std::next(vec.data(), vec.size()));}) << "<ms>" << std::endl;
    std::cout << "<insertion_sort_2>" << timeit([&]{dg::sort_variants::quicksort::quicksort(vec2.data(), std::next(vec2.data(), vec2.size()));}) << "<ms>" << std::endl;

    std::cout << "<insertion_sort_1>" << timeit([&]{std::sort(vec.data(), std::next(vec.data(), vec.size()));}) << "<ms>" << std::endl;
    std::cout << "<insertion_sort_2>" << timeit([&]{dg::sort_variants::quicksort::quicksort(vec2.data(), std::next(vec2.data(), vec2.size()));}) << "<ms>" << std::endl;

    // std::cout << "<insertion_sort_1>" << timeit([&]{std::sort(vec.data(), std::next(vec.data(), vec.size()));}) << "<ms>" << std::endl;
    // std::cout << "<insertion_sort_2>" << timeit([&]{dg::sort_variants::quicksort::quicksort(vec2.data(), std::next(vec2.data(), vec2.size()));}) << "<ms>" << std::endl;

    // std::cout << "<insertion_sort_1>" << timeit([&]{std::sort(vec.data(), std::next(vec.data(), vec.size()));}) << "<ms>" << std::endl;
    // std::cout << "<insertion_sort_2>" << timeit([&]{dg::sort_variants::quicksort::quicksort(vec2.data(), std::next(vec2.data(), vec2.size()));}) << "<ms>" << std::endl;

    // for (size_t i = 0u; i < vec2.size(); ++i){
    //     if (vec[i] != vec2[i]){
    //         std::cout << i << "<>" << vec[i] << "<>" << vec2[i] << std::endl;
    //     }
    // }
    // for (uint32_t e: vec2){
        // std::cout << e << std::endl;
    // }
    // std::sort(vec3.begin(), vec3.end());

    // std::cout << dg::sort_variants::quicksort::counter;

    assert(vec == vec2);

    // const size_t SZ_BITSET = size_t{1} << 30;

    // std::vector<uint8_t> bitset_vec(SZ_BITSET);
    // std::generate(bitset_vec.begin(), bitset_vec.end(), std::bind(std::uniform_int_distribution<uint8_t>{}, std::mt19937{}));
    // uint64_t total = 0u; 

    // auto task = [&]{
    //     for (uint8_t bitset: bitset_vec){
    //         total += bitset;
    //     }
    // };

    // std::cout << "<countr_zero>" << timeit(task) << "<ms>" << "<>" << total << std::endl; 

    // std::cout << dg::sort_variants::quicksort::COUNTING_SZ << std::endl; 

    // assert(vec2 == vec3);

    //our agenda today is to work on the frame + resolutor + quicksort (to improve the sorting speed of the heap allocator, we got a feedback about this insertion sort feature a while ago, roughly 1-2 months ago)

    //what did we learn?
    //equivalency of logics
    //build a compute tree, try different mathematical equivalency transformations, and benchmark
    //this would be our main cuda compute optimization engine
    //we'll devote 3 months on this problem

    //quicksearch implementation
    //everything starts with a standard, a composer, and a hinter
    //a composer tries his best to turn the music to the standard, and hints the other composers to steer in the direction of linearity (this hint becomes the standard)
    //the hint problem is a hard problem to solve, as I already told that we need to come up with a special minimal coding, remember that we only need to "preserve" the order (of sorted x -> f(x)), not the actual numerical value, because if the x follows the hints of the order, then linearity is achieved
    //the minimal coding size for THIS is n! suffix array (this is image encoding)
    //remember our stock prediction compression algorithm, we have n! suffix array then n! suffix array then another n! suffix array
    //we need to store the "linearity information", in a lossy compressed way
    //the two sum problem is usally done by PACM of lhs and rhs (uint8_t + uint8_t -> uint16_t) followed by another pacm (uint16_t + uint16_t -> uint16_t) followed by another split
    //we've been working super hard to solve the suffix compression (with extra information to allow specific "leeway" to bonsai the domain projection space) on cuda 
    //the most optimized instruction set we have found is Taylor Series precomputed intercourse operation tile x tile (involing 2, 3, 4 variables)

    //so linearity compression is about sorting the array + extracting the suffix + get slopes of sorted array + rinse and repeat
    //the suffices after certain recursive calls will be lossy or lossless compared to the orginal input data 
    //OK, this is where this gets really hard, because of logit saturation, the suffix array information will get saturated after certain number of backpropagations, we have to find the BEST way to backprop the suffix array data and converge the required suffix array data size without compromising the training phase
    //OK so what's up with all the cool stuffs rotate + linear + softmax that all the kids do? That's ... 1877 technology
    //NVIDIA is really good at deceiving those kids to devote their $BB -> the technology
    //what we would actually want is an optimized to the Moon tile intercourse operation (compiled + assemblied + tuned by our compute engines)
    //we'd want to have a FIXED flops for that because we can EXPECT that the lock will be unlocked within a certain time (we can tune this by juggling the memregion_sz) so we dont have to forever wait on a memregion to complete
    //we dont want your rotate, or multihead attentions, or softmax or multi-layer-perceptrons or layer normalization, what the hell does that even do???? 

    //I've been thinking about the suffix array
    //the suffix array propagation is transient
    //there are suffix array collisions, which is a problem of resolution at the collided nodes
    //and suffix array saturation, which is also a problem of resolution at the collided nodes

    //the problem is that we'd want to decay the "resolution" of the suffix array without compromising the final results
    //we'd attempt that by offloading the responsibility to the intercoursing tile (diffracting tile), and kind of loose the suffix array information to the immediate succeeding nodes

    //that's one dimensional

    //how do we talk in terms of 2 dimensionals, 3 dimensionals, ...
    //it's enumerated projection space compression (a single index to compress the projection space)
    //and Taylor Series linearity index

    //for every node, we'd want to mimic the projection space and reduce the linearity index by informing the succeeding guy of the expecting enumerated projection space (the finding of the enumerated projection space is truly a problem, and the intersections of the enumerated projection space need to have a rule)
    //so we only have one operation, which is the intercourse operation
    //so we have a set of crit tiles, accumulated -> a tile called instrument, the instrument would be propped down to the immediate succeeding by using "special" projection space compression and resolute the "linearity" problem
    //well well, it's gonna be at least 6 months for me to get that fully working + polished

    //I have to admit though, people would change their point of views in roughly 3months -> 6months about how advanced search + logit density mining will be actually performed
    //brother, we've been working on the neural network problem for 50 years, people made it classified it, people didnt make it publish it
    //we know the fundamental problem is the two sum problem, the locality problem and the backprop logit saturation problem

    //what is our problem?
    //we can't really find the best operations, sqrt(x) or sin(x) or cos(x) or Taylor Sum operation called intercourse(a, b)
    //this is what we are mining, the best operations (the optimal patterns)
    //we'll devote 2 months on this problem

    //all the operations are translatable to Taylor Series infinite sequence
    //Taylor Series consists of: finite
    //                           infinite convergable
    //                           inifinite not convergable

    //imagine a binary tree of logit tile intercourses
    //intercourse, hint pivots iteratively (instead of doing actual partitioning, we are hinting the left partition (left branch) to be less than a pivot, and the right partition (right branch) to be greater than a pivot)
    //for every iteration, the domain space is progressing in the direction of the pivot, the maximum complexity is n * log(n) * C iterations
    //increasing linearity of projection space

    //

}