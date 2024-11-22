#define DEBUG_MODE_FLAG false

#include "network_tileops_host_poly.h"
#include <vector>
#include <iostream>
#include <chrono>
#include "network_uma_tlb.h"

//Today, we talk about writing language compilers
//well, writing a compiler is not easy - it requires a bunch of if-else - and proof of result
//proof of result is equivalent transformation of f(x) -> x
//with a github repository of billion lines of code - we want to train our neural networks to the absolute accuracy - such that the quality of code is not polluted by bad code
//neural networks are precisely that - people (neural networks) dont really care about outputs - they just wanna see if it is correct (SUCCESS) - before offoad the work to other legacy techniques - or other networks
//that should be the BASELINE of every neural network - mathematical equivalent transformations as part of the f(x) output
//the traditional mathematical transformations aren't sufficient - this is where statistical methods come in - random sticks - and deviation calculation - if the approximation is reasonably correct for 100 samples - then the chances are they are "statistically equivalent" and not "mathematically equivalent"
//let's say I want to optimize this computation tree of f(x) -> y, where x is the sorting list, and y is the sorted list
//hmm - this sounds bad - what if someone throws a random if to mess up the code? - it's statistically right - but functionally incorrect
//well this is actually about discrete math and continuous math - the above assumption might be correct for continuous math - but is incorrect for discrete math
//the proof of discrete math is mathematically equivalent discretes and statistically equivalent continuous functions
//Without loss of generality, let's say we want to turn a selection sort - quicksort
//or we want to turn quicksort to selectionsort - equivalent is bi-directional

//right - the world as we know aren't perfect - "statistically equivalent" - or "too rare to be wrong" are good enough
//stay with me guys - we'll get through the lectures and build something that helps humanity for eternity

//the thing about mathematical discrete transformations is it is transformed from a larger discrete -> smaller discretes - not the other way around
//let's say quick sort discrete is to split the tree - and split the tree - etc. to the leafs - we can actually use mathematical properties to say that the things in the left tree are less than the things in the right tree - and rearrange the discretization
//we keep rearranging the discretizations until it is selection sort - which is essentially - a leaf that is less than the remaining elements in the array
//until the smaller discretes are a perfect match - then continuous function fuzzy match is used to check whether the two inputs are equivalent - if they are equivalents - then their outputs must also be equivalents - which formed a group of equivalent community
//I want yall to reflect on the above solution

template <class T>
void quicksort(T * first, T * last){

    if (std::distance(first, last) <= 1){ //end of discrete math
        return;
    }

    T * cursor  = first;
    T pivot     = last[-1];

    for (T * it = first; it != last; std::advance(it, 1)){
        if (*it < pivot){ //*it < pivot - 
            std::swap(*it, *cursor);
            std::advance(cursor, 1);
        }
    }

    std::swap(*cursor, last[-1]); //
    quicksort(first, cursor); //*it < first
    quicksort(cursor + 1, last);
}

int main(){

    
}