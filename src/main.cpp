#define DEBUG_MODE_FLAG false

// #include "network_tileops_host_poly.h"
#include <vector>
#include <iostream>
#include <chrono>
// #include "network_uma_tlb.h"
#include <chrono>

int main(){

    //let's prove the theorem that f(g(x)) = x, g(x1) ~= g(x2) if x1 != x2 and x1 x2 are semantically equivalent, in an optimal lossless compression environment

    //we know that in a lossless compression environment
    //the maximum worst_case compression is len(training_data_token_arr)    - data is indexed as if they are in a database
    //the minimum worst_case compression is len(training_data_token)        - data is in its original form - no compression

    //the maximum best_case compression is 1
    //the minimum best_case compression is len(training_data_token_arr)

    //let's prove three theorems: (1): the more x1 and x2 are semantically equivalent, they closer they are in the optimal compressed semantic coordinate
    //                            (2): g(x) is the semantic of x
    //                            (3): assume the loss function penaltizes the positional difference in euclidian coordinate - then the more x1 and x2 are semantically equivalent, the closer they are in the euclidian coordinate 

    //(2): g(x) has bijective relation to x (lossless compression) - so g(x) is the semantic of x
    
    //Proof by induction: Assume there exists f(g(x)) = x, g(x1) ~= g(x2) if x1 and x2 are semantically equivalent, x1 != x2
    //assume incoming input is x3 - prove that g(x3) ~= g(x4) if x3 and x4 are semantically equivalent
    //proof by contradiction: assume g(x3) \c group(g(x4)) - then g(x3) c group(g(x5))
    //assume x5 does not exist
    //then the existence of group(g(x5)) in the semantic coordinate must expand the current semantic space in the leading dimension that is greater than any of the intersected semantic dimensions of g(x3) and g(x4)
    //assume x5 exists
    //then we know that g(x5) - g(x4) > acceptible_epsilon, so g(x5) - g(x3) > acceptible epsilon which contradicts with the optimal semantic tree rule (1)
    //so it is not greedily not optimal - which contradicts with the newton approx being greedily optimal

    //let's prove the theorem that f(g(x)) = x, g(x1) = g(x2) if x1 and x2 are semantically equivalent, in an optimal lossy compression environment
    //...

    //let's prove the theorem that g(<brain_input><brain_output>) -> <brain_input>
}