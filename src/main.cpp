#define DEBUG_MODE_FLAG false
#include <unordered_set>
#include <atomic>
#include <memory>
#include <chrono>
#include <iterator>

int main(){

    //2025 is the year we are optimizing one thing: compressible_size/ neural_network_size
    //we want to maximize the compressible_size/ neural_network_size - there is gradient descend problem but we aren't worrying about that now
    //we want to stack 1024 f(g(x)) -> x
    //the problem with gradient descends is the problem of uniform influence
    //a random leaf logit has a certain influence on the output logits
    //a group of equivalent influence can have the same training rate
    //so we can describe the problem as finding the sequence of training of same influence groups
    //WLOG, influence can be found by doing 1111111111 for output layer and memset(layer, ID, sizeof(layer)) then group gradients by discretization (this is an oversimplication - there is an entire research topics about setting intial values)
    //with an arbitrary learning rate
    //WLOG, a valid training sequence is: group0 -> group0 -> group3 -> group2 -> group2, etc
    //this solution is proven to reasonably optimally approximate any neural network f(x) -> y
    //
    
}