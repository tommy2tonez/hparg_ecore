#define DEBUG_MODE_FLAG false

#include "network_tileops_host_poly.h"
#include <vector>
#include <iostream>
#include <chrono>

int main(){

    //alright guys - let's do this
    //one more VERY VERY important activator for pair operation and our thing is officially turing complete
    //alright - things are perfect - we need to leverage GPUs for very computation intensive tasks - like optimization and gradient differentiation
    //CPUs can do forward | recv packet | send_packet and friends

    //things to note is that backpropagation in a real-time brain is not as important
    //we want the backpropagation to complete without taking any additional flops -
    //this means that exact timing of tile scheduling is rather unrealistic - yet there exists an optimal scheduling with the least flops to do the back propagation
    //in the interim, we preserve the flops to mainly do forward - which is GPT job

    //I want yall to reflect on the double slit experiment and the current state of the art GPT transformer architecture
    //we have a frame x, f(x) -> y, and softmax being the slit
    //we can see that the painted picture y (the projected screen) is heavily influenced by the softmax - inteference and friends - which ruins the picture - in other words, limits the possibility space of the output
    //why is limiting the possibility space even a good thing?
    //I already explained this before - imagine Mom gave you a billion to get a girlfriend and a car and a house - this is a brainless task
    //imagine Mom gave you 10K to get a girlfriend, a car and a house - then you have to stategize - get the crew - rob the bank - get the one billion - then this becomes a brainless task

    //this works 99% of the time - but the 1% of the time - the cop came early - plan ruined - you need to take a different path - but you are trained to continue to do the job
    //this is where the pair_activator is from - you are programmed to switched to an entirely different plan - outgun the cops - run from the cops - get the helicopter and shoot everybody, etc.
    //this is where the GPU optimizers (gradient differentation) takes place - it is there to make sure that your brain is not too saturated and opened to ideas

    using namespace std::chrono;
    
    std::vector<uint8_t> lhs(256, 1);
    std::vector<uint8_t> rhs(256, 2);
    std::vector<uint8_t> dst(256, 0);
    
    size_t iter_sz{};

    std::cin >> iter_sz;

    auto now = high_resolution_clock::now();

    for (size_t i = 0u; i < iter_sz; ++i){
        dg::network_tileops_host_poly::fwd_pair(dst.data(), lhs.data(), rhs.data(), dg::network_tileops_host_poly::make_dispatch_code(dg::network_tileops_host_poly::uuu_8_8_8, dg::network_tileops_host_poly::ops_add));
    }

    auto then = high_resolution_clock::now();
    auto lapsed = duration_cast<milliseconds>(then - now).count();

    std::cout << lhs[0] << "<>" << lapsed << "<ms>" << std::endl;
}