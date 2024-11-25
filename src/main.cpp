#define DEBUG_MODE_FLAG true 

#include "network_memlock.h"
#include "network_mempress_collector.h"

int main(){

    //alright guys - this is going to be a hard task that we want to focus on this week
    //we want to 
    //(1):
    //isolate the resolutor cache (affined) to solely work on its designated tasks
    //ping signal, pong request, pong_signal, gpu dispatch init + gpu dispatch backward - need to fit in the core (or thread) L1 cache and branch prediction state machines
    //we don't want to thrash branch prediction
    //the cost of ping/pong + gpu dispatch is exactly < 5 CPU flops/ dispatch - this needs a LOT of magic to happen - given our abstraction of work - few people that can do this cleanly

    //(2): allow the program to run on solely atomic infrastructure (modern intel core guarantees this) to offset the cost of seq_cst (which is a necessity - otherwise you risk UB - or performance constraint which are equally bad)
    //(3): external tiles are the most expensive - in the sense that we need to serialize the tile + sequential access + forward it to a foreign machine
    //(4): implement transfer functionality for kernelmap_x

    //next week, we want to focus on
    //offset the cost of ping-pong by doing concurrent forward transactions
    //such that we offset the "synchronization" cost by making more transactions
    //because ping-pong are actually not expensive (in the sense of flops) - they are time-consuming
    //the ultimate goal of utilizing a machine is that - you want to utilize the flops
    //calibrate the socket performance by using a calibration network - this is getting recursive
    //rebuild cuda dispatch machine - cuda leaves us a very few options to do this
    //allow polymorphic buffer for internal allocation - buffer traits are specified to achieve sequential locality

    //polymorphic buffer choices: - fast_buffer (intended to be deallocated quickly, right after consumer's consumption) 
    //                            - persistent_buffer
    //                            - slow_buffer

    //internal allocation responsibility: WLOG, radix fast_buffer   -> fast_long_buffer + fast_short_buffer
    //                                          radix slow_buffer   -> slow_long_buffer + slow_short_buffer

    //each of these guy is allocated on a different dg_heap - which guarantees the allocation to be fragmentation-free + optiomal cyclic reuse of pages (need to specify the page_size statistical deviation - in the sense that the time it takes to get through a page is the allocation lifetime)

    //next next week, we want to focus on:
    //build a ring of f(g(x)) -> x
    //talk about transforming paths (floyed) + optimization methods + discrete math + proof of work + turing completeness of the language
    //modern encryption cracking methods by using f(g(x)) -> x
    //modern brain wave interference injection

    //next next next week, we want to focus on:
    //a synchronous brain across 3 billion devices - by using cyclic leafs
    //we don't want a big, giant brain, we want a concurrent synchronous small brain
    //Green's theorem - by reducing one big giant ring to geographically conditional open-close rings

    //after this lecture you'll be able to:
    //crack all current asymmetric encoding methods and finally get some Bitcoin for thyself - legally
    //crack all current symmetric encoding methods
    //understand the mechanics of the universe
    //modern human brain enhancement - by using brain wave interference injection - think of brain wave as collector
    //we want to radix + group the things that fire together as scheduled tile
    //we want nano techs (input | output interface) - something that parasites the human brain - we need the help of smarter guys to do this

    //able to explain 3 theoretical physic questions:
    //betweenness centrality + tensor saturation and Heisenberg's uncertainty
    //massful objects as stationary objects in an arbitrary dimension
    //communication between objects - synchronization issue - flip flop issue - centrality issue - saturated tensor issue
    //circular ring of massful objects
    //is the universe really expanding? - Heisenberg's uncertainty

}