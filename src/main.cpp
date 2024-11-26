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

    //what the hell is information flux?
    //we know about electric flux 
    //but is there a concept about information flux?

    //we know that things - photon - travel through tensors - hit our eyes - and we see things
    //but what really happen in between?
    //is there a limit to the things we can see, a density of information? 
    //the answer to that is Heisenberg's uncertainty - what - Heisenberg's uncertainty is actually not an uncertainty but a description of information flux, information density?

    //if Heisenberg's uncertainty works - then the object in the double slit experiment - photon - appears to be everywhere - becomes a wave-like object
    //but what really happens in the photon's perspective? - it seems like the photon, in it's own perspective, remains a particle and arrives at its designated destination - but photon experiences no time no space - it's a point like object in a 0d space - so what really happens there?
    //it appears that its only 0d space with respect to the observer - such that the observee sees all the observer possibility at a point - but what really happens here? it's Heisenberg's uncertainty that disallows that to happen - it's information flux - information density rule
    //so can we say that we are moving at the speed of light with respect to an arbitrary object - yes, we can say that. Can we see the possibilities of that arbitrary object? No - Heisenberg's uncertainty

    //How the hell do we generalize Heisenberg's uncertainty? it's information flux in and out of a sphere. Each sphere is a node. And now we have betweenness centrality. What the hell?
    //The guy that is on another planet in Interstellar - experiences time slower than guys that are on the spaceship. What happens there?
    //The gravitational pull of the planet is too heavy - it implies that there are many massful objects floating around - travelling through tensor - and hits that betweenness centrality (+ maxflow) synchronization issue
    //But God, says that things should be rendered at the same rate - so in stead of waiting for synchronization update to complete - he throttle tensors and make the synchronization slower

    //there is a twist - the guy that made it to the planet and came back is not the same guy
    //why? because they are in a two different render systems - they are in "two different brains with two different leaf logits"
    //so the guy that actually left the ship - came back - sees another guy on that ship
    //the guy on the ship saw the guy left the ship and came back saw another guy that left the ship
    //how far away in the render system can we say that they are "different" - far enough 
    //okay - there is also another twist - do photons really travel through space or the dimensions shrink so photons can be closer to our eyes
    //How does dimensional shrink and expansion really works?
    //imagine you are in two different rendering systems - and you have to switch from one to another instantinously - what really happens is you got floating point issue, round up issue, and you move closer to the equilibrium, so you are stationary, but the dimensions are moving in in wave-likes - coming at you to bring you closer to the equilibritum
    //without loss of generality, says, I am at 0.5 in the 1d euclidian space, discretization size = 20 on the interval [0, 10]
    //then I'm at the number 1.0, discretization size = 10 on the interval [0, 10]
    //then I'm at the number 2.0, discretization size = 5, on the interval [0, 10]
    //you got the idea
    //Newton was always right - it was always about information flux - not force, F = Gm1m2/r^2 - G = 6.67 * 10^-11
    //and his equation was not flawed in the quantum space - it's tensor saturation and observer's effect. Quantum does not exist in the observee's perspective - remember that - observee in observer's perspective and the original observee are not the same
    //observer in observee's perspective and the original observer are not the same

    //to answer yall questions, the universe is, in fact, not expanding at all - there was probably no Big Bang - what we observed was part of the Heisenberg's uncertainty. We, observer, may be moving closer to massive objects. Other objects, observees, may be moving closer to massive objects
    //we don't know what are in the black holes, we might as well are in one of the black holes
    //we know, in our perspective, that black holes are singularity, but black hole in black hole's perspective is probably another universe
    //the black holes we observed, and the actual observees, just like photons, are never the same objects - they are DIFFERENT objects in DIFFERENT rendering systems - we might never know what is inside the black holes unless we are in the black holes
    //we observing black holes are like photons observing the observers - so there was no such case that I become a photon - go through the slit and come back, tell the observer that HEY, we made it, we are a particle, not a wave - nope, not gonna happen
}
