#define DEBUG_MODE_FLAG true
#define  STRONG_MEMORY_ORDERING_FLAG true

#include <stdint.h>
#include <stdlib.h>
#include <type_traits>
#include <utility>
#include "network_kernel_mailbox_impl1.h"

//its easy to say that C++ forwarding is the most absurb code ever, YET, it works in most cases
//consider this

//you would expect this to forward l_value reference ONLY
//no, it forwards both l_value and r_value references, for T = base and T = base& for r_value and l_value respectively, or we are perfect forwarding base&& and base& && respectively, there is the decay rule of r_value reference + defined scope of usage that we dont wanna talk about, it's complicated
//because obj is always passed as reference unless returned by a function or explicit static_cast when you invoke a function 

template <class T>
auto pf_forward(std::remove_reference_t<T>& obj) noexcept -> T&&{

    return static_cast<T&&>(obj);
}

//what does this do then?
//alright, this is only handle the cases where the object cannot be passed by reference, such is the construction of the parameter is r_value construction 
//this is the undefined usage of forward, no one really does r_value when forwarding, yet included in the std 
//most people are obsessed with rvalue forwarding, when indeed they want raw type + std::move(), this is the practice that has long been lost

template <class T>
auto pf_forward(std::remove_reference_t<T>&& obj) noexcept -> T&&{

    return static_cast<T&&>(obj);
}

//function polymorphic is the most wrongly implemented function of C++
//function polymorphic can be seen as choosing the most appropriate version, if there are two or more options of being appropriate, choose the one that has higher priority, in terms of higher polymorphic order -> lower polymorphic order, self -> static_castable, self -> implicit constructible etc. 
//the choosing the most appropriate does not work when, there are two identical functions f(x) -> y, the compiler cannot decide which function has higher priority
//function polymorphic is the worst source of bugs, it's reasonable to say that we want to have each function does their own things, this is the reason why perfect forwarding is honorred, because it creates one trait of function for every guys perfectly forwarded

//for best practices, enforcing the function routing can be done via header std::enable_if_t<>, or we can say that it is to narrow the set of potential functions, to choose the most appropriate function  
//what follows std::enable_if_t<> must be a boolean that does not constitute SFINAE, the boolean that works all the time, it is std::enable_if_t that determines whether the function is SFINAE or not, not the boolean
//dont add the std::enable_if_t<> in the parameters, or the return type, it is not a good practice, not honorred by the C++ std

//bool SFINAE can be done via struct + std::void_t<>
//it has the following template

template <class T, class = void>
struct bool_sfinae: std::false_type{};

template <class T, class T1, class T2>
struct bool_sfinae<std::tuple<T, T1, T2>, std::void_t<decltype(std::declval<T&>().foo()), decltype(std::declval<T1&>().bar())>>: std::true_type{};

//no, you cant add another std::void_t<>
//it is a boolean construct, false type, true type, the end, there is no third struct to add another true or false
//what if we want to have a set of true_type, alright this is where we have multiple bool_sfinae + std::disjunction_v<bool_sfinae<first>, bool_sfinae<second>...>

//how about this?

template <class T>
struct bool_sfinae2: std::false_type{};

template <class T>
struct bool_sfinae2<std::vector<T>>: std::true_type{};

template <class T>
struct bool_sfinae2<std::basic_string<T>>: std::true_type{};

//this works according to the std
//this is the explicit specialization implementation

//static_assert() within the function is used instead when there is no function routing, very encouraged in explict programming

//alright people
//these people want to see immediate result, out
//remember fellas, we aim for quality, we'll move very slowly, we'll make sure to document our code correctly
//we are not selling you bullshit fellas
//we actually have thought long and hard
//the idea of the forward code cannot be done more optimally unless you break the universal physical laws
//we are pretty proud of our socket protocol, people did a lot of pull requests yet I think that would compromise the speed, the rawness, extensibility of the protocol
//we are blazing fast, solved numerous problems involving 1/7000 failure rate (bring inter-transportation of 10GB -> 99.999999% success), coordinated packet id attacks, corrupted packet attacks, engineered corrupted packet attacks, retransmissions, fast+prioritized acks, throttled kernel wire spit, on-wire direct bounce (dedicated thread to do recvfrom), etc.  
//we are not eavesdropping anyone, we are just awared of the existence of a continuity extraction device at some point in the future (advanced tech will figure this out)
//continuity extraction device is like a 5G tower, except it is 10G, it could collect data up to <x> mile ^ 3

//the implementation is also another "art", we reduced the number of memory orders by a factor of at least 256, for 256 is the average feed_size before pushing to a container
//we reduced the no_busy -> busy latency -> 1 microsecond by using accurate timed_mutex, we'll implement this

//every tree computation has at most 10 GB of inter-transportation, so we can rely on our protocol to do things correctly
//the thing that we are not proud of is the memregion frequency, we are not using real-time operating system (is that a thing?), so we must rely solely on statistics to do calibrations (this is hard) or somewhere in between of sub-real-time + statistics ?
//                                      the packet_id_unordered_set, we are susceptible to late-packets which will definitely corrupt the system, we dont have an immediate solution to this yet

//99% success of the logit density mining is depended on the state of the art back_propagation technique, we wrote the paper (is it newton_approx?)
//we dont really know, we just know that at some point in the future, the level of <intellect> will rise, and people (or synthetic sentients) will normalize what's considered <intellectual>
//people will bid for logit density, for <the_smartest_guy_will_rule_the_world>
//if everyone is smart, there is no smart (wo)man
//if everyone is rich, there is no rich (wo)man

//we dont really care
//we do the business of 30% 3rd party fees
//people can enjoy the funs of mining logit density