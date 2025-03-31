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
