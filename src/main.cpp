#define DEBUG_MODE_FLAG true
#define  STRONG_MEMORY_ORDERING_FLAG true

#include <stdint.h>
#include <stdlib.h>
#include <type_traits>
#include <utility>
#include "network_kernel_mailbox_impl1.h"
#include <expected>
#include <iostream>
// #include "network_producer_consumer.h"
#include "network_producer_consumer.h"

static inline intmax_t foo_counter = 0u; 

class Foo{

    public:

        Foo() noexcept{
            foo_counter += 1;
        }

        ~Foo() noexcept{
            foo_counter -= 1;
        }
};

struct Consumer: public virtual dg::network_producer_consumer::KVConsumerInterface<size_t, Foo>{

    size_t * total;

    void push(const size_t& key, std::move_iterator<Foo *> foo_arr, size_t sz) noexcept{

        *total += 1;
        // std::cout << key << std::endl;
    }
};

struct Consumer2: public virtual dg::network_producer_consumer::ConsumerInterface<Foo>{

    size_t * total;

    void push(std::move_iterator<Foo *> value, size_t sz) noexcept{

        *total += sz;
    }
};

void foo(){

    size_t total    = 0u;
    auto consumer   = Consumer{};
    consumer.total  = &total;
    char * buf      = (char *) std::malloc(dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&consumer, 1024));
    auto handle     = dg::network_producer_consumer::delvrsrv_kv_open_preallocated_handle(&consumer, 1024, buf);

    auto now        = std::chrono::high_resolution_clock::now();

    for (size_t i = 0u; i < size_t{1} << 26; ++i){
        dg::network_producer_consumer::delvrsrv_kv_deliver(handle.value(), i, Foo{});
    }

    dg::network_producer_consumer::delvrsrv_kv_close_preallocated_handle(handle.value());

    auto then       = std::chrono::high_resolution_clock::now();

    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(then - now).count() << "<ms>" << total << "<total>" << foo_counter << "<foo_sz>" << std::endl;
}

void bar(){

    size_t total    = 0u;
    auto consumer   = Consumer2{};
    consumer.total  = &total;
    char * buf      = (char *) std::malloc(dg::network_producer_consumer::delvrsrv_allocation_cost(&consumer, 512));
    auto handle     = dg::network_producer_consumer::delvrsrv_open_preallocated_handle(&consumer, 512, buf);

    auto now        = std::chrono::high_resolution_clock::now();

    for (size_t i = 0u; i < size_t{1} << 26; ++i){
        dg::network_producer_consumer::delvrsrv_deliver(handle.value(), Foo{});
    }

    dg::network_producer_consumer::delvrsrv_close_preallocated_handle(handle.value());

    auto then       = std::chrono::high_resolution_clock::now();

    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(then - now).count() << "<ms>" << total << "<total>" << foo_counter << "<foo_sz>" << std::endl;
}

int main(){

    //that was good enough
    //you heard these pieces of craps never developed a thing said
    //the only reason I haven't moved on is because I am afraid of might have happened if I apply an improved version of taylor series backpropagation
    //it's all there fellas, in between the lines of taylor_fast
    //its up to yall to release the beast
    //it's gonna be a big mess
    //sorry Dad, I rather die everyday literally than to leave yall in this situation, I have a bet to finish

    bar();
    foo();
}