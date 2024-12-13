#define DEBUG_MODE_FLAG false

#include <chrono>
#include <memory>
#include <vector>
#include <random>
#include <functional>
#include <iostream>
#include "network_producer_consumer.h"
#include <utility>
#include <array>

//alright guys - today let's do some compiler calibrations - we've been guessing the perf but there's no instrument yet 
//need to test if we can reach the theoretical 1 << 30 pingpong + init + orphan dispatchs/ core * s:
//(1): direct access radix load (10% faster than its counterpart - arithemtic access radix load)
//(2): arithmetic access radix load
//(3): radix dispatch vs switch dispatch (alrights - it does remove the branch prediction cost - offset 40ns -> 2ns per dispatch)
//(4): cold instruction cache + hot instruction cache (this is hard to test - yet the idea is simple, we want to radix load to avoid instruction cache overheads)
//(5): heap allocation of delvrsrv + heap_stack allocation of delvsrv (eseentially - we wanna guess the radix tree size and its height then open_delvrsrv_raiihandle accordingly with SZ = radix_size ** height, then we wanna reuse the space - by using heapstack bignum approach)
//(6): key [value] dispatch - right (this is a hard problem), we have 4096 concurrent memlock_regions, we wanna reduce the radix space -> 128 - uint8_t, open-addressing of value - this is hard - there are considerations
//these optimizations alone are worth more than concurrency itself
//so we must do micro optimizations (offset branch prediction overhead, offset cache access overhead) - not because we're greedy - but we are moving the direction of "host_concurrency_not_cuda_concurrency" - which is a micro optimization by itself
//alrights - I promise my brother I post the proof of concept to crack the state-of-the-art asymmetric encryption soon - so let's get back on track 

template <class Task>
auto timeit(Task task) noexcept{

    using namespace std::chrono;
    auto now = high_resolution_clock::now();
    task();
    auto then = high_resolution_clock::now();

    return duration_cast<milliseconds>(then - now).count();
}

void foo1(size_t& total){

    static std::vector<uint64_t> table = []{
        std::vector<uint64_t> rs(256);
        std::iota(rs.begin(), rs.end(), 0u);
        std::shuffle(rs.begin(), rs.end(), std::mt19937{std::chrono::high_resolution_clock::now().time_since_epoch().count()});

        return rs;
    }();

    total += table[total & 0xFF];
}

void foo2(size_t& total){

    static std::vector<uint64_t> table = []{
        std::vector<uint64_t> rs(256);
        std::iota(rs.begin(), rs.end(), 0u);
        std::shuffle(rs.begin(), rs.end(), std::mt19937{std::chrono::high_resolution_clock::now().time_since_epoch().count()});

        return rs;
    }();

    total += table[total & 0xFF];
}

void foo3(size_t& total){

    static std::vector<uint64_t> table = []{
        std::vector<uint64_t> rs(256);
        std::iota(rs.begin(), rs.end(), 0u);
        std::shuffle(rs.begin(), rs.end(), std::mt19937{std::chrono::high_resolution_clock::now().time_since_epoch().count()});

        return rs;
    }();

    total += table[total & 0xFF];
}

struct Foo1: dg::network_producer_consumer::ConsumerInterface<size_t>{

    size_t * total;

    void push(size_t * arr, size_t arr_sz) noexcept{

        for (size_t i = 0u; i < arr_sz; ++i){
            foo1(*total);
        }
    }
};

struct Foo2: dg::network_producer_consumer::ConsumerInterface<size_t>{

    size_t * total;

    void push(size_t * arr, size_t arr_sz) noexcept{

        for (size_t i = 0u; i < arr_sz; ++i){
            foo2(*total);
        }
    }
};

struct Foo3: dg::network_producer_consumer::ConsumerInterface<size_t>{

    size_t * total;

    void push(size_t * arr, size_t arr_sz) noexcept{

        for (size_t i = 0u; i < arr_sz; ++i){
            foo3(*total);
        }
    }
};

int main(){

    // size_t SZ;
    // std::cin >> SZ;
    const size_t SZ = size_t{1} << 27;
    std::vector<uint8_t> tape(SZ); 
    std::vector<size_t> tmp;
    tmp.reserve(1024);
    // std::vector<uint64_t> buf(SZ);
    // std::vector<uint8_t> radix_table(256);
    std::iota(tape.begin(), tape.end(), 0u);
    std::generate(tape.begin(), tape.end(), std::bind(std::uniform_int_distribution<uint8_t>(), std::mt19937{}));

    uint64_t total{};

    auto task3 = [&]() noexcept{
        for (size_t i = 0u; i < SZ; ++i){
            uint8_t cur_tape = tape[i] & 3u; 

            switch (cur_tape){
                case 0:
                    foo1(total);
                    break;
                case 1:
                    foo2(total);
                    break;
                case 2:
                    foo3(total);
                    break;
                default:
                    std::unreachable();
                    break;
            }

            if (tmp.size() == tmp.capacity()){
                tmp.clear();
            }

            tmp.push_back(0u);
        }
    };

    auto task2 = [&]() noexcept{
        for (size_t i = 0u; i < SZ; ++i){
            uint8_t cur_tape = tape[i] & 3u; 
            foo1(total);
        }
    };

    auto task1 = [&]() noexcept{
        Foo1 foo1_ins{};
        foo1_ins.total = &total;
        Foo2 foo2_ins{};
        foo2_ins.total = &total;
        Foo3 foo3_ins{};
        foo3_ins.total = &total;

        {
            auto foo1_delivery_handle   = dg::network_producer_consumer::delvrsrv_open_raiihandle(&foo1_ins, 32);
            auto foo2_delivery_handle   = dg::network_producer_consumer::delvrsrv_open_raiihandle(&foo2_ins, 32);
            auto foo3_delivery_handle   = dg::network_producer_consumer::delvrsrv_open_raiihandle(&foo3_ins, 32); 
            
            std::array<dg::network_producer_consumer::DeliveryHandle<size_t> *, 4> dispatch_arr{foo1_delivery_handle->get(), foo2_delivery_handle->get(), foo3_delivery_handle->get(), foo3_delivery_handle->get()}; //it seems like gcc does not like switch case as high level interface - only llvm built on switch case accepts that

            for (size_t i = 0u; i < SZ; ++i){
                size_t cur_tape = tape[i] & 3u;
                dg::network_producer_consumer::delvrsrv_deliver(dispatch_arr[cur_tape], size_t{0u});  
            }
        }
    };

    uint64_t milli_lapsed_1 = timeit(task1);
    uint64_t milli_lapsed_2 = timeit(task2);
    uint64_t milli_lapsed_3 = timeit(task3);

    std::cout << milli_lapsed_1 << "<ms>" << milli_lapsed_2 << "<ms>" << "<>" << milli_lapsed_3 << "<>" << total << std::endl;
}