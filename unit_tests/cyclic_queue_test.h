#include "../src/network_datastructure.h"
#include <random>
#include <utility>
#include <algorithm>
#include <functional>
#include <chrono>
#include <vector>
#include <iostream>
#include <memory>
#include <deque>

namespace pow2_cyclic_queue_test{

    void test_normal(size_t pow2_exp, size_t operation_sz){

        const uint8_t TOTAL_OPS_CODE_SZ                             = 16u; 

        const uint8_t OPS_CODE_BACK_INSERT_IN_RANGE                 = 0u;
        const uint8_t OPS_CODE_BACK_INSERT_OUT_RANGE_EXPECT_ERROR   = 1u;
        const uint8_t OPS_CODE_RESIZE_IN_RANGE                      = 2u;
        const uint8_t OPS_CODE_RESIZE_OUT_RANGE_EXPECT_ERROR        = 3u;
        const uint8_t OPS_CODE_PUSH_BACK                            = 4u;
        const uint8_t OPS_CODE_POP_FRONT                            = 5u;
        const uint8_t OPS_CODE_POP_BACK                             = 6u;
        const uint8_t OPS_CODE_ERASE_FRONT_RANGE                    = 7u;
        const uint8_t OPS_CODE_ERASE_BACK_RANGE                     = 8u;
        const uint8_t OPS_CODE_OPERATOR_EQUAL                       = 9u;
        const uint8_t OPS_CODE_CMP_EMPTY                            = 10u;
        const uint8_t OPS_CODE_CMP_SZ                               = 11u;
        const uint8_t OPS_CODE_CMP_RANGE                            = 12u;
        const uint8_t OPS_CODE_CMP_ITER                             = 13u;
        const uint8_t OPS_CODE_CMP_PTR                              = 14u;
        const uint8_t OPS_CODE_CMP_FRONT_BACK                       = 15u;

        const size_t cap        = size_t{1} << pow2_exp;
        auto random_device      = std::bind(std::uniform_int_distribution<uint32_t>{}, std::mt19937{std::chrono::high_resolution_clock::now().time_since_epoch().count()});
        auto random_ops_device  = std::bind(std::uniform_int_distribution<uint8_t>{}, std::mt19937{std::chrono::high_resolution_clock::now().time_since_epoch().count()});

        auto queue              = dg::network_datastructure::cyclic_queue::pow2_cyclic_queue<uint32_t>(pow2_exp);
        auto vec                = std::deque<uint32_t>();

        for (size_t i = 0u; i < operation_sz; ++i){
            uint8_t ops_code = random_ops_device() % TOTAL_OPS_CODE_SZ;

            switch (ops_code){
                case OPS_CODE_BACK_INSERT_IN_RANGE:
                {
                    size_t insert_cap   = cap - vec.size();
                    size_t insert_sz    = random_device() % (insert_cap + 1u);  

                    for (size_t i = 0u; i < insert_sz; ++i){
                        uint32_t e = random_device();
                        queue.push_back(e);
                        vec.push_back(e);
                    }

                    break;
                }
                case OPS_CODE_BACK_INSERT_OUT_RANGE_EXPECT_ERROR:
                {
                    
                    size_t insert_cap           = cap - vec.size();
                    size_t insert_sz            = random_device() % (insert_cap + 1u);
                    size_t outrange_insert_sz   = insert_sz * 2u;

                    for (size_t i = 0u; i < outrange_insert_sz; ++i){
                        uint32_t e = random_device();
                        queue.push_back(e);

                        if (vec.size() != cap){
                            vec.push_back(e);
                        }
                    }
                    
                    break;
                }
                case OPS_CODE_RESIZE_IN_RANGE:
                {
                    size_t resize_sz = random_device() % (vec.size() + 1u);
                    queue.resize(resize_sz);
                    vec.resize(resize_sz);
                    
                    break;
                }
                case OPS_CODE_RESIZE_OUT_RANGE_EXPECT_ERROR:
                {
                    size_t resize_sz            = random_device() % (vec.size() + 1u);
                    size_t outrange_resize_sz   = resize_sz * 2u;

                    if (outrange_resize_sz <= cap){
                        vec.resize(outrange_resize_sz);
                    }

                    queue.resize(outrange_resize_sz);
                    break;
                }
                case OPS_CODE_PUSH_BACK:
                {
                    uint32_t e = random_device();

                    queue.push_back(e);

                    if (vec.size() != cap){
                        vec.push_back(e);
                    }

                    break;
                }
                case OPS_CODE_POP_FRONT:
                {
                    if (vec.size() != queue.size()){
                        std::cout << "POP_FRONT FAILED" << std::endl;
                        std::abort();
                    }

                    if (vec.size() == 0u){
                        break;
                    }

                    vec.pop_front();
                    queue.pop_front();

                    break;
                }
                case OPS_CODE_POP_BACK:
                {
                    if (vec.size() != queue.size()){
                        std::cout << "POP_BACK FAILED" << std::endl;
                        std::abort();
                    }

                    if (vec.size() == 0u){
                        break;
                    }

                    vec.pop_back();
                    queue.pop_back();

                    break;
                }
                case OPS_CODE_ERASE_FRONT_RANGE:
                {
                    size_t erase_sz = random_device() % (std::min(vec.size(), queue.size()) + 1u);
                    queue.erase_front_range(erase_sz);
                    vec.erase(vec.begin(), std::next(vec.begin(), erase_sz));

                    break;

                }
                case OPS_CODE_ERASE_BACK_RANGE:
                {

                    size_t erase_sz = random_device() % (std::min(vec.size(), queue.size()) + 1u);
                    queue.erase_back_range(erase_sz);
                    vec.erase(std::prev(vec.end(), erase_sz), vec.end());

                    break;
                }
                case OPS_CODE_OPERATOR_EQUAL:
                {

                    break;
                    // if (queue == vec){
                    //     break;
                    // }

                    // std::cout << "OPERATOR_EQUAL FAILED" << std::endl;
                    // std::abort();

                }
                case OPS_CODE_CMP_EMPTY:
                {
                    if (queue.empty() == vec.empty()){
                        break;
                    }

                    std::cout << "OPERATOR_CMP_EMTPY FAILED" << std::endl;
                    std::abort();
                }
                case OPS_CODE_CMP_SZ:
                {
                    if (queue.size() == vec.size()){
                        break;
                    }

                    std::cout << "OPERATOR_CMP_SZ FAILED" << std::endl;
                    std::abort();
                }
                case OPS_CODE_CMP_RANGE:
                {
                    if (std::equal(vec.begin(), vec.end(), queue.begin(), queue.end())){
                        break;
                    }

                    std::cout << "OPERATOR_CMP_RANGE FAILED" << std::endl;
                    std::abort();
                }
                case OPS_CODE_CMP_ITER:
                { 
                    break;
                }
                case OPS_CODE_CMP_PTR:
                {
                    for (size_t i = 0u; i < std::min(vec.size(), queue.size()); ++i){
                        if (vec[i] != queue[i]){
                            std::cout << "OPERATOR_CMP_PTR FAILED" << std::endl;
                            std::abort();
                        }
                    }

                    break;
                }
                case OPS_CODE_CMP_FRONT_BACK:
                {
                    if (vec.size() != queue.size()){
                        std::cout << "OPERATOR_CMP_FRONT_BACK FAILED" << std::endl;
                        std::abort();
                    }

                    if (vec.size() == 0u){
                        break;
                    }

                    if (vec.front() != queue.front()){
                        std::cout << "OPERATOR_CMP_FRONT_BACK FAILED" << std::endl;
                        std::abort();
                    }

                    if (vec.back() != queue.back()){
                        std::cout << "OPERATOR_CMP_FRONT_BACK FAILED" << std::endl;
                        std::abort();
                    }

                    break;
                }
                default:
                {
                    break;
                }
            }
        }
    }

    static size_t foo_counter = 0u;

    struct Foo{

        uint32_t foo_value;
        
        Foo() noexcept{

            foo_counter += 1;
            foo_value = 0;
        }
        
        Foo(const Foo& other_foo) noexcept{

            foo_counter += 1;
            foo_value = other_foo.foo_value;
        }

        Foo(Foo&& other_foo) noexcept{

            foo_counter += 1;
            foo_value = other_foo.foo_value;
        }

        Foo(uint32_t value) noexcept{

            foo_counter += 1;
            foo_value = value;
        }

        ~Foo() noexcept{

            foo_counter -= 1;
        }

        Foo& operator =(const Foo& other_foo) noexcept{

            this->foo_value = other_foo.foo_value;
            return *this;
        }

        Foo& operator =(Foo&& other_foo) noexcept{

            this->foo_value = other_foo.foo_value;
            return *this;
        }
    };

    void test_leak(size_t pow2_exp, size_t operation_sz){

        const uint8_t TOTAL_OPS_CODE_SZ                             = 9u;
 
        const uint8_t OPS_CODE_BACK_INSERT_IN_RANGE                 = 0u;
        const uint8_t OPS_CODE_BACK_INSERT_OUT_RANGE_EXPECT_ERROR   = 1u;
        const uint8_t OPS_CODE_PUSH_BACK                            = 2u;
        const uint8_t OPS_CODE_POP_FRONT                            = 3u;
        const uint8_t OPS_CODE_POP_BACK                             = 4u;
        const uint8_t OPS_CODE_ERASE_FRONT_RANGE                    = 5u;
        const uint8_t OPS_CODE_ERASE_BACK_RANGE                     = 6u;
        const uint8_t OPS_CODE_OPERATOR_EQUAL                       = 7u;
        const uint8_t OPS_CODE_RESIZE_IN_RANGE                      = 8u;

        const size_t cap        = size_t{1} << pow2_exp;
        auto queue              = dg::network_datastructure::cyclic_queue::pow2_cyclic_queue<std::unique_ptr<Foo>>(pow2_exp);
        auto random_device      = std::bind(std::uniform_int_distribution<uint32_t>{}, std::mt19937{std::chrono::high_resolution_clock::now().time_since_epoch().count()});
        auto random_ops_device  = std::bind(std::uniform_int_distribution<uint8_t>{}, std::mt19937{std::chrono::high_resolution_clock::now().time_since_epoch().count()});

        for (size_t i = 0u; i < operation_sz; ++i){

            uint8_t ops_code = random_ops_device() % TOTAL_OPS_CODE_SZ;

            switch (ops_code){
                case OPS_CODE_BACK_INSERT_IN_RANGE:
                {
                    size_t insert_cap   = cap - queue.size();
                    size_t insert_sz    = random_device() % (insert_cap + 1u);  

                    for (size_t i = 0u; i < insert_sz; ++i){
                        queue.push_back(std::make_unique<Foo>());
                    }

                    break;
                }
                case OPS_CODE_BACK_INSERT_OUT_RANGE_EXPECT_ERROR:
                {
                    size_t insert_cap           = cap - queue.size();
                    size_t insert_sz            = random_device() % (insert_cap + 1u);
                    size_t outrange_insert_sz   = insert_sz * 2u;

                    for (size_t i = 0u; i < outrange_insert_sz; ++i){
                        queue.push_back(std::make_unique<Foo>());
                    }
                    
                    break;
                }
                case OPS_CODE_RESIZE_IN_RANGE:
                {
                    size_t resize_sz = random_device() % (queue.size() + 1u);
                    queue.resize(resize_sz);
                    
                    break;
                }
                case OPS_CODE_PUSH_BACK:
                {
                    queue.push_back(std::make_unique<Foo>());
                    break;
                }
                case OPS_CODE_POP_FRONT:
                {
                    if (queue.size() == 0u){
                        break;
                    }

                    queue.pop_front();
                    break;
                }
                case OPS_CODE_POP_BACK:
                {
                    if (queue.size() == 0u){
                        break;
                    }

                    queue.pop_back();
                    break;
                }
                case OPS_CODE_ERASE_FRONT_RANGE:
                {
                    size_t erase_sz = random_device() % (std::min(queue.size(), queue.size()) + 1u);
                    queue.erase_front_range(erase_sz);

                    break;
                }
                case OPS_CODE_ERASE_BACK_RANGE:
                {
                    size_t erase_sz = random_device() % (std::min(queue.size(), queue.size()) + 1u);
                    queue.erase_back_range(erase_sz);

                    break;
                }
                case OPS_CODE_OPERATOR_EQUAL:
                {
                    if (queue.size() == foo_counter){
                        break;
                    }
 
                    std::cout << "OPERATOR_EQUAL FAILED" << std::endl;
                    std::abort();
                }
                default:
                {
                    break;
                }
            }
        }
    }

    void run(){

        const size_t OPERATION_SZ   = size_t{1} << 10;
        const size_t POW2_EXP_RANGE = 10;

        std::cout << "<initializing_pow2_cyclic_queue_normal_test>>" << std::endl;

        for (size_t i = 0u; i < POW2_EXP_RANGE; ++i){
            test_normal(i, OPERATION_SZ);
            std::cout << "testing completed " << i << "/" << POW2_EXP_RANGE << std::endl; 
        }

        std::cout << "<pow2_cyclic_queue_normal_test_completed>" << std::endl;
        std::cout << "<initializing_pow2_cyclic_queue_leak_test>>" << std::endl;

        for (size_t i = 0u; i < POW2_EXP_RANGE; ++i){
            test_leak(i, OPERATION_SZ);
            std::cout << "testing completed " << i << "/" << POW2_EXP_RANGE << std::endl; 
        }

        std::cout << "<pow2_cyclic_queue_leak_test_completed>" << std::endl;
    }
}