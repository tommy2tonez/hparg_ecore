#include "../src/network_producer_consumer.h"
#include <stdint.h>
#include <stdlib.h>
#include <unordered_map>
#include <utility>
#include <algorithm>

namespace keyfeed_test{

    static size_t foo_counter{};

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

    struct Consumer: virtual dg::network_producer_consumer::ConsumerInterface<uint32_t>{

        std::vector<uint32_t> * rs;

        void push(std::move_iterator<uint32_t *> data_arr, size_t sz) noexcept{

            std::copy(data_arr, std::next(data_arr, sz), std::back_inserter(*rs));
        }
    };

    struct Consumer2: virtual dg::network_producer_consumer::ConsumerInterface<Foo>{

        std::vector<uint32_t> * rs;

        void push(std::move_iterator<Foo *> data_arr, size_t sz) noexcept{

            for (size_t i = 0u; i < sz; ++i){
                rs->push_back(data_arr[i].foo_value);
            }
        }
    };

    auto randomize_vector(size_t sz, size_t key_range) -> std::vector<uint32_t>{

        auto key_random_device = std::bind(std::uniform_int_distribution<uint32_t>(0u, key_range + 1), std::mt19937{std::chrono::high_resolution_clock::now().time_since_epoch().count()});
        std::vector<uint32_t> rs(sz);
        std::generate(rs.begin(), rs.end(), key_random_device);

        return rs;
    }  

    void test_one_feed(std::vector<uint32_t> key_feed_vec, size_t feed_sz){

        std::vector<uint32_t> consuming_vec{};

        {
            auto consumer           = Consumer{};
            consumer.rs             = &consuming_vec;
            auto raii_feed          = dg::network_producer_consumer::delvrsrv_open_raiihandle(&consumer, feed_sz);

            if (!raii_feed.has_value()){
                std::cout << "test_one_preallocated_feed_memory failed" << std::endl;
                std::abort();
            }

            for (const auto& key: key_feed_vec){
                dg::network_producer_consumer::delvrsrv_deliver(raii_feed.value().get(), key);
            }
        }

        if (key_feed_vec != consuming_vec){
            std::cout << "test_one_preallocated_feed failed" << std::endl;
            std::abort();       
        }
    }

    void test_one_preallocated_feed(std::vector<uint32_t> key_feed_vec, size_t feed_sz){

        std::vector<uint32_t> consuming_vec{};

        {
            auto consumer           = Consumer{};
            consumer.rs             = &consuming_vec;
            size_t allocation_cost  = dg::network_producer_consumer::delvrsrv_allocation_cost(&consumer, feed_sz);
            auto buf                = std::make_unique<char[]>(allocation_cost);
            auto raii_feed          = dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&consumer, feed_sz, buf.get());

            if (!raii_feed.has_value()){
                std::cout << "test_one_preallocated_feed_memory failed" << std::endl;
                std::abort();
            }

            for (const auto& key: key_feed_vec){
                dg::network_producer_consumer::delvrsrv_deliver(raii_feed.value().get(), key);
            }
        }

        if (key_feed_vec != consuming_vec){
            std::cout << "test_one_preallocated_feed failed" << std::endl;
            std::abort();       
        }
    }

    void test_leak_one_feed(std::vector<uint32_t> key_feed_vec, size_t feed_sz){

        std::vector<uint32_t> consuming_vec{};

        {
            auto consumer           = Consumer2{};
            consumer.rs             = &consuming_vec;
            auto raii_feed          = dg::network_producer_consumer::delvrsrv_open_raiihandle(&consumer, feed_sz);

            if (!raii_feed.has_value()){
                std::cout << "test_leak_one_preallocated_feed_memory failed" << std::endl;
                std::abort();
            }

            for (const auto& key: key_feed_vec){
                dg::network_producer_consumer::delvrsrv_deliver(raii_feed.value().get(), Foo(key));
            }
        }

        if (key_feed_vec != consuming_vec){
            std::cout << "test_one_preallocated_feed failed" << std::endl;
            std::abort();       
        }

        if (foo_counter != 0u){
            std::cout << "test_leak_one_preallocated_feed failed" << std::endl;
            std::abort();
        }
    }

    void test_leak_one_preallocated_feed(std::vector<uint32_t> key_feed_vec, size_t feed_sz){

        std::vector<uint32_t> consuming_vec{};

        {
            auto consumer           = Consumer2{};
            consumer.rs             = &consuming_vec;
            size_t allocation_cost  = dg::network_producer_consumer::delvrsrv_allocation_cost(&consumer, feed_sz);
            auto buf                = std::make_unique<char[]>(allocation_cost);
            auto raii_feed          = dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&consumer, feed_sz, buf.get());

            if (!raii_feed.has_value()){
                std::cout << "test_leak_one_preallocated_feed_memory failed" << std::endl;
                std::abort();
            }

            for (const auto& key: key_feed_vec){
                dg::network_producer_consumer::delvrsrv_deliver(raii_feed.value().get(), Foo(key));
            }
        }

        if (key_feed_vec != consuming_vec){
            std::cout << "test_one_preallocated_feed failed" << std::endl;
            std::abort();       
        }

        if (foo_counter != 0u){
            std::cout << "test_leak_one_preallocated_feed failed" << std::endl;
            std::abort();
        }
    }

    void run(){

        std::cout << "<initializing_feed_test>" << std::endl;

        const size_t TEST_SZ_PER_GROWTH     = 8192;
        const size_t GROWTH_FACTOR          = 1u;
        const size_t INITIAL_RANGE          = 0u;
        const size_t GROWTH_SZ              = 10;  
        const size_t MAP_RANDOM_SZ          = 32; 
        const size_t MAP_RANDOM_FEED_SZ     = 32;

        auto map_sz_random_device           = std::bind(std::uniform_int_distribution<size_t>(0u, MAP_RANDOM_SZ - 1), std::mt19937{std::chrono::high_resolution_clock::now().time_since_epoch().count()}); 
        auto map_feed_sz_random_device      = std::bind(std::uniform_int_distribution<size_t>(0u, MAP_RANDOM_FEED_SZ - 1), std::mt19937{std::chrono::high_resolution_clock::now().time_since_epoch().count()});

        for (size_t i = 0u; i < GROWTH_SZ; ++i){
            for (size_t j = 0u; j < TEST_SZ_PER_GROWTH; ++j){
                size_t current_key_range    = (size_t{1} << (GROWTH_FACTOR * i + INITIAL_RANGE)) - 1;
                size_t map_sz               = map_sz_random_device();
                size_t map_feed_sz          = map_feed_sz_random_device(); 
                auto map                    = randomize_vector(map_sz, current_key_range);

                test_one_feed(map, map_feed_sz);
                test_one_preallocated_feed(map, map_feed_sz);
                test_leak_one_feed(map, map_feed_sz);
                test_leak_one_preallocated_feed(map, map_feed_sz);
            }

            std::cout << "testing completed " << i << "/" << GROWTH_SZ << std::endl;
        }

        std::cout << "<feed_test_completed>" << std::endl;
    }

}