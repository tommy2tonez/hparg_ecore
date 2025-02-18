//
#include <random>
#include <iostream>
#include <algorithm>
#include <stdint.h>
#include <stdlib.h>
#include <vector>
#include <functional>
#include <chrono>
#include <utility>
#include <array>
#include <cstring>
#include <mutex>
#include <thread>
#include <atomic>

class Foo{

    private:

        // std::mutex mtx;
        std::atomic<std::array<size_t, 1>> arr;

    public:

        void inc() noexcept{

            // std::lock_guard<std::mutex> grd(this->mtx);
            // this->counter.fetch_add(1, std::memory_order_relaxed);
            // this->counter += 1;
            auto new_arr = this->arr.load(std::memory_order_relaxed);
            new_arr[0] += 1;
            new_arr[1] += 1;
            this->arr.exchange(new_arr, std::memory_order_relaxed);
        }

        auto read() noexcept -> size_t{

            return this->arr.load(std::memory_order_relaxed)[0];
        }
};

int main(){

    // std::chrono::utc_clock::now();
    using tp = std::chrono::time_point<std::chrono::utc_clock>;

    const size_t THREAD_SZ  = 8;
    const size_t COUNTER_SZ = size_t{1} << 24;
    std::vector<std::unique_ptr<Foo>> foo_vec{};

    for (size_t i = 0u; i < THREAD_SZ; ++i){
        foo_vec.push_back(std::make_unique<Foo>());
    }

    auto now = std::chrono::high_resolution_clock::now();

    {
        std::vector<std::thread> thr_vec{};

        for (size_t i = 0u; i < THREAD_SZ; ++i){
            auto task = [&, i]() noexcept{
                for (size_t j = 0u; j < COUNTER_SZ; ++j){
                    foo_vec[i]->inc();
                }
            };

            thr_vec.emplace_back(task);
        }

        for (size_t i = 0u; i < THREAD_SZ; ++i){
            thr_vec[i].join();
        }
    }

    auto then   = std::chrono::high_resolution_clock::now();
    auto lapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - then).count();

    for (size_t i = 0u; i < THREAD_SZ; ++i){
        std::cout << foo_vec[i]->read() << std::endl;
    }

    std::cout << lapsed << "<ms>" << std::endl;
}