#define DEBUG_MODE_FLAG false

#include <chrono>
#include <memory>
#include <vector>
#include <random>
#include <functional>
#include <iostream>
// #include "network_producer_consumer.h"
#include <utility>
#include <array>
#include "dg_map_variants.h"

template <class Task>
auto timeit(Task task) noexcept{

    using namespace std::chrono;
    auto now = high_resolution_clock::now();
    task();
    auto then = high_resolution_clock::now();

    return duration_cast<milliseconds>(then - now).count();
}

struct NullValueGenerator{

    constexpr auto operator()() const noexcept -> std::pair<size_t, size_t>{

        return {std::numeric_limits<size_t>::max(), std::numeric_limits<size_t>::max()};
    }
};

int main(){

    const size_t SZ = size_t{1} << 28; //alright - we are at 2x array access - which is decent - let's try collisions at uint8_t
    std::vector<uint16_t> buf(SZ);
    std::generate(buf.begin(), buf.end(), std::bind(std::uniform_int_distribution<uint16_t>{}, std::mt19937{}));

    dg::map_variants::unordered_unstable_fastinsert_map<size_t, size_t, NullValueGenerator, uint32_t> mmap{};
    mmap.reserve(size_t{1} << 15);

    auto task = [&]{
        for (size_t e: buf){
            if (mmap.size() == size_t{1} << 15){
                mmap.clear();
            }

            mmap[e] += 1;
        }
    };

    auto lapsed = timeit(task);
    std::cout << mmap[0] << "<>" << lapsed << "<ms>" << std::endl;
}