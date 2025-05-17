#define DEBUG_MODE_FLAG true
#define STRONG_MEMORY_ORDERING_FLAG true

// #include "network_memlock.h"
// #include "network_memlock_proxyspin.h"
#include <atomic>
#include <random>
#include <memory>
#include <functional>
#include <utility>
#include <algorithm>
#include <vector>
#include <chrono>
#include <iostream>
#include "assert.h"

//let's strategize
//what do we have

//collapse the wall (cand reduction optimization)
//block quicksort (branching optimization)
//insertion iteration up to n flops sort
//find minimum pivot greater than (by leveraging the back index of the insertion sort)
//pivoting the remaining array + continue the quicksort

template <class CallBack, class First, class Second, class ...Args>
static void insertion_sort(const CallBack& callback, First first, Second second, Args ...args){
    
    if constexpr(sizeof...(Args) == 0){
        callback(std::min(first, second), std::max(first, second));
    } else{
        auto cb_lambda = [=]<class ...AArgs>(AArgs ...aargs){
            callback(std::min(first, second), aargs...);
        };

        insertion_sort(cb_lambda, std::max(first, second), args...);
    }
} 

template <class CallBack, class First, class ...Args>
static void template_sort(const CallBack& callback, First first, Args ...args){

    if constexpr(sizeof...(Args) == 0){
        callback(first);
    } else{
        auto cb_lambda  = [=]<class ...AArgs>(AArgs ...aargs){
            insertion_sort(callback, first, aargs...);
        };

        template_sort(cb_lambda, args...);
    }
}

template <class _Ty, size_t SZ_Arg>
static void template_sort_arr(_Ty * first, const std::integral_constant<size_t, SZ_Arg>&){

    auto sort_cb    = [=]<class ...Args>(Args ...args){
        
        auto fwd_tup        = std::make_tuple(args...);
        const auto idx_seq  = std::make_index_sequence<sizeof...(Args)>{};

        [=]<class Tup, size_t ...IDX>(Tup&& tup, const std::index_sequence<IDX...>&){
            ((first[IDX]  = std::get<IDX>(tup)), ...);
        }(fwd_tup, idx_seq);

    };

    const auto idx_seq    = std::make_index_sequence<SZ_Arg>{};

    [=]<size_t ...IDX>(const std::index_sequence<IDX...>&){
        template_sort(sort_cb, first[IDX]...);
    }(idx_seq);
}

void insertion_sort_1(uint32_t * first, uint32_t * last){

    size_t sz = std::distance(first, last);
    
    //0 1

    for (size_t i = 0u; i < sz; ++i){        
        for (size_t j = 0u; j < i; ++j){
            size_t rhs_idx  = i - j;
            size_t lhs_idx  = rhs_idx - 1u;

            if (first[rhs_idx] >= first[lhs_idx]){
                break;
            }

            std::swap(first[rhs_idx], first[lhs_idx]);
        }
    }
}

//we'll do an insertion sort with sliding window problem
//such is we are reducing the number of if branches by a factor of 4

void insertion_sort_2(uint32_t * first, uint32_t * last){

    size_t sz                           = std::distance(first, last);
    constexpr size_t SLIDING_WINDOW_SZ  = 3u;

    if (sz < SLIDING_WINDOW_SZ) [[unlikely]]{
        insertion_sort_1(first, last);
        return;        
    }

    //strategize, for loop, 4 + trailing one every iteration

    for (size_t i = 0u; i < sz; ++i){
        intmax_t sorting_idx = i;

        while (true){
            if (sorting_idx == 0){
                break;
            }

            if (first[sorting_idx] >= first[sorting_idx - 1]){
                break;
            }

            sorting_idx = std::max(intmax_t{0}, static_cast<intmax_t>(sorting_idx - (SLIDING_WINDOW_SZ - 1)));
            template_sort_arr(std::next(first, sorting_idx), std::integral_constant<size_t, SLIDING_WINDOW_SZ>{});
        }
    }
}

auto insertion_sort_3(uint32_t * first, uint32_t * last, size_t insertion_sort_allowance) -> uint32_t *{

    size_t sz                           = std::distance(first, last);
    constexpr size_t SLIDING_WINDOW_SZ  = 3u;

    if (sz < SLIDING_WINDOW_SZ) [[unlikely]]{
        insertion_sort_1(first, last);
        return last;
    }

    size_t counter = 0u; 

    //strategize, for loop, 4 + trailing one every iteration

    for (size_t i = 0u; i < sz; ++i){
        intmax_t sorting_idx = i;

        if (counter >= insertion_sort_allowance){
            return std::next(first, i);
        }

        while (true){
            if (sorting_idx == 0){
                break;
            }

            if (first[sorting_idx] >= first[sorting_idx - 1]){
                break;
            }

            sorting_idx = std::max(intmax_t{0}, static_cast<intmax_t>(sorting_idx - (SLIDING_WINDOW_SZ - 1)));
            template_sort_arr(std::next(first, sorting_idx), std::integral_constant<size_t, SLIDING_WINDOW_SZ>{});
            counter += 1;
        }
    }

    return last;
}

//strategize
//doing insertion sort up to certain flops, find the last index, find the tentative pivot within the remaining array, std::max() the pivor
//collapse the wall for the remaining part, continue

//let's do a quickie

auto find_left_wall(uint32_t * first, uint32_t * last, uint32_t * pivot) -> uint32_t *{

    while (true){
        if (first == last){
            return first;
        }

        if (first == pivot){
            return first;
        }

        if (*first > *pivot){
            return first;
        }

        //<= pivot

        std::advance(first, 1u);
    }
}

auto find_right_wall(uint32_t * first, uint32_t * last, uint32_t * pivot) -> uint32_t *{

    while (true){
        if (first == last){
            return last;
        }

        if (std::prev(last) == pivot){
            return last;
        }

        if (*std::prev(last) < *pivot){ //
            return last;
        }

        //>= pivot
        // last = std::prev(last);
        std::advance(last, -1);
    }
}

//pivot is within first last

auto pivot_partition(uint32_t * first, uint32_t * last, uint32_t * pivot) -> uint32_t *{

    assert(first != last);

    //attempt to swap pivot

    std::swap(*std::prev(last), *pivot);

    uint32_t pivot_value    = *std::prev(last); 
    size_t iteration_sz     = std::distance(first, last); 
    size_t less_sz          = 0u;

    for (size_t i = 0u; i < iteration_sz; ++i){
        if (first[i] < pivot_value){
            std::swap(first[less_sz++], first[i]);
        }
    }

    std::swap(first[less_sz], *std::prev(last));

    return std::next(first, less_sz);
}

void quicksort(uint32_t * first, uint32_t * last){

    constexpr size_t SMALL_SORT_SZ = 16u;
    size_t sz = std::distance(first, last);

    if (sz <= SMALL_SORT_SZ){
        insertion_sort_2(first, last);
        return;
    }

    //hmm it seems like this is a little backward

    size_t insertion_sort_allowance = sz / 16;
    uint32_t * insert_last          = insertion_sort_3(first, last, insertion_sort_allowance); //guarantee to sort at least 1 guy
    uint32_t * new_first            = insert_last;
    size_t new_sz                   = std::distance(new_first, last);

    if (new_sz == 0u){
        return;
    }

    size_t mid_idx                  = new_sz >> 1u;

    // if (new_first[mid_idx] < new_first[0]){ // >= is fine, why?
        // std::swap(new_first[0], new_first[mid_idx]); 
    // }

    uint32_t * left_incl_wall       = find_left_wall(new_first, last, std::next(new_first, mid_idx));
    uint32_t * right_excl_wall      = find_right_wall(new_first, last, std::next(new_first, mid_idx));  
    uint32_t * pivot_ptr            = pivot_partition(left_incl_wall, right_excl_wall, std::next(new_first, mid_idx));

    quicksort(new_first, pivot_ptr);
    quicksort(std::next(pivot_ptr), last);
    std::inplace_merge(first, new_first, last); //this is insanely complicated, because the allocations pattern are actually preordered, 
}  

template <class Task>
auto timeit(Task task) -> size_t{

    auto then = std::chrono::high_resolution_clock::now();
    task();
    auto now = std::chrono::high_resolution_clock::now();

    return std::chrono::duration_cast<std::chrono::milliseconds>(now - then).count();
} 

int main(){

    const size_t SZ = size_t{1} << 23;

    std::vector<uint32_t> vec(SZ);
    std::generate(vec.begin(), vec.end(), std::bind(std::uniform_int_distribution<uint32_t>{}, std::mt19937{}));
    std::vector<uint32_t> vec2 = vec;
    std::vector<uint32_t> vec3 = vec;

    std::cout << "<insertion_sort_1>" << timeit([&]{std::sort(vec.data(), std::next(vec.data(), vec.size()));}) << "<ms>" << std::endl;
    std::cout << "<insertion_sort_2>" << timeit([&]{quicksort(vec2.data(), std::next(vec2.data(), vec2.size()));}) << "<ms>" << std::endl;

    std::cout << "<insertion_sort_1>" << timeit([&]{std::sort(vec.data(), std::next(vec.data(), vec.size()));}) << "<ms>" << std::endl;
    std::cout << "<insertion_sort_2>" << timeit([&]{quicksort(vec2.data(), std::next(vec2.data(), vec2.size()));}) << "<ms>" << std::endl;

    // for (size_t i = 0u; i < vec2.size(); ++i){
    //     if (vec[i] != vec2[i]){
    //         std::cout << i << "<>" << vec[i] << "<>" << vec2[i] << std::endl;
    //     }
    // }
    // for (uint32_t e: vec2){
        // std::cout << e << std::endl;
    // }
    // std::sort(vec3.begin(), vec3.end());

    assert(vec == vec2);
    // assert(vec2 == vec3);

    //our agenda today is to work on the frame + resolutor + quicksort (to improve the sorting speed of the heap allocator, we got a feedback about this insertion sort feature a while ago, roughly 1-2 months ago)

}