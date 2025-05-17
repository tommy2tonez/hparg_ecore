#ifndef __SORT_VARIANTS_H__
#define __SORT_VARIANTS_H__

#include <random>
#include <memory>
#include <functional>
#include <utility>
#include <algorithm>
#include <vector>
#include <chrono>
#include <iostream>
#include "assert.h"
#include "stdx.h"
#include <type_traits>
#include <bit>

namespace dg::sort_variants::quicksort{

    static inline constexpr size_t BLOCK_PIVOT_MAX_LESS_SZ      = 16u;
    static inline constexpr size_t MAX_RECURSION_DEPTH          = 64u; 
    static inline constexpr size_t COMPUTE_LEEWAY_MULTIPLIER    = 8u; 
    static inline constexpr size_t SMALL_QUICKSORT_SZ           = 16u;
    static inline constexpr size_t ASC_SORTING_RATIO            = 4u;

    template <class _Ty>
    static void insertion_sort_1(_Ty * first, _Ty * last){

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

    template <class _Ty>
    static void insertion_sort_2(_Ty * first, _Ty * last){

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

    template <class _Ty>
    static __attribute__((noinline)) auto insertion_sort_3(_Ty * first, _Ty * last, size_t insertion_sort_allowance) -> _Ty *{

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

    template <class _Ty>
    static auto find_left_wall(_Ty * first, _Ty * last, _Ty * pivot) -> _Ty *{

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

    template <class _Ty>
    static auto find_right_wall(_Ty * first, _Ty * last, _Ty * pivot) -> _Ty *{

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

    template <class _Ty>
    static auto pivot_partition(_Ty * first, _Ty * last, _Ty * pivot) -> _Ty *{

        assert(first != last);

        //attempt to swap pivot

        std::swap(*std::prev(last), *pivot);

        _Ty pivot_value         = *std::prev(last); 
        size_t iteration_sz     = std::distance(first, last); 
        size_t less_sz          = 0u;
        
        size_t blk_sz           = iteration_sz / BLOCK_PIVOT_MAX_LESS_SZ;
        size_t blk_pop_sz       = blk_sz * BLOCK_PIVOT_MAX_LESS_SZ; 
        size_t rem_sz           = iteration_sz - blk_pop_sz; 

        for (size_t i = 0u; i < blk_sz; ++i){
            _Ty * local_first   = std::next(first, i * BLOCK_PIVOT_MAX_LESS_SZ);
            uint64_t bitset     = 0u;

            for (size_t j = 0u; j < BLOCK_PIVOT_MAX_LESS_SZ; ++j){
                bitset |= static_cast<uint64_t>(local_first[j] < pivot_value) << j;
            }

            while (bitset != 0u){
                size_t j = std::countr_zero(bitset);
                bitset &= (bitset - 1u);
                std::swap(first[less_sz++], local_first[j]);      
            }
        }

        for (size_t i = 0u; i < rem_sz; ++i){
            if (first[i + blk_pop_sz] < pivot_value){
                std::swap(first[less_sz++], first[i + blk_pop_sz]);
            }
        }

        std::swap(first[less_sz], *std::prev(last));
        return std::next(first, less_sz);
    }

    template <class _Ty>
    static auto base_asc_quicksort(_Ty * first, _Ty * last, uint64_t flops, uint64_t max_flops, uint32_t stack_idx) -> uint64_t{

        size_t sz = std::distance(first, last);

        if (sz <= SMALL_QUICKSORT_SZ){
            insertion_sort_2(first, last);
            return sz * sz;
        }

        if (flops > max_flops || stack_idx >= MAX_RECURSION_DEPTH) [[unlikely]]{
            std::sort(first, last);
            return sz * stdx::ulog2(stdx::ceil2(sz));
        }

        size_t insertion_sort_allowance = sz / ASC_SORTING_RATIO;
        _Ty * insert_last               = insertion_sort_3(first, last, insertion_sort_allowance);
        _Ty * new_first                 = insert_last;
        size_t new_sz                   = std::distance(new_first, last);
        uint64_t incurred_cost          = insertion_sort_allowance; 

        if (new_sz == 0u){
            return incurred_cost;
        }
        
        size_t mid_idx                  = new_sz >> 1;

        _Ty * left_incl_wall            = find_left_wall(new_first, last, std::next(new_first, mid_idx));
        _Ty * right_excl_wall           = find_right_wall(new_first, last, std::next(new_first, mid_idx));  
        _Ty * pivot_ptr                 = pivot_partition(left_incl_wall, right_excl_wall, std::next(new_first, mid_idx));
        
        incurred_cost                   += new_sz * 2;
        incurred_cost                   += base_asc_quicksort(new_first, pivot_ptr, flops + incurred_cost, max_flops, stack_idx + 1u);
        incurred_cost                   += base_asc_quicksort(std::next(pivot_ptr), last, flops + incurred_cost, max_flops, stack_idx + 1u);

        std::inplace_merge(first, new_first, last); //this is incredibly hard to implement correctly
        incurred_cost                   += sz;

        return incurred_cost;
    }

    template <class _Ty>
    static auto base_quicksort(_Ty * first, _Ty * last, uint64_t flops, uint64_t max_flops, uint32_t stack_idx) -> uint64_t{

        size_t sz = std::distance(first, last);

        if (sz <= SMALL_QUICKSORT_SZ){
            insertion_sort_2(first, last);
            return sz * sz;
        }

        if (flops > max_flops || stack_idx >= MAX_RECURSION_DEPTH) [[unlikely]]{
            std::sort(first, last);
            return sz * stdx::ulog2(stdx::ceil2(sz));
        }

        size_t mid_idx                  = sz >> 1;
        uint64_t incurred_cost          = 0u;

        _Ty * left_incl_wall            = find_left_wall(first, last, std::next(first, mid_idx));
        _Ty * right_excl_wall           = find_right_wall(first, last, std::next(first, mid_idx));  
        _Ty * pivot_ptr                 = pivot_partition(left_incl_wall, right_excl_wall, std::next(first, mid_idx));
        
        incurred_cost                   += sz * 2;
        incurred_cost                   += base_quicksort(first, pivot_ptr, flops + incurred_cost, max_flops, stack_idx + 1u);
        incurred_cost                   += base_quicksort(std::next(pivot_ptr), last, flops + incurred_cost, max_flops, stack_idx + 1u);

        return incurred_cost;
    }

    template <class _Ty>
    __attribute__((noinline)) void asc_quicksort(_Ty * first, _Ty * last) noexcept{

        static_assert(std::is_arithmetic_v<_Ty>);

        size_t sz           = std::distance(first, last);
        size_t compute_sz   = sz * stdx::ulog2(stdx::ceil2(sz)) * COMPUTE_LEEWAY_MULTIPLIER;

        base_asc_quicksort(first, last, 0u, compute_sz, 0u);
    }

    template <class _Ty>
    __attribute__((noinline)) void quicksort(_Ty * first, _Ty * last) noexcept{

        static_assert(std::is_arithmetic_v<_Ty>);

        size_t sz           = std::distance(first, last);
        size_t compute_sz   = sz * stdx::ulog2(stdx::ceil2(sz)) * COMPUTE_LEEWAY_MULTIPLIER;

        base_quicksort(first, last, 0u, compute_sz, 0u);
    }
} 

#endif