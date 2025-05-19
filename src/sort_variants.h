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
// #include "stdx.h"
#include <type_traits>
#include <bit>

namespace dg::sort_variants::quicksort{

    static inline constexpr size_t BLOCK_PIVOT_MAX_LESS_SZ      = 8u;
    static inline constexpr size_t BLOCK_PIVOT_MAX_WALL_SZ      = 32u;
    static inline constexpr size_t MAX_RECURSION_DEPTH          = 64u; 
    static inline constexpr size_t COMPUTE_LEEWAY_MULTIPLIER    = 8u; 
    static inline constexpr size_t SMALL_QUICKSORT_SZ           = 16u;
    static inline constexpr size_t ASC_SORTING_RATIO            = 4u;
    static inline constexpr size_t SMALL_PIVOT_PARTITION_SZ     = 16u; 

    // static inline constexpr size_t BLOCK_PIVOT_BITSET_DUMP_SZ   = 4u;

    using qs_unsigned_bitset_t = uint32_t;

    template <class T, std::enable_if_t<std::is_unsigned_v<T>, bool> = true>
    constexpr auto ulog2(T val) noexcept -> T{

        return static_cast<T>(sizeof(T) * CHAR_BIT - 1u) - static_cast<T>(std::countl_zero(val));
    }

    template <class T, std::enable_if_t<std::is_unsigned_v<T>, bool> = true>
    static constexpr auto ceil2(T val) noexcept -> T{

        if (val < 2u) [[unlikely]]{
            return 1u;
        } else [[likely]]{
            T uplog_value = ulog2(static_cast<T>(val - 1u)) + 1u;
            return T{1u} << uplog_value;
        }
    }

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
        constexpr size_t SLIDING_WINDOW_SZ  = 5u;

        if (sz < SLIDING_WINDOW_SZ){
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
    static auto insertion_sort_3(_Ty * first, _Ty * last, size_t insertion_sort_allowance) -> _Ty *{

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

    //optimization ON, how to do this correctly?
    template <class _Ty>
    static auto find_left_wall(_Ty * first, _Ty * last, _Ty * pivot) -> _Ty *{

        while (true){
            // if (first == last){
                // return first;
            // }

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
            // if (first == last){
                // return last;
            // }

            if (std::prev(last) == pivot){
                return last;
            }

            if (*std::prev(last) < *pivot){ //
                return last;
            }

            //>= pivot
            last = std::prev(last);
            // std::advance(last, -1);
        }
    }

    //we'll be super happy if we could optimize this
    template <class _Ty>
    static inline auto  pivot_partition_1(_Ty * first, _Ty * last, _Ty * pivot) -> _Ty *{

        assert(first != last);

        //attempt to swap pivot

        std::swap(*std::prev(last), *pivot);

        const _Ty& pivot_value  = *std::prev(last); 
        size_t iteration_sz     = std::distance(first, last); 
        size_t less_sz          = 0u;
        
        size_t blk_sz           = iteration_sz / BLOCK_PIVOT_MAX_LESS_SZ;
        size_t blk_pop_sz       = blk_sz * BLOCK_PIVOT_MAX_LESS_SZ; 
        size_t rem_sz           = iteration_sz - blk_pop_sz; 

        for (size_t i = 0u; i < blk_sz; ++i){
            _Ty * local_first           = std::next(first, i * BLOCK_PIVOT_MAX_LESS_SZ);
            qs_unsigned_bitset_t bitset = 0u;

            for (size_t j = 0u; j < BLOCK_PIVOT_MAX_LESS_SZ; ++j){
                bitset |= static_cast<qs_unsigned_bitset_t>(local_first[j] < pivot_value) << j;
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

    //we'll be super happy if we could optimize this
    template <class _Ty>
    static __attribute__((noinline)) auto  pivot_partition_0(_Ty * first, _Ty * last, _Ty * pivot) -> _Ty *{

        assert(first != last);

        //attempt to swap pivot

        std::swap(*std::prev(last), *pivot);

        const _Ty& pivot_value  = *std::prev(last); 
        size_t iteration_sz     = std::distance(first, last); 
        size_t less_sz          = 0u;
        
        size_t blk_sz           = iteration_sz / BLOCK_PIVOT_MAX_LESS_SZ;
        size_t blk_pop_sz       = blk_sz * BLOCK_PIVOT_MAX_LESS_SZ; 
        size_t rem_sz           = iteration_sz - blk_pop_sz; 

        for (size_t i = 0u; i < blk_sz; ++i){
            _Ty * local_first           = std::next(first, i * BLOCK_PIVOT_MAX_LESS_SZ);
            qs_unsigned_bitset_t bitset = 0u;

            for (size_t j = 0u; j < BLOCK_PIVOT_MAX_LESS_SZ; ++j){
                bitset |= static_cast<qs_unsigned_bitset_t>(local_first[j] < pivot_value) << j;
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
    static inline __attribute__((always_inline)) void dg_restrict_swap(_Ty * __restrict__ lhs, _Ty * __restrict__ rhs){

        std::swap(*lhs, *rhs);
    }

    template <class Ty>
    static auto extract_greatereq_from_left(Ty * first, const Ty& pivot, const size_t sz) -> qs_unsigned_bitset_t{

        qs_unsigned_bitset_t lhs_bitset = 0u; 

        for (size_t i = 0u; i < sz; ++i) [[likely]]{
            size_t reverse_idx  = (sz - 1) - i;
            lhs_bitset          |= static_cast<qs_unsigned_bitset_t>(first[i] >= pivot) << reverse_idx; //refill_sz, we'd want to get the relative position compared to the advance -refill_sz
        }

        return lhs_bitset;
    }

    template <class Ty>
    static auto extract_lesser_from_right(Ty * last, const Ty& pivot, const size_t sz) -> qs_unsigned_bitset_t{

        qs_unsigned_bitset_t rhs_bitset = 0u;

        for (size_t i = 0u; i < sz; ++i) [[likely]]{
            size_t reverse_idx  = sz - (i + 1); 
            rhs_bitset          |= static_cast<qs_unsigned_bitset_t>(last[-(static_cast<intmax_t>(i) + 1)] < pivot) << reverse_idx;
        }

        return rhs_bitset;
    }

    template <class T, size_t SZ>
    static consteval auto low_bits(const std::integral_constant<size_t, SZ>) -> T{

        static_assert(std::is_unsigned_v<T>);

        if (SZ == std::numeric_limits<T>::digits){
            return std::numeric_limits<T>::max();
        }

        return (T{1} << SZ) - 1u;
    } 

    template <class Ty, size_t SZ>
    static inline __attribute__((always_inline)) auto extract_greatereq_from_left(Ty * first, const Ty& pivot, const std::integral_constant<size_t, SZ>) -> qs_unsigned_bitset_t{

        qs_unsigned_bitset_t lhs_bitset = 0u; 

        [&]<size_t ...IDX>(const std::index_sequence<IDX...>){
            (
                [&]{
                    (void) IDX;
                    constexpr size_t reverse_idx = (SZ - 1) - IDX;
                    lhs_bitset |= static_cast<qs_unsigned_bitset_t>(first[IDX] < pivot) << reverse_idx; //less is better, due to the population of the operation, no one ever does >
                }(), ...
            );
        }(std::make_index_sequence<SZ>{});

        return (~lhs_bitset) & low_bits<qs_unsigned_bitset_t>(std::integral_constant<size_t, SZ>{});
    }

    template <class Ty, size_t SZ>
    static inline __attribute__((always_inline)) auto extract_lesser_from_right(Ty * last, const Ty& pivot, const std::integral_constant<size_t, SZ>) -> qs_unsigned_bitset_t{

        qs_unsigned_bitset_t rhs_bitset = 0u;
        Ty * first                      = std::prev(last, SZ);

        [&]<size_t ...IDX>(const std::index_sequence<IDX...>){
            (
                [&]{
                    (void) IDX;
                    rhs_bitset |= static_cast<qs_unsigned_bitset_t>(first[IDX] < pivot) << IDX; //refill_sz, we'd want to get the relative position compared to the advance -refill_sz        
                }(), ...
            );
        }(std::make_index_sequence<SZ>{});

        return rhs_bitset;
    }

    //we are 15% from optimal as clued by our peers
    //where how
    //without SIMD, I dont think I could further optimize the code, as hinted by our measures
    //this function performance is fine, if first last size is > 128 or 256, the optimality of this function is reached at the point,
    //if there are performance issues, we'd want to have another dispatch code for another size
    //we have given up on actually solving the problem, it's very super complicated
    //the theoerical limit is probably around 30% w.r.t to the std sort for uniform distributed data, we are still 20% deviated from the optimal code, we have done a fine job, not the best pivot_partition_1 could be further optimized or another radix of the implementation
    //we have reached 37.8% w.r.t to the std implementation without using SIMD, this is quite an achievement, we'll be back
    //180ms is the optimal threshold, we'll try (we have reached 179ms guys!!!)
    //for the very first time, we've reached 189ms
    //we'll stop the optimizations fellas
    //I think this is good enough, we'll process the pull requests later

    template <class _Ty>
    static inline auto pivot_partition_2(_Ty * first, _Ty * last, _Ty * pivot) -> _Ty *{

        //this is complicated to write, I'll try

        std::swap(*std::prev(last), *pivot);

        const _Ty& pivot_value          = *std::prev(last);
        _Ty * ffirst                    = first;
        _Ty * llast                     = std::prev(last);

        qs_unsigned_bitset_t lhs_bitset = 0u;
        qs_unsigned_bitset_t rhs_bitset = 0u;

        //this loop is not optimized

        //the programming strategy we used here is called tranactional programming
        //assume the valid state is first, last, lhs_bitset, rhs_bitset
        //for every transaction, first last must be a valid interval representing the not-considered partition
        //and lhs_bitset and rhs_bitset must be representing the trailing information of those first last

        //what before first must be less (when the bitset is empty)
        //what after or equal last must be greater or equal (when the bitset is empty)

        //we dont really care why the transaction is there, as long as it snaps the program into a valid state + the loop break is reached after certain valid transactions

        while (true){
            while (lhs_bitset != 0u && rhs_bitset != 0u){
                size_t lhs_back_idx     = std::countr_zero(lhs_bitset) + 1u;
                size_t rhs_forward_idx  = std::countr_zero(rhs_bitset);

                lhs_bitset              &= lhs_bitset - 1u;
                rhs_bitset              &= rhs_bitset - 1u;

                dg_restrict_swap(std::prev(ffirst, lhs_back_idx), std::next(llast, rhs_forward_idx));
            }

            if (std::distance(ffirst, llast) < BLOCK_PIVOT_MAX_WALL_SZ){
                break;
            }

            if (lhs_bitset == 0u){
                lhs_bitset = extract_greatereq_from_left(ffirst, pivot_value, std::integral_constant<size_t, BLOCK_PIVOT_MAX_WALL_SZ>{});
                std::advance(ffirst, BLOCK_PIVOT_MAX_WALL_SZ);
            } else{
                rhs_bitset = extract_lesser_from_right(llast, pivot_value, std::integral_constant<size_t, BLOCK_PIVOT_MAX_WALL_SZ>{});
                std::advance(llast, -static_cast<intmax_t>(BLOCK_PIVOT_MAX_WALL_SZ));
            }
        }

        //while loop is broken when the not considered range < BLOCK_PIVOT_MAX_WALL_SZ
        //and the while (lhs_bitset + rhs_bitset) is guaranteed to be the immediate previous reached point 

        size_t after_sz = std::distance(ffirst, llast);

        //lhs_bitset == 0u or rhs_bitset == 0u
        if (lhs_bitset == 0u){
            lhs_bitset      = extract_greatereq_from_left(ffirst, pivot_value, after_sz);                
            std::advance(ffirst, after_sz);
        } else{
            rhs_bitset      = extract_lesser_from_right(llast, pivot_value, after_sz);
            std::advance(llast, -static_cast<intmax_t>(after_sz));
        }

        while (lhs_bitset != 0u && rhs_bitset != 0u){
            size_t lhs_back_idx     = std::countr_zero(lhs_bitset) + 1u;
            size_t rhs_forward_idx  = std::countr_zero(rhs_bitset);
            lhs_bitset              &= lhs_bitset - 1u;
            rhs_bitset              &= rhs_bitset - 1u;

            dg_restrict_swap(std::prev(ffirst, lhs_back_idx), std::next(llast, rhs_forward_idx));
        }

        //we'll do another optimization here
        //three cases
        //lhs_bitset == 0u (rhs_bitset != 0u)
        //lhs_bitset == 0u (lhs_bitset != 0u)
        //lhs_bitset == 0u && rhs_bitset == 0u (other)

        if (lhs_bitset != 0u){
            _Ty * back_ptr = std::prev(llast, 1u); //we are doing greater partition, since all the llast forward are greater + eq, and all the prev_ffirst backward are lesser, the only partition to be considered is what's left in the lhs_bitset  

            while (lhs_bitset != 0u){
                size_t back_idx = std::countr_zero(lhs_bitset);
                lhs_bitset      &= lhs_bitset - 1u;
                std::iter_swap(back_ptr, std::prev(ffirst, back_idx + 1u));
                std::advance(back_ptr, -1);
            }

            std::iter_swap(std::next(back_ptr), std::prev(last));
            return std::next(back_ptr);
        }

        if (rhs_bitset != 0u){
            _Ty * front_ptr = ffirst; //we are doing lesser partition, ...

            while (rhs_bitset != 0u){
                size_t front_idx    = std::countr_zero(rhs_bitset);
                rhs_bitset          &= rhs_bitset - 1u;
                std::iter_swap(front_ptr, std::next(llast, front_idx));
                std::advance(front_ptr, 1u);
            }

            std::iter_swap(front_ptr, std::prev(last));
            return front_ptr;
        }

        std::iter_swap(llast, std::prev(last)); //first == last, according to our logic, all the last including itself forward (up to the std::prev(last)) are greater or equal, if it is std::prev(last) then it is definitely equal, we can safely do an iter_swap
        return llast;
    }

    template <class _Ty>
    static inline auto pivot_partition(_Ty * first, _Ty * last, _Ty * pivot) -> _Ty *{

        size_t sz = std::distance(first, last);

        if (sz <= SMALL_PIVOT_PARTITION_SZ){
            return pivot_partition_1(first, last, pivot);
        } else{
            return pivot_partition_2(first, last, pivot);
        }
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
            return sz * ulog2(ceil2(sz));
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
        _Ty * pivot_ptr                 = pivot_partition_1(left_incl_wall, right_excl_wall, std::next(new_first, mid_idx));
        
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
            return sz * ulog2(ceil2(sz));
        } else [[likely]]{
            static_assert(SMALL_QUICKSORT_SZ >= 4);

            constexpr size_t PIVOT_CAND_SZ  = 3u;
            size_t mid_idx                  = sz >> 1;
            uint64_t incurred_cost          = 0u;

            _Ty * mid_ptr                   = std::next(first, mid_idx);
            _Ty * previous_mid_ptr          = std::prev(mid_ptr);

            template_sort_arr(previous_mid_ptr, std::integral_constant<size_t, PIVOT_CAND_SZ>{});

            _Ty * pivot_ptr                 = pivot_partition(first, last, mid_ptr);    

            incurred_cost                   += sz;
            incurred_cost                   += base_quicksort(first, pivot_ptr, flops + incurred_cost, max_flops, stack_idx + 1u);
            incurred_cost                   += base_quicksort(std::next(pivot_ptr), last, flops + incurred_cost, max_flops, stack_idx + 1u);

            return incurred_cost;    
        }
    }

    template <class _Ty>
    __attribute__((noinline)) void asc_quicksort(_Ty * first, _Ty * last) noexcept{

        static_assert(std::is_arithmetic_v<_Ty>);

        size_t sz           = std::distance(first, last);
        size_t compute_sz   = sz * ulog2(ceil2(sz)) * COMPUTE_LEEWAY_MULTIPLIER;

        base_asc_quicksort(first, last, 0u, compute_sz, 0u);
    }

    //we are very proud of what people have recommended at 
    //https://llvm.org/devmtg/2023-02-25/slides/A-New-Implementation-for-std-sort.pdf
    //our good friends implementation is here: https://github.com/llvm/llvm-project/blob/main/libcxx/include/__algorithm/sort.h
    //we have made a further progress on the optimization (by re-branching + reimplementing insertion sort for arithmetic types)
    //we dont really have applications for the arithmetic sort except for our heap implementation which is a very crucial backbone of all things
    //for the first time, we can sort our heap segments 3 times faster without polluting CPUs (we have completed the assignment!!!)
    //we'll move on, because this implementation is already uptodate-blackart-optimized, I can't find another instruction just yet, we'll circle back to this problem, I think 15% squeeze could be performed as clued by our peers
    //the only thing left to implement is a quad swap
    //quad swap is a merge_sort (or quicksort) approach of swapping, reducing the number of supposedly 6 cmps -> 5 cmps
    //this when used in conjunction with our template sort can be very useful 

    template <class _Ty>
    __attribute__((noinline)) void quicksort(_Ty * first, _Ty * last) noexcept{

        static_assert(std::is_arithmetic_v<_Ty>);

        size_t sz           = std::distance(first, last);
        size_t compute_sz   = sz * ulog2(ceil2(sz)) * COMPUTE_LEEWAY_MULTIPLIER;

        base_quicksort(first, last, 0u, compute_sz, 0u);
    }
} 

#endif