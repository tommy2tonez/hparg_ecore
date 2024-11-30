#ifndef __DG_MAP_VARIANTS_H__
#define __DG_MAP_VARIANTS_H__

#include <stdint.h>
#include <stdlib.h>
#include <vector>
#include <unordered_map>
#include <limits>
#include <climits>
#include <utility>
#include <ratio>
#include <algorithm>
#include <functional>
#include <iostream>

namespace dg::map_variants{

    template <class T, std::enable_if_t<std::is_unsigned_v<T>, bool> = true>
    static constexpr auto ulog2(T val) noexcept -> size_t{

        return static_cast<size_t>(sizeof(T) * CHAR_BIT - 1) - static_cast<size_t>(std::countl_zero(val));
    }

    template <class T, std::enable_if_t<std::is_unsigned_v<T>, bool> = true>
    static constexpr auto least_pow2_greater_equal_than(T val) noexcept -> T{

        if (val == 0u) [[unlikely]]{
            return 1u;
        }

        size_t max_log2     = ulog2(val);
        size_t min_log2     = std::countr_zero(val);
        size_t cand_log2    = max_log2 + ((max_log2 ^ min_log2) != 0u);

        return T{1u} << cand_log2;
    }

    //there is a slight problem
    //insert factor == 1    => 1 - (e^-1) virtual load factor
    //insert_factor = 2     => 1 - (e^-2) virtual load factor - which is a decent load factor
    //with the actual load factor of 3/4, and insert_factor of 2, we can expect to have the least operation count of 2 / (3/4) = 8/3 = 2.6666 * size() before rehashing happens
    //this is not expensive - in the sense of statistic - like garbage collection - unless you are wiring money that requires certain latency otherwise people would die - I recommend to use this map

    template <class Key, class Mapped, class SizeType = std::size_t, class Hasher = std::hash<Key>, class Pred = std::equal_to<Key>, class Allocator = std::allocator<std::pair<Key, Mapped>>, class LoadFactor = std::ratio<3, 4>, class InsertFactor = std::ratio<2, 1>>
    class unordered_unstable_map{

        private:

            std::vector<std::pair<Key, Mapped>, typename std::allocator_traits<Allocator>::template rebind_alloc<std::pair<Key, Mapped>>> node_vec;
            std::vector<SizeType, typename std::allocator_traits<Allocator>::template rebind_alloc<SizeType>> bucket_vec;
            Hasher _hasher;
            Pred pred;
            Allocator allocator;
            SizeType erase_count;

            using bucket_iterator               = typename std::vector<SizeType, typename std::allocator_traits<Allocator>::template rebind_alloc<SizeType>>::iterator;
            using bucket_const_iterator         = typename std::vector<SizeType, typename std::allocator_traits<Allocator>::template rebind_alloc<SizeType>>::const_iterator;

            static inline constexpr SizeType NULL_VIRTUAL_ADDR          = std::numeric_limits<SizeType>::max();
            static inline constexpr SizeType ORPHANED_VIRTUAL_ADDR      = std::numeric_limits<SizeType>::max() - 1;
            static inline constexpr std::size_t REHASH_CHK_MODULO       = 16u;
            static inline constexpr std::size_t LAST_MOHICAN_SZ         = 16u;

            static constexpr auto is_insertable(SizeType virtual_addr) noexcept -> bool{

                return (virtual_addr | SizeType{1u}) == std::numeric_limits<SizeType>::max();
            }

        public:

            static constexpr inline double MIN_MAX_LOAD_FACTOR      = 0.05;
            static constexpr inline double MAX_MAX_LOAD_FACTOR      = 0.95; 
            static constexpr inline double MIN_MAX_INSERT_FACTOR    = 1;
            static constexpr inline double MAX_MAX_INSERT_FACTOR    = 32; 

            static_assert(std::is_unsigned_v<SizeType>);
            static_assert(noexcept(std::declval<const Hasher&>()(std::declval<const Key&>())));
            static_assert(noexcept(std::declval<Hasher&>()(std::declval<const Key&>())));
            // static_assert(noexcept(std::declval<const Pred&>()(std::declval<const Key&>(), std::declval<const Key&>())));
            // static_assert(noexcept(std::declval<Pred&>()(std::declval<const Key&>(), std::declval<const Key&>())));
            static_assert(std::is_nothrow_destructible_v<std::pair<Key, Mapped>>);

            using key_type                      = Key;
            using value_type                    = std::pair<Key, Mapped>; 
            using mapped_type                   = Mapped;
            using hasher                        = Hasher;
            using key_equal                     = Pred; 
            using allocator_type                = Allocator;
            using reference                     = value_type&;
            using const_reference               = const value_type&;
            using pointer                       = typename std::allocator_traits<Allocator>::pointer; 
            using const_pointer                 = typename std::allocator_traits<Allocator>::const_pointer;
            using iterator                      = typename std::vector<std::pair<Key, Mapped>, typename std::allocator_traits<Allocator>::template rebind_alloc<std::pair<Key, Mapped>>>::iterator; 
            using const_iterator                = typename std::vector<std::pair<Key, Mapped>, typename std::allocator_traits<Allocator>::template rebind_alloc<std::pair<Key, Mapped>>>::const_iterator; 
            using reverse_iterator              = typename std::vector<std::pair<Key, Mapped>, typename std::allocator_traits<Allocator>::template rebind_alloc<std::pair<Key, Mapped>>>::reverse_iterator;
            using const_reverse_iterator        = typename std::vector<std::pair<Key, Mapped>, typename std::allocator_traits<Allocator>::template rebind_alloc<std::pair<Key, Mapped>>>::const_reverse_iterator;
            using size_type                     = SizeType;
            using difference_type               = intmax_t;
            using self                          = unordered_unstable_map;
            using load_factor_ratio             = typename LoadFactor::type; 
            using insert_factor_ratio           = typename InsertFactor::type;

            static consteval auto max_load_factor() noexcept -> double{

                return static_cast<double>(load_factor_ratio::num) / load_factor_ratio::den;
            }

            static consteval auto max_insert_factor() noexcept -> double{

                return static_cast<double>(insert_factor_ratio::num) / insert_factor_ratio::den;
            }

            static_assert(std::clamp(self::max_load_factor(), MIN_MAX_LOAD_FACTOR, MAX_MAX_LOAD_FACTOR) == self::max_load_factor());
            static_assert(std::clamp(self::max_insert_factor(), MIN_MAX_INSERT_FACTOR, MAX_MAX_INSERT_FACTOR) == self::max_insert_factor());

            constexpr unordered_unstable_map(): node_vec(), 
                                                bucket_vec(self::min_capacity() + LAST_MOHICAN_SZ, NULL_VIRTUAL_ADDR),
                                                _hasher(),
                                                pred(),
                                                allocator(),
                                                erase_count(0u){}

            constexpr explicit unordered_unstable_map(size_type bucket_count, 
                                                      const Hasher& _hasher = Hasher(), 
                                                      const Pred& pred = Pred(), 
                                                      const Allocator& allocator = Allocator()): node_vec(allocator),
                                                                                                 bucket_vec(std::max(self::min_capacity(), dg::map_variants::least_pow2_greater_equal_than(bucket_count)) + LAST_MOHICAN_SZ, NULL_VIRTUAL_ADDR, allocator),
                                                                                                 _hasher(_hasher),
                                                                                                 pred(pred),
                                                                                                 allocator(allocator),
                                                                                                 erase_count(0u){
                
                node_vec.reserve(estimate_size(capacity()));
            }

            constexpr unordered_unstable_map(size_type bucket_count, 
                                             const Allocator& allocator): node_vec(allocator),
                                                                          bucket_vec(std::max(self::min_capacity(), dg::map_variants::least_pow2_greater_equal_than(bucket_count)) + LAST_MOHICAN_SZ, NULL_VIRTUAL_ADDR, allocator),
                                                                          _hasher(),
                                                                          pred(),
                                                                          allocator(allocator),
                                                                          erase_count(0u){
                node_vec.reserve(estimate_size(capacity()));                                                                
            }

            constexpr unordered_unstable_map(size_type bucket_count, 
                                             const Hasher& _hasher, 
                                             const Allocator& allocator): node_vec(allocator),
                                                                          bucket_vec(std::max(self::min_capacity(), dg::map_variants::least_pow2_greater_equal_than(bucket_count)) + LAST_MOHICAN_SZ, NULL_VIRTUAL_ADDR, allocator),
                                                                          _hasher(_hasher),
                                                                          pred(),
                                                                          allocator(allocator),
                                                                          erase_count(0u){

                node_vec.reserve(estimate_size(capacity()));
            }

            constexpr explicit unordered_unstable_map(const Allocator& allocator): node_vec(allocator),
                                                                                   bucket_vec(min_capacity() + LAST_MOHICAN_SZ, NULL_VIRTUAL_ADDR, allocator),
                                                                                   _hasher(),
                                                                                   pred(),
                                                                                   allocator(allocator),
                                                                                   erase_count(0u){

                node_vec.reserve(estimate_size(capacity()));
            }

            template <class InputIt>
            constexpr unordered_unstable_map(InputIt first, 
                                             InputIt last, 
                                             size_type bucket_count, 
                                             const Hasher& _hasher = Hasher(), 
                                             const Pred& pred = Pred(), 
                                             const Allocator& allocator = Allocator()): unordered_unstable_map(bucket_count, _hasher, pred, allocator){
                
                insert(first, last);
            }

            template <class InputIt>
            constexpr unordered_unstable_map(InputIt first, 
                                             InputIt last, 
                                             size_type bucket_count, 
                                             const Allocator& allocator): unordered_unstable_map(first, last, bucket_count, Hasher(), Pred(), allocator){}

            constexpr unordered_unstable_map(std::initializer_list<value_type> init_list, 
                                             size_type bucket_count, 
                                             const Allocator& allocator): unordered_unstable_map(init_list.begin(), init_list.end(), bucket_count, allocator){}

            constexpr unordered_unstable_map(std::initializer_list<value_type> init_list, 
                                             size_type bucket_count, 
                                             const Hasher& _hasher, 
                                             const Allocator& allocator): unordered_unstable_map(init_list.begin(), init_list.end(), bucket_count, _hasher, allocator){}

            constexpr void clear() noexcept{

                std::fill(bucket_vec.begin(), bucket_vec.end(), NULL_VIRTUAL_ADDR);
                node_vec.clear();
            }

            constexpr void rehash(size_type tentative_new_cap, bool force_rehash = false){

                if (!force_rehash && tentative_new_cap <= capacity()){
                    return;
                }

                decltype(bucket_vec) bucket_proxy = std::move(bucket_vec); 

                try{
                    while (true){
                        size_t new_cap  = std::max(self::min_capacity(), dg::map_variants::least_pow2_greater_equal_than(tentative_new_cap)) + LAST_MOHICAN_SZ;
                        bool bad_bit    = false; 
                        bucket_vec.resize(new_cap, NULL_VIRTUAL_ADDR);

                        for (size_t i = 0u; i < node_vec.size(); ++i){
                            auto it = bucket_efind(node_vec[i].first); //its fine to invoke internal method here because we are in a valid state - in other words, node_vec can contain allocations that aren't referenced by one of the buckets
                            if (it != std::prev(bucket_vec.end())) [[likely]]{
                                *it = i; 
                            } else [[unlikely]]{
                                tentative_new_cap = new_cap * 2;
                                bad_bit = true;
                                break;
                            }
                        }

                        if (!bad_bit){
                            break;
                        }
                    }

                    erase_count = 0u;
                } catch (...){
                    bucket_vec = std::move(bucket_proxy);
                    std::rethrow_exception(std::current_exception());
                }
            }

            constexpr void reserve(size_type new_sz){
 
                if (new_sz <= size()){
                    return;
                }

                rehash(estimate_capacity(new_sz));
            }

            constexpr void swap(self& other) noexcept(std::allocator_traits<Allocator>::is_always_equal
                                                      && std::is_nothrow_swappable<Hasher>::value
                                                      && std::is_nothrow_swappable<Pred>::value){
                
                std::swap(node_vec, other.node_vec);
                std::swap(bucket_vec, other.bucket_vec);
                std::swap(_hasher, other._hasher);
                std::swap(pred, other.pred);
                std::swap(allocator, other.allocator);
                std::swap(erase_count, other.erase_count);
            }

            template <class ...Args>
            constexpr auto emplace(Args&& ...args) -> std::pair<iterator, bool>{

                return internal_insert_or_assign(value_type(std::forward<Args>(args)...));
            }

            template <class KeyLike, class ...Args>
            constexpr auto try_emplace(KeyLike&& key, Args&& ...args) -> std::pair<iterator, bool>{

                return internal_insert(value_type(std::piecewise_construct, std::forward_as_tuple(std::forward<KeyLike>(key)), std::forward_as_tuple(std::forward<Args>(args)...)));
            }

            template <class ValueLike = value_type> //the only problem we were trying to solve was adding implicit initialization of value_type - so this should solve it
            constexpr auto insert(ValueLike&& value) -> std::pair<iterator, bool>{

                return internal_insert(std::forward<ValueLike>(value));
            }

            template <class Iterator>
            constexpr void insert(Iterator first, Iterator last){

                while (first != last){
                    internal_insert(*first);
                    std::advance(first, 1);
                }
            }

            constexpr void insert(std::initializer_list<value_type> init_list){

                insert(init_list.begin(), init_list.end());
            }

            template <class KeyLike>
            constexpr auto insert_or_assign(KeyLike&& key, mapped_type&& mapped) -> std::pair<iterator, bool>{

                return internal_insert_or_assign(value_type(std::forward<KeyLike>(key), std::forward<mapped_type>(mapped)));
            }

            constexpr auto erase(const_iterator it) noexcept -> iterator{

                return internal_erase(it);
            }

            template <class KeyLike>
            constexpr auto erase(const KeyLike& key) noexcept(noexcept(internal_erase(std::declval<const KeyLike&>()))) -> size_t{

                return internal_erase(key);
            }

            template <class Iterator>
            constexpr auto erase(Iterator first, Iterator last) noexcept(noexcept(internal_erase(*std::declval<Iterator>()))){

                while (first != last){
                    internal_erase(*first);
                    std::advance(first, 1);
                }
            }

            template <class KeyLike>
            constexpr auto operator[](KeyLike&& key) -> mapped_type&{

                return internal_find_or_default(std::forward<KeyLike>(key))->second;
            }

            template <class KeyLike>
            constexpr auto contains(const KeyLike& key) const noexcept(bucket_find(std::declval<const KeyLike&>())) -> bool{

                return *bucket_find(key) != NULL_VIRTUAL_ADDR;
            }

            template <class KeyLike>
            constexpr auto count(const KeyLike& key) const noexcept(bucket_find(std::declval<const KeyLike&>())) -> size_type{

                return *bucket_find(key) != NULL_VIRTUAL_ADDR;
            }

            template <class KeyLike>
            constexpr auto exist_find(const KeyLike& key) noexcept(noexcept(bucket_exist_find(std::declval<const KeyLike&>()))) -> iterator{

                return std::next(node_vec.begin(), *bucket_exist_find(key));
            }

            template <class KeyLike>
            constexpr auto exist_find(const KeyLike& key) const noexcept(noexcept(bucket_exist_find(std::declval<const KeyLike&>()))) -> const_iterator{

                return std::next(node_vec.begin(), *bucket_exist_find(key));
            }

            template <class KeyLike>
            constexpr auto find(const KeyLike& key) noexcept(noexcept(bucket_find(std::declval<const KeyLike&>()))) -> iterator{

                size_type virtual_addr = *bucket_find(key);

                if (virtual_addr == NULL_VIRTUAL_ADDR){
                    return node_vec.end();
                }

                return std::next(node_vec.begin(), virtual_addr);
            }

            template <class KeyLike>
            constexpr auto find(const KeyLike& key) const noexcept(noexcept(bucket_find(std::declval<const KeyLike&>()))) -> const_iterator{

                size_type virtual_addr = *bucket_find(key);

                if (virtual_addr == NULL_VIRTUAL_ADDR){
                    return node_vec.end();
                }

                return std::next(node_vec.begin(), virtual_addr);
            }

            template <class KeyLike>
            constexpr auto at(const KeyLike& key) const noexcept(noexcept(exist_find(std::declval<const KeyLike&>()))) -> const mapped_type&{

                return exist_find(key)->second;
            }

            template <class KeyLike>
            constexpr auto at(const KeyLike& key) noexcept(noexcept(exist_find(std::declval<const KeyLike&>()))) -> mapped_type&{

                return exist_find(key)->second;
            }

            constexpr auto empty() const noexcept -> bool{

                return node_vec.empty();
            }

            constexpr auto min_capacity() noexcept -> size_type{

                return 32u;
            }

            constexpr auto capacity() noexcept -> size_type{

                return bucket_vec.size() - LAST_MOHICAN_SZ;
            }

            constexpr auto size() const noexcept -> size_type{

                return node_vec.size();
            }

            constexpr auto insert_size() const noexcept -> size_type{

                return size() + erase_count;
            }

            constexpr auto max_size() const noexcept -> size_type{

                return std::numeric_limits<size_type>::max();
            }

            constexpr auto hash_function() const & noexcept -> const Hasher&{

                return _hasher;
            }

            constexpr auto key_eq() const & noexcept -> const Pred&{

                return pred;
            }

            constexpr auto hash_function() && noexcept -> Hasher&&{

                return static_cast<Hasher&&>(_hasher);
            }

            constexpr auto key_eq() && noexcept -> Pred&&{

                return static_cast<Pred&&>(pred);
            }

            constexpr auto load_factor() const noexcept -> double{

                return size() / static_cast<double>(capacity());
            }

            constexpr auto insert_factor() const noexcept -> double{

                return (size() + erase_count) / static_cast<double>(capacity());
            }

            constexpr auto begin() noexcept -> iterator{

                return node_vec.begin();
            }

            constexpr auto begin() const noexcept -> const_iterator{

                return node_vec.begin();
            }

            constexpr auto cbegin() noexcept -> reverse_iterator{

                return node_vec.cbegin();
            }

            constexpr auto cbegin() const noexcept -> const_reverse_iterator{

                return node_vec.cbegin();
            }

            constexpr auto end() noexcept -> iterator{

                return node_vec.end();
            }

            constexpr auto end() const noexcept -> const_iterator{

                return node_vec.end();
            }

            constexpr auto cend() noexcept -> reverse_iterator{

                return node_vec.cend();
            }

            constexpr auto cend() const noexcept -> const_reverse_iterator{

                return node_vec.cend();
            }

        private:

            constexpr void maybe_check_for_rehash(){

                if (((size() + erase_count) % REHASH_CHK_MODULO) != 0u) [[likely]]{
                    return;
                } else{
                    if (estimate_capacity(node_vec.size()) <= capacity() && insert_size() <= estimate_insert_capacity(capacity())) [[likely]]{
                        return;
                    } else [[unlikely]]{
                        //either cap > size or insert_cap > max_insert_cap or both - if both - extend
                        if (estimate_capacity(node_vec.size()) > capacity()){
                            size_type new_cap = capacity() * 2;
                            rehash(new_cap, true);
                        } else{
                            size_type new_cap = estimate_capacity(node_vec.size());
                            rehash(new_cap, true);
                        }
                    }
                }
            }

            constexpr void force_uphash(){

                size_type new_cap = capacity() * 2;
                rehash(new_cap, true);
            }

            constexpr auto estimate_size(size_type cap) const noexcept -> size_type{

                return cap * load_factor_ratio::num / load_factor_ratio::den;
            }

            constexpr auto estimate_capacity(size_type sz) const noexcept -> size_type{

                return sz * load_factor_ratio::den / load_factor_ratio::num;
            }

            constexpr auto estimate_insert_capacity(size_type cap) const noexcept -> size_type{

                return cap * insert_factor_ratio::num / insert_factor_ratio::den;
            }

            constexpr auto to_bucket_index(size_type hashed_value) const noexcept -> size_type{

                return hashed_value & (bucket_vec.size() - (LAST_MOHICAN_SZ + 1u));
            }

            template <class KeyLike>
            constexpr auto bucket_exist_find(const KeyLike& key) noexcept(noexcept(std::declval<Hasher&>()(std::declval<const KeyLike&>()))
                                                                          && noexcept(std::declval<Pred&>()(std::declval<const Key&>(), std::declval<const KeyLike&>()))) -> bucket_iterator{

                auto it = std::next(bucket_vec.begin(), to_bucket_index(_hasher(key)));

                while (true){
                    if (*it != ORPHANED_VIRTUAL_ADDR && pred(static_cast<const Key&>(node_vec[*it].first), key)){
                        return it;
                    }

                    std::advance(it, 1u);
                }
            }

            template <class KeyLike>
            constexpr auto bucket_exist_find(const KeyLike& key) const noexcept(noexcept(std::declval<const Hasher&>()(std::declval<const KeyLike&>()))
                                                                                 && noexcept(std::declval<const Pred&>()(std::declval<const Key&>(), std::declval<const KeyLike&>()))) -> bucket_const_iterator{

                auto it = std::next(bucket_vec.begin(), to_bucket_index(_hasher(key)));

                while (true){
                    if (*it != ORPHANED_VIRTUAL_ADDR && pred(node_vec[*it].first, key)){
                        return it;
                    }

                    std::advance(it, 1u);
                }
            }

            template <class KeyLike>
            constexpr auto bucket_find(const KeyLike& key) noexcept(noexcept(std::declval<Hasher&>()(std::declval<const KeyLike&>()))
                                                                    && noexcept(std::declval<Pred&>()(std::declval<const Key&>(), std::declval<const KeyLike&>()))) -> bucket_iterator{

                auto it = std::next(bucket_vec.begin(), to_bucket_index(_hasher(key)));

                while (true){
                    if (*it == NULL_VIRTUAL_ADDR || (*it != ORPHANED_VIRTUAL_ADDR && pred(static_cast<const Key&>(node_vec[*it].first), key))){
                        return it;
                    }

                    std::advance(it, 1u);
                }
            }

            template <class KeyLike>
            constexpr auto bucket_find(const KeyLike& key) const noexcept(noexcept(std::declval<const Hasher&>()(std::declval<const KeyLike&>()))
                                                                          && noexcept(std::declval<const Pred&>()(std::declval<const Key&>(), std::declval<const KeyLike&>()))) -> bucket_const_iterator{

                auto it = std::next(bucket_vec.begin(), to_bucket_index(_hasher(key)));

                while (true){
                    if (*it == NULL_VIRTUAL_ADDR || (*it != ORPHANED_VIRTUAL_ADDR && pred(node_vec[*it].first, key))){
                        return it;
                    }

                    std::advance(it, 1u);
                }
            }

            template <class KeyLike>
            constexpr auto bucket_efind(const KeyLike& key) noexcept(noexcept(std::declval<Hasher&>()(std::declval<const KeyLike&>()))) -> bucket_iterator{

                auto it = std::next(bucket_vec.begin(), to_bucket_index(_hasher(key)));

                while (true){
                    if (*it == NULL_VIRTUAL_ADDR){
                        return it;
                    }

                    std::advance(it, 1u);
                }
            }

            template <class KeyLike>
            constexpr auto bucket_efind(const KeyLike& key) const noexcept(noexcept(std::declval<Hasher&>()(std::declval<const KeyLike&>()))) -> bucket_const_iterator{

                auto it = std::next(bucket_vec.begin(), to_bucket_index(_hasher(key)));

                while (true){
                    if (*it == NULL_VIRTUAL_ADDR){
                        return it;
                    }

                    std::advance(it, 1u);
                }
            }

            template <class KeyLike>
            constexpr auto bucket_ifind(const KeyLike& key) noexcept(noexcept(std::declval<Hasher&>()(std::declval<const KeyLike&>()))) -> bucket_iterator{

                auto it = std::next(bucket_vec.begin(), to_bucket_index(_hasher(key)));

                while (true){
                    if (is_insertable(*it)){
                        return it;
                    }

                    std::advance(it, 1u);
                }
            }

            template <class ValueLike>
            constexpr auto do_insert_at(bucket_iterator it, ValueLike&& value) -> iterator{

                size_type addr = node_vec.size();
                node_vec.push_back(std::forward<ValueLike>(value));
                *it = addr;
                return std::prev(node_vec.end());
            }

            template <class ValueLike>
            constexpr auto internal_noexist_insert(ValueLike&& value) -> iterator{

                maybe_check_for_rehash();

                while (true){
                    bucket_iterator it = bucket_ifind(value.first);

                    if (it != std::prev(bucket_vec.end())) [[likely]]{
                        return do_insert_at(it, std::forward<ValueLike>(value));
                    } else [[unlikely]]{
                        force_uphash();
                    }
                }
            } 

            template <class ValueLike>
            constexpr auto internal_insert_or_assign(ValueLike&& value) -> std::pair<iterator, bool>{

                bucket_iterator it = bucket_find(value.first);

                if (*it != NULL_VIRTUAL_ADDR){
                    iterator rs = std::next(node_vec.begin(), *it);
                    *rs = std::forward<ValueLike>(value);
                    return std::make_pair(rs, false);
                }

                return std::make_pair(internal_noexist_insert(std::forward<ValueLike>(value)), true);
            }

            template <class ValueLike>
            constexpr auto internal_insert(ValueLike&& value) -> std::pair<iterator, bool>{

                bucket_iterator it = bucket_find(value.first);

                if (*it != NULL_VIRTUAL_ADDR){
                    return std::make_pair(std::next(node_vec.begin(), *it), false);
                }

                return std::make_pair(internal_noexist_insert(std::forward<ValueLike>(value)), true);
            }

            template <class KeyLike, class Arg = mapped_type, std::enable_if_t<std::is_default_constructible_v<Arg>, bool> = true>
            constexpr auto internal_find_or_default(KeyLike&& key, Arg * compiler_hint = nullptr) -> iterator{

                bucket_iterator it = bucket_find(key);

                if (*it != NULL_VIRTUAL_ADDR){
                    return std::next(node_vec.begin(), *it);
                }

                return internal_noexist_insert(value_type(std::forward<KeyLike>(key), mapped_type()));
            }

            template <class KeyLike>
            constexpr auto internal_erase(const KeyLike& key) noexcept(noexcept(std::declval<Hasher&>()(std::declval<const KeyLike&>()))) -> size_type{

                bucket_iterator erasing_bucket_it       = bucket_find(key);
                size_type erasing_bucket_virtual_addr   = *erasing_bucket_it;

                if (erasing_bucket_virtual_addr != NULL_VIRTUAL_ADDR){
                    bucket_iterator swapee_bucket_it = bucket_exist_find(node_vec.back().first); 
                    std::iter_swap(std::next(node_vec.begin(), erasing_bucket_virtual_addr), std::prev(node_vec.end()));
                    node_vec.pop_back();
                    *swapee_bucket_it   = erasing_bucket_virtual_addr;
                    *erasing_bucket_it  = ORPHANED_VIRTUAL_ADDR;
                    erase_count         += 1;

                    return 1u;
                }

                return 0u;
            }

            template <class KeyLike>
            constexpr void internal_exist_erase(const KeyLike& key) noexcept(noexcept(std::declval<Hasher&>()(std::declval<const KeyLike&>()))){

                bucket_iterator erasing_bucket_it       = bucket_exist_find(key);
                size_type erasing_bucket_virtual_addr   = *erasing_bucket_it;
                bucket_iterator swapee_bucket_it        = bucket_exist_find(node_vec.back().first);

                std::iter_swap(std::next(node_vec.begin(), erasing_bucket_virtual_addr), std::prev(node_vec.end()));
                node_vec.pop_back();
                *swapee_bucket_it   = erasing_bucket_virtual_addr;
                *erasing_bucket_it  = ORPHANED_VIRTUAL_ADDR;
                erase_count         += 1;
            }

            constexpr auto internal_erase(const_iterator it) noexcept -> iterator{

                if (it == node_vec.end()){
                    return it;
                }

                auto dif = std::distance(node_vec.begin(), it);
                internal_exist_erase(it->first);

                if (dif != node_vec.size()) [[likely]]{
                    return std::next(node_vec.begin(), dif);
                } else [[unlikely]]{
                    return node_vec.begin();
                }
            }
    };

    //this map is extremely fast if you use it for const lookup purposes - where there is no insert, erase and the usage of at(const KeyLike&) is pivotal
    template <class Key, class Mapped, class NullValueGenerator, class SizeType = std::size_t, class Hasher = std::hash<Key>, class Pred = std::equal_to<Key>, class Allocator = std::allocator<std::pair<Key, Mapped>>, class LoadFactor = std::ratio<3, 4>, class InsertFactor = std::ratio<2, 1>>
    class unordered_unstable_fast_map{

        private:

            std::vector<std::pair<Key, Mapped>, typename std::allocator_traits<Allocator>::template rebind_alloc<std::pair<Key, Mapped>>> node_vec;
            std::vector<SizeType, typename std::allocator_traits<Allocator>::template rebind_alloc<SizeType>> bucket_vec;
            Hasher _hasher;
            Pred pred;
            Allocator allocator;
            SizeType erase_count;

            using bucket_iterator               = typename std::vector<SizeType, typename std::allocator_traits<Allocator>::template rebind_alloc<SizeType>>::iterator;
            using bucket_const_iterator         = typename std::vector<SizeType, typename std::allocator_traits<Allocator>::template rebind_alloc<SizeType>>::const_iterator;

            static inline constexpr SizeType ORPHANED_VIRTUAL_ADDR      = std::numeric_limits<SizeType>::min();
            static inline constexpr SizeType NULL_VIRTUAL_ADDR          = std::numeric_limits<SizeType>::max();
            static inline constexpr std::size_t REHASH_CHK_MODULO       = 16u;
            static inline constexpr std::size_t LAST_MOHICAN_SZ         = 16u;

            static constexpr auto is_insertable(SizeType virtual_addr) noexcept -> bool{

                return virtual_addr == NULL_VIRTUAL_ADDR || virtual_addr == ORPHANED_VIRTUAL_ADDR;
            }

        public:

            static_assert(std::is_unsigned_v<SizeType>);
            static_assert(noexcept(std::declval<const Hasher&>()(std::declval<const Key&>())));
            static_assert(noexcept(std::declval<Hasher&>()(std::declval<const Key&>())));
            // static_assert(noexcept(std::declval<const Pred&>()(std::declval<const Key&>(), std::declval<const Key&>())));
            // static_assert(noexcept(std::declval<Pred&>()(std::declval<const Key&>(), std::declval<const Key&>())));
            static_assert(std::is_nothrow_destructible_v<std::pair<Key, Mapped>>);

            static constexpr inline double MIN_MAX_LOAD_FACTOR      = 0.05;
            static constexpr inline double MAX_MAX_LOAD_FACTOR      = 0.95; 
            static constexpr inline double MIN_MAX_INSERT_FACTOR    = 0.05;
            static constexpr inline double MAX_MAX_INSERT_FACTOR    = 8; 

            using key_type                      = Key;
            using value_type                    = std::pair<Key, Mapped>; 
            using mapped_type                   = Mapped;
            using hasher                        = Hasher;
            using key_equal                     = Pred; 
            using allocator_type                = Allocator;
            using reference                     = value_type&;
            using const_reference               = const value_type&;
            using pointer                       = typename std::allocator_traits<Allocator>::pointer; 
            using const_pointer                 = typename std::allocator_traits<Allocator>::const_pointer;
            using iterator                      = typename std::vector<std::pair<Key, Mapped>, typename std::allocator_traits<Allocator>::template rebind_alloc<std::pair<Key, Mapped>>>::iterator; 
            using const_iterator                = typename std::vector<std::pair<Key, Mapped>, typename std::allocator_traits<Allocator>::template rebind_alloc<std::pair<Key, Mapped>>>::const_iterator; 
            using reverse_iterator              = typename std::vector<std::pair<Key, Mapped>, typename std::allocator_traits<Allocator>::template rebind_alloc<std::pair<Key, Mapped>>>::reverse_iterator;
            using const_reverse_iterator        = typename std::vector<std::pair<Key, Mapped>, typename std::allocator_traits<Allocator>::template rebind_alloc<std::pair<Key, Mapped>>>::const_reverse_iterator;
            using size_type                     = SizeType;
            using difference_type               = intmax_t;
            using self                          = unordered_unstable_fast_map;
            using load_factor_ratio             = typename LoadFactor::type; 
            using insert_factor_ratio           = typename InsertFactor::type;

            static consteval auto max_load_factor() noexcept -> double{

                return static_cast<double>(load_factor_ratio::num) / load_factor_ratio::den;
            }

            static consteval auto max_insert_factor() noexcept -> double{

                return static_cast<double>(insert_factor_ratio::num) / insert_factor_ratio::den;
            }

            static_assert(std::clamp(self::max_load_factor(), MIN_MAX_LOAD_FACTOR, MAX_MAX_LOAD_FACTOR) == self::max_load_factor());
            static_assert(std::clamp(self::max_insert_factor(), MIN_MAX_INSERT_FACTOR, MAX_MAX_INSERT_FACTOR) == self::max_insert_factor());

            constexpr unordered_unstable_fast_map(): node_vec(), 
                                                     bucket_vec(self::min_capacity() + LAST_MOHICAN_SZ, NULL_VIRTUAL_ADDR),
                                                     _hasher(),
                                                     pred(),
                                                     allocator(),
                                                     erase_count(0u){

                node_vec.push_back(NullValueGenerator{}());
            }

            constexpr explicit unordered_unstable_fast_map(size_type bucket_count, 
                                                           const Hasher& _hasher = Hasher(), 
                                                           const Pred& pred = Pred(), 
                                                           const Allocator& allocator = Allocator()): node_vec(allocator),
                                                                                                      bucket_vec(std::max(self::min_capacity(), dg::map_variants::least_pow2_greater_equal_than(bucket_count)) + LAST_MOHICAN_SZ, NULL_VIRTUAL_ADDR, allocator),
                                                                                                      _hasher(_hasher),
                                                                                                      pred(pred),
                                                                                                      allocator(allocator),
                                                                                                      erase_count(0u){
                
                node_vec.reserve(estimate_size(capacity()) + 1u);
                node_vec.push_back(NullValueGenerator{}());
            }

            constexpr unordered_unstable_fast_map(size_type bucket_count, 
                                                  const Allocator& allocator): node_vec(allocator),
                                                                               bucket_vec(std::max(self::min_capacity(), dg::map_variants::least_pow2_greater_equal_than(bucket_count)) + LAST_MOHICAN_SZ, NULL_VIRTUAL_ADDR, allocator),
                                                                               _hasher(),
                                                                               pred(),
                                                                               allocator(allocator),
                                                                               erase_count(0u){
                
                node_vec.reserve(estimate_size(capacity()) + 1u);
                node_vec.push_back(NullValueGenerator{}());
            }

            constexpr unordered_unstable_fast_map(size_type bucket_count, 
                                                  const Hasher& _hasher, 
                                                  const Allocator& allocator): node_vec(allocator),
                                                                               bucket_vec(std::max(self::min_capacity(), dg::map_variants::least_pow2_greater_equal_than(bucket_count)) + LAST_MOHICAN_SZ, NULL_VIRTUAL_ADDR, allocator),
                                                                               _hasher(_hasher),
                                                                               pred(),
                                                                               allocator(allocator),
                                                                               erase_count(0u){
                node_vec.reserve(estimate_size(capacity()) + 1u);
                node_vec.push_back(NullValueGenerator{}());
            }

            constexpr explicit unordered_unstable_fast_map(const Allocator& allocator): node_vec(allocator),
                                                                                        bucket_vec(min_capacity() + LAST_MOHICAN_SZ, NULL_VIRTUAL_ADDR, allocator),
                                                                                        _hasher(),
                                                                                        pred(),
                                                                                        allocator(allocator),
                                                                                        erase_count(0u){

                node_vec.reserve(estimate_size(capacity()) + 1u);
                node_vec.push_back(NullValueGenerator{}());
            }

            template <class InputIt>
            constexpr unordered_unstable_fast_map(InputIt first, 
                                                  InputIt last, 
                                                  size_type bucket_count, 
                                                  const Hasher& _hasher = Hasher(), 
                                                  const Pred& pred = Pred(), 
                                                  const Allocator& allocator = Allocator()): unordered_unstable_fast_map(bucket_count, _hasher, pred, allocator){
                
                insert(first, last);
            }

            template <class InputIt>
            constexpr unordered_unstable_fast_map(InputIt first, 
                                                  InputIt last, 
                                                  size_type bucket_count, 
                                                  const Allocator& allocator): unordered_unstable_fast_map(first, last, bucket_count, Hasher(), Pred(), allocator){}

            constexpr unordered_unstable_fast_map(std::initializer_list<value_type> init_list, 
                                                  size_type bucket_count, 
                                                  const Allocator& allocator): unordered_unstable_fast_map(init_list.begin(), init_list.end(), bucket_count, allocator){}

            constexpr unordered_unstable_fast_map(std::initializer_list<value_type> init_list, 
                                                  size_type bucket_count, 
                                                  const Hasher& _hasher, 
                                                  const Allocator& allocator): unordered_unstable_fast_map(init_list.begin(), init_list.end(), bucket_count, _hasher, allocator){}

            constexpr void clear() noexcept{

                std::fill(bucket_vec.begin(), bucket_vec.end(), NULL_VIRTUAL_ADDR);
                node_vec.resize(1u);
            }

            constexpr void rehash(size_type tentative_new_cap, bool force_rehash = false){

                if (!force_rehash && tentative_new_cap <= capacity()){
                    return;
                }

                decltype(bucket_vec) bucket_proxy = std::move(bucket_vec); 

                try{
                    while (true){
                        size_t new_cap  = std::max(self::min_capacity(), dg::map_variants::least_pow2_greater_equal_than(tentative_new_cap)) + LAST_MOHICAN_SZ;
                        bool bad_bit    = false; 
                        bucket_vec.resize(new_cap, NULL_VIRTUAL_ADDR);

                        for (size_t i = 1u; i < node_vec.size(); ++i){
                            auto it = bucket_efind(node_vec[i].first); //its fine to invoke internal method here because we are in a valid state - in other words, node_vec can contain allocations that aren't referenced by one of the buckets
                            if (it != std::prev(bucket_vec.end())) [[likely]]{
                                *it = i; 
                            } else [[unlikely]]{
                                tentative_new_cap = new_cap * 2;
                                bad_bit = true;
                                break;
                            }
                        }

                        if (!bad_bit){
                            break;
                        }
                    }

                    erase_count = 0u;
                } catch (...){
                    bucket_vec = std::move(bucket_proxy);
                    std::rethrow_exception(std::current_exception());
                }
            }

            constexpr void reserve(size_type new_sz){
 
                if (new_sz < node_vec.size()){
                    return;
                }

                rehash(estimate_capacity(new_sz));
            }

            constexpr void swap(self& other) noexcept(std::allocator_traits<Allocator>::is_always_equal
                                                      && std::is_nothrow_swappable<Hasher>::value
                                                      && std::is_nothrow_swappable<Pred>::value){
                
                std::swap(node_vec, other.node_vec);
                std::swap(bucket_vec, other.bucket_vec);
                std::swap(_hasher, other._hasher);
                std::swap(pred, other.pred);
                std::swap(allocator, other.allocator);
                std::swap(erase_count, other.erase_count);
            }

            template <class ...Args>
            constexpr auto emplace(Args&& ...args) -> std::pair<iterator, bool>{

                return internal_insert_or_assign(value_type(std::forward<Args>(args)...));
            }

            template <class KeyLike, class ...Args>
            constexpr auto try_emplace(KeyLike&& key, Args&& ...args) -> std::pair<iterator, bool>{

                return internal_insert(value_type(std::piecewise_construct, std::forward_as_tuple(std::forward<KeyLike>(key)), std::forward_as_tuple(std::forward<Args>(args)...)));
            }

            template <class ValueLike = value_type>
            constexpr auto insert(ValueLike&& value) -> std::pair<iterator, bool>{

                return internal_insert(std::forward<ValueLike>(value));
            }

            template <class Iterator>
            constexpr void insert(Iterator first, Iterator last){

                while (first != last){
                    internal_insert(*first);
                    std::advance(first, 1);
                }
            }

            constexpr void insert(std::initializer_list<value_type> init_list){

                insert(init_list.begin(), init_list.end());
            }

            template <class KeyLike>
            constexpr auto insert_or_assign(KeyLike&& key, mapped_type&& mapped) -> std::pair<iterator, bool>{

                return internal_insert_or_assign(value_type(std::forward<KeyLike>(key), std::forward<mapped_type>(mapped)));
            }

            constexpr auto erase(const_iterator it) noexcept -> iterator{

                return internal_erase(it);
            }

            template <class KeyLike>
            constexpr auto erase(const KeyLike& key) noexcept(noexcept(internal_erase(std::declval<const KeyLike&>()))) -> size_t{

                return internal_erase(key);
            }

            template <class Iterator>
            constexpr auto erase(Iterator first, Iterator last) noexcept(noexcept(internal_erase(*std::declval<Iterator>()))){

                while (first != last){
                    internal_erase(*first);
                    std::advance(first, 1);
                }
            }

            template <class KeyLike>
            constexpr auto operator[](KeyLike&& key) -> mapped_type&{

                return internal_find_or_default(std::forward<KeyLike>(key))->second;
            }

            template <class KeyLike>
            constexpr auto contains(const KeyLike& key) const noexcept(bucket_find(std::declval<const KeyLike&>())) -> bool{

                return *bucket_find(key) != NULL_VIRTUAL_ADDR;
            }

            template <class KeyLike>
            constexpr auto count(const KeyLike& key) const noexcept(bucket_find(std::declval<const KeyLike&>())) -> size_type{

                return *bucket_find(key) != NULL_VIRTUAL_ADDR;
            }

            template <class KeyLike>
            constexpr auto exist_find(const KeyLike& key) noexcept(noexcept(bucket_exist_find(std::declval<const KeyLike&>()))) -> iterator{

                return std::next(node_vec.begin(), *bucket_exist_find(key));
            }

            template <class KeyLike>
            constexpr auto exist_find(const KeyLike& key) const noexcept(noexcept(bucket_exist_find(std::declval<const KeyLike&>()))) -> const_iterator{

                return std::next(node_vec.begin(), *bucket_exist_find(key));
            }

            template <class KeyLike>
            constexpr auto find(const KeyLike& key) noexcept(noexcept(bucket_find(std::declval<const KeyLike&>()))) -> iterator{

                size_type virtual_addr = *bucket_find(key);

                if (virtual_addr == NULL_VIRTUAL_ADDR){
                    return node_vec.end();
                }

                return std::next(node_vec.begin(), virtual_addr);
            }

            template <class KeyLike>
            constexpr auto find(const KeyLike& key) const noexcept(noexcept(bucket_find(std::declval<const KeyLike&>()))) -> const_iterator{

                size_type virtual_addr = *bucket_find(key);

                if (virtual_addr == NULL_VIRTUAL_ADDR){
                    return node_vec.end();
                }

                return std::next(node_vec.begin(), virtual_addr);
            }

            template <class KeyLike>
            constexpr auto at(const KeyLike& key) const noexcept(noexcept(exist_find(std::declval<const KeyLike&>()))) -> const mapped_type&{

                return exist_find(key)->second;
            }

            template <class KeyLike>
            constexpr auto at(const KeyLike& key) noexcept(noexcept(exist_find(std::declval<const KeyLike&>()))) -> mapped_type&{

                return exist_find(key)->second;
            }

            constexpr auto empty() const noexcept -> bool{

                return size() != 0u;
            }

            constexpr auto min_capacity() noexcept -> size_type{

                return 32u;
            }

            constexpr auto capacity() noexcept -> size_type{

                return bucket_vec.size() - LAST_MOHICAN_SZ;
            }

            constexpr auto size() const noexcept -> size_type{

                return node_vec.size() - 1u;
            }

            constexpr auto insert_size() const noexcept -> size_type{

                return size() + erase_count;
            }

            constexpr auto max_size() const noexcept -> size_type{

                return std::numeric_limits<size_type>::max();
            }

            constexpr auto hash_function() const & noexcept -> const Hasher&{

                return _hasher;
            }

            constexpr auto key_eq() const & noexcept -> const Pred&{

                return pred;
            }

            constexpr auto hash_function() && noexcept -> Hasher&&{

                return static_cast<Hasher&&>(_hasher);
            }

            constexpr auto key_eq() && noexcept -> Pred&&{

                return static_cast<Pred&&>(pred);
            }
            
            constexpr auto load_factor() const noexcept -> double{

                return size() / static_cast<double>(capacity());
            }

            constexpr auto insert_factor() const noexcept -> double{

                return (size() + erase_count) / static_cast<double>(capacity());
            }

            constexpr auto begin() noexcept -> iterator{

                return std::next(node_vec.begin());
            }

            constexpr auto begin() const noexcept -> const_iterator{

                return std::next(node_vec.begin());
            }

            constexpr auto cbegin() noexcept -> reverse_iterator{

                return node_vec.cbegin();
            }

            constexpr auto cbegin() const noexcept -> const_reverse_iterator{

                return node_vec.cbegin();
            }

            constexpr auto end() noexcept -> iterator{

                return node_vec.end();
            }

            constexpr auto end() const noexcept -> const_iterator{

                return node_vec.end();
            }

            constexpr auto cend() noexcept -> reverse_iterator{

                return std::prev(node_vec.cend());
            }

            constexpr auto cend() const noexcept -> const_reverse_iterator{

                return std::prev(node_vec.cend());
            }

        private:

            constexpr void maybe_check_for_rehash(){

                if (((size() + erase_count) % REHASH_CHK_MODULO) != 0u) [[likely]]{
                    return;
                } else [[unlikely]]{
                    if (estimate_capacity(size()) <= capacity() && insert_size() <= estimate_insert_capacity(capacity())) [[likely]]{
                        return;
                    } else [[unlikely]]{
                        //either cap > size or insert_cap > max_insert_cap or both - if both - extend
                        if (estimate_capacity(size()) > capacity()){
                            size_type new_cap = capacity() * 2;
                            rehash(new_cap, true);
                        } else{
                            size_type new_cap = estimate_capacity(size());
                            rehash(new_cap, true);
                        }
                    }
                }
            }

            constexpr void force_uphash(){

                size_type new_cap = capacity() * 2;
                rehash(new_cap, true);
            }

            constexpr auto estimate_size(size_type cap) const noexcept -> size_type{

                return cap * load_factor_ratio::num / load_factor_ratio::den;
            }

            constexpr auto estimate_capacity(size_type sz) const noexcept -> size_type{

                return sz * load_factor_ratio::den / load_factor_ratio::num;
            }

            constexpr auto estimate_insert_capacity(size_type cap) const noexcept -> size_type{

                return cap * insert_factor_ratio::num / insert_factor_ratio::den;
            }

            constexpr auto to_bucket_index(size_type hashed_value) const noexcept -> size_type{

                return hashed_value & (bucket_vec.size() - (LAST_MOHICAN_SZ + 1));
            }

            template <class KeyLike>
            constexpr auto bucket_exist_find(const KeyLike& key) noexcept(noexcept(std::declval<Hasher&>()(std::declval<const KeyLike&>()))
                                                                          && noexcept(std::declval<Pred&>()(std::declval<const Key&>(), std::declval<const KeyLike&>()))) -> bucket_iterator{

                //GCC sets branch prediction 1(unlikely)/10(likely) 
                //assume load_factor of 50% - avg - and reasonable hash function
                //50% ^ 3 = 1/8 - which is == branch predictor

                auto it = std::next(bucket_vec.begin(), to_bucket_index(_hasher(key)));

                if (pred(static_cast<const Key&>(node_vec[*it].first), key)){ //this optimization might not be as important in CPU arch but very important in GPU arch - where you want to minimize branch prediction by using block_quicksort approach
                    return it;
                }

                std::advance(it, 1u);

                if (pred(static_cast<const Key&>(node_vec[*it].first), key)){
                    return it;
                }

                std::advance(it, 1u);

                while (true){
                    if (pred(static_cast<const Key&>(node_vec[*it].first), key)) [[likely]]{
                        return it;
                    }

                    std::advance(it, 1u);
                }
            }

            template <class KeyLike>
            constexpr auto bucket_exist_find(const KeyLike& key) const noexcept(noexcept(std::declval<const Hasher&>()(std::declval<const KeyLike&>()))
                                                                                 && noexcept(std::declval<const Pred&>()(std::declval<const Key&>(), std::declval<const KeyLike&>()))) -> bucket_const_iterator{

                //GCC sets branch prediction 1(unlikely)/10(likely)
                //assume load_factor of 50% - avg - and reasonable hash function
                //50% ^ 3 = 1/8 - which is == branch predictor

                auto it = std::next(bucket_vec.begin(), to_bucket_index(_hasher(key)));

                if (pred(node_vec[*it].first, key)){
                    return it;
                }

                std::advance(it, 1u);

                if (pred(node_vec[*it].first, key)){
                    return it;
                }

                std::advance(it, 1u);

                while (true){
                    if (pred(node_vec[*it].first, key)) [[likely]]{
                        return it;
                    }

                    std::advance(it, 1u);
                }
            }

            template <class KeyLike>
            constexpr auto bucket_find(const KeyLike& key) noexcept(noexcept(std::declval<Hasher&>()(std::declval<const KeyLike&>()))
                                                                    && noexcept(std::declval<Pred&>()(std::declval<const Key&>(), std::declval<const KeyLike&>()))) -> bucket_iterator{

                auto it = std::next(bucket_vec.begin(), to_bucket_index(_hasher(key)));

                while (true){
                    if (*it == NULL_VIRTUAL_ADDR || pred(static_cast<const Key&>(node_vec[*it].first), key)){
                        return it;
                    }

                    std::advance(it, 1u);
                }
            }

            template <class KeyLike>
            constexpr auto bucket_find(const KeyLike& key) const noexcept(noexcept(std::declval<const Hasher&>()(std::declval<const KeyLike&>()))
                                                                          && noexcept(std::declval<const Pred&>()(std::declval<const Key&>(), std::declval<const KeyLike&>()))) -> bucket_const_iterator{

                auto it = std::next(bucket_vec.begin(), to_bucket_index(_hasher(key)));

                while (true){
                    if (*it == NULL_VIRTUAL_ADDR || pred(node_vec[*it].first, key)){
                        return it;
                    }

                    std::advance(it, 1u);
                }
            }

            template <class KeyLike>
            constexpr auto bucket_efind(const KeyLike& key) noexcept(noexcept(std::declval<Hasher&>()(std::declval<const KeyLike&>()))) -> bucket_iterator{

                auto it = std::next(bucket_vec.begin(), to_bucket_index(_hasher(key)));

                while (true){
                    if (*it == NULL_VIRTUAL_ADDR){
                        return it;
                    }

                    std::advance(it, 1u);
                }
            }

            template <class KeyLike>
            constexpr auto bucket_efind(const KeyLike& key) const noexcept(noexcept(std::declval<Hasher&>()(std::declval<const KeyLike&>()))) -> bucket_const_iterator{

                auto it = std::next(bucket_vec.begin(), to_bucket_index(_hasher(key)));

                while (true){
                    if (*it == NULL_VIRTUAL_ADDR){
                        return it;
                    }

                    std::advance(it, 1u);
                }
            }

            template <class KeyLike>
            constexpr auto bucket_ifind(const KeyLike& key) noexcept(noexcept(std::declval<Hasher&>()(std::declval<const KeyLike&>()))) -> bucket_iterator{

                auto it = std::next(bucket_vec.begin(), to_bucket_index(_hasher(key)));

                while (true){
                    if (is_insertable(*it)){
                        return it;
                    }

                    std::advance(it, 1u);
                }
            }

            template <class ValueLike>
            constexpr auto do_insert_at(bucket_iterator it, ValueLike&& value) -> iterator{

                size_type addr = node_vec.size();
                node_vec.push_back(std::forward<ValueLike>(value));
                *it = addr;
                return std::prev(node_vec.end());
            }

            template <class ValueLike>
            constexpr auto internal_noexist_insert(ValueLike&& value) -> iterator{

                maybe_check_for_rehash();

                while (true){
                    bucket_iterator it = bucket_ifind(value.first);

                    if (it != std::prev(bucket_vec.end())) [[likely]]{
                        return do_insert_at(it, std::forward<ValueLike>(value));
                    } else [[unlikely]]{
                        force_uphash();
                    }
                }
            }

            template <class ValueLike>
            constexpr auto internal_insert_or_assign(ValueLike&& value) -> std::pair<iterator, bool>{

                bucket_iterator it = bucket_find(value.first);

                if (*it != NULL_VIRTUAL_ADDR){
                    iterator rs = std::next(node_vec.begin(), *it);
                    *rs = std::forward<ValueLike>(value);
                    return std::make_pair(rs, false);
                }

                return std::make_pair(internal_noexist_insert(std::forward<ValueLike>(value)), true);
            }

            template <class ValueLike>
            constexpr auto internal_insert(ValueLike&& value) -> std::pair<iterator, bool>{

                bucket_iterator it = bucket_find(value.first);

                if (*it != NULL_VIRTUAL_ADDR){
                    return std::make_pair(std::next(node_vec.begin(), *it), false);
                }

                return std::make_pair(internal_noexist_insert(std::forward<ValueLike>(value)), true);
            }

            template <class KeyLike, class Arg = mapped_type, std::enable_if_t<std::is_default_constructible_v<Arg>, bool> = true>
            constexpr auto internal_find_or_default(KeyLike&& key, Arg * compiler_hint = nullptr) -> iterator{

                bucket_iterator it = bucket_find(key);

                if (*it != NULL_VIRTUAL_ADDR){
                    return std::next(node_vec.begin(), *it);
                }

                return internal_noexist_insert(value_type(std::forward<KeyLike>(key), mapped_type()));
            }

            template <class KeyLike>
            constexpr auto internal_erase(const KeyLike& key) noexcept(noexcept(std::declval<Hasher&>()(std::declval<const KeyLike&>()))) -> size_type{

                bucket_iterator erasing_bucket_it       = bucket_find(key);
                size_type erasing_bucket_virtual_addr   = *erasing_bucket_it;

                if (erasing_bucket_virtual_addr != NULL_VIRTUAL_ADDR){
                    bucket_iterator swapee_bucket_it = bucket_exist_find(node_vec.back().first); 
                    std::iter_swap(std::next(node_vec.begin(), erasing_bucket_virtual_addr), std::prev(node_vec.end()));
                    node_vec.pop_back();
                    *swapee_bucket_it   = erasing_bucket_virtual_addr;
                    *erasing_bucket_it  = ORPHANED_VIRTUAL_ADDR;
                    erase_count         += 1;

                    return 1u;
                }

                return 0u;
            }

            template <class KeyLike>
            constexpr void internal_exist_erase(const KeyLike& key) noexcept(noexcept(std::declval<Hasher&>()(std::declval<const KeyLike&>()))){

                bucket_iterator erasing_bucket_it       = bucket_exist_find(key);
                size_type erasing_bucket_virtual_addr   = *erasing_bucket_it;
                bucket_iterator swapee_bucket_it        = bucket_exist_find(node_vec.back().first);

                std::iter_swap(std::next(node_vec.begin(), erasing_bucket_virtual_addr), std::prev(node_vec.end()));
                node_vec.pop_back();
                *swapee_bucket_it   = erasing_bucket_virtual_addr;
                *erasing_bucket_it  = ORPHANED_VIRTUAL_ADDR;
                erase_count         += 1;
            }

            constexpr auto internal_erase(const_iterator it) noexcept -> iterator{

                if (it == node_vec.end()){
                    return it;
                }

                auto dif = std::distance(node_vec.begin(), it);
                internal_exist_erase(it->first);

                if (dif != node_vec.size()) [[likely]]{
                    return std::next(node_vec.begin(), dif);
                } else [[unlikely]]{
                    return std::next(node_vec.begin());
                }
            }
    };

    //this map only works as if there is no erase - user must control the erase by using setter clear() and getter size() - this has a very specialized application - like dg_heap_allocation
    //erase was provided in the user interface - yet their usages aren't recommended 
    //this map is specialized for cache-purpose, where the CACHE is not FIFO - but cleared at cap
    //InsertFactor is now the new LoadFactor - and there is nothing we can do about decreasing the LoadFactor - except for calling clear() - otherwise it would tick per every insert + erase operation
    template <class Key, class Mapped, class NullValueGenerator, class SizeType = std::size_t, class Hasher = std::hash<Key>, class Pred = std::equal_to<Key>, class Allocator = std::allocator<std::pair<Key, Mapped>>, class LoadFactor = std::ratio<1, 2>, class InsertFactor = std::ratio<3, 4>>
    class unordered_unstable_fastinsert_map{

        private:

            std::vector<std::pair<Key, Mapped>, typename std::allocator_traits<Allocator>::template rebind_alloc<std::pair<Key, Mapped>>> node_vec;
            std::vector<SizeType, typename std::allocator_traits<Allocator>::template rebind_alloc<SizeType>> bucket_vec;
            Hasher _hasher;
            Pred pred;
            Allocator allocator;
            SizeType erase_count;

            using bucket_iterator               = typename std::vector<SizeType, typename std::allocator_traits<Allocator>::template rebind_alloc<SizeType>>::iterator;
            using bucket_const_iterator         = typename std::vector<SizeType, typename std::allocator_traits<Allocator>::template rebind_alloc<SizeType>>::const_iterator;

            static inline constexpr SizeType ORPHANED_VIRTUAL_ADDR      = std::numeric_limits<SizeType>::min();
            static inline constexpr SizeType NULL_VIRTUAL_ADDR          = std::numeric_limits<SizeType>::max();
            static inline constexpr std::size_t REHASH_CHK_MODULO       = 16u; //this is important - because % 256 == a read of the address instead of an arithmetic operation
            static inline constexpr std::size_t LAST_MOHICAN_SZ         = 16u;

            static constexpr auto is_insertable(SizeType virtual_addr) noexcept -> bool{

                return virtual_addr == NULL_VIRTUAL_ADDR || virtual_addr == ORPHANED_VIRTUAL_ADDR;
            }

        public:

            static_assert(std::is_unsigned_v<SizeType>);
            static_assert(noexcept(std::declval<const Hasher&>()(std::declval<const Key&>())));
            static_assert(noexcept(std::declval<Hasher&>()(std::declval<const Key&>())));
            // static_assert(noexcept(std::declval<const Pred&>()(std::declval<const Key&>(), std::declval<const Key&>())));
            // static_assert(noexcept(std::declval<Pred&>()(std::declval<const Key&>(), std::declval<const Key&>())));
            static_assert(std::is_nothrow_destructible_v<std::pair<Key, Mapped>>);

            static constexpr inline double MIN_MAX_LOAD_FACTOR      = 0.05;
            static constexpr inline double MAX_MAX_LOAD_FACTOR      = 0.95; 
            static constexpr inline double MIN_MAX_INSERT_FACTOR    = 0.05;
            static constexpr inline double MAX_MAX_INSERT_FACTOR    = 8; 

            using key_type                      = Key;
            using value_type                    = std::pair<Key, Mapped>; 
            using mapped_type                   = Mapped;
            using hasher                        = Hasher;
            using key_equal                     = Pred; 
            using allocator_type                = Allocator;
            using reference                     = value_type&;
            using const_reference               = const value_type&;
            using pointer                       = typename std::allocator_traits<Allocator>::pointer; 
            using const_pointer                 = typename std::allocator_traits<Allocator>::const_pointer;
            using iterator                      = typename std::vector<std::pair<Key, Mapped>, typename std::allocator_traits<Allocator>::template rebind_alloc<std::pair<Key, Mapped>>>::iterator; 
            using const_iterator                = typename std::vector<std::pair<Key, Mapped>, typename std::allocator_traits<Allocator>::template rebind_alloc<std::pair<Key, Mapped>>>::const_iterator; 
            using reverse_iterator              = typename std::vector<std::pair<Key, Mapped>, typename std::allocator_traits<Allocator>::template rebind_alloc<std::pair<Key, Mapped>>>::reverse_iterator;
            using const_reverse_iterator        = typename std::vector<std::pair<Key, Mapped>, typename std::allocator_traits<Allocator>::template rebind_alloc<std::pair<Key, Mapped>>>::const_reverse_iterator;
            using size_type                     = SizeType;
            using difference_type               = intmax_t;
            using self                          = unordered_unstable_fastinsert_map;
            using load_factor_ratio             = typename LoadFactor::type; 
            using insert_factor_ratio           = typename InsertFactor::type;

            static consteval auto max_load_factor() noexcept -> double{

                return static_cast<double>(load_factor_ratio::num) / load_factor_ratio::den;
            }

            static consteval auto max_insert_factor() noexcept -> double{

                return static_cast<double>(insert_factor_ratio::num) / insert_factor_ratio::den;
            }

            static_assert(std::clamp(self::max_load_factor(), MIN_MAX_LOAD_FACTOR, MAX_MAX_LOAD_FACTOR) == self::max_load_factor());
            static_assert(std::clamp(self::max_insert_factor(), MIN_MAX_INSERT_FACTOR, MAX_MAX_INSERT_FACTOR) == self::max_insert_factor());

            constexpr unordered_unstable_fastinsert_map(): node_vec(), 
                                                           bucket_vec(self::min_capacity() + LAST_MOHICAN_SZ, NULL_VIRTUAL_ADDR),
                                                           _hasher(),
                                                           pred(),
                                                           allocator(),
                                                           erase_count(0u){

                node_vec.push_back(NullValueGenerator{}());
            }

            constexpr explicit unordered_unstable_fastinsert_map(size_type bucket_count, 
                                                                 const Hasher& _hasher = Hasher(), 
                                                                 const Pred& pred = Pred(), 
                                                                 const Allocator& allocator = Allocator()): node_vec(allocator),
                                                                                                            bucket_vec(std::max(self::min_capacity(), dg::map_variants::least_pow2_greater_equal_than(bucket_count)) + LAST_MOHICAN_SZ, NULL_VIRTUAL_ADDR, allocator),
                                                                                                            _hasher(_hasher),
                                                                                                            pred(pred),
                                                                                                            allocator(allocator),
                                                                                                            erase_count(0u){
                
                node_vec.reserve(estimate_size(capacity()) + 1u);
                node_vec.push_back(NullValueGenerator{}());
            }

            constexpr unordered_unstable_fastinsert_map(size_type bucket_count, 
                                                        const Allocator& allocator): node_vec(allocator),
                                                                                     bucket_vec(std::max(self::min_capacity(), dg::map_variants::least_pow2_greater_equal_than(bucket_count)) + LAST_MOHICAN_SZ, NULL_VIRTUAL_ADDR, allocator),
                                                                                     _hasher(),
                                                                                     pred(),
                                                                                     allocator(allocator),
                                                                                     erase_count(0u){
                
                node_vec.reserve(estimate_size(capacity()) + 1u);
                node_vec.push_back(NullValueGenerator{}());
            }

            constexpr unordered_unstable_fastinsert_map(size_type bucket_count, 
                                                        const Hasher& _hasher, 
                                                        const Allocator& allocator): node_vec(allocator),
                                                                                     bucket_vec(std::max(self::min_capacity(), dg::map_variants::least_pow2_greater_equal_than(bucket_count)) + LAST_MOHICAN_SZ, NULL_VIRTUAL_ADDR, allocator),
                                                                                     _hasher(_hasher),
                                                                                     pred(),
                                                                                     allocator(allocator),
                                                                                     erase_count(0u){
                node_vec.reserve(estimate_size(capacity()) + 1u);
                node_vec.push_back(NullValueGenerator{}());
            }

            constexpr explicit unordered_unstable_fastinsert_map(const Allocator& allocator): node_vec(allocator),
                                                                                              bucket_vec(min_capacity() + LAST_MOHICAN_SZ, NULL_VIRTUAL_ADDR, allocator),
                                                                                              _hasher(),
                                                                                              pred(),
                                                                                              allocator(allocator),
                                                                                              erase_count(0u){

                node_vec.reserve(estimate_size(capacity()) + 1u);
                node_vec.push_back(NullValueGenerator{}());
            }

            template <class InputIt>
            constexpr unordered_unstable_fastinsert_map(InputIt first, 
                                                        InputIt last, 
                                                        size_type bucket_count, 
                                                        const Hasher& _hasher = Hasher(), 
                                                        const Pred& pred = Pred(), 
                                                        const Allocator& allocator = Allocator()): unordered_unstable_fastinsert_map(bucket_count, _hasher, pred, allocator){
                
                insert(first, last);
            }

            template <class InputIt>
            constexpr unordered_unstable_fastinsert_map(InputIt first, 
                                                  InputIt last, 
                                                  size_type bucket_count, 
                                                  const Allocator& allocator): unordered_unstable_fastinsert_map(first, last, bucket_count, Hasher(), Pred(), allocator){}

            constexpr unordered_unstable_fastinsert_map(std::initializer_list<value_type> init_list, 
                                                  size_type bucket_count, 
                                                  const Allocator& allocator): unordered_unstable_fastinsert_map(init_list.begin(), init_list.end(), bucket_count, allocator){}

            constexpr unordered_unstable_fastinsert_map(std::initializer_list<value_type> init_list, 
                                                  size_type bucket_count, 
                                                  const Hasher& _hasher, 
                                                  const Allocator& allocator): unordered_unstable_fastinsert_map(init_list.begin(), init_list.end(), bucket_count, _hasher, allocator){}

            constexpr void clear() noexcept{

                std::fill(bucket_vec.begin(), bucket_vec.end(), NULL_VIRTUAL_ADDR);
                node_vec.resize(1u);
            }

            constexpr void rehash(size_type tentative_new_cap, bool force_rehash = false){

                if (!force_rehash && tentative_new_cap <= capacity()){
                    return;
                }

                decltype(bucket_vec) bucket_proxy = std::move(bucket_vec); 

                try{
                    while (true){
                        size_t new_cap  = std::max(self::min_capacity(), dg::map_variants::least_pow2_greater_equal_than(tentative_new_cap)) + LAST_MOHICAN_SZ;
                        bool bad_bit    = false;
                        bucket_vec.resize(new_cap, NULL_VIRTUAL_ADDR);

                        for (size_t i = 1u; i < node_vec.size(); ++i){
                            auto it = bucket_efind(node_vec[i].first); //its fine to invoke internal method here because we are in a valid state - in other words, node_vec can contain allocations that aren't referenced by one of the buckets
                            if (it != std::prev(bucket_vec.end())) [[likely]]{
                                *it = i; 
                            } else [[unlikely]]{
                                tentative_new_cap = new_cap * 2;
                                bad_bit = true;
                                break;
                            }
                        }

                        if (!bad_bit){
                            break;
                        }
                    }

                    erase_count = 0u;
                } catch (...){
                    bucket_vec = std::move(bucket_proxy);
                    std::rethrow_exception(std::current_exception());
                }
            }

            constexpr void reserve(size_type new_sz){
 
                if (new_sz <= size()){
                    return;
                }

                rehash(estimate_capacity(new_sz));
            }

            constexpr void swap(self& other) noexcept(std::allocator_traits<Allocator>::is_always_equal
                                                      && std::is_nothrow_swappable<Hasher>::value
                                                      && std::is_nothrow_swappable<Pred>::value){
                
                std::swap(node_vec, other.node_vec);
                std::swap(bucket_vec, other.bucket_vec);
                std::swap(_hasher, other._hasher);
                std::swap(pred, other.pred);
                std::swap(allocator, other.allocator);
                std::swap(erase_count, other.erase_count);
            }

            template <class ...Args>
            constexpr auto emplace(Args&& ...args) -> std::pair<iterator, bool>{

                return internal_insert_or_assign(value_type(std::forward<Args>(args)...));
            }

            template <class KeyLike, class ...Args>
            constexpr auto try_emplace(KeyLike&& key, Args&& ...args) -> std::pair<iterator, bool>{

                return internal_insert(value_type(std::piecewise_construct, std::forward_as_tuple(std::forward<KeyLike>(key)), std::forward_as_tuple(std::forward<Args>(args)...)));
            }

            template <class ValueLike = value_type>
            constexpr auto noexist_insert(ValueLike&& value) -> iterator{

                return internal_noexist_insert(std::forward<ValueLike>(value));
            } 

            template <class ValueLike = value_type>
            constexpr auto insert(ValueLike&& value) -> std::pair<iterator, bool>{

                return internal_insert(std::forward<ValueLike>(value));
            }

            template <class Iterator>
            constexpr void insert(Iterator first, Iterator last){

                while (first != last){
                    internal_insert(*first);
                    std::advance(first, 1);
                }
            }

            constexpr void insert(std::initializer_list<value_type> init_list){

                insert(init_list.begin(), init_list.end());
            }

            template <class KeyLike>
            constexpr auto insert_or_assign(KeyLike&& key, mapped_type&& mapped) -> std::pair<iterator, bool>{

                return internal_insert_or_assign(value_type(std::forward<KeyLike>(key), std::forward<mapped_type>(mapped)));
            }

            constexpr auto erase(const_iterator it) noexcept -> iterator{

                return internal_erase(it);
            }

            template <class KeyLike>
            constexpr auto erase(const KeyLike& key) noexcept(noexcept(internal_erase(std::declval<const KeyLike&>()))) -> size_t{

                return internal_erase(key);
            }

            template <class Iterator>
            constexpr auto erase(Iterator first, Iterator last) noexcept(noexcept(internal_erase(*std::declval<Iterator>()))){

                while (first != last){
                    internal_erase(*first);
                    std::advance(first, 1);
                }
            }

            template <class KeyLike>
            constexpr auto operator[](KeyLike&& key) -> mapped_type&{

                return internal_find_or_default(std::forward<KeyLike>(key))->second;
            }

            template <class KeyLike>
            constexpr auto contains(const KeyLike& key) const noexcept(bucket_find(std::declval<const KeyLike&>())) -> bool{

                return *bucket_find(key) != NULL_VIRTUAL_ADDR;
            }

            template <class KeyLike>
            constexpr auto count(const KeyLike& key) const noexcept(bucket_find(std::declval<const KeyLike&>())) -> size_type{

                return *bucket_find(key) != NULL_VIRTUAL_ADDR;
            }

            template <class KeyLike>
            constexpr auto exist_find(const KeyLike& key) noexcept(noexcept(bucket_exist_find(std::declval<const KeyLike&>()))) -> iterator{

                return std::next(node_vec.begin(), *bucket_exist_find(key));
            }

            template <class KeyLike>
            constexpr auto exist_find(const KeyLike& key) const noexcept(noexcept(bucket_exist_find(std::declval<const KeyLike&>()))) -> const_iterator{

                return std::next(node_vec.begin(), *bucket_exist_find(key));
            }

            template <class KeyLike>
            constexpr auto find(const KeyLike& key) noexcept(noexcept(bucket_find(std::declval<const KeyLike&>()))) -> iterator{

                size_type virtual_addr = *bucket_find(key);

                if (virtual_addr == NULL_VIRTUAL_ADDR){
                    return node_vec.end();
                }

                return std::next(node_vec.begin(), virtual_addr);
            }

            template <class KeyLike>
            constexpr auto find(const KeyLike& key) const noexcept(noexcept(bucket_find(std::declval<const KeyLike&>()))) -> const_iterator{

                size_type virtual_addr = *bucket_find(key);

                if (virtual_addr == NULL_VIRTUAL_ADDR){
                    return node_vec.end();
                }

                return std::next(node_vec.begin(), virtual_addr);
            }

            template <class KeyLike>
            constexpr auto at(const KeyLike& key) const noexcept(noexcept(exist_find(std::declval<const KeyLike&>()))) -> const mapped_type&{

                return exist_find(key)->second;
            }

            template <class KeyLike>
            constexpr auto at(const KeyLike& key) noexcept(noexcept(exist_find(std::declval<const KeyLike&>()))) -> mapped_type&{

                return exist_find(key)->second;
            }

            constexpr auto empty() const noexcept -> bool{

                return size() != 0u;
            }

            constexpr auto min_capacity() noexcept -> size_type{

                return 32u;
            }

            constexpr auto capacity() noexcept -> size_type{

                return bucket_vec.size() - LAST_MOHICAN_SZ;
            }

            constexpr auto size() const noexcept -> size_type{

                return node_vec.size() - 1u;
            }

            constexpr auto insert_size() const noexcept -> size_type{

                return size() + erase_count;
            }

            constexpr auto max_size() const noexcept -> size_type{

                return std::numeric_limits<size_type>::max();
            }

            constexpr auto hash_function() const & noexcept -> const Hasher&{

                return _hasher;
            }

            constexpr auto key_eq() const & noexcept -> const Pred&{

                return pred;
            }

            constexpr auto hash_function() && noexcept -> Hasher&&{

                return static_cast<Hasher&&>(_hasher);
            }

            constexpr auto key_eq() && noexcept -> Pred&&{

                return static_cast<Pred&&>(pred);
            }
            
            constexpr auto load_factor() const noexcept -> double{

                return size() / static_cast<double>(capacity());
            }

            constexpr auto insert_factor() const noexcept -> double{

                return (size() + erase_count) / static_cast<double>(capacity());
            }

            constexpr auto begin() noexcept -> iterator{

                return std::next(node_vec.begin());
            }

            constexpr auto begin() const noexcept -> const_iterator{

                return std::next(node_vec.begin());
            }

            constexpr auto cbegin() noexcept -> reverse_iterator{

                return node_vec.cbegin();
            }

            constexpr auto cbegin() const noexcept -> const_reverse_iterator{

                return node_vec.cbegin();
            }

            constexpr auto end() noexcept -> iterator{

                return node_vec.end();
            }

            constexpr auto end() const noexcept -> const_iterator{

                return node_vec.end();
            }

            constexpr auto cend() noexcept -> reverse_iterator{

                return std::prev(node_vec.cend());
            }

            constexpr auto cend() const noexcept -> const_reverse_iterator{

                return std::prev(node_vec.cend());
            }

        private:

            constexpr void check_for_rehash(){

                if (estimate_capacity(size()) <= capacity() && insert_size() <= estimate_insert_capacity(capacity())) [[likely]]{
                    return;
                } else [[unlikely]]{
                   //either cap > size or insert_cap > max_insert_cap or both - if both - extend
                    if (estimate_capacity(size()) > capacity()){
                        size_type new_cap = capacity() * 2;
                        rehash(new_cap, true);
                    } else{
                        size_type new_cap = estimate_capacity(size());
                        rehash(new_cap, true);
                    }
                }
            }

            constexpr void force_uphash(){ //problems

                size_type new_cap = capacity() * 2;
                rehash(new_cap, true);
            }

            constexpr auto estimate_size(size_type cap) const noexcept -> size_type{

                return cap * load_factor_ratio::num / load_factor_ratio::den;
            }

            constexpr auto estimate_capacity(size_type sz) const noexcept -> size_type{

                return sz * load_factor_ratio::den / load_factor_ratio::num;
            }

            constexpr auto estimate_insert_capacity(size_type cap) const noexcept -> size_type{

                return cap * insert_factor_ratio::num / insert_factor_ratio::den;
            }

            constexpr auto to_bucket_index(size_type hashed_value) const noexcept -> size_type{

                return hashed_value & (bucket_vec.size() - (LAST_MOHICAN_SZ + 1));
            }

            template <class KeyLike>
            constexpr auto bucket_exist_find(const KeyLike& key) noexcept(noexcept(std::declval<Hasher&>()(std::declval<const KeyLike&>()))
                                                                          && noexcept(std::declval<Pred&>()(std::declval<const Key&>(), std::declval<const KeyLike&>()))) -> bucket_iterator{

                //GCC sets branch prediction 1(unlikely)/10(likely) 
                //assume load_factor of 50% - avg - and reasonable hash function
                //50% ^ 3 = 1/8 - which is == branch predictor

                auto it = std::next(bucket_vec.begin(), to_bucket_index(_hasher(key))); //we don't care where the bucket is pointing to - as long as it is a fixed random position and it does not pass the last of the last mohicans

                if (pred(static_cast<const Key&>(node_vec[*it].first), key)){ //this optimization might not be as important in CPU arch but very important in GPU arch - where you want to minimize branch prediction by using block_quicksort approach
                    return it;
                }

                std::advance(it, 1u);

                if (pred(static_cast<const Key&>(node_vec[*it].first), key)){
                    return it;
                }

                std::advance(it, 1u);

                while (true){
                    if (pred(static_cast<const Key&>(node_vec[*it].first), key)) [[likely]]{
                        return it;
                    }

                    std::advance(it, 1u);
                }
            }

            template <class KeyLike>
            constexpr auto bucket_exist_find(const KeyLike& key) const noexcept(noexcept(std::declval<const Hasher&>()(std::declval<const KeyLike&>()))
                                                                                 && noexcept(std::declval<const Pred&>()(std::declval<const Key&>(), std::declval<const KeyLike&>()))) -> bucket_const_iterator{

                //GCC sets branch prediction 1(unlikely)/10(likely)
                //assume load_factor of 50% - avg - and reasonable hash function
                //50% ^ 3 = 1/8 - which is == branch predictor

                auto it = std::next(bucket_vec.begin(), to_bucket_index(_hasher(key)));

                if (pred(node_vec[*it].first, key)){
                    return it;
                }

                std::advance(it, 1u);

                if (pred(node_vec[*it].first, key)){
                    return it;
                }

                std::advance(it, 1u);

                while (true){
                    if (pred(node_vec[*it].first, key)) [[likely]]{
                        return it;
                    }

                    std::advance(it, 1u);
                }
            }

            template <class KeyLike>
            constexpr auto bucket_find(const KeyLike& key) noexcept(noexcept(std::declval<Hasher&>()(std::declval<const KeyLike&>()))
                                                                    && noexcept(std::declval<Pred&>()(std::declval<const Key&>(), std::declval<const KeyLike&>()))) -> bucket_iterator{

                auto it = std::next(bucket_vec.begin(), to_bucket_index(_hasher(key)));

                while (true){
                    if (*it == NULL_VIRTUAL_ADDR || pred(static_cast<const Key&>(node_vec[*it].first), key)) [[likely]]{
                        return it;
                    }

                    std::advance(it, 1u);
                }
            }

            template <class KeyLike>
            constexpr auto bucket_find(const KeyLike& key) const noexcept(noexcept(std::declval<const Hasher&>()(std::declval<const KeyLike&>()))
                                                                          && noexcept(std::declval<const Pred&>()(std::declval<const Key&>(), std::declval<const KeyLike&>()))) -> bucket_const_iterator{

                auto it = std::next(bucket_vec.begin(), to_bucket_index(_hasher(key)));

                while (true){
                    if (*it == NULL_VIRTUAL_ADDR || pred(node_vec[*it].first, key)) [[]]{
                        return it;
                    }

                    std::advance(it, 1u);
                }
            }

            template <class KeyLike>
            constexpr auto bucket_efind(const KeyLike& key) noexcept(noexcept(std::declval<Hasher&>()(std::declval<const KeyLike&>()))) -> bucket_iterator{

                auto it = std::next(bucket_vec.begin(), to_bucket_index(_hasher(key)));

                while (true){
                    if (*it == NULL_VIRTUAL_ADDR){
                        return it;
                    }

                    std::advance(it, 1u);
                }
            }

            template <class KeyLike>
            constexpr auto bucket_efind(const KeyLike& key) const noexcept(noexcept(std::declval<Hasher&>()(std::declval<const KeyLike&>()))) -> bucket_const_iterator{

                auto it = std::next(bucket_vec.begin(), to_bucket_index(_hasher(key)));

                while (true){
                    if (*it == NULL_VIRTUAL_ADDR){
                        return it;
                    }

                    std::advance(it, 1u);
                }
            }

            template <class KeyLike>
            constexpr auto bucket_ifind(const KeyLike& key) noexcept(noexcept(std::declval<Hasher&>()(std::declval<const KeyLike&>()))) -> bucket_iterator{

                return bucket_efind(key);
            }

            template <class ValueLike>
            constexpr auto do_insert_at(bucket_iterator it, ValueLike&& value) -> iterator{

                size_type addr = node_vec.size();
                node_vec.push_back(std::forward<ValueLike>(value));
                *it = addr;
                return std::prev(node_vec.end());
            }

            template <class ValueLike>
            constexpr auto internal_noexist_insert(ValueLike&& value) -> iterator{

                check_for_rehash();

                while (true){
                    bucket_iterator it = bucket_ifind(value.first);

                    if (it != std::prev(bucket_vec.end())) [[likely]]{
                        return do_insert_at(it, std::forward<ValueLike>(value));
                    } else [[unlikely]]{
                        force_uphash();
                    }
                }
            }

            template <class ValueLike>
            constexpr auto internal_insert_or_assign(ValueLike&& value) -> std::pair<iterator, bool>{

                bucket_iterator it = bucket_find(value.first);

                if (*it != NULL_VIRTUAL_ADDR){
                    iterator rs = std::next(node_vec.begin(), *it);
                    *rs = std::forward<ValueLike>(value);
                    return std::make_pair(rs, false);
                }

                if (insert_size() % REHASH_CHK_MODULO != 0u && it != std::prev(bucket_vec.end())) [[likely]]{
                    return std::make_pair(do_insert_at(it, std::forward<ValueLike>(value)), true);
                } else [[unlikely]]{
                    return std::make_pair(internal_noexist_insert(std::forward<ValueLike>(value)), true);
                }
            }

            template <class ValueLike>
            constexpr auto internal_insert(ValueLike&& value) -> std::pair<iterator, bool>{

                bucket_iterator it = bucket_find(value.first);

                if (*it != NULL_VIRTUAL_ADDR){
                    return std::make_pair(std::next(node_vec.begin(), *it), false);
                }

                if (insert_size() % REHASH_CHK_MODULO != 0u && it != std::prev(bucket_vec.end())) [[likely]]{
                    return std::make_pair(do_insert_at(it, std::forward<ValueLike>(value)), true);
                } else [[unlikely]]{
                    return std::make_pair(internal_noexist_insert(std::forward<ValueLike>(value)), true);
                }            
            }

            template <class KeyLike, class Arg = mapped_type, std::enable_if_t<std::is_default_constructible_v<Arg>, bool> = true>
            constexpr auto internal_find_or_default(KeyLike&& key, Arg * compiler_hint = nullptr) -> iterator{

                bucket_iterator it = bucket_find(key);

                if (*it != NULL_VIRTUAL_ADDR){
                    return std::next(node_vec.begin(), *it);
                }

                if (insert_size() % REHASH_CHK_MODULO != 0u && it != std::prev(bucket_vec.end())) [[likely]]{
                    return do_insert_at(it, value_type(std::forward<KeyLike>(key), mapped_type()));
                } else [[unlikely]]{
                    return internal_noexist_insert(value_type(std::forward<KeyLike>(key), mapped_type()));
                }
            }

            template <class KeyLike>
            constexpr auto internal_erase(const KeyLike& key) noexcept(noexcept(std::declval<Hasher&>()(std::declval<const KeyLike&>()))) -> size_type{

                bucket_iterator erasing_bucket_it       = bucket_find(key);
                size_type erasing_bucket_virtual_addr   = *erasing_bucket_it;

                if (erasing_bucket_virtual_addr != NULL_VIRTUAL_ADDR){
                    bucket_iterator swapee_bucket_it = bucket_exist_find(node_vec.back().first); 
                    std::iter_swap(std::next(node_vec.begin(), erasing_bucket_virtual_addr), std::prev(node_vec.end()));
                    node_vec.pop_back();
                    *swapee_bucket_it   = erasing_bucket_virtual_addr;
                    *erasing_bucket_it  = ORPHANED_VIRTUAL_ADDR;
                    erase_count         += 1;

                    return 1u;
                }

                return 0u;
            }

            template <class KeyLike>
            constexpr void internal_exist_erase(const KeyLike& key) noexcept(noexcept(std::declval<Hasher&>()(std::declval<const KeyLike&>()))){

                bucket_iterator erasing_bucket_it       = bucket_exist_find(key);
                size_type erasing_bucket_virtual_addr   = *erasing_bucket_it;
                bucket_iterator swapee_bucket_it        = bucket_exist_find(node_vec.back().first);

                std::iter_swap(std::next(node_vec.begin(), erasing_bucket_virtual_addr), std::prev(node_vec.end()));
                node_vec.pop_back();
                *swapee_bucket_it   = erasing_bucket_virtual_addr;
                *erasing_bucket_it  = ORPHANED_VIRTUAL_ADDR;
                erase_count         += 1;
            }

            constexpr auto internal_erase(const_iterator it) noexcept -> iterator{

                if (it == node_vec.end()){
                    return it;
                }

                auto dif = std::distance(node_vec.begin(), it);
                internal_exist_erase(it->first);

                if (dif != node_vec.size()) [[likely]]{
                    return std::next(node_vec.begin(), dif);
                } else [[unlikely]]{
                    return std::next(node_vec.begin());
                }
            }
    };
}

#endif