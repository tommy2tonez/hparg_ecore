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

namespace dg::map_variants{

    static constexpr inline double unordered_unstable_map_min_max_load_factor = 0.3;
    static constexpr inline double unordered_unstable_map_max_max_load_factor = 0.875; 

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

    //there is an exotic case of orphans - such that one could exploit by reaching max_load_factor and erase and repeat
    //it's an unlikely event but for performance constraints - this implementation does not include such case - one could solve the problem performantly by using # of inserted before check for rehash - it's recommended

    template <class Key, class Mapped, class SizeType = std::size_t, class Hasher = std::hash<Key>, class Pred = std::equal_to<Key>, class Allocator = std::allocator<std::pair<Key, Mapped>>, class LoadFactor = std::ratio<7, 8>>
    class unordered_unstable_map{

        private:

            std::vector<std::pair<Key, Mapped>, typename std::allocator_traits<Allocator>::template rebind_alloc<std::pair<Key, Mapped>>> node_vec;
            std::vector<SizeType, typename std::allocator_traits<Allocator>::template rebind_alloc<SizeType>> bucket_vec;
            Hasher _hasher;
            Pred pred;
            Allocator allocator;

            using bucket_iterator               = typename std::vector<SizeType, typename std::allocator_traits<Allocator>::template rebind_alloc<SizeType>>::iterator;
            using bucket_const_iterator         = typename std::vector<SizeType, typename std::allocator_traits<Allocator>::template rebind_alloc<SizeType>>::const_iterator;

            static inline constexpr SizeType null_virtual_addr      = std::numeric_limits<SizeType>::max() - 1;
            static inline constexpr SizeType orphaned_virtual_addr  = std::numeric_limits<SizeType>::max();

            static constexpr auto is_insertable(SizeType virtual_addr) noexcept -> bool{

                return (virtual_addr | SizeType{1u}) == std::numeric_limits<SizeType>::max();
            }

        public:

            static_assert(std::is_unsigned_v<SizeType>);
            static_assert(noexcept(std::declval<const Hasher&>()(std::declval<const Key&>())));
            static_assert(noexcept(std::declval<Hasher&>()(std::declval<const Key&>())));
            // static_assert(noexcept(std::declval<const Pred&>()(std::declval<const Key&>(), std::declval<const Key&>())));
            // static_assert(noexcept(std::declval<Pred&>()(std::declval<const Key&>(), std::declval<const Key&>())));

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

            static consteval auto max_load_factor() noexcept -> double{

                return static_cast<double>(load_factor_ratio::num) / load_factor_ratio::den;
            }

            static_assert(std::clamp(self::max_load_factor(), unordered_unstable_map_min_max_load_factor, unordered_unstable_map_max_max_load_factor) == self::max_load_factor());

            constexpr unordered_unstable_map(): node_vec(), 
                                                bucket_vec(self::min_capacity() + 1, null_virtual_addr),
                                                _hasher(),
                                                pred(),
                                                allocator(){}

            constexpr explicit unordered_unstable_map(size_type bucket_count, 
                                                      const Hasher& _hasher = Hasher(), 
                                                      const Pred& pred = Pred(), 
                                                      const Allocator& allocator = Allocator()): node_vec(allocator),
                                                                                                 bucket_vec(std::max(self::min_capacity(), dg::map_variants::least_pow2_greater_equal_than(bucket_count)) + 1, null_virtual_addr, allocator),
                                                                                                 _hasher(_hasher),
                                                                                                 pred(pred),
                                                                                                 allocator(allocator){}

            constexpr unordered_unstable_map(size_type bucket_count, 
                                             const Allocator& allocator): node_vec(allocator),
                                                                          bucket_vec(std::max(self::min_capacity(), dg::map_variants::least_pow2_greater_equal_than(bucket_count)) + 1, null_virtual_addr, allocator),
                                                                          _hasher(),
                                                                          pred(),
                                                                          allocator(allocator){}

            constexpr unordered_unstable_map(size_type bucket_count, 
                                             const Hasher& _hasher, 
                                             const Allocator& allocator): node_vec(allocator),
                                                                          bucket_vec(std::max(self::min_capacity(), dg::map_variants::least_pow2_greater_equal_than(bucket_count)) + 1, null_virtual_addr, allocator),
                                                                          _hasher(_hasher),
                                                                          pred(),
                                                                          allocator(allocator){}

            constexpr explicit unordered_unstable_map(const Allocator& allocator): node_vec(allocator),
                                                                                   bucket_vec(min_capacity() + 1, null_virtual_addr, allocator),
                                                                                   _hasher(),
                                                                                   pred(),
                                                                                   allocator(allocator){}

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

                if (virtual_addr == null_virtual_addr){
                    return node_vec.end();
                }

                return std::next(node_vec.begin(), virtual_addr);
            }

            template <class KeyLike>
            constexpr auto find(const KeyLike& key) const noexcept(noexcept(bucket_find(std::declval<const KeyLike&>()))) -> const_iterator{

                size_type virtual_addr = *bucket_find(key);

                if (virtual_addr == null_virtual_addr){
                    return node_vec.end();
                }

                return std::next(node_vec.begin(), virtual_addr);
            }

            constexpr void clear() noexcept{

                std::fill(bucket_vec.begin(), bucket_vec.end(), null_virtual_addr);
                node_vec.clear();
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
                    std::advance(first, 1u);
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
                    std::advance(first, 1u);
                }
            }

            template <class KeyLike>
            constexpr auto at(const KeyLike& key) const noexcept(noexcept(exist_find(std::declval<const KeyLike&>()))) -> const mapped_type&{

                return exist_find(key)->second;
            }

            template <class KeyLike>
            constexpr auto at(const KeyLike& key) noexcept(noexcept(exist_find(std::declval<const KeyLike&>()))) -> mapped_type&{

                return exist_find(key)->second;
            }

            template <class KeyLike>
            constexpr auto operator[](KeyLike&& key) -> mapped_type&{

                return internal_find_or_default(std::forward<KeyLike>(key)).first->second;
            }

            template <class KeyLike>
            constexpr auto contains(const KeyLike& key) const noexcept(bucket_find(std::declval<const KeyLike&>())) -> bool{

                return *bucket_find(key) != null_virtual_addr;
            }

            template <class KeyLike>
            constexpr auto count(const KeyLike& key) const noexcept(bucket_find(std::declval<const KeyLike&>())) -> size_type{

                return *bucket_find(key) != null_virtual_addr;
            }

            constexpr void reserve(size_type new_sz){
 
                if (new_sz <= node_vec.size()){
                    return;
                }

                self proxy = self(estimate_capacity(new_sz), std::move(_hasher), std::move(pred), std::move(allocator));

                for (auto& node: node_vec){
                    proxy.internal_noexist_insert(std::move(node));
                }

                *this = std::move(proxy);
            }

            constexpr void rehash(size_type tentative_new_cap){

                if (tentative_new_cap < bucket_vec.size()){
                    return;
                }

                self proxy = self(tentative_new_cap, std::move(_hasher), std::move(pred), std::move(allocator));

                for (auto& node: node_vec){
                    proxy.internal_noexist_insert(std::move(node));
                }

                *this = std::move(proxy);
            }

            constexpr auto empty() const noexcept -> bool{

                return node_vec.empty();
            }

            constexpr auto min_capacity() noexcept -> size_type{

                return 8u;
            } 

            constexpr auto size() const noexcept -> size_type{

                return node_vec.size();
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

                return node_vec.size() / static_cast<double>(bucket_vec.size() - 1);
            }

            constexpr auto begin() noexcept -> iterator{

                return node_vec.begin();
            }

            constexpr auto begin() const noexcept -> const_iterator{

                return node_vec.begin();
            }

            constexpr auto begin(size_type off) noexcept -> iterator{

                return std::next(node_vec.begin(), off);
            }

            constexpr auto begin(size_type off) const noexcept -> const_iterator{

                return std::next(node_vec.begin(), off);
            }

            constexpr auto cbegin() noexcept -> reverse_iterator{

                return node_vec.cbegin();
            }

            constexpr auto cbegin() const noexcept -> const_reverse_iterator{

                return node_vec.cbegin();
            }

            constexpr auto cbegin(size_type off) noexcept -> reverse_iterator{

                return std::next(node_vec.cbegin(), off);
            }

            constexpr auto cbegin(size_type off) const noexcept -> const_reverse_iterator{

                return std::next(node_vec.cbegin(), off);
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

            constexpr void check_for_rehash(){

                if (estimate_capacity(node_vec.size()) < bucket_vec.size()) [[likely]]{
                    return;
                } else [[unlikely]]{
                    size_type new_cap = (bucket_vec.size() - 1) * 2;
                    rehash(new_cap);
                }
            }

            constexpr void force_uphash(){

                size_type new_cap = (bucket_vec.size() - 1) * 2;
                rehash(new_cap);
            }

            constexpr auto estimate_capacity(size_type sz) const noexcept -> size_type{

                return sz * load_factor_ratio::num / load_factor_ratio::den;
            }

            constexpr auto to_bucket_index(size_type hashed_value) const noexcept -> size_type{

                return hashed_value & (bucket_vec.size() - 2u);
            }

            template <class KeyLike>
            constexpr auto bucket_exist_find(const KeyLike& key) noexcept(noexcept(std::declval<Hasher&>()(std::declval<const KeyLike&>()))
                                                                          && noexcept(std::declval<Pred&>()(std::declval<const Key&>(), std::declval<const KeyLike&>()))) -> bucket_iterator{

                auto it = std::next(bucket_vec.begin(), to_bucket_index(_hasher(key)));

                while (true){
                    if (*it != orphaned_virtual_addr && pred(static_cast<const Key&>(node_vec[*it].first), key)){
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
                    if (*it != orphaned_virtual_addr && pred(node_vec[*it].first, key)){
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
                    if (*it == null_virtual_addr || (*it != orphaned_virtual_addr && pred(static_cast<const Key&>(node_vec[*it].first), key))){
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
                    if (*it == null_virtual_addr || (*it != orphaned_virtual_addr && pred(node_vec[*it].first, key))){
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
            constexpr auto internal_noexist_insert(ValueLike&& value) -> iterator{

                check_for_rehash();

                while (true){
                    bucket_iterator it = bucket_ifind(value.first);

                    if (it != std::prev(bucket_vec.end())) [[likely]]{
                        size_type addr = node_vec.size();
                        node_vec.push_back(std::forward<ValueLike>(value));
                        *it = addr;
                        return std::prev(node_vec.end());
                    } else [[unlikely]]{
                        force_uphash();
                    }
                }
            } 

            template <class ValueLike>
            constexpr auto internal_insert_or_assign(ValueLike&& value) -> std::pair<iterator, bool>{

                bucket_iterator it = bucket_find(value.first);

                if (*it != null_virtual_addr){
                    iterator rs = std::next(node_vec.begin(), *it);
                    *rs = std::forward<ValueLike>(value);
                    return std::make_pair(rs, false);
                }

                return std::make_pair(internal_noexist_insert(std::forward<ValueLike>(value)), true);
            }

            template <class ValueLike>
            constexpr auto internal_insert(ValueLike&& value) -> std::pair<iterator, bool>{

                bucket_iterator it = bucket_find(value.first);

                if (*it != null_virtual_addr){
                    return std::make_pair(std::next(node_vec.begin(), *it), false);
                }

                return std::make_pair(internal_noexist_insert(std::forward<ValueLike>(value)), true);
            }

            template <class KeyLike, class Arg = mapped_type, std::enable_if_t<std::is_default_constructible_v<Arg>, bool> = true>
            constexpr auto internal_find_or_default(KeyLike&& key, Arg * compiler_hint = nullptr) -> std::pair<iterator, bool>{

                bucket_iterator it = bucket_find(key);

                if (*it != null_virtual_addr){
                    return std::make_pair(std::next(node_vec.begin(), *it), false);
                }

                return std::make_pair(internal_noexist_insert(value_type(std::forward<KeyLike>(key), mapped_type())), true);
            }

            template <class KeyLike>
            constexpr auto internal_erase(const KeyLike& key) noexcept(noexcept(std::declval<Hasher&>()(std::declval<const KeyLike&>()))) -> size_type{

                bucket_iterator erasing_bucket_it       = bucket_find(key);
                size_type erasing_bucket_virtual_addr   = *erasing_bucket_it;

                if (erasing_bucket_virtual_addr != null_virtual_addr){
                    bucket_iterator swapee_bucket_it = bucket_exist_find(node_vec.back().first); 
                    std::iter_swap(std::next(node_vec.begin(), erasing_bucket_virtual_addr), std::prev(node_vec.end()));
                    node_vec.pop_back();
                    *swapee_bucket_it   = erasing_bucket_virtual_addr;
                    *erasing_bucket_it  = orphaned_virtual_addr;

                    return 1u;
                }

                return 0u;
            }

            template <class KeyLike>
            constexpr void internal_exist_erase(const KeyLike& key) noexcept(noexcept(std::declval<Hasher&>()(std::declval<const KeyLike&>()))){

                bucket_iterator erasing_bucket_it       = bucket_exist_find(key);
                size_type erasing_bucket_virtual_addr   = std::exchange(*erasing_bucket_it, orphaned_virtual_addr);
                bucket_iterator swapee_bucket_it        = bucket_exist_find(node_vec.back().first);

                std::iter_swap(std::next(node_vec.begin(), erasing_bucket_virtual_addr), std::prev(node_vec.end()));
                node_vec.pop_back();
                *swapee_bucket_it = erasing_bucket_virtual_addr;
            }

            constexpr auto internal_erase(const_iterator it) noexcept -> iterator{

                if (it == node_vec.end()){
                    return it;
                }

                auto dif = std::distance(node_vec.begin(), it);
                internal_exist_erase(it->first);

                if (dif == node_vec.size()) [[unlikely]]{
                    return node_vec.begin();
                } else [[likely]]{
                    return std::next(node_vec.begin(), dif);
                }
            }
    };
}

#endif