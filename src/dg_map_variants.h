#ifndef __DG_FLAT_MAP_H__
#define __DG_FLAT_MAP_H__

#include <stdint.h>
#include <stdlib.h>
#include <vector>
#include <unordered_map>
#include <limits.h>
#include <utility>

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

    template <class Key, class Mapped, class SizeType = std::size_t, class Hasher = std::hash<Key>, class Pred = std::equal_to<Key>, class Allocator = std::allocator<std::pair<Key, Mapped>>>
    class unordered_unstable_map{

        private:

            std::vector<std::pair<Key, Mapped>, typename std::allocator_traits<Allocator>::template rebind_alloc<std::pair<Key, Mapped>>> node_vec;
            std::vector<SizeType, typename std::allocator_traits<Allocator>::template rebind_alloc<SizeType>> bucket_vec;
            Hasher hasher;
            Pred pred;
            Allocator allocator;

            using bucket_iterator               = typename std::vector<SizeType, typename std::allocator_traits<Allocator>::template rebind_alloc<SizeType>>::iterator;
            using bucket_const_iterator         = typename std::vector<SizeType, typename std::allocator_traits<Allocator>::template rebind_alloc<SizeType>>::const_iterator;
            using bucket_reverse_iterator       = typename std::vector<SizeType, typename std::allocator_traits<Allocator>::template rebind_alloc<SizeType>>::reverse_iterator;
            using bucket_const_reverse_iterator = typename std::vector<SizeType, typename std::allocator_traits<Allocator>::template rebind_alloc<SizeType>>::const_reverse_iterator;

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
            using iterator                      = typename std::vector<std::pair<Key, Mapped>, typename std::allocator_traits<Allocator>::template rebind_alloc<std::pair<Key, Mapped>>>::iterator; 
            using const_iterator                = typename std::vector<std::pair<Key, Mapped>, typename std::allocator_traits<Allocator>::template rebind_alloc<std::pair<Key, Mapped>>>::const_iterator; 
            using reverse_iterator              = typename std::vector<std::pair<Key, Mapped>, typename std::allocator_traits<Allocator>::template rebind_alloc<std::pair<Key, Mapped>>>::reverse_iterator;
            using const_reverse_iterator        = typename std::vector<std::pair<Key, Mapped>, typename std::allocator_traits<Allocator>::template rebind_alloc<std::pair<Key, Mapped>>>::const_reverse_iterator;
            using size_type                     = SizeType;
            using difference_type               = intmax_t;
            using self                          = unordered_unstable_map;

            constexpr unordered_unstable_map(): node_vec(), 
                                                bucket_vec(min_capacity() + 1, null_virtual_addr),
                                                hasher(),
                                                pred(),
                                                allocator(){}

            constexpr explicit unordered_unstable_map(size_type bucket_count, 
                                                      const Hasher& hasher = Hasher(), 
                                                      const Pred& pred = Pred(), 
                                                      const Allocator& allocator = Allocator()): node_vec(allocator),
                                                                                                 bucket_vec(std::max(min_capacity(), least_pow2_greater_equal_than(bucket_count)) + 1, null_virtual_addr, allocator),
                                                                                                 hasher(hasher),
                                                                                                 pred(pred),
                                                                                                 allocator(allocator){}

            constexpr unordered_unstable_map(size_type bucket_count, 
                                             const Allocator& allocator): node_vec(allocator),
                                                                          bucket_vec(std::max(min_capacity(), least_pow2_greater_equal_than(bucket_count)) + 1, null_virtual_addr, allocator),
                                                                          hasher(),
                                                                          pred(),
                                                                          allocator(allocator){}

            constexpr unordered_unstable_map(size_type bucket_count, 
                                             const Hasher& hasher, 
                                             const Allocator& allocator): node_vec(allocator),
                                                                          bucket_vec(std::max(min_capacity(), least_pow2_greater_equal_than(bucket_count)) + 1, null_virtual_addr, allocator),
                                                                          hasher(hasher),
                                                                          pred(),
                                                                          allocator(allocator){}

            constexpr explicit unordered_unstable_map(const Allocator& allocator): node_vec(allocator),
                                                                                   bucket_vec(min_capacity() + 1, null_virtual_addr, allocator),
                                                                                   hasher(),
                                                                                   pred(),
                                                                                   allocator(allocator){}

            template <class InputIt>
            constexpr unordered_unstable_map(InputIt first, 
                                             InputIt last, 
                                             size_type bucket_count, 
                                             const Hasher& hasher = Hasher(), 
                                             const Pred& pred = Pred(), 
                                             const Allocator& allocator = Allocator()): unordered_unstable_map(bucket_count, hasher, pred, allocator){
                
                this->insert(first, last);
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
                                             const Hasher& hash, 
                                             const Allocator& allocator): unordered_unstable_map(init_list.begin(), init_list.end(), bucket_count, hash, allocator){}

            template <class KeyLike>
            constexpr auto exist_find(const KeyLike& key) noexcept(noexcept(this->bucket_exist_find(std::declval<const KeyLike&>()))) -> iterator{

                return std::next(this->node_vec.begin(), *this->bucket_exist_find(key));
            }

            template <class KeyLike>
            constexpr auto exist_find(const KeyLike& key) const noexcept(noexcept(this->bucket_exist_find(std::declval<const KeyLike&>()))) -> const_iterator{

                return std::next(this->node_vec.begin(), *this->bucket_exist_find(key));
            }

            template <class KeyLike>
            constexpr auto find(const KeyLike& key) noexcept(noexcept(this->bucket_find(std::declval<const KeyLike&>()))) -> iterator{

                size_type virtual_addr = *this->bucket_find(key);

                if (virtual_addr == null_virtual_addr){
                    return this->end();
                }

                return std::next(this->node_vec.begin(), virtual_addr);
            }

            template <class KeyLike>
            constexpr auto find(const KeyLike& key) const noexcept(noexcept(this->bucket_find(std::declval<const KeyLike&>()))) -> const_iterator{

                size_type virtual_addr = *this->bucket_find(key);

                if (virtual_addr == null_virtual_addr){
                    return this->end();
                }

                return std::next(this->node_vec.begin(), virtual_addr);
            }

            constexpr void clear() noexcept{

                std::fill(this->bucket_vec.begin(), this->bucket_vec.end(), null_virtual_addr);
                this->node_vec.clear();
            }

            template <class ...Args>
            constexpr auto emplace(Args&& ...args) -> std::pair<iterator, bool>{

                return this->internal_insert_or_assign(value_type(std::forward<Args>(args)...));
            }

            template <class KeyLike, class ...Args>
            constexpr auto try_emplace(KeyLike&& key, Args&& ...args) -> std::pair<iterator, bool>{

                return this->internal_insert(value_type(std::piecewise_construct, std::forward_as_tuple(std::forward<KeyLike>(key)), std::forward_as_tuple(std::forward<Args>(args)...)));
            }

            template <class ValueLike>
            constexpr auto insert(ValueLike&& value) -> std::pair<iterator, bool>{

                return this->internal_insert(std::forward<ValueLike>(value));
            }

            template <class Iterator>
            constexpr void insert(Iterator first, Iterator last){

                for (auto it = first; it != last; ++it){
                    this->insert(*it);
                }
            }

            constexpr void insert(std::initializer_list<value_type> init_list){

                this->insert(init_list.begin(), init_list.end());
            }

            template <class KeyLike>
            constexpr auto insert_or_assign(KeyLike&& key, mapped_type&& mapped) -> std::pair<iterator, bool>{

                return this->internal_insert_or_assign(value_type(std::forward<KeyLike>(key), std::forward<mapped_type>(mapped)));
            }

            constexpr auto erase(const_iterator it) noexcept -> iterator{

                return this->internal_erase(it);
            }

            template <class KeyLike>
            constexpr auto erase(const KeyLike& key) noexcept(noexcept(this->internal_erase(std::declval<const KeyLike&>()))) -> size_t{

                return this->internal_erase(key);
            }

            template <class KeyLike>
            constexpr auto at(const KeyLike& key) const noexcept(noexcept(this->exist_find(std::declval<const KeyLike&>()))) -> const mapped_type&{

                return this->exist_find(key)->second;
            }

            template <class KeyLike>
            constexpr auto at(const KeyLike& key) noexcept(noexcept(this->exist_find(std::declval<const KeyLike&>()))) -> mapped_type&{

                return this->exist_find(key)->second;
            }

            template <class KeyLike>
            constexpr auto operator[](KeyLike&& key) -> mapped_type&{

                return this->internal_find_or_default(std::forward<KeyLike>(key)).first->second;
            }

            template <class KeyLike>
            constexpr auto contains(const KeyLike& key) const noexcept(this->bucket_find(std::declval<const KeyLike&>())) -> bool{

                return *this->bucket_find(key) != null_virtual_addr;
            }

            template <class KeyLike>
            constexpr auto count(const KeyLike& key) const noexcept(this->bucket_find(std::declval<const KeyLike&>())) -> size_type{

                return *this->bucket_find(key) != null_virtual_addr;
            }

            constexpr void reserve(size_type new_sz){
 
                if (new_sz <= this->node_vec.size()){
                    return;
                }

                self proxy = self(this->estimate_capacity(new_sz), std::move(this->hasher), std::move(this->pred), std::move(this->allocator));

                for (auto& node: this->node_vec){
                    proxy.internal_noexist_insert(std::move(node));
                }

                *this = std::move(proxy);
            }

            constexpr void rehash(size_type tentative_new_cap){

                if (tentative_new_cap < this->bucket_vec.size()){
                    return;
                }

                self proxy = self(tentative_new_cap, std::move(this->hasher), std::move(this->pred), std::move(this->allocator));

                for (auto& node: this->node_vec){
                    proxy.internal_noexist_insert(std::move(node));
                }

                *this = std::move(proxy);
            }

            constexpr auto empty() const noexcept -> bool{

                return this->node_vec.empty();
            }

            constexpr auto min_capacity() noexcept -> size_type{

                return 8u;
            } 

            constexpr auto size() const noexcept -> size_type{

                return this->node_vec.size();
            }

            constexpr auto max_size() const noexcept -> size_type{

                return std::numeric_limits<size_type>::max();
            }

            constexpr auto hash_function() const & noexcept -> const Hasher&{

                return this->hasher;
            }

            constexpr auto key_eq() const & noexcept -> const Pred&{

                return this->pred;
            }

            constexpr auto hash_function() && noexcept -> Hasher&&{

                return static_cast<Hasher&&>(this->hasher);
            }

            constexpr auto key_eq() && noexcept -> Pred&&{

                return static_cast<Pred&&>(this->pred);
            }

            constexpr auto load_factor() const noexcept -> double{

                return this->node_vec.size() / static_cast<double>(this->bucket_vec.size() - 1);
            }

            consteval auto max_load_factor() const noexcept -> double{

                return 0.875;
            }

            constexpr auto begin() noexcept -> iterator{

                return this->node_vec.begin();
            }

            constexpr auto begin() const noexcept -> const_iterator{

                return this->node_vec.begin();
            }

            constexpr auto begin(size_type off) noexcept -> iterator{

                return std::next(this->node_vec.begin(), off);
            }

            constexpr auto begin(size_type off) const noexcept -> const_iterator{

                return std::next(this->node_vec.begin(), off);
            }

            constexpr auto cbegin() noexcept -> reverse_iterator{

                return this->node_vec.cbegin();
            }

            constexpr auto cbegin() const noexcept -> const_reverse_iterator{

                return this->node_vec.cbegin();
            }

            constexpr auto cbegin(size_type off) noexcept -> reverse_iterator{

                return std::next(this->node_vec.cbegin(), off);
            }

            constexpr auto cbegin(size_type off) const noexcept -> const_reverse_iterator{

                return std::next(this->node_vec.cbegin(), off);
            }

            constexpr auto end() noexcept -> iterator{

                return this->node_vec.end();
            }

            constexpr auto end() const noexcept -> const_iterator{

                return this->node_vec.end();
            }

            constexpr auto cend() noexcept -> reverse_iterator{

                return this->node_vec.cend();
            }

            constexpr auto cend() const noexcept -> const_reverse_iterator{

                return this->node_vec.cend();
            }

        private:

            constexpr void check_for_rehash(){

                if (this->estimate_capacity(this->node_vec.size()) < this->bucket_vec.size()){
                    return;
                }

                size_type new_cap = (this->bucket_vec.size() - 1) * 2;
                this->rehash(new_cap);
            }

            constexpr void force_uphash(){

                size_type new_cap = (this->bucket_vec.size() - 1) * 2;
                this->rehash(new_cap);
            }

            constexpr auto estimate_capacity(size_type sz) const noexcept -> size_type{

                return sz / this->max_load_factor();
            }

            constexpr auto to_bucket_index(size_type hashed_value) const noexcept -> size_type{

                return hashed_value & (bucket_vec.size() - 2u);
            }

            template <class KeyLike>
            constexpr auto bucket_exist_find(const KeyLike& key) noexcept(noexcept(std::declval<Hasher&>()(std::declval<const KeyLike&>()))
                                                                          && noexcept(std::declval<Pred&>()(std::declval<const Key&>(), std::declval<const KeyLike&>()))) -> bucket_iterator{

                auto it = std::next(this->bucket_vec.begin(), this->to_bucket_index(this->hasher(key)));

                while (true){
                    if (*it != orphaned_virtual_addr && this->pred(this->node_vec[*it].first, key)){
                        return it;
                    }

                    std::advance(it, 1u);
                }
            }

            template <class KeyLike>
            constexpr auto bucket_exist_find(const KeyLike& key) const noexcept(noexcept(std::declval<const Hasher&>()(std::declval<const KeyLike&>()))
                                                                                 && noexcept(std::declval<const Pred&>()(std::declval<const Key&>(), std::declval<const KeyLike&>()))) -> bucket_const_iterator{

                auto it = std::next(this->bucket_vec.begin(), this->to_bucket_index(this->hasher(key)));

                while (true){
                    if (*it != orphaned_virtual_addr && this->pred(this->node_vec[*it].first, key)){
                        return it;
                    }

                    std::advance(it, 1u);
                }
            }

            template <class KeyLike>
            constexpr auto bucket_find(const KeyLike& key) noexcept(noexcept(std::declval<Hasher&>()(std::declval<const KeyLike&>()))
                                                                    && noexcept(std::declval<Pred&>()(std::declval<const Key&>(), std::declval<const KeyLike&>()))) -> bucket_iterator{

                auto it = std::next(this->bucket_vec.begin(), this->to_bucket_index(this->hasher(key)));

                while (true){
                    if (*it == null_virtual_addr || (*it != orphaned_virtual_addr && this->pred(this->node_vec[*it].first, key))){ 
                        return it;
                    }

                    std::advance(it, 1u);
                }
            }

            template <class KeyLike>
            constexpr auto bucket_find(const KeyLike& key) const noexcept(noexcept(std::declval<const Hasher&>()(std::declval<const KeyLike&>()))
                                                                          && noexcept(std::declval<const Pred&>()(std::declval<const Key&>(), std::declval<const KeyLike&>()))) -> bucket_const_iterator{

                auto it = std::next(this->bucket_vec.begin(), this->to_bucket_index(this->hasher(key)));

                while (true){
                    if (*it == null_virtual_addr || (*it != orphaned_virtual_addr && this->pred(this->node_vec[*it].first, key))){
                        return it;
                    }

                    std::advance(it, 1u);
                }
            }

            template <class KeyLike>
            constexpr auto bucket_ifind(const KeyLike& key) noexcept(noexcept(std::declval<Hasher&>()(std::declval<const KeyLike&>()))) -> bucket_iterator{

                auto it = std::next(this->bucket_vec.begin(), this->to_bucket_index(this->hasher(key)));

                while (true){
                    if (this->is_insertable(*it)){
                        return it;
                    }

                    std::advance(it, 1u);
                }
            }

            template <class ValueLike>
            constexpr auto internal_noexist_insert(ValueLike&& value) -> iterator{

                this->check_for_rehash();

                while (true){
                    bucket_iterator it = this->bucket_ifind(value.first);

                    if (it != std::prev(this->bucket_vec.end())){
                        size_type addr = this->node_vec.size();
                        this->node_vec.push_back(std::forward<ValueLike>(value));
                        *it = addr;
                        return std::prev(this->node_vec.end());
                    }

                    this->force_uphash();
                }
            } 

            template <class ValueLike>
            constexpr auto internal_insert_or_assign(ValueLike&& value) -> std::pair<iterator, bool>{

                bucket_iterator it = this->bucket_find(value.first);

                if (*it != null_virtual_addr){
                    iterator rs = std::next(this->node_vec.begin(), *it);
                    *rs = std::forward<ValueLike>(value);
                    return std::make_pair(rs, false);
                }

                return std::make_pair(this->internal_noexist_insert(std::forward<ValueLike>(value)), true);
            }

            template <class ValueLike>
            constexpr auto internal_insert(ValueLike&& value) -> std::pair<iterator, bool>{

                bucket_iterator it = this->bucket_find(value.first);

                if (*it != null_virtual_addr){
                    return std::make_pair(std::next(this->node_vec.begin(), *it), false);
                }

                return std::make_pair(this->internal_noexist_insert(std::forward<ValueLike>(value)), true);
            }

            template <class KeyLike, class Arg = mapped_type, std::enable_if_t<std::is_default_constructible_v<mapped_type>, bool> = true>
            constexpr auto internal_find_or_default(KeyLike&& key, Arg * compiler_hint = nullptr) -> std::pair<iterator, bool>{

                bucket_iterator it = this->bucket_find(key);

                if (*it != null_virtual_addr){
                    return std::make_pair(std::next(this->node_vec.begin(), *it), false);
                }

                return std::make_pair(this->internal_noexist_insert(value_type(std::forward<KeyLike>(key), mapped_type())), true);
            }

            template <class KeyLike>
            constexpr auto internal_erase(const KeyLike& key) noexcept(noexcept(std::declval<Hasher&>()(std::declval<const KeyLike&>()))) -> size_type{

                bucket_iterator erasing_bucket_it       = this->bucket_find(key);
                size_type erasing_bucket_virtual_addr   = *erasing_bucket_it;

                if (erasing_bucket_virtual_addr != null_virtual_addr){
                    bucket_iterator swapee_bucket_it = this->bucket_exist_find(this->node_vec.back().first); 
                    std::iter_swap(std::next(this->node_vec.begin(), erasing_bucket_virtual_addr), std::prev(this->node_vec.end()));
                    this->node_vec.pop_back();
                    *swapee_bucket_it   = erasing_bucket_virtual_addr;
                    *erasing_bucket_it  = orphaned_virtual_addr;

                    return 1u;
                }

                return 0u;
            }

            template <class KeyLike>
            constexpr void internal_exist_erase(const KeyLike& key) noexcept(noexcept(std::declval<Hasher&>()(std::declval<const KeyLike&>()))){

                bucket_iterator erasing_bucket_it       = this->bucket_exist_find(key);
                size_type erasing_bucket_virtual_addr   = std::exchange(*erasing_bucket_it, orphaned_virtual_addr);
                bucket_iterator swapee_bucket_it        = this->bucket_exist_find(this->node_vec.back().first);

                std::iter_swap(std::next(this->node_vec.begin(), erasing_bucket_virtual_addr), std::prev(this->node_vec.end()));
                this->node_vec.pop_back();
                *swapee_bucket_it = erasing_bucket_virtual_addr;
            }

            constexpr auto internal_erase(const_iterator it) noexcept -> iterator{

                if (it == this->node_vec.end()){
                    return it;
                }

                auto dif = std::distance(this->node_vec.begin(), it);
                this->internal_exist_erase(it->first);

                if (dif == this->node_vec.size()){
                    return this->node_vec.begin();
                }

                return std::next(this->node_vec.begin(), dif);
            }
    };
}

#endif