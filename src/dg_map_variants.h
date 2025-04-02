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
    
    template <class T, intmax_t Num, intmax_t Den, std::enable_if_t<std::is_unsigned_v<T>, bool> = true>
    static consteval auto get_max_pow2_multiplier_in_range(std::ratio<Num, Den>) -> T{

        using promoted_t = std::size_t;

        constexpr auto find_lambda = []() constexpr{
            promoted_t rs = 1u; 

            for (size_t i = 0u; i < std::numeric_limits<T>::digits; ++i){
                promoted_t cand = promoted_t{1} << i;

                if (cand * Num / Den <= std::numeric_limits<T>::max()){
                    rs = cand;
                }
            }

            return rs;
        };

        constexpr promoted_t rs = find_lambda();
        static_assert(rs * Num / Den <= std::numeric_limits<T>::max());

        return rs;
    }

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
            static inline constexpr std::size_t REHASH_CHK_MODULO       = 8u;
            static inline constexpr std::size_t LAST_MOHICAN_SZ         = 8u;

            static constexpr auto is_insertable(SizeType virtual_addr) noexcept -> bool{

                return (virtual_addr | SizeType{1u}) == std::numeric_limits<SizeType>::max();
            }

        public:

            static constexpr inline double MIN_MAX_LOAD_FACTOR      = 0.05;
            static constexpr inline double MAX_MAX_LOAD_FACTOR      = 0.95; 
            static constexpr inline double MIN_MAX_INSERT_FACTOR    = 0.05;
            static constexpr inline double MAX_MAX_INSERT_FACTOR    = 3; //1 - e^-3

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
            using size_type                     = SizeType;
            using difference_type               = intmax_t;
            using self                          = unordered_unstable_map;
            using load_factor_ratio             = typename LoadFactor::type;
            using insert_factor_ratio           = typename InsertFactor::type;
            using erase_hint                    = bucket_const_iterator;

            static consteval auto max_load_factor() -> double{

                return static_cast<double>(load_factor_ratio::num) / load_factor_ratio::den;
            }

            static consteval auto max_insert_factor() -> double{

                return static_cast<double>(insert_factor_ratio::num) / insert_factor_ratio::den;
            }

            static consteval auto min_capacity() -> size_type{

                return 32u;
            }

            static consteval auto max_capacity() -> size_type{
                
                return std::min(static_cast<size_type>(size_type{1} << (std::numeric_limits<size_type>::digits - 1)), get_max_pow2_multiplier_in_range<size_type>(insert_factor_ratio{}));
            }

            static consteval auto max_size() -> size_type{

                return max_capacity() * load_factor_ratio::num / load_factor_ratio::den;
            }

            static consteval auto max_insert_size() -> size_type{

                return max_capacity() * insert_factor_ratio::num / insert_factor_ratio::den;
            }

            static constexpr auto estimate_size(size_type cap) noexcept -> size_type{

                return cap * load_factor_ratio::num / load_factor_ratio::den;
            }

            static constexpr auto estimate_capacity(size_type sz) noexcept -> size_type{

                return sz * load_factor_ratio::den / load_factor_ratio::num;
            }

            static constexpr auto estimate_insert_capacity(size_type cap) noexcept -> size_type{

                return cap * insert_factor_ratio::num / insert_factor_ratio::den;
            }

            static_assert(min_capacity() <= max_capacity());
            static_assert(std::is_unsigned_v<size_type>);
            static_assert(std::is_unsigned_v<decltype(std::declval<const hasher&>()(std::declval<const key_type&>()))>);
            static_assert(noexcept(std::declval<const hasher&>()(std::declval<const key_type&>())));
            // static_assert(noexcept(std::declval<const key_equal&>()(std::declval<const key_type&>(), std::declval<const key_type&>()))); its 2024 and I dont know why these arent noexcept
            static_assert(std::is_nothrow_destructible_v<value_type>);
            static_assert(std::is_nothrow_swappable_v<value_type>);

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
                
                while (first != last){
                    internal_insert(*first);
                    std::advance(first, 1);
                }
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

            constexpr void rehash(size_type tentative_new_cap, bool force_rehash = false){

                if (!force_rehash && tentative_new_cap <= capacity()){
                    return;
                }

                while (true){
                    size_t new_cap  = std::max(self::min_capacity(), dg::map_variants::least_pow2_greater_equal_than(tentative_new_cap)) + LAST_MOHICAN_SZ;
                    bool bad_bit    = false;
                    decltype(bucket_vec) tmp_bucket_vec(new_cap, NULL_VIRTUAL_ADDR, allocator);

                    for (size_t i = 0u; i < node_vec.size(); ++i){
                        size_type bucket_idx    = hash_function()(static_cast<const Key&>(node_vec[i].first)) & (tmp_bucket_vec.size() - (LAST_MOHICAN_SZ + 1u));
                        auto it                 = std::find(std::next(tmp_bucket_vec.begin(), bucket_idx), tmp_bucket_vec.end(), NULL_VIRTUAL_ADDR);
                        [[assume(it != tmp_bucket_vec.end())]];

                        if (it != std::prev(tmp_bucket_vec.end())) [[likely]]{
                            *it = i;
                        } else [[unlikely]]{
                            tentative_new_cap = new_cap * 2;
                            bad_bit = true;
                            break;
                        }
                    }

                    if (!bad_bit){
                        bucket_vec = std::move(tmp_bucket_vec);
                        erase_count = 0u;
                        return;
                    }
                }
            }

            constexpr void reserve(size_type new_sz){
 
                if (new_sz <= size()){
                    return;
                }

                rehash(estimate_capacity(new_sz));
            }

            template <class ...Args>
            constexpr auto emplace(Args&& ...args) -> std::pair<iterator, bool>{

                return internal_insert(value_type(std::forward<Args>(args)...)); //alright guys - I just read the std and emplace and try_emplace are THE SAME THING - I dont know why they add the try_ to indicate that KeyLike&& is the first arg 
            }

            template <class KeyLike, class ...Args>
            constexpr auto try_emplace(KeyLike&& key, Args&& ...args) -> std::pair<iterator, bool>{

                return internal_insert_w_key(std::forward<KeyLike>(key), std::forward<Args>(args)...);
            }

            template <class ValueLike = value_type> //the only problem we were trying to solve was adding implicit initialization of value_type - so this should solve it
            constexpr auto insert(ValueLike&& value) -> std::pair<iterator, bool>{

                return internal_insert(std::forward<ValueLike>(value));
            }

            //developer has to use insert at their own discretion - because it might compromise speed - compared to insert(ValueType&&)
            template <class Iterator>
            constexpr void insert(Iterator first, Iterator last){

                static_assert(std::is_lvalue_reference_v<decltype(*first)>);

                size_t insert_sz        = std::distance(first, last); 
                auto insert_status_vec  = std::vector<uint8_t>(insert_sz, allocator);
                Iterator cur            = first;
                size_t succeed_sz       = 0u;

                try{
                    while (cur != last){
                        auto [_, status] = internal_insert(*cur);
                        std::advance(cur, 1);
                        insert_status_vec[succeed_sz++] = status;
                    }
                } catch (...){
                    cur = first;
                    for (size_t i = 0u; i < succeed_sz; ++i){
                        if (insert_status_vec[i]){
                            erase(cur->first);
                        }
                        std::advance(cur, 1);
                    }

                    std::rethrow_exception(std::current_exception());
                }
            }

            constexpr void insert(std::initializer_list<value_type> init_list){

                insert(init_list.begin(), init_list.end());
            }

            template <class KeyLike, class MappedLike>
            constexpr auto insert_or_assign(KeyLike&& key, MappedLike&& mapped) -> std::pair<iterator, bool>{

                return internal_insert_or_assign(value_type(std::forward<KeyLike>(key), std::forward<MappedLike>(mapped)));
            }

            template <class KeyLike>
            constexpr auto operator[](KeyLike&& key) -> mapped_type&{

                return internal_find_or_default(std::forward<KeyLike>(key))->second;
            }

            //I feel like these are noexcept(auto) features - and the APIs should explicitly declare the noexcept criterias that are not encapsulated by the container - we aren't worrying about that now

            constexpr void clear() noexcept{

                static_assert(noexcept(node_vec.clear()));
                std::fill(bucket_vec.begin(), bucket_vec.end(), NULL_VIRTUAL_ADDR);
                node_vec.clear();
            }

            constexpr void swap(self& other) noexcept(std::allocator_traits<Allocator>::is_always_equal
                                                      && std::is_nothrow_swappable_v<Hasher>
                                                      && std::is_nothrow_swappable_v<Pred>){

                std::swap(node_vec, other.node_vec);
                std::swap(bucket_vec, other.bucket_vec);
                std::swap(_hasher, other._hasher);
                std::swap(pred, other.pred);
                std::swap(allocator, other.allocator);
                std::swap(erase_count, other.erase_count);
            }

            template <class EraseArg>
            constexpr auto erase(EraseArg&& erase_arg) noexcept{

                if constexpr(std::is_convertible_v<EraseArg&&, const_iterator>){
                    static_assert(std::is_nothrow_convertible_v<EraseArg&&, const_iterator>);
                    return internal_erase_it(std::forward<EraseArg>(erase_arg));
                } else{
                    // static_assert(noexcept(internal_erase(std::forward<EraseArg>(erase_arg))));
                    return internal_erase(std::forward<EraseArg>(erase_arg));
                }
            }

            constexpr auto erase(const_iterator it, erase_hint hint) noexcept{

                return internal_erase_it_w_hint(it, hint);    
            }

            template <class KeyLike>
            constexpr auto contains(const KeyLike& key) const noexcept(noexcept(bucket_find(std::declval<const KeyLike&>()))) -> bool{

                return *bucket_find(key) != NULL_VIRTUAL_ADDR;
            }

            template <class KeyLike>
            constexpr auto count(const KeyLike& key) const noexcept(noexcept(bucket_find(std::declval<const KeyLike&>()))) -> size_type{

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
            constexpr auto erase_find(const KeyLike& key) noexcept(noexcept(bucket_find(std::declval<const KeyLike&>()))) -> std::pair<iterator, erase_hint>{

                bucket_iterator bucket  = bucket_find(key);
                size_type virtual_addr  = *bucket;

                if (virtual_addr == NULL_VIRTUAL_ADDR){
                    return std::make_pair(node_vec.end(), bucket);
                }

                return std::make_pair(std::next(node_vec.begin(), virtual_addr), bucket);
            }

            template <class KeyLike>
            constexpr auto erase_find(const KeyLike& key) const noexcept(noexcept(bucket_find(std::declval<const KeyLike&>()))) -> std::pair<const_iterator, erase_hint>{

                bucket_const_iterator bucket    = bucket_find(key);
                size_type virtual_addr          = *bucket;

                if (virtual_addr == NULL_VIRTUAL_ADDR){
                    return std::make_pair(node_vec.end(), bucket);
                }

                return std::make_pair(std::next(node_vec.begin(), virtual_addr), bucket);
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

            constexpr auto capacity() const noexcept -> size_type{

                return bucket_vec.size() - LAST_MOHICAN_SZ;
            }

            constexpr auto size() const noexcept -> size_type{

                return node_vec.size();
            }

            constexpr auto insert_size() const noexcept -> size_type{

                return size() + erase_count;
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

            constexpr auto cbegin() const noexcept -> const_iterator{

                return node_vec.cbegin();
            }

            constexpr auto end() noexcept -> iterator{

                return node_vec.end();
            }

            constexpr auto end() const noexcept -> const_iterator{

                return node_vec.end();
            }

            constexpr auto cend() const noexcept -> const_iterator{

                return node_vec.cend();
            }

        private:

            constexpr void maybe_check_for_rehash(){

                if (((size() + erase_count) % REHASH_CHK_MODULO) != 0u) [[likely]]{
                    return;
                } else{
                    if (size() < estimate_size(capacity()) && insert_size() < estimate_insert_capacity(capacity())) [[likely]]{ //might be buggy - not this line but other inserts that do not lead to this line
                        return;
                    } else [[unlikely]]{
                        //either cap > size or insert_cap > max_insert_cap or both - if both - extend
                        if (size() >= estimate_size(capacity())){
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

            constexpr auto to_bucket_index(auto hashed_value) const noexcept -> size_type{

                static_assert(std::is_unsigned_v<decltype(hashed_value)>);
                return hashed_value & (bucket_vec.size() - (LAST_MOHICAN_SZ + 1u));
            }

            template <class KeyLike>
            constexpr auto bucket_exist_find(const KeyLike& key) noexcept(noexcept(std::declval<const Hasher&>()(std::declval<const KeyLike&>()))
                                                                          && noexcept(std::declval<const Pred&>()(std::declval<const Key&>(), std::declval<const KeyLike&>()))) -> bucket_iterator{

                auto it = std::next(bucket_vec.begin(), to_bucket_index(hash_function()(key)));

                while (true){
                    if (*it != ORPHANED_VIRTUAL_ADDR && key_eq()(static_cast<const Key&>(node_vec[*it].first), key)){
                        return it;
                    }

                    std::advance(it, 1u);
                }
            }

            template <class KeyLike>
            constexpr auto bucket_exist_find(const KeyLike& key) const noexcept(noexcept(std::declval<const Hasher&>()(std::declval<const KeyLike&>()))
                                                                                 && noexcept(std::declval<const Pred&>()(std::declval<const Key&>(), std::declval<const KeyLike&>()))) -> bucket_const_iterator{

                auto it = std::next(bucket_vec.begin(), to_bucket_index(hash_function()(key)));

                while (true){
                    if (*it != ORPHANED_VIRTUAL_ADDR && key_eq()(node_vec[*it].first, key)){
                        return it;
                    }

                    std::advance(it, 1u);
                }
            }

            template <class KeyLike>
            constexpr auto bucket_find(const KeyLike& key) noexcept(noexcept(std::declval<const Hasher&>()(std::declval<const KeyLike&>()))
                                                                    && noexcept(std::declval<const Pred&>()(std::declval<const Key&>(), std::declval<const KeyLike&>()))) -> bucket_iterator{

                auto it = std::next(bucket_vec.begin(), to_bucket_index(hash_function()(key)));

                while (true){
                    if (*it == NULL_VIRTUAL_ADDR || (*it != ORPHANED_VIRTUAL_ADDR && key_eq()(static_cast<const Key&>(node_vec[*it].first), key))){
                        return it;
                    }

                    std::advance(it, 1u);
                }
            }

            template <class KeyLike>
            constexpr auto bucket_find(const KeyLike& key) const noexcept(noexcept(std::declval<const Hasher&>()(std::declval<const KeyLike&>()))
                                                                          && noexcept(std::declval<const Pred&>()(std::declval<const Key&>(), std::declval<const KeyLike&>()))) -> bucket_const_iterator{

                auto it = std::next(bucket_vec.begin(), to_bucket_index(hash_function()(key)));

                while (true){
                    if (*it == NULL_VIRTUAL_ADDR || (*it != ORPHANED_VIRTUAL_ADDR && key_eq()(node_vec[*it].first, key))){
                        return it;
                    }

                    std::advance(it, 1u);
                }
            }

            template <class KeyLike>
            constexpr auto bucket_ifind(const KeyLike& key) noexcept(noexcept(std::declval<const Hasher&>()(std::declval<const KeyLike&>()))) -> bucket_iterator{

                auto it = std::next(bucket_vec.begin(), to_bucket_index(hash_function()(key)));

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

            template <class KeyLike, class ...Args>
            constexpr auto internal_insert_w_key(KeyLike&& key, Args&& ...args) -> std::pair<iterator, bool>{

                bucket_iterator it = bucket_find(key);

                if (*it != NULL_VIRTUAL_ADDR){
                    return std::make_pair(std::next(node_vec.begin(), *it), false);
                }

                auto value = value_type(std::piecewise_construct, std::forward_as_tuple(std::forward<KeyLike>(key)), std::forward_as_tuple(std::forward<Args>(args)...));
                return std::make_pair(internal_noexist_insert(std::move(value)), true);
            }

            template <class KeyLike, class Arg = mapped_type, std::enable_if_t<std::is_default_constructible_v<Arg>, bool> = true>
            constexpr auto internal_find_or_default(KeyLike&& key, Arg * compiler_hint = nullptr) -> iterator{

                bucket_iterator it = bucket_find(key);

                if (*it != NULL_VIRTUAL_ADDR){
                    return std::next(node_vec.begin(), *it);
                }

                return internal_noexist_insert(value_type(std::forward<KeyLike>(key), mapped_type()));
            }

            constexpr void internal_exist_bucket_erase(bucket_iterator erasing_bucket_it) noexcept{

                bucket_iterator swapee_bucket_it        = bucket_exist_find(node_vec.back().first);
                size_type erasing_bucket_virtual_addr   = std::exchange(*erasing_bucket_it, ORPHANED_VIRTUAL_ADDR);

                if (swapee_bucket_it != erasing_bucket_it){
                    std::iter_swap(std::next(node_vec.begin(), erasing_bucket_virtual_addr), std::prev(node_vec.end()));
                    *swapee_bucket_it = erasing_bucket_virtual_addr;
                }

                node_vec.pop_back();
                erase_count += 1;
            }

            constexpr void internal_exist_bucket_erase(bucket_const_iterator erasing_bucket_const_it) noexcept{

                bucket_iterator erasing_bucket_it = std::next(bucket_vec.begin(), std::distance(bucket_vec.cbegin(), erasing_bucket_const_it)); //I dont think compiler is allowed to optimize this - idk
                internal_exist_bucket_erase(erasing_bucket_it);
            }

            template <class KeyLike>
            constexpr auto internal_erase(const KeyLike& key) noexcept(noexcept(bucket_find(std::declval<const KeyLike&>()))) -> size_type{

                bucket_iterator erasing_bucket_it       = bucket_find(key);
                size_type erasing_bucket_virtual_addr   = *erasing_bucket_it;

                if (erasing_bucket_virtual_addr != NULL_VIRTUAL_ADDR){
                    internal_exist_bucket_erase(erasing_bucket_it);
                    return 1u;
                }

                return 0u;
            }

            template <class KeyLike>
            constexpr void internal_exist_erase(const KeyLike& key) noexcept(noexcept(bucket_exist_find(std::declval<const KeyLike&>()))){
                
                internal_exist_bucket_erase(bucket_exist_find(key));
            }

            constexpr auto internal_erase_it(const_iterator it) noexcept -> iterator{

                if (it == node_vec.end()){
                    return node_vec.end();
                }

                auto dif = std::distance(node_vec.cbegin(), it);
                internal_exist_erase(it->first);

                if (dif != node_vec.size()) [[likely]]{
                    return std::next(node_vec.begin(), dif);
                } else [[unlikely]]{
                    return node_vec.begin();
                }
            }

            constexpr auto internal_erase_it_w_hint(const_iterator it, bucket_const_iterator hint) noexcept -> iterator{

                if (it == node_vec.end()){
                    return node_vec.end();
                }

                auto dif = std::distance(node_vec.cbegin(), it);
                internal_exist_bucket_erase(hint);

                if (dif != node_vec.size()) [[likely]]{
                    return std::next(node_vec.begin(), dif);
                } else [[unlikely]]{
                    return node_vec.begin();
                }
            }
    };

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
            static inline constexpr std::size_t REHASH_CHK_MODULO       = 8u;
            static inline constexpr std::size_t LAST_MOHICAN_SZ         = 8u;

            static constexpr auto is_insertable(SizeType virtual_addr) noexcept -> bool{

                return virtual_addr == NULL_VIRTUAL_ADDR || virtual_addr == ORPHANED_VIRTUAL_ADDR;
            }

        public:

            static constexpr inline double MIN_MAX_LOAD_FACTOR      = 0.05;
            static constexpr inline double MAX_MAX_LOAD_FACTOR      = 0.95;
            static constexpr inline double MIN_MAX_INSERT_FACTOR    = 0.05;
            static constexpr inline double MAX_MAX_INSERT_FACTOR    = 3; //1 - e^-3

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
            using size_type                     = SizeType;
            using difference_type               = intmax_t;
            using self                          = unordered_unstable_fast_map;
            using load_factor_ratio             = typename LoadFactor::type; 
            using insert_factor_ratio           = typename InsertFactor::type;
            using erase_hint                    = bucket_const_iterator;

            static consteval auto max_load_factor() -> double{

                return static_cast<double>(load_factor_ratio::num) / load_factor_ratio::den;
            }

            static consteval auto max_insert_factor() -> double{

                return static_cast<double>(insert_factor_ratio::num) / insert_factor_ratio::den;
            }

            static consteval auto min_capacity() -> size_type{

                return 32u;
            }

            static consteval auto max_capacity() -> size_type{

                return std::min(static_cast<size_type>(size_type{1} << (std::numeric_limits<size_type>::digits - 1)), get_max_pow2_multiplier_in_range<size_type>(insert_factor_ratio{}));
            }

            static consteval auto max_size() -> size_type{

                return max_capacity() * load_factor_ratio::num / load_factor_ratio::den;
            }

            static consteval auto max_insert_size() -> size_type{

                return max_capacity() * insert_factor_ratio::num / insert_factor_ratio::den;
            }

            static constexpr auto estimate_size(size_type cap) noexcept -> size_type{

                return cap * load_factor_ratio::num / load_factor_ratio::den;
            }

            static constexpr auto estimate_capacity(size_type sz) noexcept -> size_type{

                return sz * load_factor_ratio::den / load_factor_ratio::num;
            }

            static constexpr auto estimate_insert_capacity(size_type cap) noexcept -> size_type{

                return cap * insert_factor_ratio::num / insert_factor_ratio::den;
            }

            static_assert(min_capacity() <= max_capacity());
            static_assert(std::is_unsigned_v<size_type>);
            static_assert(std::is_unsigned_v<decltype(std::declval<const hasher&>()(std::declval<const key_type&>()))>);
            static_assert(noexcept(std::declval<const hasher&>()(std::declval<const key_type&>())));
            // static_assert(noexcept(std::declval<const key_equal&>()(std::declval<const key_type&>(), std::declval<const key_type&>()))); its 2024 and I dont know why these arent noexcept
            static_assert(std::is_nothrow_destructible_v<value_type>);
            static_assert(std::is_nothrow_swappable_v<value_type>);

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

                while (first != last){
                    internal_insert(*first);
                    std::advance(first, 1);
                }
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

            constexpr void rehash(size_type tentative_new_cap, bool force_rehash = false){

                if (!force_rehash && tentative_new_cap <= capacity()){
                    return;
                }

                while (true){
                    size_t new_cap  = std::max(self::min_capacity(), dg::map_variants::least_pow2_greater_equal_than(tentative_new_cap)) + LAST_MOHICAN_SZ;
                    bool bad_bit    = false; 
                    decltype(bucket_vec) tmp_bucket_vec(new_cap, NULL_VIRTUAL_ADDR, allocator);

                    for (size_t i = 1u; i < node_vec.size(); ++i){
                        size_type bucket_idx    = hash_function()(static_cast<const Key&>(node_vec[i].first)) & (tmp_bucket_vec.size() - (LAST_MOHICAN_SZ + 1u));
                        auto it                 = std::find(std::next(tmp_bucket_vec.begin(), bucket_idx), tmp_bucket_vec.end(), NULL_VIRTUAL_ADDR);
                        [[assume(it != tmp_bucket_vec.end())]];

                        if (it != std::prev(tmp_bucket_vec.end())) [[likely]]{
                            *it = i;
                        } else [[unlikely]]{
                            tentative_new_cap = new_cap * 2;
                            bad_bit = true;
                            break;
                        }
                    }

                    if (!bad_bit){
                        bucket_vec = std::move(tmp_bucket_vec);
                        erase_count = 0u;
                        return;
                    }
                }
            }

            constexpr void reserve(size_type new_sz){
 
                if (new_sz <= size()){
                    return;
                }

                rehash(estimate_capacity(new_sz));
            }

            template <class ...Args>
            constexpr auto emplace(Args&& ...args) -> std::pair<iterator, bool>{

                return internal_insert(value_type(std::forward<Args>(args)...));
            }

            template <class KeyLike, class ...Args>
            constexpr auto try_emplace(KeyLike&& key, Args&& ...args) -> std::pair<iterator, bool>{

                return internal_insert_w_key(std::forward<KeyLike>(key), std::forward<Args>(args)...);
            }

            template <class ValueLike = value_type>
            constexpr auto insert(ValueLike&& value) -> std::pair<iterator, bool>{

                return internal_insert(std::forward<ValueLike>(value));
            }

            //developer has to use insert at their own discretion - because it might compromise speed - compared to insert(ValueType&&)
            template <class Iterator>
            constexpr void insert(Iterator first, Iterator last){

                static_assert(std::is_lvalue_reference_v<decltype(*first)>);

                size_t insert_sz        = std::distance(first, last);
                auto insert_status_vec  = std::vector<uint8_t>(insert_sz, allocator);
                Iterator cur            = first;
                size_t succeed_sz       = 0u;

                try{
                    while (cur != last){
                        auto [_, status] = internal_insert(*cur);
                        std::advance(cur, 1);
                        insert_status_vec[succeed_sz++] = status;
                    }
                } catch (...){
                    cur = first;
                    for (size_t i = 0u; i < succeed_sz; ++i){
                        if (insert_status_vec[i]){
                            erase(cur->first);
                        }
                        std::advance(cur, 1);
                    }

                    std::rethrow_exception(std::current_exception());
                }
            }

            constexpr void insert(std::initializer_list<value_type> init_list){

                insert(init_list.begin(), init_list.end());
            }

            template <class KeyLike, class MappedLike>
            constexpr auto insert_or_assign(KeyLike&& key, MappedLike&& mapped) -> std::pair<iterator, bool>{

                return internal_insert_or_assign(value_type(std::forward<KeyLike>(key), std::forward<MappedLike>(mapped)));
            }

            template <class KeyLike>
            constexpr auto operator[](KeyLike&& key) -> mapped_type&{

                return internal_find_or_default(std::forward<KeyLike>(key))->second;
            }

            //I feel like these are noexcept(auto) features - and the APIs should explicitly declare the noexcept criterias that are not encapsulated by the container - we aren't worrying about that now

            constexpr void clear() noexcept{

                std::fill(bucket_vec.begin(), bucket_vec.end(), NULL_VIRTUAL_ADDR);
                node_vec.resize(1u);
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

            template <class EraseArg>
            constexpr auto erase(EraseArg&& erase_arg) noexcept{

                if constexpr(std::is_convertible_v<EraseArg&&, const_iterator>){
                    static_assert(std::is_nothrow_convertible_v<EraseArg&&, const_iterator>);
                    return internal_erase_it(std::forward<EraseArg>(erase_arg));
                } else{
                    // static_assert(noexcept(internal_erase(std::forward<EraseArg>(erase_arg))));
                    return internal_erase(std::forward<EraseArg>(erase_arg));
                }
            }

            constexpr auto erase(const_iterator it, erase_hint hint) noexcept{

                return internal_erase_it_w_hint(it, hint);    
            }

            template <class KeyLike>
            constexpr auto contains(const KeyLike& key) const noexcept(noexcept(bucket_find(std::declval<const KeyLike&>()))) -> bool{

                return *bucket_find(key) != NULL_VIRTUAL_ADDR;
            }

            template <class KeyLike>
            constexpr auto count(const KeyLike& key) const noexcept(noexcept(bucket_find(std::declval<const KeyLike&>()))) -> size_type{

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
            constexpr auto erase_find(const KeyLike& key) noexcept(noexcept(bucket_find(std::declval<const KeyLike&>()))) -> std::pair<iterator, erase_hint>{

                bucket_iterator bucket  = bucket_find(key);
                size_type virtual_addr  = *bucket;

                if (virtual_addr == NULL_VIRTUAL_ADDR){
                    return std::make_pair(node_vec.end(), bucket);
                }

                return std::make_pair(std::next(node_vec.begin(), virtual_addr), bucket);
            }

            template <class KeyLike>
            constexpr auto erase_find(const KeyLike& key) const noexcept(noexcept(bucket_find(std::declval<const KeyLike&>()))) -> std::pair<const_iterator, erase_hint>{

                bucket_const_iterator bucket    = bucket_find(key);
                size_type virtual_addr          = *bucket;

                if (virtual_addr == NULL_VIRTUAL_ADDR){
                    return std::make_pair(node_vec.end(), bucket);
                }

                return std::make_pair(std::next(node_vec.begin(), virtual_addr), bucket);
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

            constexpr auto capacity() const noexcept -> size_type{

                return bucket_vec.size() - LAST_MOHICAN_SZ;
            }

            constexpr auto size() const noexcept -> size_type{

                return node_vec.size() - 1u;
            }

            constexpr auto insert_size() const noexcept -> size_type{

                return size() + erase_count;
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

            constexpr auto cbegin() const noexcept -> const_iterator{

                return std::next(node_vec.cbegin());
            }

            constexpr auto end() noexcept -> iterator{

                return node_vec.end();
            }

            constexpr auto end() const noexcept -> const_iterator{

                return node_vec.end();
            }

            constexpr auto cend() const noexcept -> const_iterator{

                return node_vec.cend();
            }

        private:

            constexpr void maybe_check_for_rehash(){

                if (((size() + erase_count) % REHASH_CHK_MODULO) != 0u) [[likely]]{
                    return;
                } else [[unlikely]]{
                    if (size() < estimate_size(capacity()) && insert_size() < estimate_insert_capacity(capacity())) [[likely]]{ //might be buggy - not this line but other inserts that do not lead to this line
                        return;
                    } else [[unlikely]]{
                        //either cap > size or insert_cap > max_insert_cap or both - if both - extend
                        if (size() >= estimate_size(capacity())){
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

            constexpr auto to_bucket_index(auto hashed_value) const noexcept -> size_type{

                static_assert(std::is_unsigned_v<decltype(hashed_value)>);
                return hashed_value & (bucket_vec.size() - (LAST_MOHICAN_SZ + 1u));
            }

            template <class KeyLike>
            constexpr auto bucket_exist_find(const KeyLike& key) noexcept(noexcept(std::declval<const Hasher&>()(std::declval<const KeyLike&>()))
                                                                          && noexcept(std::declval<const Pred&>()(std::declval<const Key&>(), std::declval<const KeyLike&>()))) -> bucket_iterator{

                //GCC sets branch prediction 1(unlikely)/10(likely) 
                //assume load_factor of 50% - avg - and reasonable hash function
                //50% ^ 3 = 1/8 - which is == branch predictor

                auto it = std::next(bucket_vec.begin(), to_bucket_index(hash_function()(key)));

                if (key_eq()(static_cast<const Key&>(node_vec[*it].first), key)){ //this optimization might not be as important in CPU arch but very important in GPU arch - where you want to minimize branch prediction by using block_quicksort approach
                    return it;
                }

                std::advance(it, 1u);

                if (key_eq()(static_cast<const Key&>(node_vec[*it].first), key)){
                    return it;
                }

                std::advance(it, 1u);

                while (true){
                    if (key_eq()(static_cast<const Key&>(node_vec[*it].first), key)) [[likely]]{
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

                auto it = std::next(bucket_vec.begin(), to_bucket_index(hash_function()(key)));

                if (key_eq()(node_vec[*it].first, key)){
                    return it;
                }

                std::advance(it, 1u);

                if (key_eq()(node_vec[*it].first, key)){
                    return it;
                }

                std::advance(it, 1u);

                while (true){
                    if (key_eq()(node_vec[*it].first, key)) [[likely]]{
                        return it;
                    }

                    std::advance(it, 1u);
                }
            }

            template <class KeyLike>
            constexpr auto bucket_find(const KeyLike& key) noexcept(noexcept(std::declval<const Hasher&>()(std::declval<const KeyLike&>()))
                                                                    && noexcept(std::declval<const Pred&>()(std::declval<const Key&>(), std::declval<const KeyLike&>()))) -> bucket_iterator{

                auto it = std::next(bucket_vec.begin(), to_bucket_index(hash_function()(key)));

                while (true){
                    if (*it == NULL_VIRTUAL_ADDR || key_eq()(static_cast<const Key&>(node_vec[*it].first), key)){ //branching is expensive - but I dont know if *it == NULL_VIRTUAL_ADDR should be optimized - I dont think so - this has a clearer branching pipeline - and increase the chance of hitting branch prediction
                        return it;
                    }

                    std::advance(it, 1u);
                }
            }

            template <class KeyLike>
            constexpr auto bucket_find(const KeyLike& key) const noexcept(noexcept(std::declval<const Hasher&>()(std::declval<const KeyLike&>()))
                                                                          && noexcept(std::declval<const Pred&>()(std::declval<const Key&>(), std::declval<const KeyLike&>()))) -> bucket_const_iterator{

                auto it = std::next(bucket_vec.begin(), to_bucket_index(hash_function()(key)));

                while (true){
                    if (*it == NULL_VIRTUAL_ADDR || key_eq()(node_vec[*it].first, key)){
                        return it;
                    }

                    std::advance(it, 1u);
                }
            }

            template <class KeyLike>
            constexpr auto bucket_ifind(const KeyLike& key) noexcept(noexcept(std::declval<const Hasher&>()(std::declval<const KeyLike&>()))) -> bucket_iterator{

                auto it = std::next(bucket_vec.begin(), to_bucket_index(hash_function()(key)));

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

            template <class KeyLike, class ...Args>
            constexpr auto internal_insert_w_key(KeyLike&& key, Args&& ...args) -> std::pair<iterator, bool>{

                bucket_iterator it = bucket_find(key);

                if (*it != NULL_VIRTUAL_ADDR){
                    return std::make_pair(std::next(node_vec.begin(), *it), false);
                }

                auto value = value_type(std::piecewise_construct, std::forward_as_tuple(std::forward<KeyLike>(key)), std::forward_as_tuple(std::forward<Args>(args)...));
                return std::make_pair(internal_noexist_insert(std::move(value)), true);
            }

            template <class KeyLike, class Arg = mapped_type, std::enable_if_t<std::is_default_constructible_v<Arg>, bool> = true>
            constexpr auto internal_find_or_default(KeyLike&& key, Arg * compiler_hint = nullptr) -> iterator{

                bucket_iterator it = bucket_find(key);

                if (*it != NULL_VIRTUAL_ADDR){
                    return std::next(node_vec.begin(), *it);
                }

                return internal_noexist_insert(value_type(std::forward<KeyLike>(key), mapped_type()));
            }

            constexpr void internal_exist_bucket_erase(bucket_iterator erasing_bucket_it) noexcept{

                bucket_iterator swapee_bucket_it        = bucket_exist_find(node_vec.back().first);
                size_type erasing_bucket_virtual_addr   = std::exchange(*erasing_bucket_it, ORPHANED_VIRTUAL_ADDR);

                if (swapee_bucket_it != erasing_bucket_it){
                    std::iter_swap(std::next(node_vec.begin(), erasing_bucket_virtual_addr), std::prev(node_vec.end()));
                    *swapee_bucket_it = erasing_bucket_virtual_addr;
                }

                node_vec.pop_back();
                erase_count += 1;
            }

            constexpr void internal_exist_bucket_erase(bucket_const_iterator erasing_bucket_const_it) noexcept{

                bucket_iterator erasing_bucket_it = std::next(bucket_vec.begin(), std::distance(bucket_vec.cbegin(), erasing_bucket_const_it)); //I dont think compiler is allowed to optimize this - idk
                internal_exist_bucket_erase(erasing_bucket_it);
            }

            template <class KeyLike>
            constexpr auto internal_erase(const KeyLike& key) noexcept(noexcept(bucket_find(std::declval<const KeyLike&>()))) -> size_type{

                bucket_iterator erasing_bucket_it       = bucket_find(key);
                size_type erasing_bucket_virtual_addr   = *erasing_bucket_it;

                if (erasing_bucket_virtual_addr != NULL_VIRTUAL_ADDR){
                    internal_exist_bucket_erase(erasing_bucket_it);
                    return 1u;
                }

                return 0u;
            }

            template <class KeyLike>
            constexpr void internal_exist_erase(const KeyLike& key) noexcept(noexcept(bucket_exist_find(std::declval<const KeyLike&>()))){

                internal_exist_bucket_erase(bucket_exist_find(key));
            }

            constexpr auto internal_erase_it(const_iterator it) noexcept -> iterator{

                if (it == node_vec.end()){
                    return node_vec.end();
                }

                auto dif = std::distance(node_vec.cbegin(), it);
                internal_exist_erase(it->first);

                if (dif != node_vec.size()) [[likely]]{
                    return std::next(node_vec.begin(), dif);
                } else [[unlikely]]{
                    return std::next(node_vec.begin());
                }
            }

            constexpr auto internal_erase_it_w_hint(const_iterator it, bucket_const_iterator hint) noexcept -> iterator{

                if (it == node_vec.end()){
                    return node_vec.end();
                }

                auto dif = std::distance(node_vec.cbegin(), it);
                internal_exist_bucket_erase(hint);

                if (dif != node_vec.size()) [[likely]]{
                    return std::next(node_vec.begin(), dif);
                } else [[unlikely]]{
                    return std::next(node_vec.begin());
                }
            }
    };

    template <class Key, class Mapped, class NullValueGenerator, class SizeType = std::size_t, class Hasher = std::hash<Key>, class Pred = std::equal_to<Key>, class Allocator = std::allocator<std::pair<Key, Mapped>>, class LoadFactor = std::ratio<1, 2>, class InsertFactor = std::ratio<3, 4>>
    class unordered_unstable_fastinsert_map{

        private:

            std::vector<std::pair<Key, Mapped>, typename std::allocator_traits<Allocator>::template rebind_alloc<std::pair<Key, Mapped>>> node_vec;
            std::vector<SizeType, typename std::allocator_traits<Allocator>::template rebind_alloc<SizeType>> bucket_vec; //I know for other OS - it is faster to do *ptr - instead of ptr[idx] - I dont know if this is an optimizable worth to make - this is a 5%-10% optimization opportunity - yet I dont think its worth the readability
            Hasher _hasher;
            Pred pred;
            Allocator allocator;
            SizeType erase_count;

            using bucket_iterator               = typename std::vector<SizeType, typename std::allocator_traits<Allocator>::template rebind_alloc<SizeType>>::iterator;
            using bucket_const_iterator         = typename std::vector<SizeType, typename std::allocator_traits<Allocator>::template rebind_alloc<SizeType>>::const_iterator;

            static inline constexpr SizeType ORPHANED_VIRTUAL_ADDR      = std::numeric_limits<SizeType>::min();
            static inline constexpr SizeType NULL_VIRTUAL_ADDR          = std::numeric_limits<SizeType>::max();
            static inline constexpr std::size_t REHASH_CHK_MODULO       = 8u;
            static inline constexpr std::size_t LAST_MOHICAN_SZ         = 8u;

        public:

            static constexpr inline double MIN_MAX_LOAD_FACTOR      = 0.05;
            static constexpr inline double MAX_MAX_LOAD_FACTOR      = 0.95;
            static constexpr inline double MIN_MAX_INSERT_FACTOR    = 0.05;
            static constexpr inline double MAX_MAX_INSERT_FACTOR    = 0.95; 

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
            using size_type                     = SizeType;
            using difference_type               = intmax_t;
            using self                          = unordered_unstable_fastinsert_map;
            using load_factor_ratio             = typename LoadFactor::type;
            using insert_factor_ratio           = typename InsertFactor::type;
            using erase_hint                    = bucket_const_iterator;

            static consteval auto max_load_factor() -> double{

                return static_cast<double>(load_factor_ratio::num) / load_factor_ratio::den;
            }

            static consteval auto max_insert_factor() -> double{

                return static_cast<double>(insert_factor_ratio::num) / insert_factor_ratio::den;
            }

            static consteval auto min_capacity() -> size_type{

                return 32u;
            }

            static consteval auto max_capacity() -> size_type{

                return std::min(static_cast<size_type>(size_type{1} << (std::numeric_limits<size_type>::digits - 1)), get_max_pow2_multiplier_in_range<size_type>(insert_factor_ratio{}));
            }

            static consteval auto max_size() -> size_type{

                return max_capacity() * load_factor_ratio::num / load_factor_ratio::den;
            }

            static consteval auto max_insert_size() -> size_type{

                return max_capacity() * insert_factor_ratio::num / insert_factor_ratio::den;
            }

            static constexpr auto estimate_size(size_type cap) noexcept -> size_type{

                return cap * load_factor_ratio::num / load_factor_ratio::den;
            }

            static constexpr auto estimate_capacity(size_type sz) noexcept -> size_type{

                return sz * load_factor_ratio::den / load_factor_ratio::num;
            }

            static constexpr auto estimate_insert_capacity(size_type cap) noexcept -> size_type{

                return cap * insert_factor_ratio::num / insert_factor_ratio::den;
            }

            static_assert(min_capacity() <= max_capacity());
            static_assert(std::is_unsigned_v<size_type>);
            static_assert(std::is_unsigned_v<decltype(std::declval<const hasher&>()(std::declval<const key_type&>()))>);
            static_assert(noexcept(std::declval<const hasher&>()(std::declval<const key_type&>())));
            // static_assert(noexcept(std::declval<const key_equal&>()(std::declval<const key_type&>(), std::declval<const key_type&>()))); its 2024 and I dont know why these arent noexcept
            static_assert(std::is_nothrow_destructible_v<value_type>);
            static_assert(std::is_nothrow_swappable_v<value_type>);

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

                while (first != last){
                    internal_insert(*first);
                    std::advance(first, 1);
                }            
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

            constexpr void rehash(size_type tentative_new_cap, bool force_rehash = false){

                if (!force_rehash && tentative_new_cap <= capacity()){
                    return;
                }

                while (true){
                    size_t new_cap  = std::max(self::min_capacity(), dg::map_variants::least_pow2_greater_equal_than(tentative_new_cap)) + LAST_MOHICAN_SZ;
                    bool bad_bit    = false;
                    decltype(bucket_vec) tmp_bucket_vec(new_cap, NULL_VIRTUAL_ADDR, allocator);

                    for (size_t i = 1u; i < node_vec.size(); ++i){
                        size_type bucket_idx    = hash_function()(static_cast<const Key&>(node_vec[i].first)) & (tmp_bucket_vec.size() - (LAST_MOHICAN_SZ + 1u));
                        auto it                 = std::find(std::next(tmp_bucket_vec.begin(), bucket_idx), tmp_bucket_vec.end(), NULL_VIRTUAL_ADDR);
                        [[assume(it != tmp_bucket_vec.end())]];

                        if (it != std::prev(tmp_bucket_vec.end())) [[likely]]{
                            *it = i;
                        } else [[unlikely]]{
                            tentative_new_cap = new_cap * 2;
                            bad_bit = true;
                            break;
                        }
                    }

                    if (!bad_bit){
                        bucket_vec = std::move(tmp_bucket_vec);
                        erase_count = 0u;
                        return;
                    }
                }
            }

            constexpr void reserve(size_type new_sz){
 
                if (new_sz <= size()){
                    return;
                }

                rehash(estimate_capacity(new_sz));
            }

            template <class ...Args>
            constexpr auto emplace(Args&& ...args) -> std::pair<iterator, bool>{

                return internal_insert(value_type(std::forward<Args>(args)...));
            }

            template <class KeyLike, class ...Args>
            constexpr auto try_emplace(KeyLike&& key, Args&& ...args) -> std::pair<iterator, bool>{

                return internal_insert_w_key(std::forward<KeyLike>(key), std::forward<Args>(args)...);
            }

            template <class ValueLike = value_type>
            constexpr auto insert(ValueLike&& value) -> std::pair<iterator, bool>{

                return internal_insert(std::forward<ValueLike>(value));
            }

            template <class Iterator>
            constexpr void insert(Iterator first, Iterator last){

                static_assert(std::is_lvalue_reference_v<decltype(*first)>);

                size_t insert_sz        = std::distance(first, last);
                auto insert_status_vec  = std::vector<uint8_t>(insert_sz, allocator);
                Iterator cur            = first;
                size_t succeed_sz       = 0u;

                try{
                    while (cur != last){
                        auto [_, status] = internal_insert(*cur);
                        std::advance(cur, 1);
                        insert_status_vec[succeed_sz++] = status;
                    }
                } catch (...){
                    cur = first;
                    for (size_t i = 0u; i < succeed_sz; ++i){
                        if (insert_status_vec[i]){
                            erase(cur->first);
                        }
                        std::advance(cur, 1);
                    }

                    std::rethrow_exception(std::current_exception());
                }
            }

            constexpr void insert(std::initializer_list<value_type> init_list){

                insert(init_list.begin(), init_list.end());
            }

            template <class KeyLike, class MappedLike>
            constexpr auto insert_or_assign(KeyLike&& key, MappedLike&& mapped) -> std::pair<iterator, bool>{

                return internal_insert_or_assign(value_type(std::forward<KeyLike>(key), std::forward<MappedLike>(mapped)));
            }

            template <class KeyLike>
            constexpr auto operator[](KeyLike&& key) -> mapped_type&{

                return internal_find_or_default(std::forward<KeyLike>(key))->second;
            }

            //I feel like these are noexcept(auto) features - and the APIs should explicitly declare the noexcept criterias that are not encapsulated by the container - we aren't worrying about that now

            constexpr void clear() noexcept{

                std::fill(bucket_vec.begin(), bucket_vec.end(), NULL_VIRTUAL_ADDR);
                node_vec.resize(1u);
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

            template <class EraseArg>
            constexpr auto erase(EraseArg&& erase_arg) noexcept{

                if constexpr(std::is_convertible_v<EraseArg&&, const_iterator>){
                    static_assert(std::is_nothrow_convertible_v<EraseArg&&, const_iterator>);
                    return internal_erase_it(std::forward<EraseArg>(erase_arg));
                } else{
                    // static_assert(noexcept(internal_erase(std::forward<EraseArg>(erase_arg))));
                    return internal_erase(std::forward<EraseArg>(erase_arg));
                }
            }

            constexpr auto erase(const_iterator it, erase_hint hint) noexcept{

                return internal_erase_it_w_hint(it, hint);    
            }

            template <class KeyLike>
            constexpr auto contains(const KeyLike& key) const noexcept(noexcept(bucket_find(std::declval<const KeyLike&>()))) -> bool{

                return *bucket_find(key) != NULL_VIRTUAL_ADDR;
            }

            template <class KeyLike>
            constexpr auto count(const KeyLike& key) const noexcept(noexcept(bucket_find(std::declval<const KeyLike&>()))) -> size_type{

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
            constexpr auto erase_find(const KeyLike& key) noexcept(noexcept(bucket_find(std::declval<const KeyLike&>()))) -> std::pair<iterator, erase_hint>{

                bucket_iterator bucket  = bucket_find(key);
                size_type virtual_addr  = *bucket;

                if (virtual_addr == NULL_VIRTUAL_ADDR){
                    return std::make_pair(node_vec.end(), bucket);
                }

                return std::make_pair(std::next(node_vec.begin(), virtual_addr), bucket);
            }

            template <class KeyLike>
            constexpr auto erase_find(const KeyLike& key) const noexcept(noexcept(bucket_find(std::declval<const KeyLike&>()))) -> std::pair<const_iterator, erase_hint>{

                bucket_const_iterator bucket    = bucket_find(key);
                size_type virtual_addr          = *bucket;

                if (virtual_addr == NULL_VIRTUAL_ADDR){
                    return std::make_pair(node_vec.end(), bucket);
                }

                return std::make_pair(std::next(node_vec.begin(), virtual_addr), bucket);
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

            constexpr auto capacity() const noexcept -> size_type{

                return bucket_vec.size() - LAST_MOHICAN_SZ;
            }

            constexpr auto size() const noexcept -> size_type{

                return node_vec.size() - 1u;
            }

            constexpr auto insert_size() const noexcept -> size_type{

                return size() + erase_count;
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

            constexpr auto cbegin() const noexcept -> const_iterator{

                return std::next(node_vec.cbegin());
            }

            constexpr auto end() noexcept -> iterator{

                return node_vec.end();
            }

            constexpr auto end() const noexcept -> const_iterator{

                return node_vec.end();
            }

            constexpr auto cend() const noexcept -> const_iterator{

                return node_vec.cend();
            }

        private:

            constexpr void check_for_rehash(){

                if (size() < estimate_size(capacity()) && insert_size() < estimate_insert_capacity(capacity())) [[likely]]{ //might be buggy - not this line but other inserts that do not lead to this line
                    return;
                } else [[unlikely]]{
                   //either cap > size or insert_cap > max_insert_cap or both - if both - extend
                    if (size() >= estimate_size(capacity())){
                        size_type new_cap = capacity() * 2;
                        rehash(new_cap, true);
                    } else{
                        size_type new_cap = estimate_capacity(size());
                        rehash(new_cap, true);
                    }
                }
            }

            constexpr void force_uphash(){

                size_type new_cap = capacity() * 2;
                rehash(new_cap, true);
            }

            constexpr auto to_bucket_index(auto hashed_value) const noexcept -> size_type{

                static_assert(std::is_unsigned_v<decltype(hashed_value)>);
                return hashed_value & (bucket_vec.size() - (LAST_MOHICAN_SZ + 1u));
            }

            template <class KeyLike>
            constexpr auto bucket_exist_find(const KeyLike& key) noexcept(noexcept(std::declval<const Hasher&>()(std::declval<const KeyLike&>()))
                                                                          && noexcept(std::declval<const Pred&>()(std::declval<const Key&>(), std::declval<const KeyLike&>()))) -> bucket_iterator{

                //GCC sets branch prediction 1(unlikely)/10(likely) 
                //assume load_factor of 50% - avg - and reasonable hash function
                //50% ^ 3 = 1/8 - which is == branch predictor

                auto it = std::next(bucket_vec.begin(), to_bucket_index(hash_function()(key))); //we don't care where the bucket is pointing to - as long as it is a fixed random position and it does not pass the last of the last mohicans

                if (key_eq()(static_cast<const Key&>(node_vec[*it].first), key)){ //this optimization might not be as important in CPU arch but very important in GPU arch - where you want to minimize branch prediction by using block_quicksort approach
                    return it;
                }

                std::advance(it, 1u);

                if (key_eq()(static_cast<const Key&>(node_vec[*it].first), key)){
                    return it;
                }

                std::advance(it, 1u);

                while (true){
                    if (key_eq()(static_cast<const Key&>(node_vec[*it].first), key)) [[likely]]{
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

                auto it = std::next(bucket_vec.begin(), to_bucket_index(hash_function()(key)));

                if (key_eq()(node_vec[*it].first, key)){
                    return it;
                }

                std::advance(it, 1u);

                if (key_eq()(node_vec[*it].first, key)){
                    return it;
                }

                std::advance(it, 1u);

                while (true){
                    if (key_eq()(node_vec[*it].first, key)) [[likely]]{
                        return it;
                    }

                    std::advance(it, 1u);
                }
            }

            template <class KeyLike>
            constexpr auto bucket_find(const KeyLike& key) noexcept(noexcept(std::declval<const Hasher&>()(std::declval<const KeyLike&>()))
                                                                    && noexcept(std::declval<const Pred&>()(std::declval<const Key&>(), std::declval<const KeyLike&>()))) -> bucket_iterator{

                auto it = std::next(bucket_vec.begin(), to_bucket_index(hash_function()(key)));

                while (true){
                    if (*it == NULL_VIRTUAL_ADDR || key_eq()(static_cast<const Key&>(node_vec[*it].first), key)){
                        return it;
                    }

                    std::advance(it, 1u);
                }
            }

            template <class KeyLike>
            constexpr auto bucket_find(const KeyLike& key) const noexcept(noexcept(std::declval<const Hasher&>()(std::declval<const KeyLike&>()))
                                                                          && noexcept(std::declval<const Pred&>()(std::declval<const Key&>(), std::declval<const KeyLike&>()))) -> bucket_const_iterator{

                auto it = std::next(bucket_vec.begin(), to_bucket_index(hash_function()(key)));

                while (true){
                    if (*it == NULL_VIRTUAL_ADDR || key_eq()(node_vec[*it].first, key)){
                        return it;
                    }

                    std::advance(it, 1u);
                }
            }

            template <class KeyLike>
            constexpr auto bucket_ifind(const KeyLike& key) noexcept(noexcept(std::declval<const Hasher&>()(std::declval<const KeyLike&>()))) -> bucket_iterator{

                auto it = std::next(bucket_vec.begin(), to_bucket_index(hash_function()(key)));

                while (true){
                    if (*it == NULL_VIRTUAL_ADDR){
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

            template <class KeyLike, class ...Args>
            constexpr auto internal_insert_w_key(KeyLike&& key, Args&& ...args) -> std::pair<iterator, bool>{

                bucket_iterator it = bucket_find(key);

                if (*it != NULL_VIRTUAL_ADDR){
                    return std::make_pair(std::next(node_vec.begin(), *it), false);
                }

                auto value = value_type(std::piecewise_construct, std::forward_as_tuple(std::forward<KeyLike>(key)), std::forward_as_tuple(std::forward<Args>(args)...));

                if (insert_size() % REHASH_CHK_MODULO != 0u && it != std::prev(bucket_vec.end())) [[likely]]{
                    return std::make_pair(do_insert_at(it, std::move(value)), true);
                } else [[unlikely]]{
                    return std::make_pair(internal_noexist_insert(std::move(value)), true);
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

            constexpr void internal_exist_bucket_erase(bucket_iterator erasing_bucket_it) noexcept{

                bucket_iterator swapee_bucket_it        = bucket_exist_find(node_vec.back().first);
                size_type erasing_bucket_virtual_addr   = std::exchange(*erasing_bucket_it, ORPHANED_VIRTUAL_ADDR);

                if (swapee_bucket_it != erasing_bucket_it){
                    std::iter_swap(std::next(node_vec.begin(), erasing_bucket_virtual_addr), std::prev(node_vec.end()));
                    *swapee_bucket_it = erasing_bucket_virtual_addr;
                }

                node_vec.pop_back();
                erase_count += 1;
            }

            constexpr void internal_exist_bucket_erase(bucket_const_iterator erasing_bucket_const_it) noexcept{

                bucket_iterator erasing_bucket_it = std::next(bucket_vec.begin(), std::distance(bucket_vec.cbegin(), erasing_bucket_const_it)); //I dont think compiler is allowed to optimize this - idk
                internal_exist_bucket_erase(erasing_bucket_it);
            }

            template <class KeyLike>
            constexpr auto internal_erase(const KeyLike& key) noexcept(noexcept(bucket_find(std::declval<const KeyLike&>()))) -> size_type{

                bucket_iterator erasing_bucket_it       = bucket_find(key);
                size_type erasing_bucket_virtual_addr   = *erasing_bucket_it;

                if (erasing_bucket_virtual_addr != NULL_VIRTUAL_ADDR){
                    internal_exist_bucket_erase(erasing_bucket_it);
                    return 1u;
                }

                return 0u;
            }

            template <class KeyLike>
            constexpr void internal_exist_erase(const KeyLike& key) noexcept(noexcept(bucket_exist_find(std::declval<const KeyLike&>()))){
                
                return internal_exist_bucket_erase(bucket_exist_find(key));
            }

            constexpr auto internal_erase_it(const_iterator it) noexcept -> iterator{

                if (it == node_vec.end()){
                    return node_vec.end();
                }

                auto dif = std::distance(node_vec.cbegin(), it);
                internal_exist_erase(it->first);

                if (dif != node_vec.size()) [[likely]]{
                    return std::next(node_vec.begin(), dif);
                } else [[unlikely]]{
                    return std::next(node_vec.begin());
                }
            }

            constexpr auto internal_erase_it_w_hint(const_iterator it, bucket_const_iterator hint) noexcept -> iterator{

                if (it == node_vec.end()){
                    return node_vec.end();
                }

                auto dif = std::distance(node_vec.cbegin(), it);
                internal_exist_bucket_erase(hint);

                if (dif != node_vec.size()) [[likely]]{
                    return std::next(node_vec.begin(), dif);
                } else [[unlikely]]{
                    return std::next(node_vec.begin());
                }
            }
    };
}

#endif