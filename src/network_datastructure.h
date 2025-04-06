#ifndef __DG_NETWORK_DATASTRUCTURE_H__
#define __DG_NETWORK_DATASTRUCTURE_H__

#include <stdint.h>
#include <stdlib.h>
#include "network_exception.h"
#include <ratio>
#include <memory>

// #include "network_log.h"

namespace dg::network_datastructure::cyclic_queue{

    //this only works if the default state of T does not hold extra semantic meaning rather than a truely empty, meaningless representation
    //std::vector<> actually uses inplace construction to make sure that is not the case
    //this is precisely the reason std::vector<> is not qualified for raw pointer arithmetic operation
    //because it is not constructed by using new[] contiguous memory operation
    //we are not doing that yet

    template <class = void>
    static inline constexpr bool FALSE_VAL = false;
 
    class pow2_cyclic_queue_index_getter_device{

        private:

            size_t off;
            size_t cap;
        
        public:

            constexpr pow2_cyclic_queue_index_getter_device() = default;
            constexpr pow2_cyclic_queue_index_getter_device(size_t off, size_t cap) noexcept: off(off), cap(cap){}

            constexpr auto operator()(size_t virtual_idx) const noexcept -> size_t{

                return (this->off + virtual_idx) & (this->cap - 1u);
            }
    };

    template <class BaseIterator>
    class pow2_cyclic_queue_iterator{

        private:

            BaseIterator iter_head;
            intmax_t virtual_idx;
            pow2_cyclic_queue_index_getter_device index_getter;
        
        public:

            using self              = pow2_cyclic_queue_iterator; 
            using difference_type   = std::ptrdiff_t;
            using value_type        = typename BaseIterator::value_type; 

            template <class T = BaseIterator, std::enable_if_t<std::is_nothrow_default_constructible_v<T>, bool> = true>
            constexpr pow2_cyclic_queue_iterator(): iter_head(), 
                                                    virtual_idx(), 
                                                    index_getter(){}

            constexpr pow2_cyclic_queue_iterator(BaseIterator iter_head,
                                                 intmax_t virtual_idx,
                                                 pow2_cyclic_queue_index_getter_device index_getter) noexcept(std::is_nothrow_move_constructible_v<BaseIterator>): iter_head(std::move(iter_head)),
                                                                                                                                                                   virtual_idx(virtual_idx),
                                                                                                                                                                   index_getter(index_getter){} 

            constexpr auto operator ++() noexcept -> self&{

                this->virtual_idx += 1;
                return *this;
            }

            constexpr auto operator ++(int) noexcept -> self{

                static_assert(std::is_nothrow_copy_constructible_v<self>);

                self rs = *this;
                this->virtual_idx += 1;
                return rs;
            }

            constexpr auto operator --() noexcept -> self&{

                this->virtual_idx -= 1;
                return *this;
            }

            constexpr auto operator --(int) noexcept -> self{

                static_assert(std::is_nothrow_copy_constructible_v<self>);

                self rs = *this;
                this->virtual_idx -= 1;
                return rs;
            }

            constexpr auto operator +(difference_type off) const noexcept -> self{

                return self(this->iter_head, this->virtual_idx + off, this->index_getter);
            }

            constexpr auto operator +=(difference_type off) noexcept -> self&{

                this->virtual_idx += off;
                return *this;
            }

            constexpr auto operator -(difference_type off) const noexcept -> self{

                return self(this->iter_head, this->virtual_idx - off, this->index_getter);
            }

            constexpr auto operator -(const self& other) const noexcept -> difference_type{

                return this->virtual_idx - other.virtual_idx;
            }
        
            constexpr auto operator -=(difference_type off) noexcept -> self&{

                this->virtual_idx -= off;
                return *this;
            }

            constexpr auto operator ==(const self& other) const noexcept -> bool{

                return (this->iter_head == other.iter_head) && (this->virtual_idx == other.virtual_idx);
            }

            constexpr auto operator !=(const self& other) const noexcept -> bool{

                return (this->iter_head != other.iter_head) || (this->virtual_idx != other.virtual_idx);
            }

            constexpr auto operator *() const noexcept -> decltype(auto){

                size_t actual_idx = this->index_getter(this->virtual_idx);
                return std::next(this->iter_head, actual_idx).operator *();
            }

            constexpr auto operator ->() const noexcept -> decltype(auto){

                size_t actual_idx = this->index_getter(this->virtual_idx);
                return std::next(this->iter_head, actual_idx).operator ->();
            }
    };

    template <class T, class Allocator = std::allocator<T>>
    class pow2_cyclic_queue{

        private:

            std::vector<T, Allocator> data_arr;
            size_t off;
            size_t sz;
            size_t cap;

            using self              = pow2_cyclic_queue;

        public:

            using value_type        = T;
            using iterator          = pow2_cyclic_queue_iterator<typename std::vector<T, Allocator>::iterator>;
            using const_iterator    = pow2_cyclic_queue_iterator<typename std::vector<T, Allocator>::const_iterator>;
            using size_type         = std::size_t;

            static inline constexpr size_t DEFAULT_POW2_EXPONENT = 10u; 

            pow2_cyclic_queue(): pow2_cyclic_queue(DEFAULT_POW2_EXPONENT){}

            pow2_cyclic_queue(size_t pow2_exponent): data_arr(size_t{1} << pow2_exponent),
                                                     off(0u),
                                                     sz(0u),
                                                     cap(size_t{1} << pow2_exponent){}

            constexpr auto front() const noexcept -> const T&{

                if constexpr(DEBUG_MODE_FLAG){
                    if (this->sz == 0u){
                        // dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                    }
                }

                size_t ptr = this->to_index(0u);
                return this->data_arr[ptr];
            }

            constexpr auto front() noexcept -> T&{

                if constexpr(DEBUG_MODE_FLAG){
                    if (this->sz == 0u){
                        // dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                    }
                }

                size_t ptr = this->to_index(0u);
                return this->data_arr[ptr];
            }

            constexpr auto back() const noexcept -> const T&{

                if constexpr(DEBUG_MODE_FLAG){
                    if (this->sz == 0u){
                        // dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                    }
                }

                size_t ptr = this->to_index(this->sz - 1u);
                return this->data_arr[ptr];
            }

            constexpr auto back() noexcept -> T&{

                if constexpr(DEBUG_MODE_FLAG){
                    if (this->sz == 0u){
                        // dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                    }
                }

                size_t ptr = this->to_index(this->sz - 1u);
                return this->data_arr[ptr];
            }

            constexpr auto empty() const noexcept -> bool{

                return this->sz == 0u;                
            }

            constexpr auto begin() const noexcept -> const_iterator{

                return const_iterator(this->data_arr.begin(), 0u, this->get_index_getter_device());
            }

            constexpr auto end() const noexcept -> const_iterator{

                return const_iterator(this->data_arr.begin(), this->sz, this->get_index_getter_device());
            }

            constexpr auto begin() noexcept -> iterator{
                
                return iterator(this->data_arr.begin(), 0u, this->get_index_getter_device());
            }

            constexpr auto end() noexcept -> iterator{

                return iterator(this->data_arr.begin(), this->sz, this->get_index_getter_device());
            }

            constexpr auto size() const noexcept -> size_t{

                return this->sz;
            }

            constexpr auto capacity() const noexcept -> size_t{

                return this->cap;
            }

            constexpr auto operator[](size_t idx) const noexcept -> const T&{

                return this->data_arr[this->to_index(idx)];
            }

            constexpr auto operator[](size_t idx) noexcept -> T&{

                return this->data_arr[this->to_index(idx)];
            }

            constexpr auto at(size_t idx) const noexcept -> const T&{

                return (*this)[idx];
            }

            constexpr auto at(size_t idx) noexcept -> T&{

                return (*this)[idx];
            }

            constexpr auto resize(size_t new_sz) noexcept -> exception_t{

                if (new_sz > this->cap){
                    return dg::network_exception::RESOURCE_EXHAUSTION;
                }

                if (new_sz >= this->sz){
                    this->sz = new_sz;
                    return dg::network_exception::SUCCESS;
                }

                this->defaultize_range(new_sz, this->sz - new_sz);
                this->sz = new_sz;

                return dg::network_exception::SUCCESS;
            }

            template <class ValueLike>
            constexpr auto push_back(ValueLike&& value) noexcept -> exception_t{

                if (this->sz == this->cap){
                    return dg::network_exception::QUEUE_FULL;
                }

                size_t ptr = this->to_index(this->sz);

                if constexpr(std::is_nothrow_assignable_v<T&, ValueLike&&>){
                    this->data_arr[ptr] = std::forward<ValueLike>(value);
                    this->sz            += 1u;
                    return dg::network_exception::SUCCESS; 
                } else{
                    try{
                        this->data_arr[ptr] = std::forward<ValueLike>(value);
                        this->sz            += 1u;
                        return dg::network_exception::SUCCESS;
                    } catch (...){
                        return dg::network_exception::wrap_std_exception(std::current_exception());
                    }
                }
            }

            constexpr void pop_front() noexcept{

                static_assert(std::is_nothrow_default_constructible_v<T> && std::is_nothrow_assignable_v<T&, T&&>);

                if constexpr(DEBUG_MODE_FLAG){
                    if (this->sz == 0u){
                        // dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                    }
                }

                size_t ptr          = this->to_index(0u);
                this->data_arr[ptr] = T{};
                this->off           += 1u;
                this->sz            -= 1u;
            }
 
            constexpr void pop_back() noexcept{

                static_assert(std::is_nothrow_default_constructible_v<T> && std::is_nothrow_assignable_v<T&, T&&>);

                if constexpr(DEBUG_MODE_FLAG){
                    if (this->sz == 0u){
                        // dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                    }
                }

                size_t ptr          = this->to_index(this->sz - 1u);
                this->data_arr[ptr] = T{};
                this->sz            -= 1u;
            } 

            constexpr void erase_front_range(size_t sz) noexcept{
                
                for (size_t i = 0u; i < sz; ++i){
                    pop_front();
                }
            }

            constexpr void erase_back_range(size_t sz) noexcept{

                for (size_t i = 0u; i < sz; ++i){
                    pop_back();
                }
            }

            constexpr auto operator ==(const self& other) const noexcept -> bool{

                return std::equal(this->begin(), this->end(), other.begin(), other.end());
            }

        private:

            constexpr auto to_index(size_t virtual_offset) const noexcept -> size_t{

                return (this->off + virtual_offset) & (this->cap - 1u);
            }

            constexpr auto get_index_getter_device() const noexcept -> pow2_cyclic_queue_index_getter_device{

                return pow2_cyclic_queue_index_getter_device(this->off, this->cap);
            }

            constexpr void defaultize_range(size_t virtual_idx, size_t sz) noexcept{

                static_assert(std::is_nothrow_default_constructible_v<T>);

                auto fill_task = [&, this]<class T1>(T1) noexcept{
                    size_t first                        = this->to_index(virtual_idx);
                    size_t last                         = first + sz;
                    size_t eoq_last                     = std::min(this->cap, last); 
                    const size_t wrapped_around_first   = 0u;
                    size_t wrapped_around_last          = last - eoq_last;

                    std::fill(std::next(this->data_arr.begin(), first), std::next(this->data_arr.begin(), eoq_last), T{});
                    std::fill(std::next(this->data_arr.begin(), wrapped_around_first), std::next(this->data_arr.begin(), wrapped_around_last), T{});
                };

                if constexpr(std::is_trivial_v<T>){
                    fill_task(int{});
                } else{
                    if constexpr(std::is_assignable_v<T&, const T&>){
                        if constexpr(std::is_nothrow_assignable_v<T&, const T&>){
                            fill_task(int{});
                        } else{
                            if constexpr(std::is_assignable_v<T&, T&&>){
                                if constexpr(std::is_nothrow_assignable_v<T&, T&&>){
                                    for (size_t i = 0u; i < sz; ++i){
                                        this->operator[](virtual_idx + i) = T{};
                                    }
                                } else{
                                    static_assert(FALSE_VAL<>);
                                }
                            } else{
                                static_assert(FALSE_VAL<>);
                            }
                        }
                    } else if constexpr(std::is_assignable_v<T&, T&&>){
                        if constexpr(std::is_nothrow_assignable_v<T&, T&&>){
                            for (size_t i = 0u; i < sz; ++i){
                                this->operator[](virtual_idx + i) = T{};
                            }      
                        } else{
                            static_assert(FALSE_VAL<>);
                        }                  
                    } else{
                        static_assert(FALSE_VAL<>);
                    }
                }

            }
    };
}

namespace dg::network_datastructure::fast_node_hash_map{

    //this is complex, first is the trivially_constructible of std::pair, second is constness of things
    //this is not std-compatible, yet it answers pretty much ALL the performance questions that we've been longing for without bending the rules
    //we'll be specific for now

    template <class T, class = void>
    struct null_addr{};

    template <class T>
    struct null_addr<T, std::void_t<std::enable_if_t<std::is_unsigned_v<T>>>>{
        static inline constexpr T value = std::numeric_limits<T>::max();
    };

    template <class T>
    static inline constexpr T null_addr_v = null_addr<T>::value; 

    template <class T, class = void>
    struct get_virtual_addr{};

    template <class T>
    struct get_virtual_addr<T, std::void_t<std::enable_if_t<std::is_unsigned_v<T>>>>{
        using type = T;
    };

    template <class T>
    using get_virtual_addr_t = typename get_virtual_addr<T>::type;

    //this is a fake Node
    //padding is important, whether this is a full word load or half word load
    //we are to not worry about that now
    //this is complicated

    //std::vector<> trivial copy only works if we are using raw std::vector<>::begin() + std::vector<>::end() and the pointing data type is trivial, period
    //we can't hack this further, this is for specialized applications, not for std-usage of things, refer to dg_map_variants to get std compatible map
    //std::memset is very fast if std::fill() is to see the filling pattern of vector

    template <class key_t, class value_t, class virtual_addr_t>
    struct Node{
        key_t first;
        value_t second;
        virtual_addr_t nxt_addr;
    };

    template <class Key, class Mapped, class SizeType = std::size_t, class VirtualAddrType = std::uint32_t, class Hasher = std::hash<Key>, class Pred = std::equal_to<Key>, class Allocator = std::allocator<Node<Key, Mapped, VirtualAddrType>>, class LoadFactor = std::ratio<3, 4>>
    class unordered_node_map{

        private:

            std::vector<Node, typename std::allocator_traits<Allocator>::template rebind_alloc<Node>> virtual_storage_vec;
            std::vector<VirtualAddrType, typename std::allocator_traits<Allocator>::template rebind_alloc<VirtualAddrType>> bucket_vec;
            Hasher _hasher;
            Pred pred;
            Allocator allocator;

        public:

            using key_type                  = Key;
            using mapped_type               = Mapped;
            using value_type                = Node;
            using hasher                    = Hasher;
            using key_equal                 = Pred;
            using allocator_type            = Allocator;
            using reference                 = value_type&;
            using const_reference           = const value_type&;
            using pointer                   = typename std::allocator_traits<Allocator>::pointer;
            using const_pointer             = typename std::allocator_traits<Allocator>::const_pointer;
            using iterator                  = typename std::vector<Node, typename std::allocator_traits<Allocator>::typename rebind_alloc<Node>>::iterator;
            using const_iterator            = typename std::vector<Node, typename std::allocator_traits<Allocator>::typename rebind_alloc<Node>>::const_iterator;
            using size_type                 = SizeType;
            using difference_type           = std::intmax_t;
            using self                      = unordered_node_map;
            using load_factor_ratio         = typename LoadFactor::type;
            using virtual_addr_t            = get_virtual_addr_t<VirtualAddrType>;
            using node_t                    = Node<key_t, value_t, virtual_addr_t>;

            static inline constexpr virtual_addr_t NULL_VIRTUAL_ADDR = null_addr_v<virtual_addr_t>;

            constexpr explicit unordered_node_map(size_type bucket_count,
                                                  const Hasher& _hasher = Hasher(),
                                                  const Pred& pred = Pred(),
                                                  const Allocator& allocator = Allocator()): virtual_storage_vec(allocator),
                                                                                             bucket_vec(std::max(self::min_capacity(), ceil2(bucket_count)), NULL_VIRTUAL_ADDR, allocator),
                                                                                             pred(pred),
                                                                                             allocator(allocator){

                this->virtual_storage_vec.reserve(estimate_size(capacity()));
            }

            constexpr unordered_node_map(size_type bucket_count,
                                         const Hasher& _hasher,
                                         const Allocator& allocator): unordered_node_map(bucket_count, _hasher, Pred(), allocator){}

            constexpr unordered_node_map(size_type bucket_count,
                                         const Allocator& allocator): unordered_node_map(bucket_count, Hasher(), allocator){}

            constexpr explicit unordered_node_map(const Allocator& allocator): unordered_node_map(self::min_capacity(), allocator){}

            constexpr unordered_node_map(): unordered_node_map(Allocator()){}

            template <class InputIt>
            constexpr unordered_node_map(InputIt first,
                                         InputIt last,
                                         size_type bucket_count,
                                         const Hasher& _hasher = Hasher(),
                                         const Pred& pred = Pred(),
                                         const Allocator& allocator = Allocator()): unordered_node_map(bucket_count, _hasher, pred, allocator){

                this->insert(first, last); //bad, leak
            }

            template <class InputIt>
            constexpr unordered_node_map(InputIt first,
                                         InputIt last,
                                         size_type bucket_count,
                                         const Allocator& allocator): unordered_node_map(first, last, bucket_count, Hasher(), Pred(), allocator){}

            template <class ValueLike = std::pair<const Key, Value>>
            constexpr unordered_node_map(std::initializer_list<value_type> init_list,
                                         size_type bucket_count,
                                         const Hasher& _hasher,
                                         const Allocator& allocator): unordered_node_map(init_list.begin(), init_list.end(), bucket_count, _hasher, Pred(), allocator){}

            template <class ValueLike = std::pair<const Key, Value>>
            constexpr unordered_node_map(std::initializer_list<value_type> init_list,
                                         size_type bucket_count,
                                         const Allocator& allocator): unordered_node_map(init_list.begin(), init_list.end(), bucket_count, Hasher(), allocator){}

            constexpr void rehash(size_type tentative_new_cap, bool force_rehash = false){

                if (!force_rehash && tentative_new_cap <= this->capacity()){
                    return;
                }

                size_t new_cap = ceil2(tentative_new_cap);

                if (new_cap >= this->capacity()){
                    if constexpr(std::is_nothrow_move_constructible_v<Node> && std::is_nothrow_move_assignable_v<Node>){
                        self proxy(std::make_move_iterator(this->virtual_storage_vec.begin()), std::make_move_iterator(this->virtual_storage_vec.end()), new_cap, this->_hasher, this->pred, this->allocator); //wrong
                        *this = std::move(proxy);
                     else{
                        //this is incredibly hard to write without leaks, and only to use move assignment, it's hard
                        self proxy(this->virtual_storage_vec.begin(), this->virtual_storage_vec.end(), new_cap, this->_hasher, this->pred, this->allocator);
                        *this = std::move(proxy);
                     }
                } else{

                }

                //we'll definitely have leak if we are to be very greedy here
            }

            constexpr void reserve(size_type new_sz){

                if (new_sz <= this->size()){
                    return;
                }

                this->rehash(estimate_capacity(new_sz));
            }

            template <class KeyLike, class ...Args>
            constexpr auto try_emplace(KeyLike&& key, Args&& ...args) -> std::pair<iterator, bool>{

                return this->internal_insert(node_t{key_t(std::forward<KeyLike>(key)), value_t(std::forward<Args>(args)...), virtual_addr_t{}});
            }

            template <class ...Args>
            constexpr auto emplace(Args&& ...args) -> std::pair<iterator, bool>{

                return this->try_emplace(std::forward<Args>(args)...);
            }

            template <class ValueLike = value_type>
            constexpr auto insert(ValueLike&& value) -> std::pair<iterator, bool>{

                return this->internal_insert(node_t{key_t(std::forward_like<ValueLike>(value.first)), value_t(std::forward_like<ValueLike>(value.second)), virtual_addr_t{}});
            }

            template <class Iterator>
            constexpr void insert(Iterator first, Iterator last){

                while (first != last){
                    this->insert(*first);
                    std::advance(first, 1u);
                }
            }

            template <class ValueLike = std::pair<const Key, Value>>
            constexpr void insert(std::initializer_list<ValueLike> init_list){

                //bad leak
                this->insert(init_list.begin(), init_list.end());
            }

            template <class KeyLike, class MappedLike>
            constexpr auto insert_or_assign(KeyLike&& key, MappedLike&& mapped) -> std::pair<iterator, bool>{

                return this->internal_insert_or_assign(node_t{key_t(std::forward<KeyLike>(key)), mapped_type(std::forward<MappedLike>(mapped)), virtual_addr_t{}});
            }

            template <class KeyLike>
            constexpr auto operator[](KeyLike&& key) -> mapped_type&{

                return *std::get<0>(this->insert_or_assign(std::forward<KeyLike>(key), mapped_type{}));
            }

            constexpr void clear() noexcept(true){ //this has to be noexcept

                static_assert(noexcept(this->virtual_storage_vec.clear()));

                this->virtual_storage_vec.clear();
                std::fill(this->bucket_vec.begin(), this->bucket_vec.end(), self::NULL_VIRTUAL_ADDR);
            }

            constexpr void swap(self& other) noexcept(true){

                std::swap(this->virtual_storage_vec, other.virtual_storage_vec);
                std::swap(this->bucket_vec, other.bucket_vec);
                std::swap(this->_hasher, other._hasher);
                std::swap(this->pred, other.pred);
                std::swap(this->allocator, other.allocator);
            }

            template <class EraseArg>
            constexpr auto erase(EraseArg&& erase_arg) noexcept(true) -> iterator{ //const noexcept

                if constexpr(std::is_convertible_v<EraseArg&&, const_iterator>){
                    return this->internal_erase_iter(std::forward<EraseArg>(erase_arg));
                } else{
                    return this->internal_erase_key(std::forward<EraseArg>(erase_arg));
                }
            }

            template <class KeyLike>
            constexpr auto contains(const KeyLike& key) const noexcept(true) -> bool{ 

                return this->find(key) != this->end();
            }

            template <class KeyLike>
            constexpr auto count(const KeyLike& key) const noexcept(true) -> size_t{

                return static_cast<size_t>(this->contains(key));
            }

            template <class KeyLike>
            constexpr auto find(const KeyLike& key) const noexcept(true) -> const_iterator{

                return this->internal_find(key);
            }

            template <class KeyLike>
            constexpr auto find(const KeyLike& key) noexcept(true) -> iterator{

                return std::next(this->virtual_storage_vec.begin(), std::distance(this->virtual_storage_vec.cbegin(), this->internal_find(key)));
            }

            template <class KeyLike>
            constexpr auto at(const KeyLike& key) const noexcept(true) -> const_reference{

                return *this->likely_find(key);
            }

            template <class KeyLike>
            constexpr auto at(const KeyLike& key) noexcept(true) -> reference{

                return *this->likely_find(key);
            }

            constexpr auto empty() const noexcept -> bool{

                return this->virtual_storage_vec.empty();
            }

            constexpr auto capacity() const noexcept -> size_type{

                return this->bucket_vec.size();
            }

            constexpr auto size() const noexcept -> size_type{

                return this->virtual_storage_vec.size();
            }

            constexpr auto hash_function() const & noexcept -> const Hasher&{

                return this->_hasher;
            }

            constexpr auto hash_function() && noexcept -> Hasher&&{

                return static_cast<Hasher&&>(this->_hasher);
            }

            constexpr auto key_eq() const & noexcept -> const Pred&{

                return this->pred;
            }

            constexpr auto key_eq() && noexcept -> Pred&&{

                return static_cast<Pred&&>(this->pred);
            }
 
            constexpr auto load_factor() const noexcept -> float{

                return float{load_factor_ratio::num} / load_factor_ratio::den;
            }

            constexpr auto begin() noexcept -> iterator{

                return this->virtual_storage_vec.begin();   
            }

            constexpr auto begin() const noexcept -> const_iterator{

                return this->virtual_storage_vec.begin();
            }

            constexpr auto cbegin() const noexcept -> const_iterator{

                return this->virtual_storage_vec.cbegin();
            }

            constexpr auto end() noexcept -> iterator{

                return this->virtual_storage_vec.end();
            }

            constexpr auto end() const noexcept -> const_iterator{

                return this->virtual_storage_vec.end();
            }

            constexpr auto cend() const noexcept -> const_iterator{

                return this->virtual_storage_vec.cend();
            }
        
        private:

            template <class ValueLike>
            constexpr auto internal_insert(ValueLike&& value) -> std::pair<iterator, bool>{

                if (this->virtual_storage_vec.size() == this->virtual_storage_vec.capacity()){
                    this->rehash(this->virtual_storage_vec.capacity() * 2);
                }

                size_t hashed_value                 = this->_hasher(value.first);
                size_t bucket_idx                   = this->to_bucket_index(hashed_value);
                virtual_addr_t node_virtual_addr    = this->bucket_vec[bucket_idx]; 

                while (true){
                    if (node_virtual_addr == NULL_VIRTUAL_ADDR){
                        value.nxt_addr = NULL_VIRTUAL_ADDR;
                        this->virtual_storage_vec.emplace_back(std::forward<ValueLike>(value));
                        return std::make_pair(std::prev(this->virtual_storage_vec.end(), 1u), true);
                    }

                    if (this->pred(this->virtual_storage_vec[node_virtual_addr].first, value.first)){
                        return std::make_pair(std::next(this->virtual_storage_vec.begin(), node_virtual_addr), false);
                    }

                    node_virtual_addr = this->virtual_storage_vec[node_virtual_addr].nxt_addr;
                }
            }

            template <class ValueLike>
            constexpr auto internal_insert_or_assign(ValueLike&& value) -> std::pair<iterator, bool>{

                if (this->virtual_storage_vec.size() == this->virtual_storage_vec.capacity()){
                    this->rehash(this->virtual_storage_vec.capacity() * 2);
                }

                size_t hashed_value                 = this->_hasher(value.first);
                size_t bucket_idx                   = this->to_bucket_index(hashed_value);
                virtual_addr_t node_virtual_addr    = this->bucket_vec[bucket_idx]; 

                while (true){
                    if (node_virtual_addr == NULL_VIRTUAL_ADDR){
                        value.nxt_addr = NULL_VIRTUAL_ADDR;
                        this->virtual_storage_vec.emplace_back(std::forward<ValueLike>(value));
                        return std::make_pair(std::prev(this->virtual_storage_vec.end(), 1u), true);
                    }

                    if (this->pred(this->virtual_storage_vec[node_virtual_addr].first, value.first)){
                        value.nxt_addr = this->virtual_storage_vec[node_virtual_addr].nxt_addr;
                        this->virtual_storage_vec[node_virtual_addr] = std::forward<ValueLike>(value);

                        return std::make_pair(std::next(this->virtual_storage_vec.begin(), node_virtual_addr), false);
                    }

                    node_virtual_addr = this->virtual_storage_vec[node_virtual_addr].nxt_addr;
                }
            }

            template <class KeyLike>
            constexpr auto internal_find(const KeyLike& key) const noexcept(true) -> const_iterator{

                size_t hashed_value                 = this->_hasher(key);
                size_t bucket_idx                   = this->to_bucket_index(hashed_value);
                virtual_addr_t node_virtual_addr    = this->bucket_vec[bucket_idx]; 

                while (true){
                    if (node_virtual_addr == NULL_VIRTUAL_ADDR){
                        return this->end();
                    }

                    if (this->pred(this->virtual_storage_vec[node_virtual_addr].first, key)){
                        return std::next(this->virtual_storage_vec.begin(), node_virtual_addr);
                    }

                    node_virtual_addr = this->virtual_storage_vec[node_virtual_addr].nxt_addr;
                }
            }

            template <class KeyLike>
            constexpr auto internal_exist_find(const KeyLike& key) const noexcept(true) -> const_iterator{

                //alright let's back to the set problem of <finding_set_sz> / <total_set_sz>
                //assume that our load is 80%, we are to find the probability that it passes the first node and hit 
                //assume that the node is hit twice, such is the bucket chain has the length of 2
                //then the probability of that to happen is:

                //total_set_sz == n ** operation_sz
                //we are to permute the two <finding key>, take that multiply with (n - 1) ** (operation_sz - 2)
                //then we take the ratio, then we'll have our probability                
                //the formula is: (n - 1) ** (operation_sz - 2) * (operation_sz - 1) * operation_sz / n ** operation_sz, I think

                size_t hashed_value                 = this->_hasher(key);
                size_t bucket_idx                   = this->to_bucket_index(hashed_value);
                virtual_addr_t node_virtual_addr    = this->bucket_vec[bucket_idx]; 

                if (this->pred(this->virtual_storage_vec[node_virtual_addr].first, key)){
                    return std::next(this->virtual_storage_vec.begin(), node_virtual_addr);
                }

                node_virtual_addr = this->virtual_storage_vec[node_virtual_addr].nxt_addr;

                while (true){
                    if (this->pred(this->virtual_storage_vec[node_virtual_addr].first, key)) [[likely]]{
                        return std::next(this->virtual_storage_vec.begin(), node_virtual_addr);
                    }

                    node_virtual_addr = this->virtual_storage_vec[node_virtual_addr].nxt_addr;
                }
            }

            //this is one of the C++ myths, we rather copy paste than to do the remove const, it's undefined
            template <class KeyLike>
            constexpr auto internal_find_bucket_reference(const KeyLike& key) const noexcept(true) -> const virtual_addr_t *{

                size_t hashed_value             = this->_hasher(key);
                size_t bucket_idx               = this->to_bucket_index(hashed_value);
                const virtual_addr_t * current  = &this->bucket_vec[bucket_idx];

                while (true){
                    if (*current == NULL_VIRTUAL_ADDR){
                        return nullptr;
                    }

                    if (this->pred(this->virtual_storage_vec[*current].first, key)){
                        return current;
                    }

                    current = &this->virtual_storage_vec[*current].nxt_addr;
                }
            }

            template <class KeyLike>
            constexpr auto internal_find_bucket_reference(const KeyLike& key) noexcept(true) -> virtual_addr_t *{

                size_t hashed_value             = this->_hasher(key);
                size_t bucket_idx               = this->to_bucket_index(hashed_value);
                virtual_addr_t * current        = &this->bucket_vec[bucket_idx];

                while (true){
                    if (*current == NULL_VIRTUAL_ADDR){
                        return nullptr;
                    }

                    if (this->pred(this->virtual_storage_vec[*current].first, key)){
                        return current;
                    }

                    current = &this->virtual_storage_vec[*current].nxt_addr;
                }
            }

            template <class KeyLike>
            constexpr auto internal_exist_find_bucket_reference(const KeyLike& key) const noexcept(true) -> const virtual_addr_t *{

                size_t hashed_value                 = this->_hasher(key);
                size_t bucket_idx                   = this->to_bucket_index(hashed_value);
                const virtual_addr_t * current      = &this->bucket_vec[bucket_idx];

                if (this->pred(this->virtual_storage_vec[*current].first, key)){
                    return current;
                }

                current = &this->virtual_storage_vec[*current].nxt_addr;

                while (true){
                    if (this->pred(this->virtual_storage_vec[*current].first, key)) [[likely]]{
                        return current;
                    }

                    current = &this->virtual_storage_vec[*current].nxt_addr;
                }
            }

            template <class KeyLike>
            constexpr auto internal_exist_find_bucket_reference(const KeyLike& key) noexcept(true) -> virtual_addr_t *{

                size_t hashed_value         = this->_hasher(key);
                size_t bucket_idx           = this->to_bucket_index(hashed_value);
                virtual_addr_t * current    = &this->bucket_vec[bucket_idx];

                if (this->pred(this->virtual_storage_vec[*current].first, key)){
                    return current;
                }

                current = &this->virtual_storage_vec[*current].nxt_addr;

                while (true){
                    if (this->pred(this->virtual_storage_vec[*current].first, key)) [[likely]]{
                        return current;
                    }

                    current = &this->virtual_storage_vec[*current].nxt_addr;
                }
            }

            template <class KeyLike>
            constexpr void internal_erase(const KeyLike& key) noexcept(true){

                virtual_addr_t * key_reference = this->internal_find_bucket_reference(key);

                if (key_reference == nullptr){
                    return;
                } 

                virtual_addr_t * swapping_reference = this->internal_exist_find_bucket_reference(this->virtual_storage_vec.back().first);

                if (swapping_reference == key_reference) [[unlikely]]{
                    *key_reference = this->virtual_storage_vec[*key_reference].nxt_addr;
                    this->virtual_storage_vec.pop_back();

                    return;
                }

                //we are advanced people, we want to confuse new comer

                virtual_addr_t removing_addr    = std::exchange(*key_reference, this->virtual_storage_vec[*key_reference].nxt_addr); 

                // virtual_addr_t removing_addr = *key_reference;
                // *key_reference               = this->virtual_storage_vec[removing_addr].nxt_addr;
                //we are to swap the removing_addr with the last guy for disposal

                std::swap(this->virtual_storage_vec[removing_addr], this->virtual_storage_vec.back());
                this->virtual_storage_vec.pop_back();

                //swapping_reference is now pointing to the wrong guy, swapping_reference is only wrong when swapping_reference == key_reference, which we made sure is not the case
                *swapping_reference             = removing_addr;

                //ok, we are done
            }

            template <class KeyLike>
            constexpr auto internal_erase_key(KeyLike&& key) noexcept(true) -> iterator{

            }

            constexpr auto internal_erase_iter(const_iterator iter) noexcept(true) -> iterator{

            }
    };
}

namespace dg::network_datastructure::node_hash_set{

}

#endif