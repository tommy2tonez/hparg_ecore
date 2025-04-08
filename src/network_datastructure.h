#ifndef __DG_NETWORK_DATASTRUCTURE_H__
#define __DG_NETWORK_DATASTRUCTURE_H__

#include <stdint.h>
#include <stdlib.h>
#include "network_exception.h"
#include <ratio>
#include <memory>
#include <bit>
#include <type_traits>
#include <vector>
#include <utility>
#include <memory>
#include <stdexcept>
#include <limits>

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

namespace dg::network_datastructure::unordered_map_variants{

    //this is complex, first is the trivially_constructible of std::pair, second is constness of things
    //this is not std-compatible, yet it answers pretty much ALL the performance questions that we've been longing for without bending the rules
    //we'll be specific for now

    //alright, this is officially the fastest implementation of std::unordered_map if we are to use insert + clear only
    //memory-footprint-wise talking, this is most efficient, this is also the only thing that we care in a massive parallel system, the RAM BUS across cores
    //clear-wise talking, it is trivially clearable, by implementing a fake Node
    //copy-wise talking, it is memcpyable
    //iterator wise talking, it's optimized to a raw pointer optimization by the compiler (std::vector<>::iterator guarantees such), which is raw performance of copying
    //we only use this for our keyvalue feed, of size 512
    //this code is clear

    //this is me, myself and I + him and I

    template <class T, class U>
    struct DGForwardLikeHelper{
        using type = std::remove_reference_t<U>;
    };

    template <class T, class U>
    struct DGForwardLikeHelper<T&, U>{
        using type = std::add_lvalue_reference_t<std::remove_reference_t<U>>;
    };

    template <class T, class U>
    using dg_forward_like_t = typename DGForwardLikeHelper<T, U>::type; 

    template <class T, class U>
    constexpr auto dg_forward_like(U&& value) noexcept -> dg_forward_like_t<T, U>&&{

        return static_cast<dg_forward_like_t<T, U>&&>(value);
    }

    template <class T, std::enable_if_t<std::is_unsigned_v<T>, bool> = true>
    constexpr auto ulog2(T val) noexcept -> size_t{

        return static_cast<size_t>(sizeof(T) * CHAR_BIT - 1) - static_cast<size_t>(std::countl_zero(val));
    }

    template <class T, std::enable_if_t<std::is_unsigned_v<T>, bool> = true>
    static constexpr auto ceil2(T val) noexcept -> size_t{

        if (val <= 1u){ [[unlikely]]
            return 1u;
        }

        //alright people complained about this code

        size_t uplog_value = unordered_map_variants::ulog2(static_cast<T>(val - 1)) + 1; //the problem of unsigned and signed arithmetic arises when ... sizeof(signed) == sizeof(unsigned) and we are casting from signed to unsigned, -1 + 1 should guarantee the always in unsigned counterpart range, so we dont have issues  

        return T{1u} << uplog_value;
    }

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

    //to not confuse our foos
    template <class key_t, class mapped_t, class virtual_addr_t>
    struct Node{
        key_t first;
        mapped_t second;
        virtual_addr_t nxt_addr;
    };

    //this should be usable for now
    //I dont see problems
    //the nxt_addr is never there if the user is to not read to program, they can actually use this component like every normal other map with type-erased pair
    //we have special applications for these guys, not for std-std or adoptabilities
    //alright I had bad feedback about the Node being incompatible (senior developers)
    //we dont see problems, because this is application, not std
    //alright we are clear

    //if we are for the try-except route, we are susceptible to move leak
    //such is a thrown exception would invalidate the content of the moved argument
    //this is in the std way of doing things
    //there is a virtue for each different way of error-handlings, I'm pro explicit exception instead of try-catch, because try-catch would distinct the try block and the catch block, which is not very convenient in cases of handling leaks
    //we'll move on for now
    //this map looks like a scam but it is not a scam, it is a type-erased value_type unordered_map, the only interface to access the data is ->first + ->second with the only downside of first-immutability being not protected by compile-time measurements
    //with the increasing popularity of auto& + const auto& + auto&&, type-erased value_type is actually preferred in the 2025 new std
    //OK, this should pass my code review

    template <class Key, class Mapped, class SizeType = std::size_t, class VirtualAddrType = std::uint32_t, class Hasher = std::hash<Key>, class Pred = std::equal_to<Key>, class Allocator = std::allocator<Node<Key, Mapped, VirtualAddrType>>, class LoadFactor = std::ratio<7, 8>>
    class unordered_node_map{

        private:

            std::vector<Node<Key, Mapped, VirtualAddrType>, typename std::allocator_traits<Allocator>::template rebind_alloc<Node<Key, Mapped, VirtualAddrType>>> virtual_storage_vec;
            std::vector<VirtualAddrType, typename std::allocator_traits<Allocator>::template rebind_alloc<VirtualAddrType>> bucket_vec;
            Hasher _hasher;
            Pred pred;
            Allocator allocator;

        public:

            //we cant implement a hasher noexcept or predicate noexcept static_assert yet
            //I dont know what took std so long or they simply just dont want to implement the feature
            //it's hard, yet the practice is that lookups + erase + clear must be noexcept
            //the program is hardly useful otherwise (according to my friend)
            //

            using key_type                  = Key;
            using mapped_type               = Mapped;
            using value_type                = Node<Key, Mapped, VirtualAddrType>;
            using hasher                    = Hasher;
            using key_equal                 = Pred;
            using allocator_type            = Allocator;
            using reference                 = value_type&;
            using const_reference           = const value_type&;
            using pointer                   = typename std::allocator_traits<Allocator>::pointer;
            using const_pointer             = typename std::allocator_traits<Allocator>::const_pointer;
            using iterator                  = typename std::vector<Node<Key, Mapped, VirtualAddrType>, typename std::allocator_traits<Allocator>::template rebind_alloc<Node<Key, Mapped, VirtualAddrType>>>::iterator;
            using const_iterator            = typename std::vector<Node<Key, Mapped, VirtualAddrType>, typename std::allocator_traits<Allocator>::template rebind_alloc<Node<Key, Mapped, VirtualAddrType>>>::const_iterator;
            using size_type                 = SizeType;
            using difference_type           = std::intmax_t;
            using self                      = unordered_node_map;
            using load_factor_ratio         = typename LoadFactor::type;
            using virtual_addr_t            = get_virtual_addr_t<VirtualAddrType>;
            using node_t                    = Node<Key, Mapped, VirtualAddrType>;

            static inline constexpr virtual_addr_t NULL_VIRTUAL_ADDR    = null_addr_v<virtual_addr_t>;
            static inline constexpr size_t POW2_GROWTH_FACTOR           = 1u;
            static inline constexpr uint64_t MIN_CAP                    = 8u;
            static inline constexpr uint64_t MAX_CAP                    = uint64_t{1} << 50;

            static_assert((std::numeric_limits<SizeType>::max() >= MAX_CAP));

            static_assert(std::disjunction_v<std::is_same<typename std::ratio<1, 8>::type, load_factor_ratio>, 
                                             std::is_same<typename std::ratio<2, 8>::type, load_factor_ratio>, 
                                             std::is_same<typename std::ratio<3, 8>::type, load_factor_ratio>, 
                                             std::is_same<typename std::ratio<4, 8>::type, load_factor_ratio>,
                                             std::is_same<typename std::ratio<5, 8>::type, load_factor_ratio>, 
                                             std::is_same<typename std::ratio<6, 8>::type, load_factor_ratio>, 
                                             std::is_same<typename std::ratio<7, 8>::type, load_factor_ratio>, 
                                             std::is_same<typename std::ratio<8, 8>::type, load_factor_ratio>>);

            constexpr explicit unordered_node_map(size_type bucket_count,
                                                  const Hasher _hasher = Hasher(),
                                                  const Pred& pred = Pred(),
                                                  const Allocator& allocator = Allocator()): virtual_storage_vec(allocator),
                                                                                             bucket_vec(std::max(self::min_capacity(), unordered_map_variants::ceil2(bucket_count)), NULL_VIRTUAL_ADDR, allocator),
                                                                                             _hasher(_hasher),
                                                                                             pred(pred),
                                                                                             allocator(allocator){

                if (this->capacity() > self::max_capacity()){
                    throw std::length_error("bad unordered_node_map capacity");
                }

                this->virtual_storage_vec.reserve(self::capacity_to_size(this->capacity()));
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

            constexpr unordered_node_map(std::initializer_list<std::pair<const Key, Mapped>> init_list,
                                         size_type bucket_count,
                                         const Hasher& _hasher,
                                         const Allocator& allocator): unordered_node_map(init_list.begin(), init_list.end(), bucket_count, _hasher, Pred(), allocator){}

            constexpr unordered_node_map(std::initializer_list<std::pair<const Key, Mapped>> init_list,
                                         size_type bucket_count,
                                         const Allocator& allocator): unordered_node_map(init_list.begin(), init_list.end(), bucket_count, Hasher(), allocator){}

            constexpr void rehash(size_type tentative_new_cap){

                if (tentative_new_cap <= this->capacity()){
                    return;
                }

                size_t new_bucket_cap               = std::max(self::min_capacity(), unordered_map_variants::ceil2(tentative_new_cap));

                if (new_bucket_cap > self::max_capacity()){
                    throw std::length_error("bad unordered_node_map capacity");
                }

                size_t new_virtual_storage_vec_cap  = self::capacity_to_size(new_bucket_cap);
                auto new_bucket_vec                 = decltype(bucket_vec)(new_bucket_cap, NULL_VIRTUAL_ADDR, this->allocator);

                this->virtual_storage_vec.reserve(new_virtual_storage_vec_cap); 

                //static_assert(noexcept(this->_hasher(key))); TODOs: compile time validation

                for (size_t i = 0u; i < this->virtual_storage_vec.size(); ++i){
                    size_t hashed_value                 = this->_hasher(this->virtual_storage_vec[i].first);
                    size_t bucket_idx                   = hashed_value & (new_bucket_cap - 1u);
                    virtual_addr_t * insert_reference   = &new_bucket_vec[bucket_idx];

                    while (true){
                        if (*insert_reference == NULL_VIRTUAL_ADDR){
                            break;
                        }

                        insert_reference = &this->virtual_storage_vec[*insert_reference].nxt_addr;
                    }

                    *insert_reference                       = static_cast<virtual_addr_t>(i);
                    this->virtual_storage_vec[i].nxt_addr   = NULL_VIRTUAL_ADDR;
                }

                this->bucket_vec = std::move(new_bucket_vec);
            }

            constexpr void reserve(size_type new_sz){

                if (new_sz <= this->size()){
                    return;
                }

                if (new_sz > self::max_size()){
                    throw std::length_error("bad unordered_node_map size");
                }

                this->rehash(self::size_to_capacity(new_sz));
            }

            template <class KeyLike, class ...Args>
            constexpr auto try_emplace(KeyLike&& key, Args&& ...args) -> std::pair<iterator, bool>{

                return this->internal_insert(node_t{key_type(std::forward<KeyLike>(key)), mapped_type(std::forward<Args>(args)...), virtual_addr_t{}});
            }

            template <class ...Args>
            constexpr auto emplace(Args&& ...args) -> std::pair<iterator, bool>{

                return this->insert(std::pair<const Key, Mapped>(std::forward<Args>(args)...));
            }

            template <class ValueLike = std::pair<const Key, Mapped>>
            constexpr auto insert(ValueLike&& value) -> std::pair<iterator, bool>{

                return this->internal_insert(node_t{key_type(dg_forward_like<ValueLike>(value.first)), mapped_type(dg_forward_like<ValueLike>(value.second)), virtual_addr_t{}});
            }

            template <class Iterator>
            constexpr void insert(Iterator first, Iterator last){

                //give the user a chance to not have leak by using proper std::move() + friends

                this->reserve(this->size() + std::distance(first, last));

                while (first != last){
                    this->insert(*first);
                    std::advance(first, 1u);
                }
            }

            constexpr void insert(std::initializer_list<std::pair<const Key, Mapped>> init_list){

                this->insert(init_list.begin(), init_list.end());
            }

            template <class KeyLike, class MappedLike>
            constexpr auto insert_or_assign(KeyLike&& key, MappedLike&& mapped) -> std::pair<iterator, bool>{

                return this->internal_insert_or_assign(node_t{key_type(std::forward<KeyLike>(key)), mapped_type(std::forward<MappedLike>(mapped)), virtual_addr_t{}});
            }

            template <class KeyLike>
            constexpr auto operator[](KeyLike&& key) -> mapped_type&{

                return std::get<0>(this->insert_or_assign(std::forward<KeyLike>(key), mapped_type{}))->second;
            }

            constexpr void clear() noexcept(true){

                // static_assert(noexcept(this->virtual_storage_vec.clear())); TODOs: compile_time validation

                this->virtual_storage_vec.clear();
                std::fill(this->bucket_vec.begin(), this->bucket_vec.end(), self::NULL_VIRTUAL_ADDR);
            }

            constexpr void swap(self& other) noexcept(true){

                // static_assert(noexcept(std::swap(this->virtual_storage_vec, other.virtual_storage_vec))); TODOs: compile_time validation
                // static_assert(noexcept(std::swap(this->bucket_vec, other.bucket_vec))); TODOs: compile_time validation
                // static_assert(noexcept(std::swap(this->_hasher, other._hasher))); TODOs: compile_time validation
                // static_assert(noexcept(std::swap(this->pred, other.pred))); TODOs: compile_time validation
                // static_assert(noexcept(std::swap(this->allocator, other.allocator))); TODOs: compile_time validation

                std::swap(this->virtual_storage_vec, other.virtual_storage_vec);
                std::swap(this->bucket_vec, other.bucket_vec);
                std::swap(this->_hasher, other._hasher);
                std::swap(this->pred, other.pred);
                std::swap(this->allocator, other.allocator);
            }

            template <class EraseArg>
            constexpr auto erase(EraseArg&& erase_arg) noexcept(true) -> iterator{

                if constexpr(std::is_convertible_v<EraseArg&&, const_iterator>){
                    return this->internal_erase_iter(std::forward<EraseArg>(erase_arg));
                } else{
                    return this->internal_erase_key(std::forward<EraseArg>(erase_arg));
                }
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
            constexpr auto contains(const KeyLike& key) const noexcept(true) -> bool{ 

                return this->find(key) != this->end();
            }

            template <class KeyLike>
            constexpr auto count(const KeyLike& key) const noexcept(true) -> size_t{

                return static_cast<size_t>(this->contains(key));
            }

            template <class KeyLike>
            constexpr auto at(const KeyLike& key) const -> const mapped_type&{

                auto ptr = this->find(key);

                if (ptr == this->end()){
                    throw std::out_of_range("unordered_node_map bad access");
                }

                return ptr->second;
            }

            template <class KeyLike>
            constexpr auto at(const KeyLike& key) -> mapped_type&{

                auto ptr = this->find(key);

                if (ptr == this->end()){
                    throw std::out_of_range("unordered_node_map bad access");
                }

                return ptr->second;
            }

            constexpr auto empty() const noexcept -> bool{

                return this->virtual_storage_vec.empty();
            }

            constexpr auto capacity() const noexcept -> size_type{

                return this->bucket_vec.size();
            }

            static consteval auto min_capacity() noexcept -> size_t{

                return MIN_CAP;
            }

            static consteval auto max_capacity() noexcept -> size_type{

                return MAX_CAP;
            }

            constexpr auto size() const noexcept -> size_type{

                return this->virtual_storage_vec.size();
            }

            static consteval auto max_size() noexcept -> size_type{

                return self::capacity_to_size(self::max_capacity()); 
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

            static consteval auto load_factor() noexcept -> double{

                return static_cast<double>(load_factor_ratio::num) / load_factor_ratio::den;
            }

            static constexpr auto capacity_to_size(size_t cap) noexcept -> size_t{

                return cap * load_factor();
            }

            static constexpr auto size_to_capacity(size_t sz) noexcept -> size_t{

                return sz / load_factor();
            }

        private:

            constexpr auto to_bucket_index(size_t hashed_value) const noexcept -> size_t{

                return hashed_value & (this->bucket_vec.size() - 1u);
            }

            //this is one of the C++ myths, we rather copy paste than to do the remove const, it's undefined
            template <class KeyLike>
            constexpr auto internal_find_bucket_reference(const KeyLike& key) const noexcept(true) -> const virtual_addr_t *{
                
                //static_assert(noexcept(this->_hasher(key))); TODOs: compile time validation
                //static_assert(noexcept(this->pred(this->virtual_storage_vec[*current].first, key))) TODOs: compile time validation

                size_t hashed_value             = this->_hasher(key);
                size_t bucket_idx               = this->to_bucket_index(hashed_value);
                const virtual_addr_t * current  = &this->bucket_vec[bucket_idx];

                while (true){
                    if (*current == NULL_VIRTUAL_ADDR){
                        return current;
                    }

                    if (this->pred(this->virtual_storage_vec[*current].first, key)){
                        return current;
                    }

                    current = &this->virtual_storage_vec[*current].nxt_addr;
                }
            }

            template <class KeyLike>
            constexpr auto internal_find_bucket_reference(const KeyLike& key) noexcept(true) -> virtual_addr_t *{

                //static_assert(noexcept(this->_hasher(key))); TODOs: compile time validation
                //static_assert(noexcept(this->pred(this->virtual_storage_vec[*current].first, key))) TODOs: compile time validation

                size_t hashed_value         = this->_hasher(key);
                size_t bucket_idx           = this->to_bucket_index(hashed_value);
                virtual_addr_t * current    = &this->bucket_vec[bucket_idx];

                while (true){
                    if (*current == NULL_VIRTUAL_ADDR){
                        return current;
                    }

                    if (this->pred(this->virtual_storage_vec[*current].first, key)){
                        return current;
                    }

                    current = &this->virtual_storage_vec[*current].nxt_addr;
                }
            }

            template <class KeyLike>
            constexpr auto internal_exist_find_bucket_reference(const KeyLike& key) const noexcept(true) -> const virtual_addr_t *{

                //static_assert(noexcept(this->_hasher(key))); TODOs: compile time validation
                //static_assert(noexcept(this->pred(this->virtual_storage_vec[*current].first, key))) TODOs: compile time validation

                size_t hashed_value             = this->_hasher(key);
                size_t bucket_idx               = this->to_bucket_index(hashed_value);
                const virtual_addr_t * current  = &this->bucket_vec[bucket_idx];

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

                //static_assert(noexcept(this->_hasher(key))); TODOs: compile time validation
                //static_assert(noexcept(this->pred(this->virtual_storage_vec[*current].first, key))) TODOs: compile time validation

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
            constexpr auto internal_find(const KeyLike& key) const noexcept(true) -> const_iterator{

                //static_assert(noexcept(this->_hasher(key))); TODOs: compile time validation
                //static_assert(noexcept(this->pred(this->virtual_storage_vec[*current].first, key))) TODOs: compile time validation

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

                //static_assert(noexcept(this->_hasher(key))); TODOs: compile time validation
                //static_assert(noexcept(this->pred(this->virtual_storage_vec[*current].first, key))) TODOs: compile time validation

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
        
            template <class ValueLike>
            constexpr auto internal_insert(ValueLike&& value) -> std::pair<iterator, bool>{

                if (this->virtual_storage_vec.size() == this->virtual_storage_vec.capacity()) [[unlikely]]{ //strong guarantee, might corrupt vector_capacity <-> bucket_vec_size ratio, signals an uphash
                    this->rehash(this->bucket_vec.size() << POW2_GROWTH_FACTOR);
                }

                virtual_addr_t * insert_reference   = this->internal_find_bucket_reference(value.first);

                if (*insert_reference == NULL_VIRTUAL_ADDR){
                    value.nxt_addr                  = NULL_VIRTUAL_ADDR;
                    virtual_addr_t appending_addr   = static_cast<virtual_addr_t>(this->virtual_storage_vec.size());
                    this->virtual_storage_vec.emplace_back(std::forward<ValueLike>(value));
                    *insert_reference               = appending_addr;

                    return std::make_pair(std::next(this->virtual_storage_vec.begin(), appending_addr), true);
                }

                return std::make_pair(std::next(this->virtual_storage_vec.begin(), *insert_reference), false);
            }

            template <class ValueLike>
            constexpr auto internal_insert_or_assign(ValueLike&& value) -> std::pair<iterator, bool>{

                auto [iter, status] = this->internal_insert(std::forward<ValueLike>(value));

                if (!status){
                    iter->second = dg_forward_like<ValueLike>(value.second);
                }

                return std::make_pair(iter, status);
            }

            template <class KeyLike>
            constexpr void internal_erase(const KeyLike& key) noexcept(true){

                // static_assert(noexcept(std::swap(std::declval<node_t&>, std::declval<node_t&>)));
                // static_assert(noexcept(this->virtual_storage_vec.pop_back()));

                virtual_addr_t * key_reference = this->internal_find_bucket_reference(key);

                if (*key_reference == NULL_VIRTUAL_ADDR){
                    return;
                } 

                virtual_addr_t * swapping_reference = this->internal_exist_find_bucket_reference(this->virtual_storage_vec.back().first);

                if (swapping_reference == key_reference) [[unlikely]]{
                    *key_reference = this->virtual_storage_vec[*key_reference].nxt_addr;
                    this->virtual_storage_vec.pop_back();

                    return;
                }

                virtual_addr_t removing_addr = std::exchange(*key_reference, this->virtual_storage_vec[*key_reference].nxt_addr); 
                std::swap(this->virtual_storage_vec[removing_addr], this->virtual_storage_vec.back());
                this->virtual_storage_vec.pop_back();
                *swapping_reference = removing_addr;
            }

            template <class KeyLike>
            constexpr auto internal_erase_key(const KeyLike& key) noexcept(true) -> iterator{

                this->internal_erase(key);
                return this->begin();
            }

            constexpr auto internal_erase_iter(const_iterator iter) noexcept(true) -> iterator{

                if (iter == this->cend()){
                    return this->begin();
                }

                return this->internal_erase_key(iter->first);
            }
    };

    template <class ...Args>
    constexpr auto operator ==(const unordered_node_map<Args...>& lhs, const unordered_node_map<Args...>& rhs) noexcept(true) -> bool{

        if (lhs.size() != rhs.size()){
            return false;
        }

        for (const auto& kv_pair: lhs){
            auto rhs_ptr = rhs.find(kv_pair.first);

            if (rhs_ptr == rhs.end()){
                return false;
            }

            if (rhs_ptr->second != kv_pair.second){
                return false;
            }
        }

        return true;
    }

    template <class ...Args>
    constexpr auto operator !=(const unordered_node_map<Args...>& lhs, const unordered_node_map<Args...>& rhs) noexcept(true) -> bool{
        
        return !(lhs == rhs);
    }
}

namespace std{

    template <class ...Args>
    constexpr void swap(dg::network_datastructure::unordered_map_variants::unordered_node_map<Args...>& lhs,
                        dg::network_datastructure::unordered_map_variants::unordered_node_map<Args...>& rhs) noexcept(noexcept(std::declval<dg::network_datastructure::unordered_map_variants::unordered_node_map<Args...>&>().swap(std::declval<dg::network_datastructure::unordered_map_variants::unordered_node_map<Args...>&>()))){

        lhs.swap(rhs);
    }

    template <class ...Args, class Pred>
    constexpr void erase_if(dg::network_datastructure::unordered_map_variants::unordered_node_map<Args...>& umap,
                            Pred pred){

        auto it = umap.begin(); 

        while (it != umap.end()){
            if (pred(*it)){
                it = umap.erase(it);
            } else{
                std::advance(it, 1u);
            }
        }
    }
}

namespace dg::network_datastructure::node_hash_set{

}

#endif