#ifndef __DG_DGBUF__
#define __DG_DGBUF__

#include "serialization.h"
#include <unordered_map>
#include <unordered_set>
#include <map>
#include <set>
#include <vector>
#include <array>
#include <stdint.h>
#include <type_traits>
#include <utility>
#include <optional>

//TODO: half-finished - need string support, noexcept qualifiers, implicit set_buf and static-checked types 

namespace dg::dgbuf::types{

    using vaddr_type    = uint64_t;
    using size_type     = uint64_t;
}

namespace dg::dgbuf::constants{

    static inline constexpr size_t MAX_TEMPLATE_RECURSIVE_DEPTH = 5;
    static inline constexpr double CAP_TO_SIZE_RATIO = 2; 
}

namespace dg::dgbuf::utility{
    
    struct modulo_key_to_idx{
        
        template <class Key>
        constexpr auto operator()(Key&& key, size_t cap) const noexcept(noexcept(key % cap)) -> size_t{

            return key % cap;
        }  
    };

    struct bitwise_and_key_to_idx{

        template <class Key>
        constexpr auto operator()(Key&& key, size_t cap) const noexcept(noexcept(key & (cap - 1))) -> size_t{

            return key & (cap - 1);
        }
    };

    static inline constexpr size_t log2(size_t x) noexcept{

        if (x == 1){
            return 0;
        }

        return log2(x >> 1) + 1;
    }

    static inline constexpr bool is_pow2(size_t x) noexcept{

        return x != 0 && (x & (x - 1)) == 0;
    } 

    static inline constexpr size_t pow2_ceil(size_t x) noexcept{

        if (x == 0){
            return 1;
        }

        if (is_pow2(x)){
            return x;
        }

        return 1 << (log2(x) + 1);
    }
}

namespace dg::dgbuf::iterator{

    //reference - const bool_vector iterator 
    template <class T>
    class vector_view_iterator: public std::iterator<std::random_access_iterator_tag, T, std::ptrdiff_t, void, T>{

        private:

            const char * buf;
            intmax_t offs;

        public:

            using self = vector_view_iterator<T>;
            using base = std::iterator<std::random_access_iterator_tag, T, std::ptrdiff_t, void, T>;

            using difference_type   = typename base::difference_type;
            using reference         = typename base::reference;
            
            constexpr vector_view_iterator() = default;

            constexpr vector_view_iterator(const char * buf, intmax_t offs) noexcept: buf(buf), offs(offs){}

            constexpr auto operator *() const noexcept -> reference{
 
                T rs{};
                dg::dgbuf_serializer::deserialize(this->buf + this->offs, rs);
                return rs; 
            }
            
            constexpr auto operator[](difference_type idx) const noexcept -> reference{

                return *(*this + idx);
            }

            constexpr auto operator ==(const self& rhs) const noexcept -> bool{

                return this->numerical_addr() == rhs.numerical_addr();
            }

            constexpr auto operator !=(const self& rhs) const noexcept -> bool{

                return this->numerical_addr() != rhs.numerical_addr();
            }

            constexpr auto operator >(const self& rhs) const noexcept -> bool{

                return this->numerical_addr() > rhs.numerical_addr();
            }

            constexpr auto operator <(const self& rhs) const noexcept -> bool{

                return this->numerical_addr() < rhs.numerical_addr();
            }

            constexpr auto operator >=(const self& rhs) const noexcept -> bool{

                return this->numerical_addr() >= rhs.numerical_addr();
            }

            constexpr auto operator <=(const self& rhs) const noexcept -> bool{

                return this->numerical_addr() <= rhs.numerical_addr();
            }

            constexpr auto operator ++() noexcept -> self&{

                this->offs += dg::dgbuf_serializer::count(T{});
                return *this;
            } 

            constexpr auto operator ++(int) noexcept -> self{

                self pre = *this;
                ++(*this);
                return pre;
            }

            constexpr auto operator --() noexcept -> self&{

                this->offs -= dg::dgbuf_serializer::count(T{});
                return *this;
            }

            constexpr auto operator --(int) noexcept -> self{

                self pre = *this;
                --(*this);
                return pre;
            }

            constexpr auto operator +(difference_type idx) const noexcept -> self{

                return self{this->buf, this->offs + dg::dgbuf_serializer::count(T{}) * idx};
            } 

            constexpr auto operator +=(difference_type idx) noexcept -> self&{

                *this = *this + idx;
                return *this;
            }

            constexpr auto operator -(difference_type idx) const noexcept -> self{

                return self{this->buf, this->offs - dg::dgbuf_serializer::count(T{}) * idx};
            }

            constexpr auto operator -=(difference_type idx) noexcept -> self&{

                *this = *this - idx;
                return *this;
            }

            constexpr auto operator -(const self& other) const noexcept -> difference_type{

                return (this->numerical_addr() - other.numerical_addr()) / dg::dgbuf_serializer::count(T{});
            }
        
        private:

            constexpr auto numerical_addr() const noexcept -> intptr_t{

                return reinterpret_cast<intptr_t>(this->buf) + this->offs;
            } 
    };

    template <class T>
    constexpr auto operator +(typename vector_view_iterator<T>::difference_type lhs, const vector_view_iterator<T>& rhs) noexcept(noexcept(rhs + lhs)) -> decltype(rhs + lhs){

        return rhs + lhs;
    }  

    template <class K, class V, class NextSeeker>
    class unordered_flat_map_view_iterator: public std::iterator<std::input_iterator_tag, std::pair<K, V>, std::ptrdiff_t, void, std::pair<K, V>>{

        private:

            vector_view_iterator<std::optional<std::pair<K, V>>> ptr;
            NextSeeker nxt_seeker; 

        public:

            using self  = unordered_flat_map_view_iterator<K, V, NextSeeker>;
            using base  = std::iterator<std::input_iterator_tag, std::pair<K, V>, std::ptrdiff_t, void, std::pair<K, V>>;
            using reference = typename base::reference;

            constexpr unordered_flat_map_view_iterator(vector_view_iterator<std::optional<std::pair<K, V>>> ptr,
                                                       NextSeeker nxt_seeker) noexcept: ptr(ptr), nxt_seeker(nxt_seeker){}

            constexpr auto operator *() const noexcept -> reference{

                return (*this->ptr).value();
            }
            
            constexpr auto operator ==(const self& rhs) const noexcept -> bool{

                return this->ptr == rhs.ptr;
            }

            constexpr auto operator !=(const self& rhs) const noexcept -> bool{

                return this->ptr != rhs.ptr;
            }

            constexpr auto operator ++() noexcept -> self&{

                this->ptr = this->nxt_seeker(this->ptr);
                return *this;
            }

            constexpr auto operator ++(int) noexcept -> self{

                self pre = *this;
                ++(*this);
                return pre;
            }
    };

    template <class K, class NextSeeker>
    class unordered_flat_set_view_iterator: public std::iterator<std::input_iterator_tag, K, std::ptrdiff_t, void, K>{

        private:

            vector_view_iterator<std::optional<K>> ptr;
            NextSeeker nxt_seeker;
        
        public:

            using self  = unordered_flat_set_view_iterator<K, NextSeeker>;
            using base  = std::iterator<std::input_iterator_tag, K, std::ptrdiff_t, void, K>;
            using reference = typename base::reference;

            constexpr unordered_flat_set_view_iterator(vector_view_iterator<std::optional<K>> ptr,
                                                       NextSeeker nxt_seeker) noexcept: ptr(ptr), nxt_seeker(nxt_seeker){}
            
            constexpr auto operator *() const noexcept -> reference{

                return (*this->ptr).value();
            }

            constexpr auto operator ==(const self& rhs) const noexcept -> bool{

                return this->ptr == rhs.ptr;
            }

            constexpr auto operator !=(const self& rhs) const noexcept -> bool{
                
                return this->ptr != rhs.ptr;
            }

            constexpr auto operator ++() noexcept -> self&{

                this->ptr = this->nxt_seeker(this->ptr);
                return *this;
            }

            constexpr auto operator ++(int) noexcept -> self{

                self pre = *this;
                ++(*this);
                return pre;
            }
    };

    template <class K, class V>
    class map_view_iterator: public std::iterator<std::input_iterator_tag, std::pair<K, V>, std::ptrdiff_t, void, std::pair<K, V>>{

        private:

            vector_view_iterator<std::pair<K, V>> ptr;
        
        public:

            using self  = map_view_iterator<K, V>;
            using base  = std::iterator<std::input_iterator_tag, std::pair<K, V>, std::ptrdiff_t, void, std::pair<K, V>>;
            using reference = typename base::reference; 

            constexpr map_view_iterator(vector_view_iterator<std::pair<K, V>> ptr) noexcept: ptr(ptr){}

            constexpr auto operator *() const noexcept -> reference{

                return *this->ptr;
            }

            constexpr auto operator ==(const self& rhs) const noexcept -> bool{

                return this->ptr == rhs.ptr;
            }

            constexpr auto operator !=(const self& rhs) const noexcept -> bool{

                return this->ptr != rhs.ptr;
            }

            constexpr auto operator ++() noexcept -> self&{

                ++this->ptr;
                return *this;
            }

            constexpr auto operator ++(int) noexcept -> self{

                auto pre = *this;
                ++(*this);
                return pre;
            }   
    };

    template <class K>
    class set_view_iterator: public std::iterator<std::input_iterator_tag, K, std::ptrdiff_t, void, K>{

        private:

            vector_view_iterator<K> ptr;
        
        public:

            using self = set_view_iterator<K>;
            using base = std::iterator<std::input_iterator_tag, K, std::ptrdiff_t, void, K>;
            using reference = typename base::reference;

            constexpr set_view_iterator(vector_view_iterator<K> ptr) noexcept: ptr(ptr){}

            constexpr auto operator *() const noexcept -> reference{

                return *this->ptr;
            }

            constexpr auto operator ==(const self& rhs) const noexcept -> bool{

                return this->ptr == rhs.ptr;
            }

            constexpr auto operator !=(const self& rhs) const noexcept -> bool{

                return this->ptr != rhs.ptr;
            }

            constexpr auto operator ++() noexcept -> self&{

                ++this->ptr;
                return *this;
            }

            constexpr auto operator ++(int) noexcept -> self{

                auto pre = *this;
                ++(*this);
                return pre;
            }     
    };

    template <class T>
    static constexpr auto seek_next_available_bucket(vector_view_iterator<std::optional<T>> first, vector_view_iterator<std::optional<T>> last) noexcept{

        for (auto it = first; it != last; ++it){
            if (*it){
                return it;
            }
        }

        return last;
    }

    template <class T>
    class propagator_device{

        private:

            vector_view_iterator<std::optional<T>> last;
        
        public:

            constexpr propagator_device() = default;

            constexpr propagator_device(vector_view_iterator<std::optional<T>> last) noexcept: last(last){}

            constexpr auto operator()(vector_view_iterator<std::optional<T>> cur) noexcept -> vector_view_iterator<std::optional<T>>{

                return seek_next_available_bucket(++cur, last);
            }
    };

    template <class T>
    static constexpr auto make_propagator_device(vector_view_iterator<std::optional<T>> last) noexcept{

        propagator_device device(last);
        return device;
    }
}

namespace dg::dgbuf::datastructure{

    template <class T>    
    class vector_view{

        private:

            types::vaddr_type data;
            types::size_type sz;
            const char * buf;

        public:
            
            static_assert(true); //T types - 

            constexpr vector_view() = default;

            constexpr vector_view(types::vaddr_type data, 
                                  types::size_type sz, 
                                  const char * buf) noexcept: data(data), sz(sz), buf(buf){}

            constexpr void set_buf(const char * buf) noexcept{

                this->buf = buf;
            } 

            constexpr auto size() const noexcept -> types::size_type{

                return this->sz;
            } 

            constexpr auto get(size_t idx) const noexcept -> T{

                const char * ptr = this->buf + this->data + idx * dg::dgbuf_serializer::count(T{});
                T rs{};
                dg::dgbuf_serializer::deserialize(ptr, rs);

                return rs;
            }

            constexpr auto operator[](size_t idx) const -> decltype(auto){

                return this->get(idx);
            }

            constexpr auto begin() const noexcept{

                iterator::vector_view_iterator<T> it(this->buf, this->data);
                return it;
            }

            constexpr auto end() const noexcept{

                iterator::vector_view_iterator<T> it(this->buf, this->data + this->sz * dg::dgbuf_serializer::count(T{}));
                return it;
            }

            template <class Reflector>
            constexpr void dg_reflect(const Reflector& reflector) const noexcept{

                reflector(this->data, this->sz);
            }

            template <class Reflector>
            constexpr void dg_reflect(const Reflector& reflector) noexcept{

                reflector(this->data, this->sz);
            }
    };

    template <class Key, class Value, class Hasher = std::hash<Key>, class KeyEq = std::equal_to<Key>, class KeyToIdx = utility::modulo_key_to_idx>
    class unordered_flat_map_view{

        private:

            vector_view<std::optional<std::pair<Key, Value>>> buckets;
            types::size_type sz;
        
        public:

            static_assert(std::is_trivially_constructible_v<Hasher>);
            static_assert(std::is_trivially_constructible_v<KeyEq>);
            static_assert(std::is_trivially_constructible_v<KeyToIdx>);

            constexpr unordered_flat_map_view() = default;

            constexpr unordered_flat_map_view(vector_view<std::optional<std::pair<Key, Value>>> buckets,
                                              types::size_type sz) noexcept: buckets(buckets), sz(sz){}

            constexpr void set_buf(const char * buf) noexcept{

                this->buckets.set_buf(buf);
            }

            constexpr auto size() const noexcept -> types::size_type{

                return this->sz;
            }

            constexpr auto begin() const noexcept{

                iterator::unordered_flat_map_view_iterator it(iterator::seek_next_available_bucket(this->buckets.begin(), this->buckets.end()), iterator::make_propagator_device(this->buckets.end()));
                return it;
            }

            constexpr auto end() const noexcept{

                iterator::unordered_flat_map_view_iterator it(this->buckets.end(), iterator::make_propagator_device(this->buckets.end()));
                return it;
            }

            template <class KeyLike>
            constexpr auto find(KeyLike&& key) const noexcept{ //TODO: noexcept qualifiers

                auto hashed = Hasher{}(key);

                for (size_t i = 0; i < this->buckets.size(); ++i){
                    size_t slot = KeyToIdx{}(hashed + i, this->buckets.size());
                    auto bucket = this->buckets.get(slot);

                    if (!bucket){
                        return this->end();
                    }

                    if (KeyEq{}(key, bucket->first)){
                        iterator::unordered_flat_map_view_iterator it(this->buckets.begin() + slot, iterator::make_propagator_device(this->buckets.end()));
                        return it;
                    }
                }

                return this->end();
            }

            template <class ...Args>
            constexpr auto operator[](Args&& ...args) noexcept -> decltype(auto){

                return (*this->find(std::forward<Args>(args)...)).second;
            } 

            template <class Reflector>
            constexpr void dg_reflect(const Reflector& reflector) const noexcept{

                reflector(this->buckets, this->sz);
            }

            template <class Reflector>
            constexpr void dg_reflect(const Reflector& reflector) noexcept{

                reflector(this->buckets, this->sz);
            }
    };

    template <class Key, class Hasher = std::hash<Key>, class KeyEq = std::equal_to<Key>, class KeyToIdx = utility::modulo_key_to_idx>
    class unordered_flat_set_view{

        private:

            vector_view<std::optional<Key>> buckets;
            types::size_type sz;
        
        public:

            static_assert(std::is_trivially_constructible_v<Hasher>);
            static_assert(std::is_trivially_constructible_v<KeyEq>);
            static_assert(std::is_trivially_constructible_v<KeyToIdx>);

            constexpr unordered_flat_set_view() = default;

            constexpr unordered_flat_set_view(vector_view<std::optional<Key>> buckets, 
                                              types::size_type sz) noexcept: buckets(buckets), sz(sz){}
            
            constexpr void set_buf(const char * buf) noexcept{

                this->buckets.set_buf(buf);
            } 

            constexpr auto size() const noexcept -> types::size_type{

                return this->sz;
            }

            constexpr auto begin() const noexcept{
                
                iterator::unordered_flat_set_view_iterator it(iterator::seek_next_available_bucket(this->buckets.begin(), this->buckets.end()), iterator::make_propagator_device(this->buckets.end()));
                return it;
            }

            constexpr auto end() const noexcept{
                
                iterator::unordered_flat_set_view_iterator it(this->buckets.end(), iterator::make_propagator_device(this->buckets.end()));
                return it;
            }

            template <class KeyLike>
            constexpr auto find(KeyLike&& key) const noexcept{ //TODO: noexcept qualifiers

                auto hashed = Hasher{}(key);

                for (size_t i = 0; i < this->sz; ++i){
                    size_t slot = KeyToIdx{}(hashed + i, this->buckets.size());
                    auto bucket = this->buckets.get(slot);
                    
                    if (!bucket){
                        return this->end();
                    }

                    if (KeyEq{}(key, bucket.value())){
                        iterator::unordered_flat_set_view_iterator it(this->buckets.begin() + slot, iterator::make_propagator_device(this->buckets.end()));
                        return it;
                    }
                }
                
                return this->end();
            }

            template <class Reflector>
            constexpr void dg_reflect(const Reflector& reflector) const noexcept{

                reflector(this->buckets, this->sz);
            }

            template <class Reflector>
            constexpr void dg_reflect(const Reflector& reflector) noexcept{

                reflector(this->buckets, this->sz);
            }
    };

    template <class Key, class Value, class Comparer = std::less<Key>>
    class map_view{

        private:

            vector_view<std::pair<Key, Value>> buckets; 
            types::size_type sz;
        
        public:

            static_assert(std::is_trivially_constructible_v<Comparer>);

            constexpr map_view() = default;
            
            constexpr map_view(vector_view<std::pair<Key, Value>> buckets, 
                               types::size_type sz) noexcept: buckets(buckets), sz(sz){}

            constexpr void set_buf(const char * buf) noexcept{

                this->buckets.set_buf(buf);
            }

            constexpr auto size() const noexcept -> types::size_type{

                return this->sz;
            }

            constexpr auto begin() const noexcept{
                
                iterator::map_view_iterator it(this->buckets.begin()); 
                return it;
            }

            constexpr auto end() const noexcept{

                iterator::map_view_iterator it(this->buckets.end());
                return it;
            }

            template <class KeyLike>
            constexpr auto find(KeyLike&& key) const noexcept{ //TODO: noexcept qualifiers
                
                auto cmp = [](std::pair<Key, Value> lhs, const auto& tgt){return Comparer{}(lhs.first, tgt);};
                auto ptr = std::lower_bound(this->buckets.begin(), this->buckets.end(), key, cmp);

                if (ptr == this->buckets.end()){
                    return this->end();
                }

                std::pair<Key, Value> found = *ptr; 

                if (!Comparer{}(found.first, key) && !Comparer{}(key, found.first)){
                    iterator::map_view_iterator it(ptr);
                    return it;
                }
                
                return this->end();
            }

            template <class Reflector>
            constexpr void dg_reflect(const Reflector& reflector) const noexcept{

                reflector(this->buckets, this->sz);
            } 

            template <class Reflector>
            constexpr void dg_reflect(const Reflector& reflector) noexcept{

                reflector(this->buckets, this->sz);
            }
    };

    template <class Key, class Comparer = std::less<Key>>
    class set_view{

        private:

            vector_view<Key> buckets;
            types::size_type sz;
        
        public:

            static_assert(std::is_trivially_constructible_v<Comparer>);
            
            constexpr set_view() = default;

            constexpr set_view(vector_view<Key> buckets, 
                              types::size_type sz) noexcept: buckets(buckets), sz(sz){}
            
            constexpr void set_buf(const char * buf) noexcept{

                this->buckets.set_buf(buf);
            }

            constexpr auto size() const noexcept -> types::size_type{

                return this->sz;
            }

            constexpr auto begin() const noexcept{
                
                iterator::set_view_iterator it(this->buckets.begin());
                return it;
            }

            constexpr auto end() const noexcept{
                
                iterator::set_view_iterator it(this->buckets.end());
                return it;
            }

            template <class KeyLike>
            constexpr auto find(KeyLike&& key) const noexcept{ //TODO: noexcept qualifiers

                auto ptr = std::lower_bound(this->buckets.begin(), this->buckets.end(), key, Comparer{});
                
                if (ptr == this->buckets.end()){
                    return this->end();
                }

                if (!Comparer{}(*ptr, key) && !Comparer{}(key, *ptr)){
                    iterator::set_view_iterator it(ptr);
                    return it;
                }

                return this->end();
            }

            template <class Reflector>
            constexpr void dg_reflect(const Reflector& reflector) const noexcept{

                reflector(this->buckets, this->sz);
            }

            template <class Reflector>
            constexpr void dg_reflect(const Reflector& reflector) noexcept{

                reflector(this->buckets, this->sz);
            }
    };
}

namespace dg::dgbuf::types_space{

    template <class T, class = void>
    struct is_std_fixed_size_container: std::false_type{};
    
    template <class T>
    struct is_std_fixed_size_container<T, std::void_t<decltype(std::tuple_size<T>::value)>>: std::true_type{};

    template <class T>
    struct is_std_optional: std::false_type{};

    template <class ...Args>
    struct is_std_optional<std::optional<Args...>>: std::true_type{}; 

    template <class T>
    struct is_std_vector: std::false_type{};

    template <class ...Args>
    struct is_std_vector<std::vector<Args...>>: std::true_type{};

    template <class T>
    struct is_std_unordered_map: std::false_type{};

    template <class ...Args>
    struct is_std_unordered_map<std::unordered_map<Args...>>: std::true_type{}; 

    template <class T>
    struct is_std_unordered_set: std::false_type{};

    template <class ...Args>
    struct is_std_unordered_set<std::unordered_set<Args...>>: std::true_type{};

    template <class T>
    struct is_std_map: std::false_type{};

    template <class ...Args>
    struct is_std_map<std::map<Args...>>: std::true_type{}; 

    template <class T>
    struct is_std_set: std::false_type{};

    template <class ...Args>
    struct is_std_set<std::set<Args...>>: std::true_type{};

    template <class T>
    struct is_dg_vector_view: std::false_type{};

    template <class ...Args>
    struct is_dg_vector_view<datastructure::vector_view<Args...>>: std::true_type{};

    template <class T>
    struct is_dg_unordered_flat_map_view: std::false_type{};

    template <class ...Args>
    struct is_dg_unordered_flat_map_view<datastructure::unordered_flat_map_view<Args...>>: std::true_type{};

    template <class T>
    struct is_dg_unordered_flat_set_view: std::false_type{};

    template <class ...Args>
    struct is_dg_unordered_flat_set_view<datastructure::unordered_flat_set_view<Args...>>: std::true_type{};

    template <class T>
    struct is_dg_map_view: std::false_type{};

    template <class ...Args>
    struct is_dg_map_view<datastructure::map_view<Args...>>: std::true_type{};

    template <class T>
    struct is_dg_set_view: std::false_type{};

    template <class ...Args>
    struct is_dg_set_view<datastructure::set_view<Args...>>: std::true_type{};

    template <class T>
    static inline constexpr bool is_std_fixed_size_container_v = is_std_fixed_size_container<T>::value;

    template <class T>
    static inline constexpr bool is_std_optional_v = is_std_optional<T>::value;

    template <class T>
    static inline constexpr bool is_std_vector_v = is_std_vector<T>::value;

    template <class T>
    static inline constexpr bool is_std_unordered_map_v = is_std_unordered_map<T>::value;

    template <class T>
    static inline constexpr bool is_std_unordered_set_v = is_std_unordered_set<T>::value;

    template <class T>
    static inline constexpr bool is_std_map_v = is_std_map<T>::value;

    template <class T>
    static inline constexpr bool is_std_set_v = is_std_set<T>::value;

    template <class T>
    static inline constexpr bool is_dg_vector_view_v = is_dg_vector_view<T>::value;

    template <class T>
    static inline constexpr bool is_dg_unordered_flat_map_view_v = is_dg_unordered_flat_map_view<T>::value;

    template <class T>
    static inline constexpr bool is_dg_unordered_flat_set_view_v = is_dg_unordered_flat_set_view<T>::value;

    template <class T>
    static inline constexpr bool is_dg_map_view_v = is_dg_map_view<T>::value;

    template <class T>
    static inline constexpr bool is_dg_set_view_v = is_dg_set_view<T>::value;

    template <class T>
    static inline constexpr bool is_dg_container_view_v = is_dg_vector_view_v<T> | is_dg_unordered_flat_map_view_v<T> | is_dg_unordered_flat_set_view_v<T> | is_dg_map_view_v<T> | is_dg_set_view_v<T>;
}

namespace dg::dgbuf::stl_to_dgbuf{

    template <size_t I = 0>
    struct type_converter{

        using successor = type_converter<I + 1>;

        template <class Key, class Value, class Hasher, class KeyEq, class ...Args>
        static auto convert(std::unordered_map<Key, Value, Hasher, KeyEq, Args...>) -> datastructure::unordered_flat_map_view<decltype(successor::convert(std::declval<Key>())), decltype(successor::convert(std::declval<Value>())), Hasher, KeyEq, utility::bitwise_and_key_to_idx>;

        template <class Key, class Hasher, class KeyEq, class ...Args>
        static auto convert(std::unordered_set<Key, Hasher, KeyEq, Args...>) -> datastructure::unordered_flat_set_view<decltype(successor::convert(std::declval<Key>())), Hasher, KeyEq, utility::bitwise_and_key_to_idx>;

        template <class Key, class Value, class Comparer, class ...Args>
        static auto convert(std::map<Key, Value, Comparer, Args...>) -> datastructure::map_view<decltype(successor::convert(std::declval<Key>())), decltype(successor::convert(std::declval<Value>())), Comparer>;

        template <class Key, class Comparer, class ...Args>
        static auto convert(std::set<Key, Comparer, Args...>) -> datastructure::set_view<decltype(successor::convert(std::declval<Key>())), Comparer>;

        template <class T, class ...Args>
        static auto convert(std::vector<T, Args...>) -> datastructure::vector_view<decltype(successor::convert(std::declval<T>()))>;

        template <class T, size_t N>
        static auto convert(std::array<T, N>) -> std::array<decltype(successor::convert(std::declval<T>())), N>;

        template <class First, class Second>
        static auto convert(std::pair<First, Second>) -> std::pair<decltype(successor::convert(std::declval<First>())), decltype(successor::convert(std::declval<Second>()))>;

        template <class ...Args>
        static auto convert(std::tuple<Args...>) -> std::tuple<decltype(successor::convert(std::declval<Args>()))...>;

        template <class T>
        static auto convert(std::optional<T>) -> std::optional<decltype(successor::convert(std::declval<T>()))>;

        template <class T, std::enable_if_t<std::is_arithmetic_v<T>, bool> = true>
        static auto convert(T) -> T; 
    };

    template <>
    struct type_converter<constants::MAX_TEMPLATE_RECURSIVE_DEPTH>{

        template <class T>
        static auto convert(T) -> T;
    };

    struct flattener{

        template <class Key, class Value, class Hasher, class ...Args, std::enable_if_t<std::is_trivially_constructible_v<Hasher>, bool> = true>
        static auto flatten(const std::unordered_map<Key, Value, Hasher, Args...>& map) -> std::vector<std::optional<std::pair<Key, Value>>>{

            auto cap = utility::pow2_ceil(map.size() * constants::CAP_TO_SIZE_RATIO);
            auto buckets = std::vector<std::optional<std::pair<Key, Value>>>(cap, std::nullopt);

            for (const auto& kv: map){
                auto hashed = Hasher{}(kv.first);
                for (size_t i = 0; i < cap; ++i){
                    size_t slot = utility::bitwise_and_key_to_idx{}(hashed + i, cap); 
                    if (!buckets[slot]){
                        buckets[slot] = kv;
                        break;
                    }
                }
            }

            return buckets;
        } 

        template <class Key, class Hasher, class ...Args, std::enable_if_t<std::is_trivially_constructible_v<Hasher>, bool> = true>
        static auto flatten(const std::unordered_set<Key, Hasher, Args...>& set) -> std::vector<std::optional<Key>>{

            auto cap = utility::pow2_ceil(set.size() * constants::CAP_TO_SIZE_RATIO);
            auto buckets = std::vector<std::optional<Key>>(cap, std::nullopt);

            for (const auto& k: set){
                auto hashed = Hasher{}(k);
                for (size_t i = 0; i < cap; ++i){
                    size_t slot = utility::bitwise_and_key_to_idx{}(hashed + i, cap);
                    if (!buckets[slot]){
                        buckets[slot] = k;
                        break;
                    }
                }
            }

            return buckets;
        }

        template <class Key, class Value, class ...Args>
        static auto flatten(const std::map<Key, Value, Args...>& map) -> std::vector<std::pair<Key, Value>>{

            auto buckets = std::vector<std::pair<Key ,Value>>();
            buckets.reserve(map.size());
            std::copy(map.begin(), map.end(), std::back_inserter(buckets));

            return buckets;
        }

        template <class Key, class ...Args>
        static auto flatten(const std::set<Key, Args...>& set) -> std::vector<Key>{

            auto buckets = std::vector<Key>();
            buckets.reserve(set.size());
            std::copy(set.begin(), set.end(), std::back_inserter(buckets));

            return buckets;
        }
    };

    struct serializer{

        template <class T, std::enable_if_t<types_space::is_std_vector_v<T>, bool> = true>
        auto serialize(const T& obj, char * head, char *& cur) -> decltype(type_converter<>::convert(obj)){

            using converted_value_type = decltype(type_converter<>::convert(std::declval<typename T::value_type>()));

            types::vaddr_type vaddr = std::distance(head, cur);
            char * ptr = cur;
            cur += obj.size() * dg::dgbuf_serializer::count(converted_value_type()); //

            for (const auto& e: obj){
                auto serialized_e = serialize(e, head, cur);
                ptr = dg::dgbuf_serializer::serialize(serialized_e, ptr);
            }

            return {vaddr, obj.size(), std::add_pointer_t<char>()};
        }

        template <class T, std::enable_if_t<types_space::is_std_unordered_map_v<T>, bool> = true>
        auto serialize(const T& obj, char * head, char *& cur) -> decltype(type_converter<>::convert(obj)){

            auto flattened  = flattener::flatten(obj);
            auto serialized = serialize(flattened, head, cur);

            return {serialized, obj.size()};
        }

        template <class T, std::enable_if_t<types_space::is_std_map_v<T>, bool> = true>
        auto serialize(const T& obj, char * head, char *& cur) -> decltype(type_converter<>::convert(obj)){

            auto flattened  = flattener::flatten(obj);
            auto serialized = serialize(flattened, head, cur);

            return {serialized, obj.size()};
        }

        template <class T, std::enable_if_t<types_space::is_std_unordered_set_v<T>, bool> = true>
        auto serialize(const T& obj, char * head, char *& cur) -> decltype(type_converter<>::convert(obj)){

            auto flattened  = flattener::flatten(obj);
            auto serialized = serialize(flattened, head, cur);

            return {serialized, obj.size()}; 
        }

        template <class T, std::enable_if_t<types_space::is_std_set_v<T>, bool> = true>
        auto serialize(const T& obj, char * head, char *& cur) -> decltype(type_converter<>::convert(obj)){

            auto flattened  = flattener::flatten(obj);
            auto serialized = serialize(flattened, head, cur);

            return {serialized, obj.size()};
        }

        template <class T, std::enable_if_t<types_space::is_std_optional_v<T>, bool> = true>
        auto serialize(const T& obj, char * head, char *& cur) -> decltype(type_converter<>::convert(obj)){

            if (!obj){
                return std::nullopt;
            }

            return serialize(obj.value(), head, cur);
        }

        template <class T, std::enable_if_t<types_space::is_std_fixed_size_container_v<T>,  bool> = true>
        auto serialize(const T& obj, char * head, char *& cur) -> decltype(type_converter<>::convert(obj)){

            auto rs = decltype(type_converter<>::convert(obj))();
            auto idx_seq = std::make_index_sequence<std::tuple_size_v<T>>();

            [&]<size_t ...IDX>(const std::index_sequence<IDX...>){
                ((std::get<IDX>(rs) = serialize(std::get<IDX>(obj), head, cur)), ...);
            }(idx_seq);

            return rs;
        }

        template <class T, std::enable_if_t<std::is_arithmetic_v<T>, bool> = true>
        auto serialize(const T& obj, char * head, char *& cur) -> decltype(type_converter<>::convert(obj)){

            return obj;
        }
    };
}

namespace dg::dgbuf::std_iterator{

    template <class Container>
    class inserter: public std::iterator<std::output_iterator_tag, void, void, void, void>{

        private:

            Container * container;
        
        public:

            using self = inserter<Container>; 

            inserter(Container& container) noexcept: container(&container){}

            template <class T>
            auto operator =(T&& e) noexcept(noexcept(container->insert(std::forward<T>(e)))) -> self&{

                container->insert(std::forward<T>(e));
                return *this;
            }

            auto operator *() noexcept -> self&{

                return *this;
            }

            auto operator++() noexcept -> self&{
                
                return *this;
            }

            auto operator++(int) noexcept -> self&{

                return *this;
            }
    };

    template <class Container>
    static auto get_std_inserter(Container& container){

        if constexpr(types_space::is_std_vector_v<Container>){
            return std::back_inserter(container);
        } else if constexpr(types_space::is_std_set_v<Container> | types_space::is_std_unordered_map_v<Container> | types_space::is_std_map_v<Container> | types_space::is_std_set_v<Container>){
            inserter isrter(container);
            return isrter;
        }
    }
}

namespace dg::dgbuf::dgbuf_to_stl{

    template <class Allocator, size_t I = 0>
    struct type_converter{

        using successor = type_converter<Allocator, I + 1>;

        template <class T>
        static auto convert(datastructure::vector_view<T>) -> std::vector<decltype(successor::convert(std::declval<T>())), typename std::allocator_traits<Allocator>::template rebind_alloc<decltype(successor::convert(std::declval<T>()))>>;

        template <class Key, class Value, class Hasher, class KeyEq, class ...Args>
        static auto convert(datastructure::unordered_flat_map_view<Key, Value, Hasher, KeyEq, Args...>) -> std::unordered_map<decltype(successor::convert(std::declval<Key>())), decltype(successor::convert(std::declval<Value>())), Hasher, KeyEq, typename std::allocator_traits<Allocator>::template rebind_alloc<std::pair<const decltype(successor::convert(std::declval<Key>())), decltype(successor::convert(std::declval<Value>()))>>>;

        template <class Key, class Hasher, class KeyEq, class ...Args>
        static auto convert(datastructure::unordered_flat_set_view<Key, Hasher, KeyEq, Args...>) -> std::unordered_set<decltype(successor::convert(std::declval<Key>())), Hasher, KeyEq, typename std::allocator_traits<Allocator>::template rebind_alloc<decltype(successor::convert(std::declval<Key>()))>>;

        template <class Key, class Value, class Comparer>
        static auto convert(datastructure::map_view<Key, Value, Comparer>) -> std::map<decltype(successor::convert(std::declval<Key>())), decltype(successor::convert(std::declval<Value>())), Comparer, typename std::allocator_traits<Allocator>::template rebind_alloc<std::pair<const decltype(successor::convert(std::declval<Key>())), decltype(successor::convert(std::declval<Value>()))>>>;

        template <class Key, class Comparer>
        static auto convert(datastructure::set_view<Key, Comparer>) -> std::set<decltype(successor::convert(std::declval<Key>())), Comparer, typename std::allocator_traits<Allocator>::template rebind_alloc<decltype(successor::convert(std::declval<Key>()))>>;

        template <class T, size_t N>
        static auto convert(std::array<T, N>) -> std::array<decltype(successor::convert(std::declval<T>())), N>;

        template <class First, class Second>
        static auto convert(std::pair<First, Second>) -> std::pair<decltype(successor::convert(std::declval<First>())), decltype(successor::convert(std::declval<Second>()))>;

        template <class ...Args>
        static auto convert(std::tuple<Args...>) -> std::tuple<decltype(std::declval<Args>())...>;

        template <class T, std::enable_if_t<std::is_arithmetic_v<T>, bool> = true>
        static auto convert(T) -> T;
    };

    template <class Allocator>
    struct type_converter<Allocator, constants::MAX_TEMPLATE_RECURSIVE_DEPTH>{

        template <class T>
        static auto convert(T) -> T;
    };

    template <class Allocator = std::allocator<void>>
    struct deserializer{
        
        using converter = type_converter<Allocator, 0>; 

        template <class T, std::enable_if_t<types_space::is_dg_container_view_v<T>, bool> = true>
        auto deserialize(const T& obj, const char * buf) -> decltype(converter::convert(obj)){

            auto cp_obj = obj;  
            auto rs     = decltype(converter::convert(obj))();
            auto transformer = [&](const auto& e){return this->deserialize(e, buf);};
            cp_obj.set_buf(buf);
            std::transform(cp_obj.begin(), cp_obj.end(), std_iterator::get_std_inserter(rs), transformer);
            
            return rs;
        }

        template <class T, std::enable_if_t<types_space::is_std_optional_v<T>, bool> = true>
        auto deserialize(const T& obj, const char * buf) -> decltype(converter::convert(obj)){

            if (!obj){
                return std::nullopt;
            }

            return deserialize(obj.value(), buf);
        }

        template <class T, std::enable_if_t<types_space::is_std_fixed_size_container_v<T>, bool> = true>
        auto deserialize(const T& obj, const char * buf) -> decltype(converter::convert(obj)){

            const auto idx_seq = std::make_index_sequence<std::tuple_size_v<T>>();
            auto rs = decltype(converter::convert(obj))();

            [&]<size_t ...IDX>(const std::index_sequence<IDX...>){
                ((std::get<IDX>(rs) = deserialize(std::get<IDX>(obj), buf)), ...);
            }(idx_seq);

            return rs;
        }

        template <class T, std::enable_if_t<std::is_arithmetic_v<T>, bool> = true>
        auto deserialize(const T& obj, const char * buf) -> decltype(converter::convert(obj)){

            return obj;
        }
    };

    //what brother desparately needs was the network_kernelmap_x and my flat_datastructure
    //yeah brother - even on CPU - you need to contain the RAM usage - you don't let it CAP the RAM and die - that's the worst approach and irreversable approach EVER
    //this is the reason you want to RAID your storage - not for ingestion speed - god damn it
    //you want to allocate everything on the memregion - make sure that it's reachability is inside the memregion
    //and locality of the immutable flat_datastructure
    //#best thing - you can dispatch to CUDA which runs the jobs for you
    //that's the definition of MPP - not multithreading and concurrency
    //multithreading and concurrency are premature optimizations - NEVER to be applicable in real life - only to be used for affined tasks (like draining kernel_network_buffer) and high latency IO tasks - other than that - NEVER use concurrency to boost your flops - that's what GPU is best at - and not CPU
    //I don't have bad intentions or whatever - I tell you the optimizables that can 10x your sales - yeah - that Neo4j after you implemented this can NEVER EVER beat the benchmark
    //truth is I dont know I spent 1 year to think about the optimizables that I could have for TigerGraph
    //the moment you followed Spark and friends was a bad moment - MPP is always about GPUs
    //I think about what you thought too - it's memregion locality - node collapses - this is actually a hard task that I haven't been able to solve yet
    //the only thing that I thought of was lambda as a service - circle infected region (by running BFS algorithms and friends) - dispatch it to distributed lambdas 
    //other non-heavy tasks like simple queries can be dispatched to the normal engine

    //I, however, pursue an entire different radix of Graph. It's dg - derivative of gradients (acceleration, jerk, snap, crackle, pop, whatever) - this is some new stuff that I will spend the next 2-3 years to work on
    //you might not see what I see yet it's always about time in this tensor transformation field
    //you want to time the backprop
    //you want to time the msgrbwd

    //I thought what you thought too - why don't I just use a counter on the tile and backprop it? It's actually going to bottleneck the future architecture of dynamic pathing - and affect locality of dispatching - if you backprop it immediately after counter reaches 0 - you risk bad locality
    //the only way to solve the locality problem is to fatten the tile - which is what PyTorch has been doing - and yeah - I just reinvented PyTorch - yay
    //so it's actually about timing and reducing LOGIT_COUNT_PER_TILE yet maintaining the GPU flops
}

#endif