#ifndef __DG_NETWORK_COMPACT_SERIALIZER_H__
#define __DG_NETWORK_COMPACT_SERIALIZER_H__

//define HEADER_CONTROL 2

#include <string>
#include <memory>
#include <vector>
#include <map>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <cstring>
#include <climits>
#include <bit>
#include <optional>
#include <numeric>
#include "network_hash.h"
#include <type_traits>
#include <array>

namespace dg::network_compact_serializer::constants{

    static constexpr auto endianness                                = std::endian::little;
    static constexpr bool IS_SAFE_INTEGER_CONVERSION_ENABLED        = true;
    static constexpr bool DESERIALIZATION_HAS_CLEAR_CONTAINER_RIGHT = true;
}

namespace dg::network_compact_serializer::types{

    using hash_type     = std::pair<uint64_t, uint64_t>; //alright this seems silly, we are to increase the chances of 10 ** 18 -> 10 ** 39 (never gonna happen), we actually chance better than RAM https://www.cs.toronto.edu/~bianca/papers/sigmetrics09.pdf, https://en.wikipedia.org/wiki/ECC_memory
                                                         //the only way to be successful in the career is to not trust everything, assume we are to deploy a massive 1 billion devices training system, we are gonna see errors very often
    using size_type     = uint64_t;

    using dgstd_unsigned_serialization_header_t = uint8_t;
}

namespace dg::network_compact_serializer::exception_space{

    struct corrupted_format: std::exception{

        inline auto what() const noexcept -> const char *{

            return "corrupted_format";
        }
    };
}

namespace dg::network_compact_serializer::types_space{

    static constexpr auto nil_lambda    = [](...){}; 

    template <class T, class = void>
    struct is_tuple: std::false_type{};
    
    template <class T>
    struct is_tuple<T, std::void_t<decltype(std::tuple_size<T>::value)>>: std::true_type{};

    template <class T>
    struct is_unique_ptr: std::false_type{};

    template <class T>
    struct is_unique_ptr<std::unique_ptr<T>>: std::bool_constant<!std::is_array_v<T>>{}; 

    template <class T>
    struct is_optional: std::false_type{};

    template <class ...Args>
    struct is_optional<std::optional<Args...>>: std::true_type{}; 

    template <class T>
    struct is_vector: std::false_type{};

    template <class ...Args>
    struct is_vector<std::vector<Args...>>: std::true_type{};

    template <class T>
    struct is_unordered_map: std::false_type{};

    template <class ...Args>
    struct is_unordered_map<std::unordered_map<Args...>>: std::true_type{}; 

    template <class T>
    struct is_unordered_set: std::false_type{};

    template <class ...Args>
    struct is_unordered_set<std::unordered_set<Args...>>: std::true_type{};

    template <class T>
    struct is_map: std::false_type{};

    template <class ...Args>
    struct is_map<std::map<Args...>>: std::true_type{}; 

    template <class T>
    struct is_set: std::false_type{};

    template <class ...Args>
    struct is_set<std::set<Args...>>: std::true_type{};

    template <class T>
    struct is_basic_string: std::false_type{};

    template <class ...Args>
    struct is_basic_string<std::basic_string<Args...>>: std::true_type{};

    template <class T, class = void>
    struct is_reflectible: std::false_type{};

    template <class T>
    struct is_reflectible<T, std::void_t<decltype(std::declval<T>().dg_reflect(nil_lambda))>>: std::true_type{};

    template <class T, class = void>
    struct is_dg_arithmetic: std::disjunction<std::is_integral<T>, std::is_floating_point<T>>{};

    template <class T>
    struct is_dg_arithmetic<T, std::void_t<std::enable_if_t<std::is_floating_point_v<T>>>>: std::bool_constant<std::numeric_limits<T>::is_iec559>{}; 
    
    template <class T, class = void>
    struct can_reserve: std::false_type{};

    template <class T>
    struct can_reserve<T, std::void_t<decltype(std::declval<T>().reserve(std::declval<size_t>()))>>: std::true_type{};

    template <class T>
    struct is_byte_stream_container: std::false_type{};

    template <class ...Args>
    struct is_byte_stream_container<std::vector<char, Args...>>: std::true_type{};

    template <class ...Args>
    struct is_byte_stream_container<std::basic_string<char, Args...>>: std::true_type{};

    template <class T, class = void>
    struct container_value_or_empty{};

    template <class T>
    struct container_value_or_empty<T, std::void_t<typename T::value_type>>{
        using type = typename T::value_type;
    };

    template <class T, class = void>
    struct container_bucket_or_empty{};

    template <class T>
    struct container_bucket_or_empty<T, std::void_t<typename T::key_type, typename T::mapped_type>>{
        using type = std::pair<typename T::key_type, typename T::mapped_type>; //I dont want to complicate this further by adding const to key_type (since this is an application) - 
    };

    template <class T>
    using containee_or_none_t = std::conditional_t<std::disjunction_v<is_vector<T>, is_unordered_set<T>, is_set<T>, is_basic_string<T>>,
                                                                      container_value_or_empty<T>,
                                                                      std::conditional_t<std::disjunction_v<is_unordered_map<T>, is_map<T>>, 
                                                                                         container_bucket_or_empty<T>, 
                                                                                         void>>;

    template <class T>
    using containee_t = typename containee_or_none_t<T>::type;

    template <class T, class = void>
    struct containee_or_empty{
        using type = void;
    };

    template <class T>
    struct containee_or_empty<T, std::void_t<containee_t<T>>>{
        using type = containee_t<T>;
    };

    template <class T>
    using containee_or_empty_t = typename containee_or_empty<T>::type; 

    //see: https://en.cppreference.com/w/cpp/language/types

    template <class T>
    static inline constexpr bool has_unique_serializable_representations_v              = std::disjunction_v<std::is_same<T, int8_t>, std::is_same<T, uint8_t>, 
                                                                                                             std::is_same<T, char>, std::is_same<T, unsigned char>, std::is_same<T, signed char>>;

    template <class T>
    static inline constexpr bool has_unique_serializable_sameendian_representations_v   = std::disjunction_v<std::is_same<T, int8_t>, std::is_same<T, uint8_t>, 
                                                                                                             std::is_same<T, char>, std::is_same<T, unsigned char>, std::is_same<T, signed char>,
                                                                                                             std::is_same<T, int16_t>, std::is_same<T, uint16_t>, 
                                                                                                             std::is_same<T, int32_t>, std::is_same<T, uint32_t>, 
                                                                                                             std::is_same<T, int64_t>, std::is_same<T, uint64_t>>;
    template <class T>
    static inline constexpr bool is_vector_v                                            = is_vector<T>::value;

    template <class T>
    static inline constexpr bool is_basic_string_v                                      = is_basic_string<T>::value;
    
    template <class T>
    static inline constexpr bool is_map_v                                               = is_map<T>::value;

    template <class T>
    static inline constexpr bool is_unordered_map_v                                     = is_unordered_map<T>::value;

    template <class T>
    static inline constexpr bool is_set_v                                               = is_set<T>::value;

    template <class T>
    static inline constexpr bool is_unordered_set_v                                     = is_unordered_set<T>::value;

    template <class T>
    static inline constexpr bool is_cpyable_linear_container_v                          = std::disjunction_v<is_vector<T>, is_basic_string<T>> && (has_unique_serializable_representations_v<containee_or_empty_t<T>> || has_unique_serializable_sameendian_representations_v<containee_or_empty_t<T>> && (std::endian::native == constants::endianness)); //this requires inter-compatible with noncpyable_linear

    template <class T>
    static inline constexpr bool is_noncpyable_linear_container_v                       = std::disjunction_v<is_vector<T>, is_basic_string<T>> && !is_cpyable_linear_container_v<T>;

    template <class T>
    static inline constexpr bool is_nonlinear_container_v                               = std::disjunction_v<is_unordered_map<T>, is_map<T>, is_unordered_set<T>, is_set<T>>;

    template <class T>
    static inline constexpr bool is_tuple_v                                             = is_tuple<T>::value; 

    template <class T>
    static inline constexpr bool is_unique_ptr_v                                        = is_unique_ptr<T>::value;

    template <class T>
    static inline constexpr bool is_optional_v                                          = is_optional<T>::value;

    template <class T>
    static inline constexpr bool is_reflectible_v                                       = is_reflectible<T>::value;

    template <class T>
    static inline constexpr bool is_dg_arithmetic_v                                     = is_dg_arithmetic<T>::value;

    template <class T>
    static inline constexpr bool is_byte_stream_container_v                             = is_byte_stream_container<T>::value;

    template <class T>
    static inline constexpr bool can_reserve_v                                          = can_reserve<T>::value;

    template <class T>
    struct base_type: std::enable_if<true, T>{};

    //alright, I dont really know if this is future-proof, let's make it defined by using defined use-cases for now, we dont have time to iterate through every possibility
    template <class T>
    struct base_type<const T>: base_type<T>{};

    template <class T>
    struct base_type<volatile T>: base_type<T>{};

    template <class T>
    struct base_type<T&>: base_type<T>{};

    template <class T>
    struct base_type<T&&>: base_type<T>{};

    template <class T>
    using base_type_t = typename base_type<T>::type;
}

namespace dg::network_compact_serializer::utility{

    using namespace network_compact_serializer::types;

    template <class = void>
    static inline constexpr bool FALSE_VAL = false;

    struct SyncedEndiannessService{
        
        static constexpr auto is_native_big      = bool{std::endian::native == std::endian::big};
        static constexpr auto is_native_little   = bool{std::endian::native == std::endian::little};
        static constexpr auto precond            = bool{(is_native_big ^ is_native_little) != 0};
        static constexpr auto deflt              = constants::endianness; 
        static constexpr auto native_uint8       = is_native_big ? uint8_t{0} : uint8_t{1}; 

        static_assert(precond); //xor

        template <class T, std::enable_if_t<std::disjunction_v<std::is_integral<T>, std::is_floating_point<T>>, bool> = true>
        static constexpr T bswap(T value){

            auto src = std::array<char, sizeof(T)>{};
            auto dst = std::array<char, sizeof(T)>{};
            const auto idx_seq  = std::make_index_sequence<sizeof(T)>();

            src = std::bit_cast<decltype(src)>(value);
            [&]<size_t ...IDX>(const std::index_sequence<IDX...>){
                ((dst[IDX] = src[sizeof(T) - IDX - 1]), ...);
            }(idx_seq);
            value = std::bit_cast<T>(dst);

            return value;
        }

        template <class T, std::enable_if_t<std::disjunction_v<std::is_integral<T>, std::is_floating_point<T>>, bool> = true>
        static constexpr void dump(char * dst, T data) noexcept{    

            if constexpr(std::endian::native != deflt){
                data = bswap(data);
            }

            std::array<char, sizeof(T)> data_buf{};
            data_buf = std::bit_cast<decltype(data_buf)>(data);
            memcpy(dst, data_buf.data(), std::integral_constant<size_t, sizeof(T)>());
        }

        template <class T, std::enable_if_t<std::disjunction_v<std::is_integral<T>, std::is_floating_point<T>>, bool> = true>
        static constexpr T load(const char * src) noexcept{

            std::array<char, sizeof(T)> tmp{};
            memcpy(tmp.data(), src, std::integral_constant<size_t, sizeof(T)>());
            T rs = std::bit_cast<T>(tmp);

            if constexpr(std::endian::native != deflt){
                rs = bswap(rs);
            }

            return rs;
        }
    };

    auto hash(const char * buf, size_t sz, uint32_t secret) noexcept -> hash_type{

        return dg::network_hash::murmur_hash_base(buf, sz, secret);
    }

    template <class T, std::enable_if_t<std::disjunction_v<types_space::is_vector<T>, 
                                                           types_space::is_basic_string<T>>, bool> = true>
    constexpr auto get_inserter() noexcept{

        auto inserter   = []<class U, class K>(U&& container, K&& arg){
            container.push_back(std::forward<K>(arg));
        };

        return inserter;
    }

    template <class T, std::enable_if_t<std::disjunction_v<types_space::is_unordered_map<T>, 
                                                           types_space::is_unordered_set<T>, 
                                                           types_space::is_map<T>, 
                                                           types_space::is_set<T>>, bool> = true>
    constexpr auto get_inserter() noexcept{ 

        auto inserter   = []<class U, class K>(U&& container, K&& args){
            container.insert(std::forward<K>(args));
        };

        return inserter;
    }

    template <class T>
    constexpr void reserve_if_possible(T&& container, size_t sz){

        if constexpr(types_space::can_reserve_v<T&&>){
            container.reserve(sz);
        }
    }

    template <class T1, class T>
    constexpr auto safe_integer_cast(T value) noexcept -> T1{

        static_assert(std::numeric_limits<T>::is_integer);
        static_assert(std::numeric_limits<T1>::is_integer);

        if constexpr(constants::IS_SAFE_INTEGER_CONVERSION_ENABLED){
            if constexpr(std::is_unsigned_v<T> && std::is_unsigned_v<T1>){
                (void) value;
            } else if constexpr(std::is_signed_v<T> && std::is_signed_v<T1>){
                (void) value;
            } else{
                if constexpr(std::is_signed_v<T>){
                    if constexpr(sizeof(T) > sizeof(T1)){
                        (void) value;
                    } else{
                        if (value < 0){
                            std::abort();
                        } else{
                            return value; //sizeof(signed) <= sizeof(unsigned)
                        }
                    }
                } else{
                    if constexpr(sizeof(T1) > sizeof(T)){
                        (void) value;
                    } else{
                        if (value > std::numeric_limits<T1>::max()){
                            std::abort();
                        } else{
                            return value; //sizeof(unsigned) >= sizeof(signed)
                        }
                    }
                }
            }

            if (value > std::numeric_limits<T1>::max()){
                std::abort();
            }

            if (value < std::numeric_limits<T1>::min()){
                std::abort();
            }

            return value;
        } else{
            return value;
        }
    }
}

namespace dg::network_compact_serializer::archive{

    struct Counter{
        
        using Self = Counter;

        template <class T, std::enable_if_t<types_space::is_dg_arithmetic_v<types_space::base_type_t<T>>, bool> = true>
        constexpr auto count(T&& data) const noexcept -> size_t{
            
            return sizeof(types_space::base_type_t<T>);
        }

        template <class T, std::enable_if_t<types_space::is_unique_ptr_v<types_space::base_type_t<T>>, bool> = true>
        constexpr auto count(T&& data) const noexcept -> size_t{

            size_t rs = this->count(bool{});

            if (data){
                rs += this->count(*data);
            }

            return rs;
        }

        template <class T, std::enable_if_t<types_space::is_optional_v<types_space::base_type_t<T>>, bool> = true>
        constexpr auto count(T&& data) const noexcept -> size_t{

            size_t rs = this->count(bool{}); 

            if (data){
                rs += this->count(*data);
            }

            return rs;
        }

        template <class T, std::enable_if_t<types_space::is_tuple_v<types_space::base_type_t<T>>, bool> = true>
        constexpr auto count(T&& data) const noexcept -> size_t{

            using base_type     = types_space::base_type_t<T>;
            const auto idx_seq  = std::make_index_sequence<std::tuple_size_v<base_type>>{};
            size_t rs           = 0u;

            [&rs]<size_t ...IDX>(T&& data, const std::index_sequence<IDX...>) noexcept{
                rs += (Self().count(std::get<IDX>(data)) + ...);
            }(std::forward<T>(data), idx_seq);

            return rs;
        }

        template <class T, std::enable_if_t<types_space::is_noncpyable_linear_container_v<types_space::base_type_t<T>> || types_space::is_nonlinear_container_v<types_space::base_type_t<T>>, bool> = true>
        constexpr auto count(T&& data) const noexcept -> size_t{

            size_t rs = this->count(types::size_type{});

            for (const auto& e: data){
                rs += this->count(e);
            }

            return rs;
        }

        template <class T, std::enable_if_t<types_space::is_cpyable_linear_container_v<types_space::base_type_t<T>>, bool> = true>
        constexpr auto count(T&& data) const noexcept -> size_t{

            return this->count(types::size_type{}) + static_cast<size_t>(data.size()) * sizeof(types_space::containee_t<types_space::base_type_t<T>>);
        }

        template <class T, std::enable_if_t<types_space::is_reflectible_v<types_space::base_type_t<T>>, bool> = true>
        constexpr auto count(T&& data) const noexcept -> size_t{

            size_t rs       = 0u;
            auto archiver   = [&rs]<class ...Args>(Args&& ...args) noexcept{
                rs += (Self().count(std::forward<Args>(args)) + ...);
            };
            data.dg_reflect(archiver);
            
            return rs;
        }
    };

    struct Forward{

        using Self = Forward;

        template <class T, std::enable_if_t<types_space::is_dg_arithmetic_v<types_space::base_type_t<T>>, bool> = true>
        constexpr void put(char *& buf, T&& data) const noexcept{
            
            network_compact_serializer::utility::SyncedEndiannessService::dump(buf, std::forward<T>(data));
            std::advance(buf, sizeof(types_space::base_type_t<T>));
        }

        template <class T, std::enable_if_t<types_space::is_unique_ptr_v<types_space::base_type_t<T>>, bool> = true>
        constexpr void put(char *& buf, T&& data) const noexcept{

            this->put(buf, static_cast<bool>(data));

            if (data){
                this->put(buf, *data);
            }
        }

        template <class T, std::enable_if_t<types_space::is_optional_v<types_space::base_type_t<T>>, bool> = true>
        constexpr void put(char *& buf, T&& data) const noexcept{

            this->put(buf, static_cast<bool>(data));

            if (data){
                this->put(buf, *data);
            }
        }

        template <class T, std::enable_if_t<types_space::is_tuple_v<types_space::base_type_t<T>>, bool> = true>
        constexpr void put(char *& buf, T&& data) const noexcept{

            using base_type     = types_space::base_type_t<T>;
            const auto idx_seq  = std::make_index_sequence<std::tuple_size_v<base_type>>{};

            []<size_t ...IDX>(char *& buf, T&& data, const std::index_sequence<IDX...>) noexcept{
                (Self().put(buf, std::get<IDX>(data)), ...);
            }(buf, std::forward<T>(data), idx_seq);
        }

        template <class T, std::enable_if_t<types_space::is_noncpyable_linear_container_v<types_space::base_type_t<T>> || types_space::is_nonlinear_container_v<types_space::base_type_t<T>>, bool> = true>
        constexpr void put(char *& buf, T&& data) const noexcept{

            this->put(buf, network_compact_serializer::utility::safe_integer_cast<types::size_type>(data.size()));

            for (const auto& e: data){
                this->put(buf, e);
            }
        }

        template <class T, std::enable_if_t<types_space::is_cpyable_linear_container_v<types_space::base_type_t<T>>, bool> = true>
        constexpr void put(char *& buf, T&& data) const noexcept{

            using base_type = types_space::base_type_t<T>;
            using elem_type = types_space::containee_t<base_type>;

            this->put(buf, network_compact_serializer::utility::safe_integer_cast<types::size_type>(data.size()));

            void * dst          = buf;
            const void * src    = data.data();
            size_t cpy_sz       = data.size() * sizeof(elem_type);  

            std::memcpy(dst, src, cpy_sz);
            std::advance(buf, cpy_sz);
        }

        template <class T, std::enable_if_t<types_space::is_reflectible_v<types_space::base_type_t<T>>, bool> = true>
        constexpr void put(char *& buf, T&& data) const noexcept{

            auto archiver = [&buf]<class ...Args>(Args&& ...args) noexcept{
                (Self().put(buf, std::forward<Args>(args)), ...);
            };

            data.dg_reflect(archiver);
        }
    };

    struct Backward{

        using Self = Backward;

        template <class T, std::enable_if_t<types_space::is_dg_arithmetic_v<types_space::base_type_t<T>>, bool> = true>
        constexpr void put(const char *& buf, T&& data) const{

            using base_type = types_space::base_type_t<T>;
            data = network_compact_serializer::utility::SyncedEndiannessService::load<base_type>(buf);
            std::advance(buf, sizeof(base_type));
        }

        template <class T, std::enable_if_t<types_space::is_unique_ptr_v<types_space::base_type_t<T>>, bool> = true>
        constexpr void put(const char *& buf, T&& data) const{

            using containee_type = typename types_space::base_type_t<T>::element_type;
            bool status = {};
            this->put(buf, status);

            if (status){
                containee_type obj;
                this->put(buf, obj);
                data = std::make_unique<containee_type>(std::move(obj));
            } else{
                data = nullptr;
            }
        }

        template <class T, std::enable_if_t<types_space::is_optional_v<types_space::base_type_t<T>>, bool> = true>
        constexpr void put(const char *& buf, T&& data) const{

            using containee_type = typename types_space::base_type_t<T>::value_type;
            bool status = {};
            this->put(buf, status);

            if (status){
                containee_type obj;
                this->put(buf, obj);
                data = std::optional<containee_type>(std::in_place_t{}, std::move(obj));
            } else{
                data = std::nullopt;
            }
        }

        template <class T, std::enable_if_t<types_space::is_tuple_v<types_space::base_type_t<T>>, bool> = true>
        constexpr void put(const char *& buf, T&& data) const{

            using base_type     = types_space::base_type_t<T>;
            const auto idx_seq  = std::make_index_sequence<std::tuple_size_v<base_type>>{};

            []<size_t ...IDX>(const char *& buf, T&& data, const std::index_sequence<IDX...>){
                (Self().put(buf, std::get<IDX>(data)), ...);
            }(buf, std::forward<T>(data), idx_seq);
        }

        template <class T, std::enable_if_t<types_space::is_noncpyable_linear_container_v<types_space::base_type_t<T>> || types_space::is_nonlinear_container_v<types_space::base_type_t<T>>, bool> = true>
        constexpr void put(const char *& buf, T&& data) const{
            
            using base_type = types_space::base_type_t<T>;
            using elem_type = types_space::containee_t<base_type>;

            //we are very tempted to do a clear operation as hinted by other programmers as bugs, we'll add this feature as optional
            //yet if they invoked deserialization on an unempty container, it is already a bug (all kinds of bugs ranging from leak bugs to memory corruption bugs to memory exhaustion bugs, to etc)
            //it's very super complicated to add a clear operation, we've yet to want to do so, we rather make a new container, put to the new container and do a move operation
            //every deserialization to an unempty container is already undefined
            //this is precisely why this is called compact serializer, we dont want to add features that wont be used

            auto sz         = types::size_type{};
            auto isrter     = network_compact_serializer::utility::get_inserter<base_type>();
            this->put(buf, sz);

            if constexpr(dg::network_compact_serializer::constants::DESERIALIZATION_HAS_CLEAR_CONTAINER_RIGHT){
                data.clear();
            }

            // data.reserve(sz);
            dg::network_compact_serializer::utility::reserve_if_possible(data, sz);

            for (size_t i = 0; i < sz; ++i){
                elem_type e;
                this->put(buf, e);
                isrter(data, std::move(e));
            }
        }

        template <class T, std::enable_if_t<types_space::is_cpyable_linear_container_v<types_space::base_type_t<T>>, bool> = true>
        constexpr void put(const char *& buf, T&& data) const{

            using base_type = types_space::base_type_t<T>;
            using elem_type = types_space::containee_t<base_type>;
            
            auto sz = types::size_type{};
            this->put(buf, sz);
            data.resize(sz);

            void * dst          = data.data();
            const void * src    = buf;
            size_t cpy_sz       = sz * sizeof(elem_type); 

            std::memcpy(dst, src, cpy_sz);
            std::advance(buf, cpy_sz);
        }

        template <class T, std::enable_if_t<types_space::is_reflectible_v<types_space::base_type_t<T>>, bool> = true>
        constexpr void put(const char *& buf, T&& data) const{

            auto archiver = [&buf]<class ...Args>(Args&& ...args){
                (Self().put(buf, std::forward<Args>(args)), ...);
            };

            data.dg_reflect(archiver);
        }
    };

    struct SafeBackward{

        using Self = SafeBackward;

        template <class T, std::enable_if_t<types_space::is_dg_arithmetic_v<types_space::base_type_t<T>>, bool> = true>
        constexpr void put(const char *& buf, size_t& buf_sz, T&& data) const{

            using base_type = types_space::base_type_t<T>;

            if (buf_sz < sizeof(base_type)){
                throw dg::network_compact_serializer::exception_space::corrupted_format();
            }

            data    = network_compact_serializer::utility::SyncedEndiannessService::load<base_type>(buf);
            std::advance(buf, sizeof(base_type));
            buf_sz  -= sizeof(base_type);
        }

        template <class T, std::enable_if_t<types_space::is_unique_ptr_v<types_space::base_type_t<T>>, bool> = true>
        constexpr void put(const char *& buf, size_t& buf_sz, T&& data) const{

            using containee_type = typename types_space::base_type_t<T>::element_type;
            bool status = {};
            this->put(buf, buf_sz, status);

            if (status){
                containee_type obj;
                this->put(buf, buf_sz, obj);
                data = std::make_unique<containee_type>(std::move(obj));
            } else{
                data = nullptr;
            }
        }

        template <class T, std::enable_if_t<types_space::is_optional_v<types_space::base_type_t<T>>, bool> = true>
        constexpr void put(const char *& buf, size_t& buf_sz, T&& data) const{

            using containee_type = typename types_space::base_type_t<T>::value_type;
            bool status = {};
            this->put(buf, buf_sz, status);

            if (status){
                containee_type obj;
                this->put(buf, buf_sz, obj);
                data = std::optional<containee_type>(std::in_place_t{}, std::move(obj));
            } else{
                data = std::nullopt;
            }
        }

        template <class T, std::enable_if_t<types_space::is_tuple_v<types_space::base_type_t<T>>, bool> = true>
        constexpr void put(const char *& buf, size_t& buf_sz, T&& data) const{

            using base_type     = types_space::base_type_t<T>;
            const auto idx_seq  = std::make_index_sequence<std::tuple_size_v<base_type>>{};

            []<size_t ...IDX>(const char *& buf, size_t& buf_sz, T&& data, const std::index_sequence<IDX...>){
                (Self().put(buf, buf_sz, std::get<IDX>(data)), ...);
            }(buf, buf_sz, std::forward<T>(data), idx_seq);
        }

        template <class T, std::enable_if_t<types_space::is_noncpyable_linear_container_v<types_space::base_type_t<T>> || types_space::is_nonlinear_container_v<types_space::base_type_t<T>>, bool> = true>
        constexpr void put(const char *& buf, size_t& buf_sz, T&& data) const{

            using base_type = types_space::base_type_t<T>;
            using elem_type = types_space::containee_t<base_type>;

            auto sz         = types::size_type{};
            auto isrter     = network_compact_serializer::utility::get_inserter<base_type>();
            this->put(buf, buf_sz, sz); 

            if constexpr(dg::network_compact_serializer::constants::DESERIALIZATION_HAS_CLEAR_CONTAINER_RIGHT){
                data.clear();
            }

            // data.reserve(sz);
            dg::network_compact_serializer::utility::reserve_if_possible(data, sz);

            for (size_t i = 0; i < sz; ++i){
                elem_type e;
                this->put(buf, buf_sz, e);
                isrter(data, std::move(e));
            }
        }

        template <class T, std::enable_if_t<types_space::is_cpyable_linear_container_v<types_space::base_type_t<T>>, bool> = true>
        constexpr void put(const char *& buf, size_t& buf_sz, T&& data) const{

            using base_type = types_space::base_type_t<T>;
            using elem_type = types_space::containee_t<base_type>;
            
            auto sz = types::size_type{};
            this->put(buf, buf_sz, sz);
            data.resize(sz);

            void * dst          = data.data();
            const void * src    = buf;
            size_t cpy_sz       = sz * sizeof(elem_type); 

            if (buf_sz < cpy_sz){
                throw dg::network_compact_serializer::exception_space::corrupted_format();
            }

            std::memcpy(dst, src, cpy_sz);
            std::advance(buf, cpy_sz);
            buf_sz -= cpy_sz;
        }

        template <class T, std::enable_if_t<types_space::is_reflectible_v<types_space::base_type_t<T>>, bool> = true>
        void put(const char *& buf, size_t& buf_sz, T&& data) const{

            auto archiver = [&buf, &buf_sz]<class ...Args>(Args&& ...args){
                (Self().put(buf, buf_sz, std::forward<Args>(args)), ...);
            };

            data.dg_reflect(archiver);
        }
    };

    template <class = void>
    static inline constexpr bool FALSE_VAL = false;

    template <class T>
    static consteval auto get_dgstd_serialization_header() noexcept -> types::dgstd_unsigned_serialization_header_t{

        if constexpr(std::is_integral_v<T>){
            if constexpr(std::is_signed_v<T>){
                if constexpr(sizeof(T) == 1u){
                    return 201u;
                } else if constexpr(sizeof(T) == 2u){
                    return 202u;
                } else if constexpr(sizeof(T) == 4u){
                    return 203u;
                } else if constexpr(sizeof(T) == 8u){
                    return 204u;
                } else if constexpr(sizeof(T) == 16u){
                    return 205u;
                } else if constexpr(sizeof(T) == 32u){
                    return 206u;
                } else{
                    static_assert(FALSE_VAL<>);
                }
            } else if constexpr(std::is_unsigned_v<T>){
                if constexpr(sizeof(T) == 1u){
                    return 207u;
                } else if constexpr(sizeof(T) == 2u){
                    return 208u;
                } else if constexpr(sizeof(T) == 4u){
                    return 209u;
                } else if constexpr(sizeof(T) == 8u){
                    return 210u;
                } else if constexpr(sizeof(T) == 16u){
                    return 211u;
                } else if constexpr(sizeof(T) == 32u){
                    return 212u;
                } else{
                    static_assert(FALSE_VAL<>);
                }
            } else{
                static_assert(FALSE_VAL<>);
            }
        } else if constexpr(std::is_floating_point_v<T>){
            if constexpr(std::numeric_limits<T>::is_iec559){
                if constexpr(sizeof(T) == 1u){
                    return 213u;
                } else if constexpr(sizeof(T) == 2u){
                    return 214u; 
                } else if constexpr(sizeof(T) == 4u){
                    return 215u;
                } else if constexpr(sizeof(T) == 8u){
                    return 216u;
                } else if constexpr(sizeof(T) == 16u){
                    return 217u;
                } else if constexpr(sizeof(T) == 32u){
                    return 218u;
                } else if constexpr(sizeof(T) == 64u){
                    return 219u;
                } else{
                    static_assert(FALSE_VAL<>);
                }
            } else{
                static_assert(FALSE_VAL<>);
            }
        } else if constexpr(types_space::is_vector_v<T>){
            return 220u;
        } else if constexpr(types_space::is_basic_string_v<T>){
            return 221u;
        } else if constexpr(types_space::is_map_v<T>){
            return 222u;
        } else if constexpr(types_space::is_unordered_map_v<T>){
            return 223u;
        } else if constexpr(types_space::is_set_v<T>){
            return 224u;
        } else if constexpr(types_space::is_unordered_set_v<T>){
            return 225u;
        } else if constexpr(types_space::is_tuple_v<T>){
            return 226u;
        } else if constexpr(types_space::is_unique_ptr_v<T>){
            return 227u;
        } else if constexpr(types_space::is_optional_v<T>){
            return 228u;
        } else if constexpr(types_space::is_reflectible_v<T>){
            return 229u;
        } else{
            static_assert(FALSE_VAL<>);
        }
    }

    struct DgStdCounter{

        using Self = DgStdCounter;

        template <class T, std::enable_if_t<types_space::is_dg_arithmetic_v<types_space::base_type_t<T>>, bool> = true>
        constexpr auto count(T&& data) const noexcept -> size_t{

            return sizeof(types::dgstd_unsigned_serialization_header_t) + sizeof(types_space::base_type_t<T>);
        }

        template <class T, std::enable_if_t<types_space::is_unique_ptr_v<types_space::base_type_t<T>>, bool> = true>
        constexpr auto count(T&& data) const noexcept -> size_t{

            size_t rs = this->count(types::dgstd_unsigned_serialization_header_t{}) + this->count(bool{});

            if (data){
                rs += this->count(*data);
            }

            return rs;
        }

        template <class T, std::enable_if_t<types_space::is_optional_v<types_space::base_type_t<T>>, bool> = true>
        constexpr auto count(T&& data) const noexcept -> size_t{

            size_t rs = this->count(types::dgstd_unsigned_serialization_header_t{}) + this->count(bool{}); 

            if (data){
                rs += this->count(*data);
            }

            return rs;
        }

        template <class T, std::enable_if_t<types_space::is_tuple_v<types_space::base_type_t<T>>, bool> = true>
        constexpr auto count(T&& data) const noexcept -> size_t{

            using base_type     = types_space::base_type_t<T>;
            const auto idx_seq  = std::make_index_sequence<std::tuple_size_v<base_type>>{};

            size_t rs           = this->count(types::dgstd_unsigned_serialization_header_t{});

            [&rs]<size_t ...IDX>(T&& data, const std::index_sequence<IDX...>) noexcept{
                rs += (Self().count(std::get<IDX>(data)) + ...);
            }(std::forward<T>(data), idx_seq);

            return rs;
        }

        template <class T, std::enable_if_t<types_space::is_noncpyable_linear_container_v<types_space::base_type_t<T>> || types_space::is_nonlinear_container_v<types_space::base_type_t<T>>, bool> = true>
        constexpr auto count(T&& data) const noexcept -> size_t{

            size_t rs = this->count(types::dgstd_unsigned_serialization_header_t{}) + this->count(types::size_type{});

            for (const auto& e: data){
                rs += this->count(e);
            }

            return rs;
        }

        template <class T, std::enable_if_t<types_space::is_cpyable_linear_container_v<types_space::base_type_t<T>>, bool> = true>
        constexpr auto count(T&& data) const noexcept -> size_t{

            return this->count(types::dgstd_unsigned_serialization_header_t{})
                   + this->count(types::dgstd_unsigned_serialization_header_t{})  
                   + this->count(types::size_type{})
                   + static_cast<size_t>(data.size()) * sizeof(types_space::containee_t<types_space::base_type_t<T>>);
        }

        template <class T, std::enable_if_t<types_space::is_reflectible_v<types_space::base_type_t<T>>, bool> = true>
        constexpr auto count(T&& data) const noexcept -> size_t{

            size_t rs = this->count(types::dgstd_unsigned_serialization_header_t{});

            auto archiver   = [&rs]<class ...Args>(Args&& ...args) noexcept{
                rs += (Self().count(std::forward<Args>(args)) + ...);
            };

            data.dg_reflect(archiver);

            return rs;
        }
    };

    //the problem we cant get over is the false positive, which we can decrease by using a begin end transaction format
    //or we'd want to reduce the chances by increasing the header types width and kind of randomize the values 
    //this is proven to be the most effective way for this type

    //the adding the header would only to serve to reduce the false positive (maybe true positive, which we dont care)

    struct DgStdForward{

        using Self = DgStdForward;

        template <class T, std::enable_if_t<types_space::is_dg_arithmetic_v<types_space::base_type_t<T>>, bool> = true>
        constexpr void put(char *& buf, T&& data) const noexcept{

            using base_type = types_space::base_type_t<T>; 

            types::dgstd_unsigned_serialization_header_t serialization_header = get_dgstd_serialization_header<base_type>();

            network_compact_serializer::utility::SyncedEndiannessService::dump(buf, serialization_header);
            std::advance(buf, sizeof(types::dgstd_unsigned_serialization_header_t));

            network_compact_serializer::utility::SyncedEndiannessService::dump(buf, std::forward<T>(data));
            std::advance(buf, sizeof(base_type));
        }

        template <class T, std::enable_if_t<types_space::is_unique_ptr_v<types_space::base_type_t<T>>, bool> = true>
        constexpr void put(char *& buf, T&& data) const noexcept{

            using base_type = types_space::base_type_t<T>;

            this->put(buf, get_dgstd_serialization_header<base_type>());
            this->put(buf, static_cast<bool>(data));

            if (data){
                this->put(buf, *data);
            }
        }

        template <class T, std::enable_if_t<types_space::is_optional_v<types_space::base_type_t<T>>, bool> = true>
        constexpr void put(char *& buf, T&& data) const noexcept{

            using base_type = types_space::base_type_t<T>; 

            this->put(buf, get_dgstd_serialization_header<base_type>());
            this->put(buf, static_cast<bool>(data));

            if (data){
                this->put(buf, *data);
            }
        }

        template <class T, std::enable_if_t<types_space::is_tuple_v<types_space::base_type_t<T>>, bool> = true>
        constexpr void put(char *& buf, T&& data) const noexcept{

            using base_type     = types_space::base_type_t<T>;
            const auto idx_seq  = std::make_index_sequence<std::tuple_size_v<base_type>>{};

            this->put(buf, get_dgstd_serialization_header<base_type>());

            []<size_t ...IDX>(char *& buf, T&& data, const std::index_sequence<IDX...>) noexcept{
                (Self().put(buf, std::get<IDX>(data)), ...);
            }(buf, std::forward<T>(data), idx_seq);
        }

        template <class T, std::enable_if_t<types_space::is_noncpyable_linear_container_v<types_space::base_type_t<T>> || types_space::is_nonlinear_container_v<types_space::base_type_t<T>>, bool> = true>
        constexpr void put(char *& buf, T&& data) const noexcept{

            using base_type     = types_space::base_type_t<T>;

            this->put(buf, get_dgstd_serialization_header<base_type>());
            this->put(buf, network_compact_serializer::utility::safe_integer_cast<types::size_type>(data.size()));

            for (const auto& e: data){
                this->put(buf, e);
            }
        }

        template <class T, std::enable_if_t<types_space::is_cpyable_linear_container_v<types_space::base_type_t<T>>, bool> = true>
        constexpr void put(char *& buf, T&& data) const noexcept{

            using base_type = types_space::base_type_t<T>;
            using elem_type = types_space::containee_t<base_type>;

            this->put(buf, get_dgstd_serialization_header<base_type>());
            this->put(buf, get_dgstd_serialization_header<elem_type>());
            this->put(buf, network_compact_serializer::utility::safe_integer_cast<types::size_type>(data.size()));

            void * dst          = buf;
            const void * src    = data.data();
            size_t cpy_sz       = data.size() * sizeof(elem_type);  

            std::memcpy(dst, src, cpy_sz);
            std::advance(buf, cpy_sz);
        }

        template <class T, std::enable_if_t<types_space::is_reflectible_v<types_space::base_type_t<T>>, bool> = true>
        constexpr void put(char *& buf, T&& data) const noexcept{

            using base_type = types_space::base_type_t<T>;

            this->put(buf, get_dgstd_serialization_header<base_type>());

            auto archiver = [&buf]<class ...Args>(Args&& ...args) noexcept{
                (Self().put(buf, std::forward<Args>(args)), ...);
            };

            data.dg_reflect(archiver);
        }
    };

    struct DgStdBackward{

        using Self = DgStdBackward;

        template <class T, std::enable_if_t<types_space::is_dg_arithmetic_v<types_space::base_type_t<T>>, bool> = true>
        constexpr void put(const char *& buf, size_t& buf_sz, T&& data) const{

            using base_type = types_space::base_type_t<T>;

            if (buf_sz < sizeof(base_type) + sizeof(types::dgstd_unsigned_serialization_header_t)){
                throw dg::network_compact_serializer::exception_space::corrupted_format();
            }

            types::dgstd_unsigned_serialization_header_t expected_header    = network_compact_serializer::utility::SyncedEndiannessService::load<types::dgstd_unsigned_serialization_header_t>(buf);
            std::advance(buf, sizeof(types::dgstd_unsigned_serialization_header_t));
            buf_sz                                                          -= sizeof(types::dgstd_unsigned_serialization_header_t);
            types::dgstd_unsigned_serialization_header_t header             = get_dgstd_serialization_header<base_type>(); 

            if (header != expected_header){
                throw dg::network_compact_serializer::exception_space::corrupted_format();
            }

            data    = network_compact_serializer::utility::SyncedEndiannessService::load<base_type>(buf);
            std::advance(buf, sizeof(base_type));
            buf_sz  -= sizeof(base_type);
        }

        template <class T, std::enable_if_t<types_space::is_unique_ptr_v<types_space::base_type_t<T>>, bool> = true>
        constexpr void put(const char *& buf, size_t& buf_sz, T&& data) const{

            using base_type         = types_space::base_type_t<T>;
            using containee_type    = typename types_space::base_type_t<T>::element_type;

            types::dgstd_unsigned_serialization_header_t expected_header    = {};
            this->put(buf, buf_sz, expected_header);
            types::dgstd_unsigned_serialization_header_t header             = get_dgstd_serialization_header<base_type>();

            if (header != expected_header){
                throw dg::network_compact_serializer::exception_space::corrupted_format();
            }

            bool status = {};
            this->put(buf, buf_sz, status);

            if (status){
                containee_type obj;
                this->put(buf, buf_sz, obj);
                data = std::make_unique<containee_type>(std::move(obj));
            } else{
                data = nullptr;
            }
        }

        template <class T, std::enable_if_t<types_space::is_optional_v<types_space::base_type_t<T>>, bool> = true>
        constexpr void put(const char *& buf, size_t& buf_sz, T&& data) const{

            using base_type         = types_space::base_type_t<T>;
            using containee_type    = typename types_space::base_type_t<T>::value_type;

            types::dgstd_unsigned_serialization_header_t expected_header    = {};
            this->put(buf, buf_sz, expected_header);
            types::dgstd_unsigned_serialization_header_t header             = get_dgstd_serialization_header<base_type>();

            if (header != expected_header){
                throw dg::network_compact_serializer::exception_space::corrupted_format();
            }

            bool status = {};
            this->put(buf, buf_sz, status);

            if (status){
                containee_type obj;
                this->put(buf, buf_sz, obj);
                data = std::optional<containee_type>(std::in_place_t{}, std::move(obj));
            } else{
                data = std::nullopt;
            }
        }

        template <class T, std::enable_if_t<types_space::is_tuple_v<types_space::base_type_t<T>>, bool> = true>
        void put(const char *& buf, size_t& buf_sz, T&& data) const{

            using base_type     = types_space::base_type_t<T>;
            const auto idx_seq  = std::make_index_sequence<std::tuple_size_v<base_type>>{};

            types::dgstd_unsigned_serialization_header_t expected_header    = {};
            this->put(buf, buf_sz, expected_header);
            types::dgstd_unsigned_serialization_header_t header             = get_dgstd_serialization_header<base_type>();

            if (header != expected_header){
                throw dg::network_compact_serializer::exception_space::corrupted_format();
            }

            []<size_t ...IDX>(const char *& buf, size_t& buf_sz, T&& data, const std::index_sequence<IDX...>){
                (Self().put(buf, buf_sz, std::get<IDX>(data)), ...);
            }(buf, buf_sz, std::forward<T>(data), idx_seq);
        }

        template <class T, std::enable_if_t<types_space::is_noncpyable_linear_container_v<types_space::base_type_t<T>> || types_space::is_nonlinear_container_v<types_space::base_type_t<T>>, bool> = true>
        constexpr void put(const char *& buf, size_t& buf_sz, T&& data) const{

            using base_type = types_space::base_type_t<T>;
            using elem_type = types_space::containee_t<base_type>;

            types::dgstd_unsigned_serialization_header_t expected_container_header  = {};
            this->put(buf, buf_sz, expected_container_header);
            types::dgstd_unsigned_serialization_header_t container_header           = get_dgstd_serialization_header<base_type>();

            if (container_header != expected_container_header){
                throw dg::network_compact_serializer::exception_space::corrupted_format();
            }

            auto sz         = types::size_type{};
            auto isrter     = network_compact_serializer::utility::get_inserter<base_type>();
            this->put(buf, buf_sz, sz); 

            if constexpr(dg::network_compact_serializer::constants::DESERIALIZATION_HAS_CLEAR_CONTAINER_RIGHT){
                data.clear();
            }

            // data.reserve(sz);
            dg::network_compact_serializer::utility::reserve_if_possible(data, sz);

            for (size_t i = 0; i < sz; ++i){
                elem_type e;
                this->put(buf, buf_sz, e);
                isrter(data, std::move(e));
            }
        }

        template <class T, std::enable_if_t<types_space::is_cpyable_linear_container_v<types_space::base_type_t<T>>, bool> = true>
        constexpr void put(const char *& buf, size_t& buf_sz, T&& data) const{

            using base_type = types_space::base_type_t<T>;
            using elem_type = types_space::containee_t<base_type>;

            types::dgstd_unsigned_serialization_header_t expected_container_header  = {};
            this->put(buf, buf_sz, expected_container_header);
            types::dgstd_unsigned_serialization_header_t container_header           = get_dgstd_serialization_header<base_type>();

            if (container_header != expected_container_header){
                throw dg::network_compact_serializer::exception_space::corrupted_format();
            }

            types::dgstd_unsigned_serialization_header_t expected_elem_header       = {};
            this->put(buf, buf_sz, expected_elem_header);
            types::dgstd_unsigned_serialization_header_t elem_header                = get_dgstd_serialization_header<elem_type>();

            if (elem_header != expected_elem_header){
                throw dg::network_compact_serializer::exception_space::corrupted_format();
            }

            auto sz = types::size_type{};
            this->put(buf, buf_sz, sz);
            data.resize(sz);

            void * dst          = data.data();
            const void * src    = buf;
            size_t cpy_sz       = sz * sizeof(elem_type); 

            if (buf_sz < cpy_sz){
                throw dg::network_compact_serializer::exception_space::corrupted_format();
            }

            std::memcpy(dst, src, cpy_sz);
            std::advance(buf, cpy_sz);
            buf_sz -= cpy_sz;
        }

        template <class T, std::enable_if_t<types_space::is_reflectible_v<types_space::base_type_t<T>>, bool> = true>
        constexpr void put(const char *& buf, size_t& buf_sz, T&& data) const{

            using base_type = types_space::base_type_t<T>;

            types::dgstd_unsigned_serialization_header_t expected_header    = {};
            this->put(buf, buf_sz, expected_header);
            types::dgstd_unsigned_serialization_header_t container_header   = get_dgstd_serialization_header<base_type>();

            if (container_header != expected_header){
                throw dg::network_compact_serializer::exception_space::corrupted_format();
            }

            auto archiver = [&buf, &buf_sz]<class ...Args>(Args&& ...args){
                (Self().put(buf, buf_sz, std::forward<Args>(args)), ...);
            };

            data.dg_reflect(archiver);
        }
    };
}

namespace dg::network_compact_serializer{

    template <class T>
    constexpr auto size(const T& obj) noexcept -> size_t{

        return dg::network_compact_serializer::archive::Counter{}.count(obj);
    }

    template <class T>
    constexpr auto serialize_into(char * buf, const T& obj) noexcept -> char *{

        dg::network_compact_serializer::archive::Forward{}.put(buf, obj);
        return buf;
    } 

    template <class T>
    constexpr auto deserialize_into(T& obj, const char * buf) -> const char *{

        dg::network_compact_serializer::archive::Backward{}.put(buf, obj);
        return buf;
    }

    template <class T>
    constexpr auto integrity_size(const T& obj) noexcept -> size_t{

        return network_compact_serializer::size(obj) + network_compact_serializer::size(types::hash_type{});
    }

    template <class T>
    constexpr auto integrity_serialize_into(char * buf, const T& obj, uint32_t secret = 0u) noexcept -> char *{ 

        char * first                = buf;
        char * last                 = dg::network_compact_serializer::serialize_into(first, obj);
        types::hash_type hashed     = dg::network_compact_serializer::utility::hash(first, std::distance(first, last), secret);
        char * llast                = dg::network_compact_serializer::serialize_into(last, hashed);

        return llast;
    }

    template <class T>
    constexpr void integrity_deserialize_into(T& obj, const char * buf, size_t sz, uint32_t secret = 0u){

        if (sz < dg::network_compact_serializer::size(types::hash_type{})){
            throw dg::network_compact_serializer::exception_space::corrupted_format();
        }

        size_t content_sz           = sz - dg::network_compact_serializer::size(types::hash_type{});
        const char * first          = buf;
        const char * last           = std::next(first, content_sz);
        types::hash_type expecting  = {};
        types::hash_type reality    = dg::network_compact_serializer::utility::hash(first, content_sz, secret);

        dg::network_compact_serializer::deserialize_into(expecting, last);

        if (expecting != reality){
            throw dg::network_compact_serializer::exception_space::corrupted_format();
        }

        {
            const char * iterating_ptr  = first;
            size_t deserializing_sz     = content_sz;
            dg::network_compact_serializer::archive::SafeBackward{}.put(iterating_ptr, deserializing_sz, obj);

            if (deserializing_sz != 0u){
                throw dg::network_compact_serializer::exception_space::corrupted_format();
            }
        }
    }

    template <class T>
    constexpr auto capintegrity_size(const T& obj) noexcept -> size_t{

        return dg::network_compact_serializer::size(uint64_t{}) + dg::network_compact_serializer::integrity_size(obj);
    }

    template <class T>
    constexpr auto capintegrity_serialize_into(char * buf, const T& obj, uint32_t secret = 0u) noexcept -> char *{

        char * first    = std::next(buf, dg::network_compact_serializer::size(uint64_t{}));
        char * last     = dg::network_compact_serializer::integrity_serialize_into(first, obj, secret);
        uint64_t sz     = std::distance(first, last);
        dg::network_compact_serializer::serialize_into(buf, sz); 

        return last;
    }

    template <class T>
    constexpr auto capintegrity_deserialize_into(T& obj, const char * buf, size_t sz, uint32_t secret = 0u) -> const char *{

        if (sz < dg::network_compact_serializer::size(uint64_t{})){
            throw dg::network_compact_serializer::exception_space::corrupted_format();
        }

        uint64_t obj_sz     = {};
        const char * first  = dg::network_compact_serializer::deserialize_into(obj_sz, buf);
        dg::network_compact_serializer::integrity_deserialize_into(obj, first, obj_sz, secret);

        return std::next(first, obj_sz);
    }

    template <class T>
    constexpr auto dgstd_size(const T& obj) noexcept -> size_t{

        return dg::network_compact_serializer::archive::DgStdCounter{}.count(obj) + network_compact_serializer::size(types::hash_type{});
    }

    template <class T>
    constexpr auto dgstd_serialize_into(char * buf, const T& obj, uint32_t secret = 0u) noexcept -> char *{

        char * first                = buf;
        char * last                 = first;
        dg::network_compact_serializer::archive::DgStdForward{}.put(last, obj);
        types::hash_type hashed     = dg::network_compact_serializer::utility::hash(first, std::distance(first, last), secret);
        char * llast                = dg::network_compact_serializer::serialize_into(last, hashed);

        return llast;
    }

    template <class T>
    constexpr void dgstd_deserialize_into(T& obj, const char * buf, size_t sz, uint32_t secret = 0u){

        if (sz < dg::network_compact_serializer::size(types::hash_type{})){
            throw dg::network_compact_serializer::exception_space::corrupted_format();
        }

        size_t content_sz                   = sz - dg::network_compact_serializer::size(types::hash_type{});
        const char * content_first          = buf;
        const char * content_last           = std::next(content_first, content_sz);
        const char * hash_first             = content_last;
        types::hash_type expecting_value    = {};
        types::hash_type reality_value      = dg::network_compact_serializer::utility::hash(content_first, content_sz, secret);

        dg::network_compact_serializer::deserialize_into(expecting_value, hash_first);

        if (expecting_value != reality_value){
            throw dg::network_compact_serializer::exception_space::corrupted_format();
        }

        {
            const char * iterating_ptr  = content_first;
            size_t deserializing_sz     = content_sz;
            dg::network_compact_serializer::archive::DgStdBackward{}.put(iterating_ptr, deserializing_sz, obj);

            if (deserializing_sz != 0u){
                throw dg::network_compact_serializer::exception_space::corrupted_format();
            }
        }
    }

    template <class Stream, class T, std::enable_if_t<types_space::is_byte_stream_container_v<Stream>, bool> = true>
    constexpr auto serialize(const T& obj) -> Stream{

        Stream stream{};
        stream.resize(dg::network_compact_serializer::size(obj));
        dg::network_compact_serializer::serialize_into(stream.data(), obj);

        return stream;
    }

    template <class T, class Stream, std::enable_if_t<types_space::is_byte_stream_container_v<Stream>, bool> = true>
    constexpr auto deserialize(const Stream& stream) -> T{

        T rs;
        dg::network_compact_serializer::deserialize_into(rs, stream.data());

        return rs;
    }

    template <class Stream, class T, std::enable_if_t<types_space::is_byte_stream_container_v<Stream>, bool> = true>
    constexpr auto integrity_serialize(const T& obj, uint32_t secret = 0u) -> Stream{

        Stream stream{};
        stream.resize(dg::network_compact_serializer::integrity_size(obj));
        dg::network_compact_serializer::integrity_serialize_into(stream.data(), obj, secret);
        
        return stream;
    }

    template <class T, class Stream, std::enable_if_t<types_space::is_byte_stream_container_v<Stream>, bool> = true>
    constexpr auto integrity_deserialize(const Stream& stream, uint32_t secret = 0u) -> T{

        T rs;
        dg::network_compact_serializer::integrity_deserialize_into(rs, stream.data(), stream.size(), secret);

        return rs;
    }

    template <class Stream, class T, std::enable_if_t<types_space::is_byte_stream_container_v<Stream>, bool> = true>
    constexpr auto dgstd_serialize(const T& obj, uint32_t secret = 0u) -> Stream{

        Stream stream{};
        stream.resize(dg::network_compact_serializer::dgstd_size(obj));
        dg::network_compact_serializer::dgstd_serialize_into(stream.data(), obj);

        return stream;
    }

    template <class T, class Stream, std::enable_if_t<types_space::is_byte_stream_container_v<Stream>, bool> = true>
    constexpr auto dgstd_deserialize(const Stream& stream, uint32_t secret = 0u) -> T{

        T rs;
        dg::network_compact_serializer::dgstd_deserialize_into(rs, stream.data(), stream.size(), secret);

        return rs;
    }
}

#endif