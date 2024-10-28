#ifndef __STD_X_H__
#define __STD_X_H__

#include <memory>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <map>
#include <set>
#include <deque>
#include <atomic>
#include <mutex>
#include <functional>

namespace stdx{
    
    using max_signed_t = __int128_t; //macro

    static inline constexpr bool IS_SAFE_MEMORY_ORDER_ENABLED       = true; 
    static inline constexpr bool IS_SAFE_INTEGER_CONVERSION_ENABLED = true;

    auto lock_guard(std::atomic_flag& lck) noexcept{

        static int i    = 0;
        auto destructor = [&](int *) noexcept{
            if constexpr(IS_SAFE_MEMORY_ORDER_ENABLED){
                std::atomic_thread_fence(std::memory_order_acq_rel);
            }
            lck.clear(std::memory_order_release);
        };

        while (!lck.test_and_set(std::memory_order_acquire)){}
        if constexpr(IS_SAFE_MEMORY_ORDER_ENABLED){
            std::atomic_thread_fence(std::memory_order_acq_rel);
        }  

        return std::unique_ptr<int, decltype(destructor)>(&i, destructor);
    }

    auto lock_guard(std::mutex& lck) noexcept{

        static int i    = 0;
        auto destructor = [&](int *) noexcept{
            if constexpr(IS_SAFE_MEMORY_ORDER_ENABLED){
                std::atomic_thread_fence(std::memory_order_acq_rel);
            }
            lck.unlock();
        };

        lck.lock();
        if constexpr(IS_SAFE_MEMORY_ORDER_ENABLED){
            std::atomic_thread_fence(std::memory_order_acq_rel);
        }

        return std::unique_ptr<int, decltype(destructor)>(&i, destructor);
    }

    template <class Destructor>
    auto resource_guard(Destructor destructor) noexcept{
        
        static_assert(std::is_nothrow_move_constructible_v<Destructor>);
        static_assert(std::is_nothrow_invocable_v<Destructor>);

        static int i    = 0;
        auto backout_ld = [destructor_arg = std::move(destructor)](int *) noexcept{
            destructor_arg();
        };

        return std::unique_ptr<int, decltype(backout_ld)>(&i, std::move(backout_ld));
    }

    template <class T, class T1>
    constexpr auto pow2mod_unsigned(T lhs, T1 rhs) noexcept -> std::conditional_t<(sizeof(T) > sizeof(T1)), T, T1>{

        static_assert(std::is_unsigned_v<T>);
        static_assert(std::is_unsigned_v<T1>);
        
        using promoted_t = std::conditional_t<(sizeof(T) > sizeof(T1)), T, T1>;
        return static_cast<promoted_t>(lhs) & static_cast<promoted_t>(rhs - 1);
    }

    template <class T, std::enable_if_t<std::is_unsigned_v<T>, bool> = true>
    constexpr auto ulog2_aligned(T val) noexcept -> size_t{

        return std::countr_zero(val);
    }

    template <class T, std::enable_if_t<std::is_unsigned_v<T>, bool> = true>
    constexpr auto ulog2(T val) noexcept -> size_t{

        return static_cast<size_t>(sizeof(T) * CHAR_BIT - 1) - static_cast<size_t>(std::countl_zero(val));
    }

    template <class T, std::enable_if_t<std::is_unsigned_v<T>, bool> = true>
    constexpr auto is_pow2(T val) noexcept -> bool{

        return val != 0u && (val & (val - 1)) == 0u;
    } 

    template <class T, std::enable_if_t<std::is_unsigned_v<T>, bool> = true>
    constexpr auto least_pow2_greater_equal_than(T val) noexcept -> T{

        if (val == 0u){ [[unlikely]]
            return 1u;
        }

        size_t max_log2     = stdx::ulog2(val);
        size_t min_log2     = std::countr_zero(val);
        size_t cand_log2    = max_log2 + ((max_log2 ^ min_log2) != 0u);

        return T{1u} << cand_log2; 
    } 

    template <class T, class T1>
    constexpr auto safe_integer_cast(T1 value) noexcept -> T{

        static_assert(std::numeric_limits<T>::is_integer);
        static_assert(std::numeric_limits<T1>::is_integer);

        if constexpr(IS_SAFE_INTEGER_CONVERSION_ENABLED){
            using promoted_t = stdx::max_signed_t; 

            static_assert(sizeof(promoted_t) > sizeof(T));
            static_assert(sizeof(promoted_t) > sizeof(T1));

            if (std::clamp(static_cast<promoted_t>(value), static_cast<promoted_t>(std::numeric_limits<T>::min()), static_cast<promoted_t>(std::numeric_limits<T>::max())) != static_cast<promoted_t>(value)){
                std::abort();
            }
        }

        return value;
    }

    template <size_t BIT_SZ, class T, std::enable_if_t<std::is_unsigned_v<T>, bool> = true>
    constexpr auto low_bit(T value) noexcept -> T{

        constexpr size_t MAX_BIT_CAP = sizeof(T) * CHAR_BIT;
        static_assert(BIT_SZ <= MAX_BIT_CAP);

        if constexpr(BIT_SZ == MAX_BIT_CAP){
            return std::numeric_limits<T>::max(); 
        } else{
            constexpr T low_mask = (T{1u} << BIT_SZ) - 1;
            return value & low_mask;
        }
    }

    template <class T>
    struct safe_integer_cast_wrapper{

        static_assert(std::numeric_limits<T>::is_integer);
        T value;

        template <class U>
        constexpr operator U() const noexcept{

            return stdx::safe_integer_cast<U>(this->value);
        }
    };

    template <class T>
    constexpr auto wrap_safe_integer_cast(T value) noexcept{

        return stdx::safe_integer_cast_wrapper<T>{value};
    }

    template <class Iterator>
    constexpr auto advance(Iterator it, intmax_t diff) noexcept -> Iterator{

        std::advance(it, diff); //I never knew what drug std was on
        return it;
    }

    auto utc_timestamp() noexcept -> std::chrono::nanoseconds{

    }

    auto unix_timestamp() noexcept -> std::chrono::nanoseconds{

    }

    template <class ...Args>
    struct vector_convertible{

        private:

            std::tuple<Args...> tup;
        
        public:

            vector_convertible(std::tuple<Args...> tup) noexcept: tup(std::move(tup)){}

            template <class ...AArgs>
            operator std::vector<AArgs...>(){

                auto rs = std::vector<AArgs...>();
                rs.reserve(sizeof...(Args));

                [&]<size_t ...IDX>(const std::index_sequence<IDX...>){
                    (
                        [&]{
                            rs.emplace_back(std::move(std::get<IDX>(this->tup)));
                        }(), ...
                    );
                }(std::make_index_sequence<sizeof...(Args)>{});

                return rs;
            }
    };

    template <class ...Args>
    auto make_vector_convertible(Args ...args) noexcept{
        
        static_assert(std::conjunction_v<std::is_nothrow_move_constructible<Args>...>);

        auto tup = std::make_tuple(std::move(args)...);
        vector_convertible rs(std::move(tup));

        return rs;
    }

    class basicstr_converitble{

        private:

            std::string_view view;

        public:

            constexpr basicstr_converitble() = default;

            constexpr basicstr_converitble(std::string_view view) noexcept: view(view){}

            template <class ...Args>
            operator std::basic_string<Args...>() const{
                
                std::basic_string<Args...> rs(view.begin(), view.end());
                return rs;
            }
    };

    auto to_basicstr_convertible(std::string_view view) noexcept -> basicstr_converitble{

        return basicstr_converitble(view);
    }


    template <class ...Args>
    auto backsplit_str(std::basic_string<Args...> s, size_t sz) -> std::pair<std::basic_string<Args...>, std::basic_string<Args...>>{

        size_t rhs_sz = std::min(s.size(), sz); 
        std::basic_string<Args...> rhs(rhs_sz, ' ');

        for (size_t i = rhs_sz; i != 0u; --i){
            rhs[i - 1] = s.back();
            s.pop_back();
        }

        return std::make_pair(std::move(s), std::move(rhs));
    }
}

#endif