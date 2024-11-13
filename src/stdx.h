#ifndef __STD_X_H__
#define __STD_X_H__

//define HEADER_CONTROL 0

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
#include <chrono>
#include "network_raii_x.h"

namespace stdx{

    struct polymorphic_launderer{
        virtual auto ptr() noexcept -> void *{
            return nullptr;
        }
    };

    template <class T>
    struct launderer{}; 

    template <>
    struct launderer<uint8_t>: polymorphic_launderer{
        uint8_t * value;

        virtual auto ptr() noexcept -> void *{
            return value;
        }
    };

    template <>
    struct launderer<uint16_t>: polymorphic_launderer{
        uint16_t * value;

        virtual auto ptr() noexcept -> void *{
            return value;
        }
    };

    template <>
    struct launderer<uint32_t>: polymorphic_launderer{
        uint32_t * value;

        virtual auto ptr() noexcept -> void *{
            return value;
        }
    };

    template <>
    struct launderer<uint64_t>: polymorphic_launderer{
        uint64_t * value;

        virtual auto ptr() noexcept -> void *{
            return value;
        }
    };

    template <>
    struct launderer<int8_t>: polymorphic_launderer{
        int8_t * value;

        virtual auto ptr() noexcept -> void *{
            return value;
        }
    };

    template <>
    struct launderer<int16_t>: polymorphic_launderer{
        int16_t * value;

        virtual auto ptr() noexcept -> void *{
            return value;
        }
    };

    template <>
    struct launderer<int32_t>: polymorphic_launderer{
        int32_t * value;

        virtual auto ptr() noexcept -> void *{
            return value;
        }
    };

    template <>
    struct launderer<int64_t>: polymorphic_launderer{
        int64_t * value;

        virtual auto ptr() noexcept -> void *{
            return value;
        }
    };

    template <>
    struct launderer<float>: polymorphic_launderer{
        float * value;

        virtual auto ptr() noexcept -> void *{
            return value;
        }
    };

    template <>
    struct launderer<double>: polymorphic_launderer{
        double * value;

        virtual auto ptr() noexcept -> void *{
            return value;
        }
    };

    template <>
    struct launderer<void>: polymorphic_launderer{
        void * value;

        virtual auto ptr() noexcept -> void *{
            return value;
        }
    };

    struct polymorphic_const_launderer{
        virtual auto ptr() noexcept -> const void *{
            return nullptr;
        }
    };

    template <class T>
    struct const_launderer{};

    template <>
    struct const_launderer<uint8_t>: polymorphic_const_launderer{
        const uint8_t * value;

        virtual auto ptr() noexcept -> const void *{

            return value;
        }
    };

    template <>
    struct const_launderer<uint16_t>: polymorphic_const_launderer{
        const uint16_t * value; 

        virtual auto ptr() noexcept -> const void *{

            return value;
        }
    };

    template <>
    struct const_launderer<uint32_t>: polymorphic_const_launderer{
        const uint32_t * value;

        virtual auto ptr() noexcept -> const void *{
            
            return value;
        }
    };

    template <>
    struct const_launderer<uint64_t>: polymorphic_const_launderer{
        const uint64_t * value;

        virtual auto ptr() noexcept -> const void *{

            return value;
        }
    };

    template <>
    struct const_launderer<int8_t>: polymorphic_const_launderer{
        const int8_t * value;

        virtual auto ptr() noexcept -> const void *{

            return value;
        } 
    };

    template <>
    struct const_launderer<int16_t>: polymorphic_const_launderer{
        const int16_t * value;

        virtual auto ptr() noexcept -> const void *{
            
            return value;
        }
    };

    template <>
    struct const_launderer<int32_t>: polymorphic_const_launderer{
        const int32_t * value;

        virtual auto ptr() noexcept -> const void *{

            return value;
        }
    };

    template <>
    struct const_launderer<int64_t>: polymorphic_const_launderer{
        const int64_t * value;

        virtual auto ptr() noexcept -> const void *{

            return value;
        }
    };

    template <>
    struct const_launderer<float>: polymorphic_const_launderer{
        const float * value;

        virtual auto ptr() noexcept -> const void *{

            return value;
        } 
    };

    template <>
    struct const_launderer<double>: polymorphic_const_launderer{
        const double * value;

        virtual auto ptr() noexcept -> const void *{

            return value;
        }
    };

    template <>
    struct const_launderer<void>: polymorphic_const_launderer{
        const void * value;

        virtual auto ptr() noexcept -> const void *{

            return value;
        }
    };

    static inline constexpr bool IS_SAFE_MEMORY_ORDER_ENABLED       = true; 
    static inline constexpr bool IS_SAFE_INTEGER_CONVERSION_ENABLED = true;

    template <class Lock>
    class xlock_guard{};

    template <>
    class xlock_guard<std::atomic_flag>{

        private:

            std::atomic_flag * volatile mtx; 

        public:

            __attribute__((always_inline)) xlock_guard(std::atomic_flag& mtx) noexcept: mtx(&mtx){

                this->mtx->test_and_set();
                std::atomic_thread_fence(std::memory_order_seq_cst);
           }

            xlock_guard(const xlock_guard&) = delete;
            xlock_guard(xlock_guard&&) = delete;

            __attribute__((always_inline)) ~xlock_guard() noexcept{

                std::atomic_thread_fence(std::memory_order_seq_cst);
                this->mtx->clear();
            }

            xlock_guard& operator =(const xlock_guard&) = delete;
            xlock_guard& operator =(xlock_guard&&) = delete;
    };

    template <>
    class xlock_guard<std::mutex>{

        private:

            std::mutex * volatile mtx;
        
        public:

            __attribute__((always_inline)) xlock_guard(std::mutex& mtx) noexcept: mtx(&mtx){

                this->mtx->lock();
                std::atomic_thread_fence(std::memory_order_seq_cst);
            }

            xlock_guard(const xlock_guard&) = delete;
            xlock_guard(xlock_guard&&) = delete;

            __attribute__((always_inline)) ~xlock_guard() noexcept{

                std::atomic_thread_fence(std::memory_order_seq_cst);
                this->mtx->unlock();
            }

            xlock_guard& operator =(const xlock_guard&) = delete;
            xlock_guard& operator =(xlock_guard&&) = delete;
    };

    //this is for deprecating volatile * (not to worry for now)
    //though std::add_volatile_t<std::add_pointer_t<T>> is more appropriate and accurate 
    
    //assume that memory fence only works if the compiler flow control reaches the fence statement - we call this consume - every variables + optimizables that in the tmp compiler buffer have to be flushed before/after the statement
    //this is not the assumption but it is actually how it works as of gcc-14 but not documented by the std
    
    //call an arbitrary concurrent tranaction __transaction__
    //within this __transaction__: radix all statements as - inline-ables and not inline-ables

    //inlineables: fine - covered by stdx::xlock_guard<polymorphic_lock_type>
    //not inlinables:   - stateless:    - depends solely on the arguments (we can assume that self is also an argument - python) - guaranteed to be covered as if we are referencing object * volatile
    //                  - stateful:     - stateful (reference inline global variables) but the states are not for concurrent usages - volatiles are not required
    //                                                                                         states are for concurrent usages - volatiles are required  
    //with volatile deprecating - seq_cst_transaction is the only suitable replacement

    //programs that adhere to the above rules are guaranteed to be defined by GCC-14

    class seq_cst_transaction{

        public:

            __attribute__((always_inline)) seq_cst_transaction() noexcept{
            
                std::atomic_signal_fence(std::memory_order_seq_cst);
            }

            seq_cst_transaction(const seq_cst_transaction&) = delete;
            seq_cst_transaction(seq_cst_transaction&&) = delete;

            __attribute__((always_inline)) ~seq_cst_transaction() noexcept{

                std::atomic_signal_fence(std::memory_order_seq_cst);
            }

            seq_cst_transaction& operator =(const seq_cst_transaction&) = delete;
            seq_cst_transaction& operator =(seq_cst_transaction&&) = delete;
    };
    
    inline __attribute__((always_inline)) void atomic_optional_thread_fence() noexcept{

        if constexpr(IS_SAFE_MEMORY_ORDER_ENABLED){
            std::atomic_thread_fence(std::memory_order_seq_cst);
        } else{
            (void) IS_SAFE_INTEGER_CONVERSION_ENABLED;
        }
    }

    //as-of gcc-14 - this launder works as if ptr is unique_ptr<void * __restrict__> and the scope of usage is the same as the callee scope
    //the implementation is defined, according to STD - this is not a Russian hack
    //extra instructions were introduced (std::atomic_signal_fence() + void * volatile) to be gcc-compiler-compliant

    template <class T>
    inline __attribute__((always_inline)) auto launder_pointer(void * volatile ptr) noexcept -> T *{

        static_assert(std::disjunction_v<std::is_same<T, uint8_t>, std::is_same<T, uint16_t>, std::is_same<T, uint32_t>, std::is_same<T, uint64_t>,
                                         std::is_same<T, int8_t>, std::is_same<T, int16_t>, std::is_same<T, int32_t>, std::is_same<T, int64_t>,
                                         std::is_same<T, float>, std::is_same<T, double>, std::is_same<T, void>, std::is_same<T, char>>);

        std::atomic_signal_fence(std::memory_order_seq_cst);
        launderer<void> launder_machine{};
        launder_machine.value = ptr;

        return static_cast<T *>(std::launder(&launder_machine)->ptr()); //this is a correct way of laundering defined by the std - force ptr() to read the polymorphic header
    }

    template <class T>
    inline __attribute__((always_inline)) auto launder_pointer(const void * volatile ptr) noexcept -> const T *{

        static_assert(std::disjunction_v<std::is_same<T, uint8_t>, std::is_same<T, uint16_t>, std::is_same<T, uint32_t>, std::is_same<T, uint64_t>,
                                         std::is_same<T, int8_t>, std::is_same<T, int16_t>, std::is_same<T, int32_t>, std::is_same<T, int64_t>,
                                         std::is_same<T, float>, std::is_same<T, double>, std::is_same<T, void>, std::is_same<T, char>>);

        std::atomic_signal_fence(std::memory_order_seq_cst);
        const_launderer<void> launder_machine{};
        launder_machine.value = ptr;

        return static_cast<const T *>(std::launder(&launder_machine)->ptr()); //this is a correct way of laundering defined by the std - force ptr() to read the polymorphic header
    }

    template <class Destructor>
    inline auto resource_guard(Destructor destructor) noexcept{
        
        static_assert(std::is_nothrow_move_constructible_v<Destructor>);
        static_assert(std::is_nothrow_invocable_v<Destructor>);
        
        auto backout_ld = [destructor_arg = std::move(destructor)](int) noexcept{
            destructor_arg();
        };

        return dg::unique_resource<int, decltype(backout_ld)>(int{0}, std::move(backout_ld));
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

    template <class T1, class T>
    constexpr auto safe_integer_cast(T value) noexcept -> T1{

        static_assert(std::numeric_limits<T>::is_integer);
        static_assert(std::numeric_limits<T1>::is_integer);

        if constexpr(IS_SAFE_INTEGER_CONVERSION_ENABLED){
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
    constexpr auto advance(Iterator it, intmax_t diff) noexcept(noexcept(std::advance(it, diff))) -> Iterator{

        static_assert(std::is_nothrow_move_constructible_v<Iterator>);
        std::advance(it, diff); //I never knew what drug std was on
        return it;
    }

    template <class T>
    struct is_chrono_dur: std::false_type{};

    template <class ...Args>
    struct is_chrono_dur<std::chrono::duration<Args...>>: std::true_type{};

    template <class T>
    static inline constexpr bool is_chrono_dur_v = is_chrono_dur<T>::value;

    struct safe_timestamp_cast_wrapper{

        std::chrono::nanoseconds caster;

        constexpr safe_timestamp_cast_wrapper(std::chrono::nanoseconds caster) noexcept: caster(std::move(caster)){}

        template <class U, std::enable_if_t<stdx::is_chrono_dur_v<U>, bool> = true>
        constexpr operator U() const noexcept{

            return std::chrono::duration_cast<U>(this->caster);
        }

        template <class U, std::enable_if_t<std::numeric_limits<U>::is_integer, bool> = true>
        constexpr operator U() const noexcept{
            auto counter = caster.count();
            static_assert(std::numeric_limits<decltype(counter)>::is_integer);
            return safe_integer_cast<U>(counter);
        }
    };

    auto utc_timestamp() noexcept -> std::chrono::nanoseconds{

        return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::utc_clock::now().time_since_epoch());
    }

    auto unix_timestamp() noexcept -> std::chrono::nanoseconds{

        return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch());
    }
    
    auto timestamp_conversion_wrap(std::chrono::nanoseconds dur) noexcept -> safe_timestamp_cast_wrapper{

        return safe_timestamp_cast_wrapper(dur);
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
    inline auto make_vector_convertible(Args ...args) noexcept -> vector_convertible<Args...>{
        
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

    inline auto to_basicstr_convertible(std::string_view view) noexcept -> basicstr_converitble{

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

    template <class ID, class T>
    class singleton{

        private:

            static inline T obj{};
        
        public:

            static inline auto get() noexcept -> T&{

                return obj;
            }
    };

    class VirtualResourceGuard{

        public:

            virtual ~VirtualResourceGuard() noexcept = default;
            virtual void release() noexcept = 0;
    };

    template <class ...Args>
    class UniquePtrVirtualGuard: public virtual VirtualResourceGuard{

        private:

            std::unique_ptr<Args...> resource;
        
        public:

            UniquePtrVirtualGuard(std::unique_ptr<Args...> resource) noexcept: resource(std::move(resource)){}

            void release() noexcept{

                static_assert(noexcept(this->resource.release()));
                this->resource.release();
            }
    };

    template <class Destructor>
    auto vresource_guard(Destructor destructor) noexcept -> std::unique_ptr<VirtualResourceGuard>{ //mem-exhaustion is not an error here - it's bad to have it as an error

        auto resource_grd = resource_guard(std::move(destructor));
        UniquePtrVirtualGuard virt_guard(std::move(resource_grd));
        return std::make_unique<decltype(virt_guard)>(std::move(virt_guard));
    }

    template <class T>
    struct hdi_container{
        alignas(std::max(std::hardware_destructive_interference_size, alignof(std::max_align_t))) T value;
    };
}

#endif
