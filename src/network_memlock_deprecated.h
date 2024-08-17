
namespace dg::network_memlock::utility{

    template <class T, size_t BIT_SZ, std::enable_if_t<std::is_unsigned_v<T>, bool> = true>
    static constexpr auto low(const std::integral_constant<size_t, BIT_SZ>) -> T{

        if constexpr(BIT_SZ == sizeof(T) * CHAR_BIT){
            return ~T{0u};
        } else{
            return (T{1} << BIT_SZ) - 1;
        }
    }

    template <class CallBack, class First, class Second, class ...Args>
    static void insertion_sort(const CallBack& callback, First first, Second second, Args ...args){
        
        if constexpr(sizeof...(Args) == 0){
            callback(std::min(first, second), std::max(first, second));
        } else{
            auto cb_lambda = [=]<class ...AArgs>(AArgs ...aargs){
                callback(std::min(first, second), aargs...);
            };

            insertion_sort(cb_lambda, std::max(first, second), args...);
        }
    } 

    template <class CallBack, class First, class ...Args>
    static void template_sort(const CallBack& callback, First first, Args ...args){

        if constexpr(sizeof...(Args) == 0){
            callback(first);
        } else{
            auto cb_lambda  = [=]<class ...AArgs>(AArgs ...aargs){
                insertion_sort(callback, first, aargs...);
            };

            template_sort(cb_lambda, args...);
        }
    }

    template <class _Ty, size_t SZ_Arg>
    static void template_sort_arr(_Ty * first, const std::integral_constant<size_t, SZ_Arg>&){

        auto sort_cb    = [=]<class ...Args>(Args ...args){
            
            auto fwd_tup        = std::make_tuple(args...);
            const auto idx_seq  = std::make_index_sequence<sizeof...(Args)>{};

            [=]<class Tup, size_t ...IDX>(Tup&& tup, const std::index_sequence<IDX...>&){
                ((first[IDX]  = std::get<IDX>(tup)), ...);
            }(fwd_tup, idx_seq);

        };

        const auto idx_seq = std::make_index_sequence<SZ_Arg>{};

        [=]<size_t ...IDX>(const std::index_sequence<IDX...>&){
            template_sort(sort_cb, first[IDX]...);
        }(idx_seq);
    }
}


    struct acquire_tag{
        const void * ptr;
    };

    struct reference_tag{
        const void * ptr;
    };

    constexpr auto tag_acquire(const void * ptr) noexcept -> acquire_tag{

        return {ptr};
    }

    constexpr auto tag_reference(const void * ptr) noexcept -> reference_tag{

        return {ptr};
    }

    template <class T, class = void>
    struct is_acqref: std::false_type{};

    template <class T>
    struct is_acqref<T, std::void_t<std::enable_if_t<std::is_same_v<acquire_tag, T> || std::is_same_v<reference_tag, T>, bool>>>: std::true_type{};

    template <class T>
    struct is_acq_tag: std::false_type{};

    template <>
    struct is_acq_tag<acquire_tag>: std::true_type{}; 

    template <class T>
    struct is_ref_tag: std::false_type{};

    template <>
    struct is_ref_tag<reference_tag>: std::true_type{};

    template <class ...LHS_Args, class ...RHS_Args>
    static constexpr auto is_acqref_tup_pair_compatible_func(std::tuple<LHS_Args...> l, std::tuple<RHS_Args...> r){

        if constexpr(sizeof...(LHS_Args) != sizeof...(RHS_Args)){
            return std::bool_constant<false>{};
        } else{
            constexpr bool is_same_pair_type = [=]<size_t ...IDX>(std::index_sequence<IDX...>){
                return std::conjunction_v<std::is_same<decltype(std::get<IDX>(l)), decltype(std::get<IDX>(r))>...>;
            }(std::make_index_sequence<sizeof...(LHS_Args)>{});

            constexpr bool is_non_zero = sizeof...(LHS_Args) != 0;
            constexpr bool is_acqref_type = std::conjunction_v<is_acqref<LHS_Args>...>;
            constexpr bool rs = is_non_zero & is_same_pair_type & is_acqref_type;

            return std::bool_constant<rs>{};
        }
    } 

    template <class LHS, class RHS>
    struct is_acqref_tup_pair_compatible: std::false_type{};

    template <class ...LHS_Args, class ...RHS_Args>
    struct is_acqref_tup_pair_compatible<std::tuple<LHS_Args...>, std::tuple<RHS_Args...>>: decltype(is_acqref_tup_pair_compatible_func(std::declval<std::tuple<LHS_Args...>>(), std::declval<std::tuple<RHS_Args...>>())){};

    template <class T>
    static inline constexpr bool is_acq_tag_v = is_acq_tag<T>::value;

    template <class T>
    static inline constexpr bool is_ref_tag_v = is_ref_tag<T>::value;

    template <class LTup, class RTup>
    static inline constexpr bool is_acqref_tup_pair_compatible_v = is_acqref_tup_pair_compatible<LTup, RTup>::value;


    template <class T>
    struct MemoryReferenceLockXPInterface{

        using interface_t = MemoryReferenceLockXPInterface<T>;

        template <class ...Args, std::enable_if_t<std::conjunction_v<std::is_same<Args, const void *>...> && (sizeof...(Args) > 0), bool> = true>
        static inline auto acquire_try_many(Args ...args) noexcept -> bool{

            return T::acquire_try_many(args...);
        }

        template <class ...Args, std::enable_if_t<std::conjunction_v<std::is_same<Args, const void *>...> && (sizeof...(Args) > 0), bool> = true>
        static inline void acquire_wait_many(Args ...args) noexcept{

            T::acquire_wait_many(args...);
        }

        template <class ...Args, std::enable_if_t<std::conjunction_v<std::is_same<Args, const void *>...> && (sizeof...(Args) > 0), bool> = true>
        static inline void acquire_release_many(Args ...args) noexcept{

            T::acquire_release_many(args...);
        }

        template <size_t SZ, std::enable_if_t<(SZ > 0), bool> = true>
        static inline void acquire_transfer_wait_many(std::array<const void *, SZ> dst, std::array<const void *, SZ> src) noexcept{

            T::acquire_transfer_wait_many(dst, src);
        }

        template <class ...Args, std::enable_if_t<std::conjunction_v<std::is_same<Args, const void *>...> && (sizeof...(Args) > 0), bool> = true>
        static inline auto reference_try_many(Args ...args) noexcept -> bool{

            return T::reference_try_many(args...);
        }
        
        template <class ...Args, std::enable_if_t<std::conjunction_v<std::is_same<Args, const void *>...> && (sizeof...(Args) > 0), bool> = true>
        static inline void reference_wait_many(Args ...args) noexcept{

            T::reference_wait_many(args...);
        }

        template <class ...Args, std::enable_if_t<std::conjunction_v<std::is_same<Args, const void *>...> && (sizeof...(Args) > 0), bool> = true>
        static inline void reference_release_many(Args ...args) noexcept{

            T::reference_release_many(args...);
        }

        template <size_t SZ, std::enable_if_t<(SZ > 0), bool> = true>
        static inline void reference_transfer_wait_many(std::array<const void *, SZ> dst, std::array<const void *, SZ> src) noexcept{

            T::reference_transfer_wait_many(dst, src);
        }

        template <size_t SZ, std::enable_if_t<(SZ > 0), bool> = true>
        static inline auto transfer_try_many(std::array<const void *, SZ> dst, std::array<const void *, SZ> src) noexcept -> bool{

            return T::transfer_try_many(dst, src);
        }
    };

    template <class T>
    struct MemoryAcquireReferenceLockInterface{

        using interface_t = MemoryAcquireReferenceLockInterface<T>;

        template <class ...Args, std::enable_if_t<std::conjunction_v<is_acqref<Args>...> && (sizeof...(Args) > 0), bool> = true>
        static inline auto acqref_try_many(Args ...args) noexcept -> bool{

            return T::acqref_try_many(args...);
        }

        template <class ...Args, std::enable_if_t<std::conjunction_v<is_acqref<Args>...> && (sizeof...(Args) > 0), bool> = true>
        static inline void acqref_wait_many(Args ...args) noexcept{

            T::acqref_wait_many(args...);
        }

        template <class ...Args, std::enable_if_t<std::conjunction_v<is_acqref<Args>...> && (sizeof...(Args) > 0), bool> = true>
        static inline void release_many(Args ...args) noexcept{

            T::release_many(args...);
        }

        template <size_t SZ, std::enable_if_t<(SZ > 0), bool> = true>
        static inline auto transfer_try_many(std::array<const void *, SZ> dst, std::array<const void *, SZ> src) noexcept -> bool{

            return T::transfer_try_many(dst, src);
        }

        template <class ...LHS_Args, class ...RHS_Args, std::enable_if_t<is_acqref_tup_pair_compatible_v<std::tuple<LHS_Args...>, std::tuple<RHS_Args...>>, bool> = true>
        static inline auto transfer_try_many(std::tuple<LHS_Args...> dst, std::tuple<RHS_Args...> src) noexcept -> bool{

            return T::transfer_try_many(dst, src);
        }

        template <class ...LHS_Args, class ...RHS_Args, std::enable_if_t<is_acqref_tup_pair_compatible_v<std::tuple<LHS_Args...>, std::tuple<RHS_Args...>>, bool> = true>
        static inline void transfer_wait_many(std::tuple<LHS_Args...> dst, std::tuple<RHS_Args...> src) noexcept{

            T::transfer_wait_many(dst, src);
        }
    };

        template <class ID, class T, class SubRegionSize>
    struct HierarchicalMemoryReferenceLock{};

    template <class ID, class T, size_t SUB_REGION_SZ>
    struct HierarchicalMemoryReferenceLock<ID, MemoryReferenceLockInterface<T>, std::integral_constant<size_t, SUB_REGION_SZ>>: MemoryReferenceLockInterface<HierarchicalMemoryReferenceLock<ID, MemoryReferenceLockInterface<T>, std::integral_constant<size_t, SUB_REGION_SZ>>>{

        private:
            
            using self      = HierarchicalMemoryReferenceLock;
            using parent    = MemoryReferenceLockInterface<T>;
            using base      = MemoryReferenceLock<ID, std::integral_constant<size_t, SUB_REGION_SZ>>;

        public:

            static void init(const void * buf, size_t buf_sz) noexcept{

                base::init(buf, buf_sz);
            }

            static inline auto transfer_try(const void * new_ptr, const void * old_ptr) noexcept -> bool{
                
                return base::transfer_try(new_ptr, old_ptr);
            }

            static inline auto acquire_try(const void * ptr) noexcept -> bool{
                
                if (!parent::reference_try(ptr)){
                    return false;
                }

                if (!base::acquire_try(ptr)){
                    parent::reference_release(ptr);
                    return false;
                }

                return true;
            }

            static inline void acquire_wait(const void * ptr) noexcept{

                parent::reference_wait(ptr);
                base::acquire_wait(ptr);
            }

            static inline void acquire_release(const void * ptr) noexcept{

                base::acquire_release(ptr);
                parent::reference_release(ptr);
            }

            static inline void acquire_transfer_wait(const void * new_ptr, const void * old_ptr) noexcept{

                if (transfer_try(new_ptr, old_ptr)){
                    return;
                }

                acquire_release(old_ptr);
                acquire_wait(new_ptr);
            }

            static inline auto reference_try(const void * ptr) noexcept -> bool{

                if (!parent::reference_try(ptr)){
                    return false;
                }

                if (!base::reference_try(ptr)){
                    parent::reference_release(ptr);
                    return false;
                }

                return true;
            }

            static inline void reference_wait(const void * ptr) noexcept{

                parent::reference_wait(ptr);
                base::reference_wait(ptr);
            } 

            static inline void reference_release(const void * ptr) noexcept{

                base::reference_release(ptr);
                parent::reference_release(ptr);
            }

            static inline void reference_transfer_wait(const void * new_ptr, const void * old_ptr) noexcept{

                if (transfer_try(new_ptr, old_ptr)){
                    return;
                }

                reference_release(old_ptr);
                reference_wait(new_ptr);
            }
    };



    template <class T, class MaxArgSize>
    struct MemoryReferenceLockX_Unique{}; 

    template <class T, size_t MAX_ARG_SIZE>
    struct MemoryReferenceLockX_Unique<MemoryReferenceLockInterface<T>, std::integral_constant<size_t, MAX_ARG_SIZE>>: MemoryReferenceLockXInterface<MemoryReferenceLockX_Unique<MemoryReferenceLockInterface<T>, std::integral_constant<size_t, MAX_ARG_SIZE>>>{

        using base = MemoryReferenceLockInterface<T>;

        static inline auto transfer_try_many(const void ** dst, const void ** src) noexcept -> bool{
            
            while (*dst){
                if (!base::transfer_try(*dst, *src)){
                    return false;
                }

                ++dst;
                ++src;
            }

            return true;
        }

        static inline auto acquire_try_many(const void ** ptrs) noexcept -> bool{

            std::array<const void *, MAX_ARG_SIZE> acquired_arr{};
            const void ** last = acquired_arr.data();

            while (*ptrs){
                if (!base::acquire_try(*ptrs)){
                    break;
                }

                *(last++) = *(ptrs++);
            }

            if (!static_cast<bool>(*ptrs)){
                return true;
            }
            
            for (auto i = acquired_arr.data(); i != last; ++i){
                base::acquire_release(*i);
            }

            return false;
        }

        static inline void acquire_wait_many(const void ** ptrs) noexcept{

            while (!acquire_try_many(ptrs)){};
        }

        static inline void acquire_release_many(const void ** ptrs) noexcept{

            while (*ptrs){
                base::acquire_release(*ptrs);
                ++ptrs;
            }
        } 

        static inline void acquire_transfer_wait_many(const void ** dst, const void ** src) noexcept{

            if (transfer_try_many(dst, src)){
                return;
            }

            acquire_release_many(src);
            acquire_wait_many(dst);
        }

        static inline auto reference_try_many(const void ** ptrs) noexcept -> bool{

            std::array<const void *, MAX_ARG_SIZE> acquired_arr{};
            const void ** last = acquired_arr.data();

            while (*ptrs){
                if (!base::reference_try(*ptrs)){
                    break;
                }

                *(last++) = *(ptrs++);
            }

            if (!static_cast<bool>(*ptrs)){
                return true;
            }
            
            for (auto i = acquired_arr.data(); i != last; ++i){
                base::reference_release(*i);
            }
            
            return false;
        }

        static inline void reference_wait_many(const void ** ptrs) noexcept{

            while (!reference_try_many(ptrs)){}
        }

        static inline void reference_release_many(const void ** ptrs) noexcept{

            while (*ptrs){
                base::reference_release(*ptrs);
                ++ptrs;
            }
        } 

        static inline void reference_transfer_wait_many(const void ** dst, const void ** src) noexcept{

            if (transfer_try_many(dst, src)){
                return;
            }

            reference_release_many(src);
            reference_wait_many(dst);
        }
    };

    template <class T>
    struct MemoryReferenceLockXP_Unique{}; 

    template <class T>
    struct MemoryReferenceLockXP_Unique<MemoryReferenceLockInterface<T>>: MemoryReferenceLockXPInterface<MemoryReferenceLockXP_Unique<MemoryReferenceLockInterface<T>>>{

        using base = MemoryReferenceLockInterface<T>;

        template <size_t SZ, std::enable_if_t<(SZ > 0), bool> = true>
        static inline auto transfer_try_many(std::array<const void *, SZ> dst, std::array<const void *, SZ> src) noexcept -> bool{

            return [&]<size_t ...IDX>(const std::index_sequence<IDX...>){
                return (base::transfer_try(dst[IDX], src[IDX]) && ...);   
            }(std::make_index_sequence<SZ>{});        
        }

        template <class ...Args, std::enable_if_t<std::conjunction_v<std::is_same<Args, const void *>...> && (sizeof...(Args) > 0), bool> = true>
        static inline auto acquire_try_many(Args ...args) noexcept -> bool{

            std::array<const void *, sizeof...(Args)> acquired_arr{};
            std::array<const void *, sizeof...(Args)> args_arr{args...};
            const void ** last = acquired_arr.data();
            size_t j{};

            for (j = 0; j < sizeof...(Args); ++j){
                if (!base::acquire_try(args_arr[j])){
                    break;
                }

                *(last++) = args_arr[j];
            }

            if (j == sizeof...(Args)){
                return true;
            }

            for (auto i = acquired_arr.data(); i != last; ++i){
                base::acquire_release(*i);
            }

            return false; 
        }

        template <class ...Args, std::enable_if_t<std::conjunction_v<std::is_same<Args, const void *>...> && (sizeof...(Args) > 0), bool> = true>
        static inline void acquire_wait_many(Args ...args) noexcept{

            while (!acquire_try_many(args...));
        }
        
        template <class ...Args, std::enable_if_t<std::conjunction_v<std::is_same<Args, const void *>...> && (sizeof...(Args) > 0), bool> = true>
        static inline void acquire_release_many(Args ...args) noexcept{

            ((base::acquire_release(args)), ...);
        }

        template <size_t SZ, std::enable_if_t<(SZ > 0), bool> = true>
        static inline void acquire_transfer_wait_many(std::array<const void *, SZ> dst, std::array<const void *, SZ> src) noexcept{

            if (transfer_try_many(dst, src)){
                return;
            }

            [=]<size_t ...IDX>(const std::index_sequence<IDX...>){
                acquire_release_many(src[IDX]...);
                acquire_wait_many(dst[IDX]...);
            }(std::make_index_sequence<SZ>{});        
        }

        template <class ...Args, std::enable_if_t<std::conjunction_v<std::is_same<Args, const void *>...> && (sizeof...(Args) > 0), bool> = true>
        static inline auto reference_try_many(Args ...args) noexcept -> bool{

            std::array<const void *, sizeof...(Args)> acquired_arr{};
            std::array<const void *, sizeof...(Args)> args_arr{args...};
            const void ** last = acquired_arr.data();
            size_t j{}; 

            for (j = 0; j < sizeof...(Args); ++j){
                if (!base::reference_try(args_arr[j])){
                    break;
                }

                *(last++) = args_arr[j];
            }

            if (j == sizeof...(Args)){
                return true;
            }

            for (auto i = acquired_arr.data(); i != last; ++i){
                base::reference_release(*i);
            }

            return false;       
        }

        template <class ...Args, std::enable_if_t<std::conjunction_v<std::is_same<Args, const void *>...> && (sizeof...(Args) > 0), bool> = true>
        static inline void reference_wait_many(Args ...args) noexcept{

            while (!reference_try_many(args...)){}
        }

        template <class ...Args, std::enable_if_t<std::conjunction_v<std::is_same<Args, const void *>...> && (sizeof...(Args) > 0), bool> = true>
        static inline void reference_release_many(Args ...args) noexcept{

            ((base::reference_release(args)), ...);
        }

        template <size_t SZ, std::enable_if_t<(SZ > 0), bool> = true>
        static inline void reference_transfer_wait_many(std::array<const void *, SZ> dst, std::array<const void *, SZ> src) noexcept{

            if (transfer_try_many(dst, src)){
                return;
            }

            [=]<size_t ...IDX>(const std::index_sequence<IDX...>){
                reference_release_many(src[IDX]...);
                reference_wait_many(dst[IDX]...);
            }(std::make_index_sequence<SZ>{});        
        }
    };

    template <class T>
    struct MemoryReferenceLockX_NoBranch_Unique{};

    template <class T>
    struct MemoryReferenceLockX_NoBranch_Unique<MemoryReferenceLockInterface<T>>: MemoryReferenceLockXPInterface<MemoryReferenceLockX_NoBranch_Unique<MemoryReferenceLockInterface<T>>>{

        using base = MemoryReferenceLockInterface<T>;

        template <size_t SZ, std::enable_if_t<(SZ > 0), bool> = true>
        static inline auto transfer_try_many(std::array<const void *, SZ> dst, std::array<const void *, SZ> src) noexcept -> bool{

            return [&]<size_t ...IDX>(const std::index_sequence<IDX...>){
                return (base::transfer_try(dst[IDX], src[IDX]) & ...);   
            }(std::make_index_sequence<SZ>{});        
        }

        template <class ...Args, std::enable_if_t<std::conjunction_v<std::is_same<Args, const void *>...> && (sizeof...(Args) > 0), bool> = true>
        static inline auto acquire_try_many(Args ...args) noexcept -> bool{

            std::array<const void *, sizeof...(Args)> ptr_array{};
            const void ** last  = ptr_array.data(); 
            const void ** cap   = ptr_array.data() + sizeof...(Args);

            ([&]{
                *last = args;
                last += base::acquire_try(args);
            }(), ...);

            if (last == cap){
                return true;
            }

            *last = nullptr;

            for (size_t i = 0; i < sizeof...(Args); ++i){
                base::acquire_release(ptr_array[i]); //serial access at nullptr
            }

            return false;        
        }

        template <class ...Args, std::enable_if_t<std::conjunction_v<std::is_same<Args, const void *>...> && (sizeof...(Args) > 0), bool> = true>
        static inline void acquire_wait_many(Args ...args) noexcept{

            while (!acquire_try_many(args...));
        }

        template <class ...Args, std::enable_if_t<std::conjunction_v<std::is_same<Args, const void *>...> && (sizeof...(Args) > 0), bool> = true>
        static inline void acquire_release_many(Args ...args) noexcept{

            ((base::acquire_release(args)), ...);
        }

        template <size_t SZ, std::enable_if_t<(SZ > 0), bool> = true>
        static inline void acquire_transfer_wait_many(std::array<const void *, SZ> dst, std::array<const void *, SZ> src) noexcept{

            if (transfer_try_many(dst, src)){
                return;
            }

            [&]<size_t ...IDX>(const std::index_sequence<IDX...>){
                acquire_release_many(src[IDX]...);
                acquire_wait_many(dst[IDX]...);
            }(std::make_index_sequence<SZ>{});        
        }

        template <class ...Args, std::enable_if_t<std::conjunction_v<std::is_same<Args, const void *>...> && (sizeof...(Args) > 0), bool> = true>
        static inline auto reference_try_many(Args ...args) noexcept -> bool{

            std::array<const void *, sizeof...(Args)> ptr_array{};
            const void ** last  = ptr_array.data(); 
            const void ** cap   = ptr_array.data() + sizeof...(Args);

            ([&]{
                *last = args;
                last += base::reference_try(args);
            }(), ...);

            if (last == cap){
                return true;
            }

            *last = nullptr;

            for (size_t i = 0; i < sizeof...(Args); ++i){
                base::reference_release(ptr_array[i]); //serial access at nullptr
            }

            return false;           
        }

        template <class ...Args, std::enable_if_t<std::conjunction_v<std::is_same<Args, const void *>...> && (sizeof...(Args) > 0), bool> = true>
        static inline void reference_wait_many(Args ...args) noexcept{

            while (!reference_try_many(args...)){}
        }

        template <class ...Args, std::enable_if_t<std::conjunction_v<std::is_same<Args, const void *>...> && (sizeof...(Args) > 0), bool> = true>
        static inline void reference_release_many(Args ...args) noexcept{

            ((base::reference_release(args)), ...);
        }

        template <size_t SZ, std::enable_if_t<(SZ > 0), bool> = true>
        static inline void reference_transfer_wait_many(std::array<const void *, SZ> dst, std::array<const void *, SZ> src) noexcept{

            if (transfer_try_many(dst, src)){
                return;
            }

            [&]<size_t ...IDX>(const std::index_sequence<IDX...>){
                reference_release_many(src[IDX]...);
                reference_wait_many(dst[IDX]...);
            }(std::make_index_sequence<SZ>{});        
        }
    };

    template <class T, class MemRegionSize, class AddrBitSpace>
    struct MemoryAcquireReferenceLock{};

    template <class T, size_t MEM_REGION_SIZE, size_t ADDR_BIT_SPACE>
    struct MemoryAcquireReferenceLock<MemoryReferenceLockXInterface<T>, std::integral_constant<size_t, MEM_REGION_SIZE>, std::integral_constant<size_t, ADDR_BIT_SPACE>>: MemoryAcquireReferenceLockInterface<MemoryAcquireReferenceLock<MemoryReferenceLockXInterface<T>, std::integral_constant<size_t, MEM_REGION_SIZE>, std::integral_constant<size_t, ADDR_BIT_SPACE>>>{

        private:

            using base          = MemoryReferenceLockXInterface<T>;
            using abstract_t    = uint64_t;
            using poly_t        = uint8_t;

            static inline constexpr size_t ACTUAL_ADDR_BIT_SPACE    = (sizeof(abstract_t) - sizeof(poly_t)) * CHAR_BIT;
            static inline constexpr size_t POLY_BIT_SPACE           = sizeof(poly_t) * CHAR_BIT;
            static inline constexpr poly_t EMP_POLY_ID              = poly_t{0u};
            static inline constexpr poly_t ACQ_POLY_ID              = poly_t{1}; 
            static inline constexpr poly_t REF_POLY_ID              = poly_t{2};
            static inline constexpr abstract_t ABSTRACT_NULL        = 0u; 

            static_assert(POLY_BIT_SPACE + ADDR_BIT_SPACE <= sizeof(abstract_t) * CHAR_BIT);

            static constexpr auto region(const void * ptr) noexcept -> const void *{

                constexpr size_t BITMASK = ~(MEM_REGION_SIZE - 1); 
                return reinterpret_cast<const void *>(reinterpret_cast<uintptr_t>(ptr) & BITMASK);
            }

            static constexpr auto abstractize(uintptr_t addr, poly_t poly_id) noexcept -> abstract_t{

                return (static_cast<abstract_t>(addr) << POLY_BIT_SPACE) | static_cast<abstract_t>(poly_id);
            }

            static constexpr auto poly_id(abstract_t val) noexcept -> poly_t{

                return val & utility::low<abstract_t>(std::integral_constant<size_t, POLY_BIT_SPACE>{});
            }

            static constexpr auto numerical_addr(abstract_t val) noexcept -> uintptr_t{

                return val >> POLY_BIT_SPACE; 
            }

            static constexpr auto addr(abstract_t val) noexcept -> const void *{

                return reinterpret_cast<const void *>(numerical_addr(val));
            }

            static constexpr auto has_different_addr(abstract_t lhs, abstract_t rhs) noexcept -> bool{
                
                constexpr abstract_t CMP_MASK = utility::low<abstract_t>(std::integral_constant<size_t, ACTUAL_ADDR_BIT_SPACE>{}) << POLY_BIT_SPACE;
                return ((lhs & CMP_MASK) ^ (rhs & CMP_MASK)) != 0u;
            } 

            template <class ...Args, std::enable_if_t<std::conjunction_v<is_acqref<Args>...> && (sizeof...(Args) > 0), bool> = true> 
            static inline auto abstractize_regionize(Args ...args) noexcept -> std::array<abstract_t, sizeof...(Args)>{

                std::array<abstract_t, sizeof...(Args)> rs{};
                size_t sz = 0;

                ([&]{
                    if constexpr(is_acq_tag_v<decltype(args)>){
                        rs[sz++] = abstractize(reinterpret_cast<uintptr_t>(region(args.ptr)), ACQ_POLY_ID);
                    } else{
                        rs[sz++] = abstractize(reinterpret_cast<uintptr_t>(region(args.ptr)), REF_POLY_ID);
                    }
                }(), ...);

                return rs;
            }

            template <size_t SZ, std::enable_if_t<(SZ > 0), bool> = true>
            static inline void remove_duplicate(std::array<abstract_t, SZ>& arr) noexcept{
                
                static_assert(ABSTRACT_NULL == abstract_t{});

                utility::template_sort_arr(arr.data(), std::integral_constant<size_t, SZ>{});
                std::array<abstract_t, SZ + 1> tmp{};
                abstract_t * last = tmp.data();

                for (size_t i = 0; i < SZ; ++i){
                    *last = arr[i];

                    if (i == 0){
                        last += 1;
                    } else{
                        last += has_different_addr(arr[i], arr[i - 1]);
                    }
                }

                *last = ABSTRACT_NULL;
                std::memcpy(arr.data(), tmp.data(), SZ * sizeof(abstract_t));
            }

            template <size_t SZ>
            static inline void partition_acqref(std::array<abstract_t, SZ>& abs_arr, 
                                                std::array<const void *, SZ>& emp_arr,
                                                std::array<const void *, SZ + 1>& acq_arr,
                                                std::array<const void *, SZ + 1>& ref_arr) noexcept{
                
                std::array<const void **, 3> output_arr{emp_arr.data(), acq_arr.data(), ref_arr.data()};

                for (size_t i = 0; i < SZ; ++i){
                    *(output_arr[poly_id(abs_arr[i])]++) = addr(abs_arr[i]);
                }

                *output_arr[1] = nullptr;
                *output_arr[2] = nullptr;
            }
            
            template <class ...Args, std::enable_if_t<std::conjunction_v<is_acqref<Args>...> && (sizeof...(Args) > 0), bool> = true> 
            static inline auto to_ptr_array(std::tuple<Args...> tup) noexcept -> std::array<const void *, sizeof...(Args)>{

                std::array<const void *, sizeof...(Args)> rs{};
                size_t sz = 0;

                std::apply(tup, [&](auto e){ //compiler does not inline reference - 
                    rs[sz++] = e.ptr;
                });

                return rs;
            }

        public:

            template <class ...Args, std::enable_if_t<std::conjunction_v<is_acqref<Args>...> && (sizeof...(Args) > 0), bool> = true>
            static inline auto acqref_try_many(Args ...args) noexcept -> bool{
                
                std::array<abstract_t, sizeof...(Args)> abstract_args{abstractize_regionize(args...)};
                std::array<const void *, sizeof...(Args)> empty_args{};
                std::array<const void *, sizeof...(Args) + 1> acquire_args{};  
                std::array<const void *, sizeof...(Args) + 1> reference_args{};

                remove_duplicate(abstract_args);
                partition_acqref(abstract_args, empty_args, acquire_args, reference_args);

                if (!base::acquire_try_many(acquire_args.data())){
                    return false;
                }

                if (!base::reference_try_many(reference_args.data())){
                    base::acquire_release_many(acquire_args.data());
                    return false;
                }

                return true;
            }

            template <class ...Args, std::enable_if_t<std::conjunction_v<is_acqref<Args>...> && (sizeof...(Args) > 0), bool> = true>
            static inline void acqref_wait_many(Args ...args) noexcept{

                while (!acqref_try_many(args...)){}
            }

            template <class ...Args, std::enable_if_t<std::conjunction_v<is_acqref<Args>...> && (sizeof...(Args) > 0), bool> = true>
            static inline void release_many(Args ...args) noexcept{
      
                std::array<abstract_t, sizeof...(Args)> abstract_args{abstractize_regionize(args...)};
                std::array<const void *, sizeof...(Args)> empty_args{};
                std::array<const void *, sizeof...(Args) + 1> acquire_args{};  
                std::array<const void *, sizeof...(Args) + 1> reference_args{};

                remove_duplicate(abstract_args);
                partition_acqref(abstract_args, empty_args, acquire_args, reference_args);

                base::reference_release_many(reference_args.data());
                base::acquire_release_many(acquire_args.data());
            }

            template <size_t SZ, std::enable_if_t<(SZ > 0), bool> = true>
            static inline auto transfer_try_many(std::array<const void *, SZ> dst, std::array<const void *, SZ> src) noexcept -> bool{

                return base::transfer_try_many(dst, src);
            }

            template <class ...LHS_Args, class ...RHS_Args, std::enable_if_t<is_acqref_tup_pair_compatible_v<std::tuple<LHS_Args...>, std::tuple<RHS_Args...>>, bool> = true>
            static inline auto transfer_try_many(std::tuple<LHS_Args...> dst, std::tuple<RHS_Args...> src) noexcept -> bool{

                return transfer_try_many(to_ptr_array(dst), to_ptr_array(src));
            }

            template <class ...LHS_Args, class ...RHS_Args, std::enable_if_t<is_acqref_tup_pair_compatible_v<std::tuple<LHS_Args...>, std::tuple<RHS_Args...>>, bool> = true>
            static inline void transfer_wait_many(std::tuple<LHS_Args...> dst, std::tuple<RHS_Args...> src) noexcept{

                if (transfer_try_many(dst, src)){
                    return;
                }

                [&]<size_t ...IDX>(const std::index_sequence<IDX...>){
                    release_many(std::get<IDX>(src)...);
                    acqref_wait_many(std::get<IDX>(dst)...);
                }(std::make_index_sequence<sizeof...(LHS_Args)>{});
            }
    };