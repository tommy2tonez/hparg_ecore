#ifndef __NETWORK_TILE_TLB_H__
#define __NETWORK_TILE_TLB_H__

namespace dg::network_tile_tlb{
    
    template <class T>
    struct TranslatorInterface{

        using vma_ptr_t = typename T::vma_ptr_t; 

        static void add(vma_ptr_t dst, vma_ptr_t src) noexcept{ //whether using direct hash_table or using vector_table + retranslation

            T::add(dst, src);
        } 

        static void remove(vma_ptr_t ptr) noexcept{
            
            T::remove(ptr);
        }

        static auto translate(vma_ptr_t ptr) noexcept -> vma_ptr_t{

            return T::translate(ptr);
        } 
    };

    template <class ID>
    struct AtomicTranslator: TranslatorInterface<AtomicTranslator<ID>>{

    };

    template <class ID>
    struct MtxTranslator: TranslatorInterface<MtxTranslator<ID>>{

    };

    static inline constexpr bool IS_ATOMIC_OPERATION_PREFERRED = true;

    template <class ID>
    using Translator = std::conditional_t<IS_ATOMIC_OPERATION_PREFERRED,
                                          AtomicTranslator<ID>,
                                          MtxTranslator<ID>>; 

}

#endif