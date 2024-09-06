#ifndef __NETWORK_TILE_TLB_H__
#define __NETWORK_TILE_TLB_H__

namespace dg::network_tile_tlb{

    //this should be a component - dispatcher only does internal address - this is hard 
    //there are so many design questions to be answered 
    //where do I do global_addr validation
    //how is the reference of global_addr solved

    //if at at setter - getter level 
    //then consistency of setter - getter should be guaranteed by translation_block_guard
    //who is responsible for calling translation_block_guard
    //should translation_block_guard be recursive_lock 

    //if at the dispatcher level (that the dispatcher is responsible for doing tile_addr checking)
    //then translation_block_guard is no longer necessary nor global_get_setter - which is very buggy

    template <class T>
    struct TileTLBInterface{
        
        static auto add(uma_ptr_t key, uma_ptr_t value) noexcept -> exception_t{

            T::add(key, value);
        }

        static void add_nothrow(uma_ptr_t key, uma_ptr_t value) noexcept{

            T::add_nothrow(key, value);
        }

        static auto translate(uma_ptr_t key) noexcept -> std::optional<uma_ptr_t>{

            return T::translate(key);
        }
        
        static void remove(uma_ptr_t key) noexcept{

            T::remove(key);
        }
    };
}

#endif