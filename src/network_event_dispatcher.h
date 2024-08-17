#ifndef __EVENT_DISPATCHER_H__
#define __EVENT_DISPATCHER_H__

#include <stdint.h>
#include <stddef.h>
#include <network_addr_lookup.h>
#include "network_function_concurrent_buffer.h"
#include "network_tile_member_access.h"  
#include "network_producer_consumer.h"

namespace dg::event_dispatcher{
    
    template <class ...Args>
    struct tags{};

    template <class T, size_t MAX_DISPATCH_SIZE>
    struct ForwardDispatcher{

        using self = ForwardDispatcher; 
        
        template <class TileMemberAccess>
        static void dispatch_leaf(void ** events, const TileMemberAccess) noexcept{

        } 

        template <class TileMemberAccess>
        static void dispatch_mono(void ** events, const TileMemberAccess) noexcept{

        } 

        template <class TileMemberAccess>
        static void dispatch_pair(void ** events, const TileMemberAccess) noexcept{

        }

        template <class TileMemberAccess>
        static void dispatch_dair(void ** events, const TileMemberAccess) noexcept{

        }

        template <class TileMemberAccess>
        static void dispatch_uacm(void ** events, const TileMemberAccess) noexcept{

        }

        static void dispatch(void ** events, size_t sz) noexcept{

            using namespace dg::network_tile_member_access; 
            std::array<void *, TYPE_COUNT> ptr_array{};

            for (size_t i = 0; i < sz; ++i){
                *(ptr_array[tile_id(events[i])]++) = events[i];
            }

            for (size_t i = 0; i < TYPE_COUNT; ++i){
                *ptr_array[i] = nullptr;
            }

            for (size_t i = 0; i < TYPE_COUNT; ++i){
                switch (i){
                    case leaf_8_id:
                        dispatch_leaf_8(ptr_array[leaf_8_id]);
                        break;
                    case mono_8_id:
                        dispatch_mono_8(ptr_array[mono_8_id]);
                        break;
                    case uacm_8_id:
                        dispatch_uacm_8(ptr_array[uacm_8_id]);
                        break;
                    case pacm_8_id:
                        dispatch_pacm_8(ptr_array[pacm_8_id]);
                        break;
                    case dair_8_id:
                        dispatch_dair(ptr_array[dair_8_id]);
                        break;
                    case leaf_16_id:
                        dispatch_leaf_16(ptr_array[leaf_16_id]);
                        break;
                    case mono_16_id:
                        dispatch_mono_16(ptr_array[mono_16_id]);
                        break;
                    case uacm_16_id:
                        dispatch_uacm_16(ptr_array[uacm_16_id]);
                        break;
                    case pacm_16_id:
                        dispatch_pacm_16(ptr_array[pacm_16_id]);
                        break;
                    case dair_16_id:
                        dispatch_dair_16(ptr_array[dair_16_id]);
                        break;
                    default:
                        std::abort();
                        break;
                }
            }
        }
    };

    //attempt to backward
    //if backwardable:      - backward + update src_grad_ver_ctrl + update dst_grad_idx + zero_grad if dst_grad_idx == width +  ping backward
    //if not backwardable   - discard
    
    template <class T, size_t MAX_DISPATCH_SIZE>
    struct BackwardDispatcher{

        using self = BackwardDispatcher; 

        template <class TileMemberAccess>
        static void dispatch_leaf(void ** events, const TileMemberAccess) noexcept{

        } 

        template <class TileMemberAccess>
        static void dispatch_mono(void ** events, const TileMemberAccess) noexcept{

        } 

        template <class TileMemberAccess>
        static void dispatch_pair(void ** events, const TileMemberAccess) noexcept{

        }

        template <class TileMemberAccess>
        static void dispatch_dair(void ** events, const TileMemberAccess) noexcept{

        }

        template <class TileMemberAccess>
        static void dispatch_uacm(void ** events, const TileMemberAccess) noexcept{

        }

        static void dispatch(void ** events, size_t sz) noexcept{

            using namespace dg::network_tile_member_access; 
            std::array<void *, TYPE_COUNT> ptr_array{};

            for (size_t i = 0; i < sz; ++i){
                *(ptr_array[tile_id(events[i])]++) = events[i];
            }

            for (size_t i = 0; i < TYPE_COUNT; ++i){
                *ptr_array[i] = nullptr;
            }

            for (size_t i = 0; i < TYPE_COUNT; ++i){
                switch (i){
                    case leaf_8_id:
                        dispatch_leaf_8(ptr_array[leaf_8_id]);
                        break;
                    case mono_8_id:
                        dispatch_mono_8(ptr_array[mono_8_id]);
                        break;
                    case uacm_8_id:
                        dispatch_uacm_8(ptr_array[uacm_8_id]);
                        break;
                    case pacm_8_id:
                        dispatch_pacm_8(ptr_array[pacm_8_id]);
                        break;
                    case dair_8_id:
                        dispatch_dair(ptr_array[dair_8_id]);
                        break;
                    case leaf_16_id:
                        dispatch_leaf_16(ptr_array[leaf_16_id]);
                        break;
                    case mono_16_id:
                        dispatch_mono_16(ptr_array[mono_16_id]);
                        break;
                    case uacm_16_id:
                        dispatch_uacm_16(ptr_array[uacm_16_id]);
                        break;
                    case pacm_16_id:
                        dispatch_pacm_16(ptr_array[pacm_16_id]);
                        break;
                    case dair_16_id:
                        dispatch_dair_16(ptr_array[dair_16_id]);
                        break;
                    default:
                        std::abort();
                        break;
                }
            }
        }
    };

    //lock rcu + read bit_control 
    //if initialized - discard, 
    //if not initialized - backward ping  

    template <class T, size_t MAX_DISPATCH_SIZE>
    struct ForwardPingDispatcher{

        template <class TileMemberAccess>
        static void dispatch_leaf(void ** events, const TileMemberAccess) noexcept{

        } 

        template <class TileMemberAccess>
        static void dispatch_mono(void ** events, const TileMemberAccess) noexcept{

        } 

        template <class TileMemberAccess>
        static void dispatch_pair(void ** events, const TileMemberAccess) noexcept{

        }

        template <class TileMemberAccess>
        static void dispatch_dair(void ** events, const TileMemberAccess) noexcept{

        }

        template <class TileMemberAccess>
        static void dispatch_uacm(void ** events, const TileMemberAccess) noexcept{

        }

        static void dispatch(void ** events, size_t sz) noexcept{

            using namespace dg::network_tile_member_access; 
            std::array<void *, TYPE_COUNT> ptr_array{};

            for (size_t i = 0; i < sz; ++i){
                *(ptr_array[tile_id(events[i])]++) = events[i];
            }

            for (size_t i = 0; i < TYPE_COUNT; ++i){
                *ptr_array[i] = nullptr;
            }

            for (size_t i = 0; i < TYPE_COUNT; ++i){
                switch (i){
                    case leaf_8_id:
                        dispatch_leaf_8(ptr_array[leaf_8_id]);
                        break;
                    case mono_8_id:
                        dispatch_mono_8(ptr_array[mono_8_id]);
                        break;
                    case uacm_8_id:
                        dispatch_uacm_8(ptr_array[uacm_8_id]);
                        break;
                    case pacm_8_id:
                        dispatch_pacm_8(ptr_array[pacm_8_id]);
                        break;
                    case dair_8_id:
                        dispatch_dair(ptr_array[dair_8_id]);
                        break;
                    case leaf_16_id:
                        dispatch_leaf_16(ptr_array[leaf_16_id]);
                        break;
                    case mono_16_id:
                        dispatch_mono_16(ptr_array[mono_16_id]);
                        break;
                    case uacm_16_id:
                        dispatch_uacm_16(ptr_array[uacm_16_id]);
                        break;
                    case pacm_16_id:
                        dispatch_pacm_16(ptr_array[pacm_16_id]);
                        break;
                    case dair_16_id:
                        dispatch_dair_16(ptr_array[dair_16_id]);
                        break;
                    default:
                        std::abort();
                        break;
                }
            }
        }
    };

    //pong_state = 0, 1, ...
    //pong_state = 0 is set by forward_dispatcher
    //lock rcu
    //if not intialized - if pong_countdown > 1 decrease pong_countdown 
    //                  - [[fall_through]] if pong_countdown == 1, send forward request 
    //if initialized - discard

    template <class T, size_t MAX_DISPATCH_SIZE>
    struct ForwardPongDispatcher{

        template <class TileMemberAccess>
        static void dispatch_leaf(void ** events, const TileMemberAccess) noexcept{

        } 

        template <class TileMemberAccess>
        static void dispatch_mono(void ** events, const TileMemberAccess) noexcept{

        } 

        template <class TileMemberAccess>
        static void dispatch_pair(void ** events, const TileMemberAccess) noexcept{

        }

        template <class TileMemberAccess>
        static void dispatch_dair(void ** events, const TileMemberAccess) noexcept{

        }

        template <class TileMemberAccess>
        static void dispatch_uacm(void ** events, const TileMemberAccess) noexcept{

        }

        static void dispatch(void ** events, size_t sz) noexcept{

            using namespace dg::network_tile_member_access; 
            std::array<void *, TYPE_COUNT> ptr_array{};

            for (size_t i = 0; i < sz; ++i){
                *(ptr_array[tile_id(events[i])]++) = events[i];
            }

            for (size_t i = 0; i < TYPE_COUNT; ++i){
                *ptr_array[i] = nullptr;
            }

            for (size_t i = 0; i < TYPE_COUNT; ++i){
                switch (i){
                    case leaf_8_id:
                        dispatch_leaf_8(ptr_array[leaf_8_id]);
                        break;
                    case mono_8_id:
                        dispatch_mono_8(ptr_array[mono_8_id]);
                        break;
                    case uacm_8_id:
                        dispatch_uacm_8(ptr_array[uacm_8_id]);
                        break;
                    case pacm_8_id:
                        dispatch_pacm_8(ptr_array[pacm_8_id]);
                        break;
                    case dair_8_id:
                        dispatch_dair(ptr_array[dair_8_id]);
                        break;
                    case leaf_16_id:
                        dispatch_leaf_16(ptr_array[leaf_16_id]);
                        break;
                    case mono_16_id:
                        dispatch_mono_16(ptr_array[mono_16_id]);
                        break;
                    case uacm_16_id:
                        dispatch_uacm_16(ptr_array[uacm_16_id]);
                        break;
                    case pacm_16_id:
                        dispatch_pacm_16(ptr_array[pacm_16_id]);
                        break;
                    case dair_16_id:
                        dispatch_dair_16(ptr_array[dair_16_id]);
                        break;
                    default:
                        std::abort();
                        break;
                }
            }
        }
    };
    
}

#endif