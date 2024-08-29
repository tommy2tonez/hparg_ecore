#ifndef __NETWORK_MEMCOMMIT_MEMPRESS_DISPATCHER_H__
#define __NETWORK_MEMCOMMIT_MEMPRESS_DISPATCHER_H__

#include "network_intmemcommit_dropbox.h"
#include "network_mempress.h" 

namespace dg::network_intmemcommit_mempress_dispatcher{

    //2hrs
    
    using event_loop_register_t = void (*)(void (*)(void) noexcept); 

    template <class ID, class DropBox, class MemPress>
    struct ForwardDelivery{};

    template <class ID, class T, class T1>
    struct ForwardDelivery<ID, dg::network_intmemcommit_dropbox::DropBoxInterface<T>, dg::network_mempress::UnboundedContainerInterface<T1>>{ //weird name 

        static void run() noexcept{

        }

        static void init(event_loop_register_t event_loop_register){

            event_loop_register(run);
        }
    };

    template <class ID, class DropBox, class MemPress>
    struct BackwardDelivery{};

    template <class ID, class T, class T1>
    struct BackwardDelivery<ID, dg::network_intmemcommit_dropbox::DropBoxInterface<T>, dg::network_mempress::UnboundedContainerInterface<T1>>{

        static void run() noexcept{

        }

        static void init(event_loop_register_t event_loop_register){

            event_loop_register(run);
        }
    };

    template <class ID, class DropBox, class MemPress>
    struct BackwardSignalDelivery{};

    template <class ID, class T, class T1>
    struct BackwardSignalDelivery<ID, dg::network_intmemcommit_dropbox::DropBoxInterface<T>, dg::network_mempress::UnboundedContainerInterface<T1>>{

    };


} 

#endif