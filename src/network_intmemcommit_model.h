#ifndef __DG_NETWORK_MEMCOMMIT_MODEL_H__
#define __DG_NETWORK_MEMCOMMIT_MODEL_H__

#include <stdint.h>
#include <stdlib.h>
#include <tuple>

namespace dg::network_intmemcommit_taxonomy{

    using memory_event_t            = uint8_t;

    enum memory_event_option: memory_event_t{
        forward_ping_signal                 = 0u,  //signal initialization
        forward_pong_request                = 1u,  //request pong post-initialization
        forward_pingpong_request            = 2u,  //signal initialization + request pong post-initialization
        forward_pong_signal                 = 3u,  //signal pong
        forward_ready_signal                = 4u,  //signal initable if pong_countdown == 0u //unprio by scan
        forward_init_signal                 = 5u,  //decay initable -> init if inbound, response -> init if outbound
        forward_load_request                = 6u,  //request injection (decay initable -> request)
        forward_load_response               = 7u,  //resolve injection_response (decay request -> response) //prio by scan
        forward_load_redirect_request       = 8u,  //decay request -> redirect_request (send peers requesting vma_ptr_t), decay redirect request -> redirect request 
        backward_ready_signal               = 9u,  //decay forward_init -> ready_signal if crit, decay backward_do_signal -> ready //unprio by scan
        backward_do_signal                  = 10u, //decay response -> do, decay ready -> do
        backward_load_request               = 11u, //request injection (decay backward_signal -> request)
        backward_load_response              = 12u, //resolve injection_response (decay request -> response)  //prio by scan
        backward_load_redirect_request      = 13u  //decay request -> redirect_request (send peers f(g(x)) = x request), decay redirect_request -> redirect_request
    };
}

namespace dg::network_intmemcommit_factory{

    //1hr
    
    using namespace dg::network_intmemcommit_taxonomy;
    using vma_ptr_t                 = uint64_t;
    using virtual_memory_event_t    = uint64_t;

    auto make_event_forward_ping_signal(vma_ptr_t signalee) noexcept -> virtual_memory_event_t{

    }

    auto make_event_forward_pong_request(vma_ptr_t requestee, vma_ptr_t requestor) noexcept -> virtual_memory_event_t{

    }

    auto make_event_forward_pingpong_request(vma_ptr_t requestee, vma_ptr_t requestor) noexcept -> virtual_memory_event_t{

    }

    auto make_event_forward_pong_signal(vma_ptr_t signalee, vma_ptr_t signaler) noexcept -> virtual_memory_event_t{

    } 

    auto make_event_forward_init_signal(vma_ptr_t signalee) noexcept -> virtual_memory_event_t{

    } 

    auto make_event_forward_load_request(vma_ptr_t injector, vma_ptr_t injectee) noexcept -> virtual_memory_event_t{

    }

    auto make_event_forward_load_response(vma_ptr_t injectee, vma_ptr_t injector) noexcept -> virtual_memory_event_t{

    }

    auto make_event_forward_load_redirect_request(vma_ptr_t injector, vma_ptr_t injectee) noexcept -> virtual_memory_event_t{

    } 

    auto make_event_backward_ready_signal(vma_ptr_t signalee, vma_ptr_t signaler) noexcept -> virtual_memory_event_t{

    }

    auto make_event_backward_do_signal(vma_ptr_t signalee, vma_ptr_t signaler) noexcept -> virtual_memory_event_t{

    } 

    auto make_event_backward_load_request(vma_ptr_t injector, vma_ptr_t injectee) noexcept -> virtual_memory_event_t{

    }

    auto make_event_backward_load_response(vma_ptr_t injectee, vma_ptr_t injector) noexcept -> virtual_memory_event_t{

    }

    auto make_event_backward_load_redirect_request(vma_ptr_t injector, vma_ptr_t injectee) noexcept -> virtual_memory_event_t{

    } 

    auto read_event_forward_ping_signal(virtual_memory_event_t) noexcept -> std::tuple<vma_ptr_t>{

    }

    auto read_event_forward_pong_request(virtual_memory_event_t) noexcept -> std::tuple<vma_ptr_t, vma_ptr_t>{

    } 

    auto read_event_forward_pingpong_request(virtual_memory_event_t) noexcept -> std::tuple<vma_ptr_t, vma_ptr_t>{

    }

    auto read_event_forward_pong_signal(virtual_memory_event_t) noexcept -> std::tuple<vma_ptr_t, vma_ptr_t>{

    }

    auto read_event_forward_init_signal(virtual_memory_event_t) noexcept -> std::tuple<vma_ptr_t>{

    }

    auto read_event_forward_load_request(virtual_memory_event_t) noexcept -> std::tuple<vma_ptr_t, vma_ptr_t>{

    }

    auto read_event_forward_load_response(virtual_memory_event_t) noexcept -> std::tuple<vma_ptr_t, vma_ptr_t>{

    }

    auto read_event_forward_load_redirect_request(virtual_memory_event_t) noexcept-> std::tuple<vma_ptr_t, vma_ptr_t>{

    }

    auto read_event_backward_ready_signal(virtual_memory_event_t) noexcept -> std::tuple<vma_ptr_t, vma_ptr_t>{

    }

    auto read_event_backward_do_signal(virtual_memory_event_t) noexcept -> std::tuple<vma_ptr_t, vma_ptr_t>{

    }

    auto read_event_backward_load_request(virtual_memory_event_t) noexcept -> std::tuple<vma_ptr_t, vma_ptr_t>{

    }

    auto read_event_backward_load_response(virtual_memory_event_t) noexcept -> std::tuple<vma_ptr_t, vma_ptr_t>{

    }

    auto read_event_backward_load_redirect_request(virtual_memory_event_t) noexcept -> std::tuple<vma_ptr_t, vma_ptr_t>{

    }

    auto read_event_taxonomy(virtual_memory_event_t) noexcept -> memory_event_t{

    }
}

#endif