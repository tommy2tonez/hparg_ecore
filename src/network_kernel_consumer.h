#ifndef __NETWORK_KERNEL_CONSUMER_H__
#define __NETWORK_KERNEL_CONSUMER_H__

namespace dg::kernel_consumer{

    //throttle + group signals (send as acked_packets)
    //group tile_injection requests (send as no-acked - should be user-configurable - tile-level logic)
    //tile_injection == backward or forward 
    //forward + backward transportation optimization should be a base component of kernel router

    //forward has a unique ping ID <operatable_idx, vma_ptr_t> (this is const qualified - if operatable_idx is 1-1 to data lifetime)
    //forward request relays the packet at some site - not necessarily directly from the requestee (this is necessary - not optimizable - this should be a <logic_component> extended by the kernel)

    //backward does not have unique ping ID
    //backward offloading by peeking current version ID of neighbor -> send f(g(x)) = x request to peers

    //this seems to be an allocator + scan problem 
    //an attempt to solve the problem here may interfere with the optimization step
}

#endif